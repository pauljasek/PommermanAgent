# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import os
import sys
from threading import Thread

from collections import defaultdict


import tensorflow.contrib.slim as slim

from tensorflow.python import debug as tf_debug

# import dmlab30
import environments
import numpy as np
import py_process
import sonnet as snt
import tensorflow as tf
import vtrace

try:
    import dynamic_batching
except tf.errors.NotFoundError:
    tf.logging.warning('Running without dynamic batching.')

from six.moves import range

nest = tf.contrib.framework.nest

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = tf.app.flags.FLAGS

    flags.DEFINE_string('logdir', 'agents/pretrained/', 'TensorFlow log directory.')
    flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

    # Flags used for testing.
    flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

    # Flags used for distributed training.
    flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
    flags.DEFINE_integer('rank', 0, 'Rank.')
    flags.DEFINE_integer('processes', 12, 'Number of processes to start.')
    flags.DEFINE_integer('agent1', 0, 'Index of agent 1.')
    flags.DEFINE_integer('agent2', 0, 'Index of agent 2.')
    flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                      'Job name. Ignored when task is set to -1.')

    # Training.
    flags.DEFINE_integer('total_environment_frames', int(100e9),
                         'Total environment frames to train for.')
    flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
    flags.DEFINE_integer('save_interval', 25000000, 'Number of frames between parameter mutations.')
    flags.DEFINE_integer('burnin', 3000000000, 'Number of frames before first parameter mutation.')
    flags.DEFINE_float('threshold', 0.1, 'Parameter copy threshold.')
    flags.DEFINE_integer('num_agents', 6, 'Number of agents in population.')
    flags.DEFINE_integer('num_adversarial_agents', 2, 'Number of adversarial agents in population.')
    flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
    flags.DEFINE_integer('unroll_length', 50, 'Unroll length in agent steps.')
    flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')
    flags.DEFINE_integer('seed', 1, 'Random seed.')

    # Loss settings.
    flags.DEFINE_float('entropy_cost', 0.2, 'Entropy cost/multiplier.')
    flags.DEFINE_float('message_entropy_multiplier', 0.001, 'Additional multiplier applied to message')
    flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
    flags.DEFINE_float('discounting', .99, 'Discounting factor.')
    flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                      'Reward clipping.')

    # Optimizer settings.
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
    flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
    flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
    flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')




# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'agent_state env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


def is_single_machine():
    return FLAGS.task == -1


def fc(input, output_shape, activation_fn=tf.nn.relu, name="fc"):
    output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
    return output

MULTI_ENT = False
SMALL_NETWORK = False
if SMALL_NETWORK:
    LSTM_SIZE = 32
else:
    LSTM_SIZE = 128

class Agent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(Agent, self).__init__(name='agent')

        self._num_actions = num_actions

        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(LSTM_SIZE)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, observations = env_output

        #board, feature = observations

        board, bomb_blast_strength, bomb_life, feature = observations

        board = tf.one_hot(
            board,
            14,
            name='one_hot_board'
        )

        board = tf.concat([board, tf.expand_dims(bomb_blast_strength, axis=-1), tf.expand_dims(bomb_life, axis=-1)], axis=-1)

        board_shape = tf.shape(board)
        batch_size = board_shape[0]
        x_dim = board_shape[1]
        y_dim = board_shape[2]
        f_dim = board_shape[3]

        feature_size = tf.shape(feature)[1]

        if SMALL_NETWORK:
            conv_out = snt.Conv2D(16, 3, stride=1, padding='VALID')(board)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(32, 3, stride=1, padding='VALID')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(32, 3, stride=1, padding='VALID')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(32, 3, stride=1, padding='VALID')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            f = tf.reshape(conv_out, [batch_size, 32])
        else:
            conv_out = snt.Conv2D(16, 3, stride=1, padding='VALID')(board)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(32, 3, stride=1, padding='VALID')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(64, 3, stride=1, padding='VALID')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = snt.Conv2D(128, 3, stride=1, padding='VALID')(conv_out)
            conv_out = tf.nn.relu(conv_out)
            f = tf.reshape(conv_out, [batch_size, 128])

        #conv_out = snt.Conv2D(256, 3, stride=1, padding='VALID')(conv_out)
        #conv_out = tf.nn.relu(conv_out)

        f = tf.concat([f, feature], axis=-1)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        last_action_list = [last_action[:, i] for i in range(len(self._num_actions))]
        one_hot_last_action = tf.concat([tf.one_hot(a, n) for a, n in zip(last_action_list, self._num_actions)],
                                        axis=-1)
        return tf.concat([f, clipped_reward, one_hot_last_action], axis=1)

    def _head(self, core_output):
        logits = snt.Linear(sum(self._num_actions))(
            core_output)

        policy_logits = tf.split(logits, self._num_actions, axis=-1, name='policy_logits')
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.concat(list(map(lambda logits: tf.multinomial(logits, num_samples=1,
                                                                      output_dtype=tf.int32), policy_logits)), -1,
                               name='new_action')
        # new_action = tf.squeeze(new_action, 1, name='new_action')

        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                  (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            if True:
                core_output, core_state = self._core(input_, core_state)
                core_output_list.append(core_output)
            else:
                core_output_list.append(input_)

        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state


def build_actor(agent1, agent2, env):
    """Builds the actor loop."""
    # Initial values.
    initials = []
    for i, env_initial in enumerate(env.initial()):
        agent = agent1 if i % 2 == 0 else agent2
        initial_env_output, initial_env_state = env_initial
        initial_agent_state = agent.initial_state(1)
        initial_action = tf.zeros([1, 3], dtype=tf.int32)
        # print(initial_env_output.observation)
        dummy_agent_output, _ = agent1(
            (initial_action,
             nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
            initial_agent_state)
        initial_agent_output = nest.map_structure(
            lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)
        initials.append((initial_env_state, initial_env_output, initial_agent_state,
                         initial_agent_output))

    # All state that needs to persist across training iterations. This includes
    # the last environment output, agent state and last agent output. These
    # variables should never go on the parameter servers.
    def create_state(t):
        # Creates a unique variable scope to ensure the variable name is unique.
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    persistent_state = nest.map_structure(
        create_state, initials)

    def step(inputs_, unused_i):
        """Steps through the agent and the environment."""
        agent_outputs = []
        for i, input_ in enumerate(inputs_):
            agent = agent1 if i % 2 == 0 else agent2

            env_state, env_output, agent_state, agent_output = input_

            # Run agent.
            action = agent_output[0]
            # print(action)
            batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                    env_output)
            agent_output, agent_state = agent((action, batched_env_output), agent_state)
            agent_outputs.append((agent_output, agent_state))

        env_outputs = env.step([agent_output[0][0] for agent_output in agent_outputs],
                               [input_[0] for input_ in inputs_])

        return [(env_state, env_output, agent_state, agent_output) for
                (env_output, env_state), (agent_output, agent_state) in zip(env_outputs, agent_outputs)]

    # Run the unroll. `read_value()` is needed to make sure later usage will
    # return the first values and not a new snapshot of the variables.
    first_valuess = nest.map_structure(lambda v: v.read_value(), persistent_state)

    # Use scan to apply `step` multiple times, therefore unrolling the agent
    # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
    # the output of each call of `step` as input of the subsequent call of `step`.
    # The unroll sequence is initialized with the agent and environment states
    # and outputs as stored at the end of the previous unroll.
    # `output` stores lists of all states and outputs stacked along the entire
    # unroll. Note that the initial states and outputs (fed through `initializer`)
    # are not in `output` and will need to be added manually later.
    outputs = tf.scan(step, tf.range(FLAGS.unroll_length), first_valuess)

    # Update persistent state with the last output from the loop.
    assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                    persistent_state, outputs)

    # The control dependency ensures that the final agent and environment states
    # and outputs are stored in `persistent_state` (to initialize next unroll).
    with tf.control_dependencies(nest.flatten(assign_ops)):
        rets = []
        for output, first_values in zip(outputs, first_valuess):
            _, env_outputs, _, agent_outputs = output
            _, first_env_output, first_agent_state, first_agent_output = first_values

            # Remove the batch dimension from the agent state/output.
            first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
            first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
            agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

            # Concatenate first output and the unroll along the time dimension.
            full_agent_outputs, full_env_outputs = nest.map_structure(
                lambda first, rest: tf.concat([[first], rest], 0),
                (first_agent_output, first_env_output), (agent_outputs, env_outputs))

            rets.append(ActorOutput(
                agent_state=first_agent_state,
                env_outputs=full_env_outputs, agent_outputs=full_agent_outputs))

        # No backpropagation should be done here.
        return nest.map_structure(tf.stop_gradient, rets)


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits, entropy_cost):
    res = 0

    for i in range(len(logits)):
        policy = tf.nn.softmax(logits[i])
        log_policy = tf.nn.log_softmax(logits[i])

        # Apply a form of weighted entropy to the output logits
        if MULTI_ENT:
            if i == 0:
                policy *= tf.expand_dims(tf.expand_dims(tf.stack([entropy_cost[0],
                                                                  entropy_cost[1],
                                                                  entropy_cost[1],
                                                                  entropy_cost[1],
                                                                  entropy_cost[1],
                                                                  entropy_cost[2]], axis=0), axis=0), axis=0)
            else:
                policy *= entropy_cost[3]
        else:
            policy *= entropy_cost[0]
            if i != 0:
                policy *= FLAGS.message_entropy_multiplier

        entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
        entropy_loss = -tf.reduce_sum(entropy_per_timestep)
        #if i > 0:
        #    entropy_loss *= FLAGS.message_entropy_multiplier
        res += entropy_loss
    return res


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions[:, :, i], logits=logits[i]) for i in range(len(logits)))
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return tf.reduce_sum(policy_gradient_loss_per_timestep)


def build_learner(agent, agent_state, env_outputs, agent_outputs, optimizer, num_env_frames, learning_rate, epsilon, entropy_cost, index=0):
    """Builds the learner loop.

    Args:
      agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
        `unroll` call for computing the outputs for a whole trajectory.
      agent_state: The initial agent state for each sequence in the batch.
      env_outputs: A `StepOutput` namedtuple where each field is of shape
        [T+1, ...].
      agent_outputs: An `AgentOutput` namedtuple where each field is of shape
        [T+1, ...].

    Returns:
      A tuple of (done, infos, and environment frames) where
      the environment frames tensor causes an update.
    """
    learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs,
                                      agent_state)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.
    agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
    rewards, infos, done, _ = nest.map_structure(
        lambda t: t[1:], env_outputs)
    learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

    if FLAGS.reward_clipping == 'abs_one':
        clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    elif FLAGS.reward_clipping == 'soft_asymmetric':
        squeezed = tf.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

    discounts = tf.to_float(~done) * FLAGS.discounting

    # Compute V-trace returns and weights.
    # Note, this is put on the CPU because it's faster than on GPU. It can be
    # improved further with XLA-compilation or with a custom TensorFlow operation.
    with tf.device('/cpu'):
        vtrace_returns = vtrace.from_logits(
            behaviour_policy_logits=agent_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=agent_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value)

    # Compute loss as a weighted sum of the baseline loss, the policy gradient
    # loss and an entropy regularization term.
    total_loss = compute_policy_gradient_loss(
        learner_outputs.policy_logits, agent_outputs.action,
        vtrace_returns.pg_advantages)
    total_loss += FLAGS.baseline_cost * compute_baseline_loss(
        vtrace_returns.vs - learner_outputs.baseline)
    total_loss += compute_entropy_loss(
        learner_outputs.policy_logits, entropy_cost)

    # Optimization
    train_op = optimizer.minimize(total_loss)

    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        num_env_frames_and_train = num_env_frames.assign_add(
            FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)

    # Adding a few summaries.
    tf.summary.scalar('total_loss/' + str(index), total_loss)
    #tf.summary.histogram('action', agent_outputs.action[0][0])
    #tf.summary.histogram('message_1', agent_outputs.action[0][1])
    #tf.summary.histogram('message_2', agent_outputs.action[0][2])

    return done, infos, num_env_frames_and_train


def create_environment(config, is_test=False):
    """Creates an environment wrapped in a `FlowEnvironment`."""
    p = py_process.PyProcess(environments.PyProcessPommerMan, config=config)
    return environments.FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""

    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get('collections', None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope('', custom_getter=getter) as vs:
        yield vs


def train():
    """Train."""

    if is_single_machine():
        local_job_device = ''
        shared_job_device = ''
        is_actor_fn = lambda i, j: True
        is_learner = True
        global_variable_device = '/gpu'
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        if False:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            #rank = FLAGS.rank


            if rank == 0:
                job_name = 'learner'
                task = 0
            else:
                job_name = 'actor'
                task = rank - 1
        else:
            job_name = FLAGS.job_name
            task = FLAGS.task

        agent1 = task // FLAGS.num_agents
        agent2 = task % FLAGS.num_agents

        local_job_device = '/job:%s/task:%d' % (job_name, task)
        shared_job_device = '/job:learner/task:0'
        is_actor_fn = lambda i, j: job_name == 'actor' and i == agent1 and j == agent2
        is_learner = job_name == 'learner'

        # Placing the variable on CPU, makes it cheaper to send it to all the
        # actors. Continual copying the variables from the GPU is slow.
        global_variable_device = shared_job_device + '/cpu'

        # cluster = tf.train.ClusterSpec({
        #    'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_agents ** 2)],
        #    'learner': ['localhost:8000']
        # })

        # cluster = tf.train.ClusterSpec({
        #    'actor': ['10.1.2.25:8000', '10.1.2.24:8000', '10.1.2.15:8000'],
        #    'learner': ['10.1.2.22:8000']
        # })

        #cluster = tf.train.ClusterSpec({
        #    'actor': ['10.1.2.25:%d' % (8001 + i) for i in range(FLAGS.num_agents ** 2)] + [
        #        '10.1.2.24:%d' % (8001 + i) for i in range(FLAGS.num_agents ** 2)] + [
        #                 '10.1.2.15:%d' % (8001 + i) for i in range(FLAGS.num_agents ** 2)],
        #    'learner': ['10.1.2.22:8000']
        #})

        '''
        cluster = tf.train.ClusterSpec({
            'actor': ['10.1.2.25:%d' % (8000 + i) for i in range(FLAGS.num_agents ** 2)] +
                     ['10.1.2.24:%d' % (8000 + i) for i in range(FLAGS.num_agents ** 2)],
            'learner': ['10.1.2.22:8000']
        }) '''

        nodefile = FLAGS.logdir + '/nodeslist.txt'
        with open(nodefile, 'r') as f:
            nodes = f.readlines()
        nodes = [x.strip().split('.')[0] for x in nodes]

        #nodes = comm.allgather(MPI.Get_processor_name())
        counts = defaultdict(int)

        if False:
            processes = []
            for i, node in enumerate(nodes):
                processes.append(node + ':' + str(14000 + counts[node]))
                counts[node] += 1
        else:
            processes = []
            for i, node in enumerate(nodes):
                if i == 0:
                    processes.append(node + ':14000')
                else:
                    for j in range(FLAGS.processes):
                        processes.append(node + ':' + str(14000 + j))

        cluster = tf.train.ClusterSpec({
            'actor': processes[1:],
            'learner': [processes[0]]
        })

        import socket

        print(job_name, task, socket.gethostname())
        print({
            'actor': processes[1:],
            'learner': [processes[0]]
        })

        sys.stdout.flush()

        server = tf.train.Server(cluster, job_name=job_name,
                                 task_index=task)

        print('created server')

        sys.stdout.flush()

        filters = [shared_job_device, local_job_device]



    # Only used to find the actor output structure.
    with tf.Graph().as_default():
        agent = Agent((6, 8, 8))
        env = create_environment({'adversarial': False})
        structure = build_actor(agent, agent, env)
        structure = [structure[0], structure[2]]
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    with tf.Graph().as_default(), \
         tf.device(local_job_device + '/cpu'), \
         pin_global_variables(global_variable_device):
        tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            agents = []
            queues = []
            for i in range(FLAGS.num_agents):
                with tf.variable_scope('agent_' + str(i)):
                    agent = Agent((6, 8, 8))

                queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer_' + str(i))

                if is_single_machine() and 'dynamic_batching' in sys.modules:
                    # For single machine training, we use dynamic batching for improved GPU
                    # utilization. The semantics of single machine training are slightly
                    # different from the distributed setting because within a single unroll
                    # of an environment, the actions may be computed using different weights
                    # if an update happens within the unroll.
                    old_build = agent._build

                    @dynamic_batching.batch_fn
                    def build(*args):
                        with tf.device('/gpu'):
                            return old_build(*args)

                    tf.logging.info('Using dynamic batching.')
                    agent._build = build

                agents.append(agent)
                queues.append(queue)

        # Build actors and ops to enqueue their output.
        enqueue_ops = [[] for i in range(FLAGS.num_agents)]
        for i in range(FLAGS.num_agents):
            for j in range(0, FLAGS.num_agents):
                if is_actor_fn(i, j):
                    tf.logging.info('Creating actor %d %d', i, j)
                    config = {'adversarial': False}
                    #if i >= FLAGS.num_agents - FLAGS.num_adversarial_agents:
                    #    config['adversarial'] = True
                    env = create_environment(config)
                    actor_output = build_actor(agents[i], agents[j], env)
                    actor1_output = [actor_output[0], actor_output[2]]
                    # actor2_output = [actor_output[1], actor_output[3]]


                    with tf.device(shared_job_device):
                        enqueue_ops[i].append(queues[i].enqueue(nest.flatten(actor1_output)))
                        # enqueue_ops[j].append(queues[j].enqueue(nest.flatten(actor2_output)))

        # If running in a single machine setup, run actors with QueueRunners
        # (separate threads).
        if is_single_machine():
            if is_learner and enqueue_ops:
                for i in range(FLAGS.num_agents):
                    tf.train.add_queue_runner(tf.train.QueueRunner(queues[i], enqueue_ops[i]))

        # Build learner.
        if is_learner:
            # Create global step, which is the number of environment frames processed.
            tf.get_variable(
                'num_environment_frames',
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            # Create batch (time major) and recreate structure.

            def make_time_major(s):
                return nest.map_structure(
                    lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)

            dequeued = []
            for i in range(FLAGS.num_agents):
                dequeue = queues[i].dequeue_many(FLAGS.batch_size)
                dequeue = nest.pack_sequence_as(structure, dequeue)
                dequeue = [d._replace(
                    env_outputs=make_time_major(d.env_outputs),
                    agent_outputs=make_time_major(d.agent_outputs)) for d in dequeue]

                dequeued.append(dequeue)

            with tf.device('/gpu'):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.

                num_env_frames = tf.train.get_global_step()
                #learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                #                                          FLAGS.total_environment_frames, 0)

                learning_rates = [tf.Variable(tf.constant(FLAGS.learning_rate * np.random.uniform(1, 1)), trainable=False) for i in range(FLAGS.num_agents)]
                epsilons = [tf.Variable(tf.constant(FLAGS.epsilon * np.random.uniform(1, 1)), trainable=False) for i in range(FLAGS.num_agents)]
                if MULTI_ENT:
                    entropy_costs = [[tf.Variable(tf.constant(FLAGS.entropy_cost * np.random.uniform(1, 1) * (FLAGS.message_entropy_multiplier if x == 3 else 1)), trainable=False) for x in range(4)] for i in range(FLAGS.num_agents)]
                else:
                    entropy_costs = [
                        [tf.Variable(tf.constant(FLAGS.entropy_cost * np.random.uniform(1, 1)), trainable=False)] for i in
                        range(FLAGS.num_agents)]

                for index in range(FLAGS.num_agents):
                    tf.summary.scalar('learning_rate/' + str(index), learning_rates[index])
                    tf.summary.scalar('epsilon/' + str(index), epsilons[index])
                    if MULTI_ENT:
                        for i, entropy_cost in enumerate(entropy_costs[index]):
                            tf.summary.scalar('entropy_cost/entropy_cost_' + str(i) + '/' + str(index), entropy_cost)
                    else:
                        tf.summary.scalar('entropy_cost/' + str(index), entropy_costs[index][0])

                stage_ops = []
                outputss = []
                for i in range(FLAGS.num_agents):
                    optimizer = tf.train.RMSPropOptimizer(learning_rates[i], FLAGS.decay,
                                                          FLAGS.momentum, epsilons[i])

                    flattened_output = nest.flatten(dequeued[i])
                    area = tf.contrib.staging.StagingArea(
                        [t.dtype for t in flattened_output],
                        [t.shape for t in flattened_output])
                    stage_op = area.put(flattened_output)

                    stage_ops.append(stage_op)

                    data_from_actorss = nest.pack_sequence_as(structure, area.get())

                    outputs = []
                    for data_from_actors in data_from_actorss:
                        # Unroll agent on sequence, create losses and update ops.
                        print('building learner', i)
                        outputs.append(build_learner(agents[i],
                                                     data_from_actors.agent_state,
                                                     data_from_actors.env_outputs,
                                                     data_from_actors.agent_outputs,
                                                     optimizer,
                                                     num_env_frames,
                                                     learning_rates[i],
                                                     epsilons[i],
                                                     entropy_costs[i],
                                                     index=i))

                    outputss.append(outputs)

            agent_parameters = tuple(
                [tf.trainable_variables('agent_' + str(i) + '/agent') + [learning_rates[i], epsilons[i]] + entropy_costs[i] for i in range(FLAGS.num_agents)])
            copy_params = [[[tf.assign(new, old) for (new, old) in zip(agent_parameters[i], agent_parameters[j])] for j in range(FLAGS.num_agents)] for i in range(FLAGS.num_agents)]

            #parameter_placeholders = tuple([tf.placeholder(dtype=p.dtype, shape=p.shape) for p in agent_parameters[0]])
            #assign_agent_parameters = [[tf.assign(parameter, placeholder, validate_shape=False, use_locking=False) for parameter, placeholder in zip(parameters, parameter_placeholders)] for parameters in agent_parameters]

            mutate_param = [[tf.assign(hyperparam, hyperparam * tf.math.pow(1.2, tf.to_float(
                tf.random.uniform([], 0, 2, dtype=tf.int32) * 2 - 1)), validate_shape=False, use_locking=False) for hyperparam in
                             [learning_rates[i], epsilons[i]] + entropy_costs[i]] for i in range(FLAGS.num_agents)]

            #test_assign = tf.assign(tf.Variable(tf.ones([1,2])), tf.zeros([1,2]))

        # Create MonitoredSession (to run the graph, checkpoint and log).
        tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters) #, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))

        with tf.train.MonitoredTrainingSession(
                server.target,
                is_chief=is_learner,
                checkpoint_dir=FLAGS.logdir,
                save_checkpoint_secs=120,
                save_summaries_secs=30,
                log_step_count_steps=50000,
                config=config,
                hooks=[py_process.PyProcessHook()]) as session:

            tf.get_default_graph().finalize()

            if is_learner:
                # Logging.
                summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

                # Prepare data for first run.
                session.run_step_fn(
                    lambda step_context: step_context.session.run(stage_ops))

                # Execute learning and track performance.
                num_env_frames_v = 0
                smoothed_performance = [0 for i in range(FLAGS.num_agents)]
                alpha = 0.001
                save_interval = FLAGS.save_interval
                last_save = FLAGS.burnin - save_interval

                while num_env_frames_v < FLAGS.total_environment_frames:
                    summary = tf.summary.Summary()
                    # print('outputss, stage_ops')
                    output_valuess, _ = session.run((outputss, stage_ops))
                    for i in range(len(output_valuess)):
                        for j, output in enumerate(output_valuess[i]):
                            done_v, infos_v, num_env_frames_v = output

                            for episode_return, episode_step in zip(
                                    infos_v.episode_return[done_v],
                                    infos_v.episode_step[done_v]):
                                episode_frames = episode_step * FLAGS.num_action_repeats

                                if j == 0:
                                    #tf.logging.info('Episode return: %f', episode_return)
                                    smoothed_performance[i] = smoothed_performance[i] * (1 - alpha) + episode_return * alpha

                                    summary.value.add(tag='/episode_return_' + str(i) + '_' + str(j),
                                                      simple_value=episode_return)
                                    summary.value.add(tag='/episode_frames_' + str(i) + '_' + str(j),
                                                      simple_value=episode_frames)

                    summary_writer.add_summary(summary, num_env_frames_v)

                    if num_env_frames_v - last_save >= save_interval:
                        last_save = num_env_frames_v
                        ranking = sorted([(perf, i) for i, perf in enumerate(smoothed_performance)])

                        copied = []
                        for _, i in ranking:
                            other_index = np.random.randint(0, FLAGS.num_agents)
                            if smoothed_performance[other_index] - smoothed_performance[i] > FLAGS.threshold:
                                print('copying from', other_index, 'to', i, '(', smoothed_performance[other_index], 'to', smoothed_performance[i], ')')
                                #session.run(copy_params[i][other_index])
                                #session.run(assign_agent_parameters[i], feed_dict={parameter_placeholders: session.run(agent_parameters[other_index])})
                                session.run_step_fn(lambda step_context: step_context.session.run(copy_params[i][other_index]))
                                copied.append((i, j))

                        for i, j in copied:
                            smoothed_performance[i] = smoothed_performance[j]

                        for i in range(FLAGS.num_agents):
                            for j in range(len([learning_rates[i], epsilons[i]] + entropy_costs[i])):
                                if np.random.rand() < 1/3:
                                    session.run_step_fn(lambda step_context: step_context.session.run(mutate_param[i][j]))
                                    #session.run(])

                        #for index in range(FLAGS.num_agents):
                        #    tf.summary.scalar('learning_rate/' + str(index), learning_rates[index])
                        #    tf.summary.scalar('epsilon/' + str(index), epsilons[index])
                        #    for i, entropy_cost in enumerate(entropy_costs[index]):
                        #        tf.summary.scalar('entropy_cost/entropy_cost_' + str(i) + '/' + str(index), entropy_cost[i])

                        #for i in range(FLAGS.num_agents):
                        #    print('test_assign')
                        #    session.run(test_assign)

                        #session.run(stage_ops)
                        #session.run_step_fn(lambda step_context: step_context.session.run(stage_ops))


            else:
                # Execute actors (they just need to enqueue their output).

                while True:
                    #if np.random.rand() < 0.0001:
                    #    print('.', end='')
                    #    sys.stdout.flush()
                    session.run(enqueue_ops)



def test():
    """Test."""

    with tf.Graph().as_default():
        agent = Agent((6, 8, 8))

        env = create_environment({'adversarial': False}, is_test=True)
        outputs = build_actor(agent, env)[0]

        returns = []

        with tf.train.SingularMonitoredSession(
                checkpoint_dir=FLAGS.logdir,
                hooks=[py_process.PyProcessHook()]) as session:
            tf.logging.info('Testing:')
            while True:
                done_v, infos_v = session.run((
                    outputs.env_outputs.done,
                    outputs.env_outputs.info
                ))
                returns.extend(infos_v.episode_return[1:][done_v[1:]])

                if len(returns) >= FLAGS.test_num_episodes:
                    tf.logging.info('Mean episode return: %f', np.mean(returns))
                    break


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.mode == 'train':
        train()
    else:
        test()


if __name__ == '__main__':
    tf.app.run()
