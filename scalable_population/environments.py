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

"""Environments and environment helper classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import tensorflow as tf
import random
import traceback

import pommerman
from pommerman import agents


nest = tf.contrib.framework.nest


class LocalLevelCache(object):
  """Local level cache."""

  def __init__(self, cache_dir='/tmp/level_cache'):
    self._cache_dir = cache_dir
    tf.gfile.MakeDirs(cache_dir)

  def fetch(self, key, pk3_path):
    path = os.path.join(self._cache_dir, key)
    if tf.gfile.Exists(path):
      tf.gfile.Copy(path, pk3_path, overwrite=True)
      return True
    return False

  def write(self, key, pk3_path):
    path = os.path.join(self._cache_dir, key)
    if not tf.gfile.Exists(path):
      tf.gfile.Copy(pk3_path, path)


class PyProcessPommerMan(object):
  """Pommerman wrapper for PyProcess."""

  def __init__(self, config=None):
    #self._observation_spec = ['board', 'bomb_blast_strength', 'bomb_life', 'position', 'ammo', 'blast_strength', 'can_kick', 'teammate', 'enemies', 'message']

    self.adversarial = config['adversarial']

    # Create a set of agents (exactly four)
    agent_list = [
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
    ]

    # Make the "Free-For-All" environment using the agent list
    self._env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    #self._env.set_training_agent(self.agent)

  def _reset(self):
    #self.__init__(self.agent_object)
    self.state = self._env.reset()
    if random.random() < 0.5:
        self.agent_mapping = [0, 1, 2, 3]
    else:
        self.agent_mapping = [1, 0, 3, 2]
    return self.state

  def _remap(self, li):
      return [li[x] for x in self.agent_mapping]

  def _parse_observation(self, observations):
      rets = []
      for agent, obs in zip(range(10,14), observations):
          teammate = obs["teammate"].value
          enemies = [e.value for e in obs["enemies"][:2]]

          centered_board = np.ones((9, 9), dtype=np.int32)
          centered_bomb_blast_strength = np.ones((9, 9), dtype=np.float32)
          centered_bomb_life = np.ones((9, 9), dtype=np.float32)

          x, y = obs["position"]
          centered_board[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["board"][
                                                                                          max(0, x - 4):min(11, x + 5),
                                                                                          max(0, y - 4):min(11, y + 5)]

          centered_board_copy = np.copy(centered_board)
          centered_board[centered_board_copy == agent] = 10
          centered_board[centered_board_copy == teammate] = 11
          centered_board[centered_board_copy == enemies[0]] = 12
          centered_board[centered_board_copy == enemies[1]] = 13

          centered_bomb_blast_strength[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["bomb_blast_strength"].astype(np.float32)[
                                                                                                         max(0, x - 4):min(11, x + 5),
                                                                                                         max(0, y - 4):min(11, y + 5)]

          centered_bomb_life[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["bomb_life"].astype(np.float32)[
                                                                                               max(0, x - 4):min(11, x + 5),
                                                                                               max(0, y - 4):min(11, y + 5)]

          position = np.array(obs["position"], dtype=np.float32) / 10 * 2 - 1
          ammo = np.array([obs["ammo"]], dtype=np.float32)
          blast_strength = np.array([obs["blast_strength"]], dtype=np.float32)
          can_kick = np.array([obs["can_kick"]], dtype=np.float32)
          teammate_alive = np.array([teammate in obs["alive"]], dtype=np.float32)
          two_enemies = np.array([enemies[0] in obs["alive"] and enemies[1] in obs["alive"]], dtype=np.float32)

          message = np.zeros((2,8), dtype=np.float32)
          message[np.arange(2), obs["message"]] = 1
          message = message.reshape(-1)

          feature = np.concatenate([position, ammo, blast_strength, can_kick, teammate_alive, two_enemies, message])

          rets.append((centered_board, centered_bomb_blast_strength, centered_bomb_life, feature))
      return rets


  def initial(self):
    return self._remap(self._parse_observation(self._reset()))

  def step(self, actions):
    #actions = self._env.act(self.state)
    #actions[self.agent - 10] = tuple(action)
    actions = self._remap(actions)

    actions = [tuple(action[0]) for action in actions]
    observation, reward, done, info = self._env.step(actions)
    self.state = observation

    if reward[0] == reward[1]:
        reward = [0, 0, 0, 0]

    observation = self._parse_observation(observation)

    if self.adversarial:
        reward = np.array(reward[::-1], dtype=np.float32) * -1
    else:
        reward = np.array(reward, dtype=np.float32)

    ret = self._remap([(r, done, obs) for r, obs in zip(reward, observation)])

    if done:
      self._reset()

    return ret

  def close(self):
    self._env.close()

  @staticmethod
  def _tensor_specs(method_name):
    """Returns a nest of `TensorSpec` with the method's output specification."""

    observation_spec = (
        tf.contrib.framework.TensorSpec([9, 9], tf.int32),    # board
        tf.contrib.framework.TensorSpec([9, 9], tf.float32),  # board
        tf.contrib.framework.TensorSpec([9, 9], tf.float32),  # board
        tf.contrib.framework.TensorSpec([23], tf.float32),    # feature
    )

    if method_name == 'initial':
      return [observation_spec] * 4
    elif method_name == 'step':
      return [(
          tf.contrib.framework.TensorSpec([], tf.float32),
          tf.contrib.framework.TensorSpec([], tf.bool),
          observation_spec,
      )] * 4


StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')


class FlowEnvironment(object):
  """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.
  """

  def __init__(self, env):
    """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation). `observation`
        should be the observation after the step is taken. If `done` is
        True, the observation should be the first observation in the next
        episode.
    """
    self._env = env

  def initial(self):
    """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
    with tf.name_scope('flow_environment_initial'):
      initial_observation = self._env.initial()

      rets = []
      for initial_obs in initial_observation:
          initial_reward = tf.constant(0.)
          initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0))
          initial_done = tf.constant(True)

          initial_output = StepOutput(
              initial_reward,
              initial_info,
              initial_done,
              initial_obs)

          # Control dependency to make sure the next step can't be taken before the
          # initial output has been read from the environment.
          with tf.control_dependencies(nest.flatten(initial_output)):
            initial_flow = tf.constant(0, dtype=tf.int64)
          initial_state = (initial_flow, initial_info)

          rets.append((initial_output, initial_state))
      return rets

  def step(self, actions, states):
    """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """
    with tf.name_scope('flow_environment_step'):
      states = nest.map_structure(tf.convert_to_tensor, states)

      # Make sure the previous step has been executed before running the next
      # step.
      with tf.control_dependencies([state[0] for state in states]):
        rets = []
        env_results = self._env.step(actions)
        for i in range(len(env_results)):
          reward, done, observation = env_results[i]
          flow, info = states[i]
          with tf.control_dependencies(nest.flatten(observation)):
            new_flow = tf.add(flow, 1)

          # When done, include the reward in the output info but not in the
          # state for the next step.
          new_info = StepOutputInfo(tf.squeeze(info.episode_return + reward),
                                    info.episode_step + 1)
          new_state = new_flow, nest.map_structure(
              lambda a, b: tf.where(done, a, b),
              StepOutputInfo(tf.constant(0.), tf.constant(0)),
              new_info)

          output = StepOutput(reward, new_info, done, observation)
          rets.append((output, new_state))
        return rets
