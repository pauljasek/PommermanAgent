
'''An agent that preforms a random action each step'''
import pommerman
from pommerman import characters
from pommerman import agents
from pommerman.agents import BaseAgent

import numpy as np
import tensorflow as tf
import collections
import time

try:
    from experiment import Agent, LSTM_SIZE
except:
    pass

nest = tf.contrib.framework.nest

StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')

class ScalableAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, agent=None, character=characters.Bomber, checkpoint_dir='agents/pretrained', agent_num=0, printing=False, disable_message=False, old=False):
        super(ScalableAgent, self).__init__(character)

        self.printing = printing
        self.disable_message = disable_message

        self.old = old

        with tf.Graph().as_default():

            if self.old:
                for i in range(agent_num + 1):
                    agent = Agent((6, 8, 8))
                self.observation = (tf.placeholder(tf.int32, [9, 9]), tf.placeholder(tf.float32, [9, 9]),
                                    tf.placeholder(tf.float32, [9, 9]), tf.placeholder(tf.float32, [22]))

            else:
                with tf.variable_scope('agent_' + str(agent_num)):
                    agent = Agent((6, 8, 8))
                self.observation = (tf.placeholder(tf.int32, [9, 9]), tf.placeholder(tf.float32, [9, 9]),
                                    tf.placeholder(tf.float32, [9, 9]), tf.placeholder(tf.float32, [23]))

            #self.session = session

            reward = tf.constant(0.)
            done = tf.constant(False)
            info = StepOutputInfo(tf.constant(0.), tf.constant(0))
            self.last_action = tf.placeholder(tf.int32, [1, 3])

            self.core_state = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [1, LSTM_SIZE]), tf.placeholder(tf.float32, [1, LSTM_SIZE]))

            env_output = StepOutput(
                reward,
                info,
                done,
                self.observation)

            batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                    env_output)
            self.agent_output, self.agent_state = agent((self.last_action, batched_env_output), self.core_state)

            self.state = (np.zeros((1,LSTM_SIZE), dtype=np.float32), np.zeros((1, LSTM_SIZE), dtype=np.float32))
            self.previous_action = np.zeros((1, 3), dtype=np.int32)

            self.session = tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir, config=tf.ConfigProto(device_count={'GPU': 0}))


    def _parse_observation(self, obs):
        agent = obs["board"][obs["position"][0]][obs["position"][1]]
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

        centered_bomb_blast_strength[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs[
                                                                                                       "bomb_blast_strength"].astype(
            np.float32)[
                                                                                                   max(0,
                                                                                                       x - 4):min(
                                                                                                       11, x + 5),
                                                                                                   max(0,
                                                                                                       y - 4):min(
                                                                                                       11, y + 5)]

        centered_bomb_life[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["bomb_life"].astype(
            np.float32)[
                                                                                         max(0, x - 4):min(11,
                                                                                                           x + 5),
                                                                                         max(0, y - 4):min(11,
                                                                                                           y + 5)]

        position = np.array(obs["position"], dtype=np.float32) / 10 * 2 - 1
        ammo = np.array([obs["ammo"]], dtype=np.float32)
        blast_strength = np.array([obs["blast_strength"]], dtype=np.float32)
        can_kick = np.array([obs["can_kick"]], dtype=np.float32)
        if self.old:
            teammate_alive = np.array([teammate in ["alive"]], dtype=np.float32)
        else:
            teammate_alive = np.array([teammate in obs["alive"]], dtype=np.float32)
        two_enemies = np.array([enemies[0] in obs["alive"] and enemies[1] in obs["alive"]], dtype=np.float32)

        #print(teammate_alive, two_enemies)

        message = np.zeros((2, 8), dtype=np.float32)
        message[np.arange(2), obs["message"]] = 1
        message = message.reshape(-1)

        if self.old:
            feature = np.concatenate([position, ammo, blast_strength, can_kick, teammate_alive, message])
        else:
            feature = np.concatenate([position, ammo, blast_strength, can_kick, teammate_alive, two_enemies, message])

        return (centered_board, centered_bomb_blast_strength, centered_bomb_life, feature)

    def act(self, obs, action_space):
        if self.disable_message:
            obs["message"] = (0,0)

        if self.printing:
            if obs["message"][0] != 0 or obs["message"][1] != 0:
                print(obs["message"])

        parsed_observation = self._parse_observation(obs)
        output, state = self.session.run([self.agent_output, self.agent_state],
                                         feed_dict={ self.observation: parsed_observation,
                                                     self.last_action: self.previous_action,
                                                     self.core_state: self.state})

        #print(obs["message"])

        if self.printing:
            dist_index = 2
            probs = (np.exp(output[1][dist_index][0])/sum(np.exp(output[1][dist_index][0])))
            #print('|' + '|'.join(str(i) * int(prob * 100) for i, prob in enumerate(probs)) + '|')
            #print(tuple(output[0][0])[dist_index])
            #if output[0][0][1] != 0 or output[0][0][2] != 0:
            #    print(output[0][0][1], output[0][0][2])

        action = tuple(output[0][0])
        self.previous_action = output[0]
        self.state = state

        return tuple(int(a) for a in action)

    def episode_end(self, reward):
        self.state = (np.zeros((1, LSTM_SIZE), dtype=np.float32), np.zeros((1, LSTM_SIZE), dtype=np.float32))
        self.previous_action = np.zeros((1, 3), dtype=np.int32)

import random
class RandomNoBombsAgent(BaseAgent):
    def act(self, obs, action_space):
        return (random.randint(0, 4), random.randint(0,7), random.randint(0,7))


def search_agents(search_dir, num_agents, ties=True, simple_agent=True):
    from tqdm import tqdm, trange
    search_dirs = [search_dir]
    num_agents_list = [num_agents]

    left_episodes = 50
    right_episodes = 50

    scores = {}

    for dir, num_agents in tqdm(zip(search_dirs, num_agents_list)):
        for agent_num in trange(num_agents):
            name = dir + '/' + str(agent_num)
            scores[name] = 0

            if simple_agent:
                agent_list = [
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False),
                    agents.SimpleAgent(),
                    #agents.PlayerAgent(agent_control="arrows"),
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False),
                    agents.SimpleAgent(),
                    #agents.PlayerAgent(agent_control="arrows"),
                ]
            else:
                agent_list = [
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False, old=False),
                    ScalableAgent(checkpoint_dir='old_agents/continued_variable_ent_ties_population_10_ent_0.001_lr_0.0002', agent_num=9, printing=False, old=True),
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False, old=False),
                    ScalableAgent(checkpoint_dir='old_agents/continued_variable_ent_ties_population_10_ent_0.001_lr_0.0002', agent_num=9, printing=False, old=True),
                ]

            # Make the "Free-For-All" environment using the agent list
            env = pommerman.make('PommeRadioCompetition-v2', agent_list)


            # Run the episodes just like OpenAI Gym
            for i_episode in trange(left_episodes):
                state = env.reset()
                done = False
                while not done:
                    actions = env.act(state)
                    state, reward, done, info = env.step(actions)
                    if ties:
                        if reward[0] == reward[1]:
                            reward = [0, 0, 0, 0]
                    scores[name] += reward[0]
            env.close()

            if simple_agent:
                agent_list = [
                    agents.SimpleAgent(),
                    #agents.PlayerAgent(agent_control="arrows"),
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False),
                    agents.SimpleAgent(),
                    #agents.PlayerAgent(agent_control="arrows"),
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False),
                ]
            else:
                agent_list = [
                    ScalableAgent(checkpoint_dir='old_agents/continued_variable_ent_ties_population_10_ent_0.001_lr_0.0002', agent_num=9, printing=False, old=True),
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False, old=False),
                    ScalableAgent(checkpoint_dir='old_agents/continued_variable_ent_ties_population_10_ent_0.001_lr_0.0002', agent_num=9, printing=False, old=True),
                    ScalableAgent(checkpoint_dir=dir, agent_num=agent_num, printing=False, old=False),
                ]

            # Make the "Free-For-All" environment using the agent list
            env = pommerman.make('PommeRadioCompetition-v2', agent_list)


            # Run the episodes just like OpenAI Gym
            for i_episode in trange(right_episodes):
                state = env.reset()
                done = False
                while not done:
                    actions = env.act(state)
                    state, reward, done, info = env.step(actions)
                    if ties:
                        if reward[0] == reward[1]:
                            reward = [0, 0, 0, 0]
                    scores[name] += reward[1]
            env.close()

            scores[name] /= left_episodes + right_episodes

            print(sorted(scores.items(), key=lambda kv: -kv[1]))

        results = sorted(scores.items(), key=lambda kv: -kv[1])
        for result in results:
            print(result[0] + ': ' + str(result[1]))

import sys, os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__



def save_video(env, output='videos/output.avi'):
    import cv2, pyglet

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 10.0, (738, 620))

    MESSAGE_COLORS = [(255, 255, 255),
                      (255, 0, 0),
                      (255, 127, 0),
                      (255, 255, 0),
                      (0, 255, 0),
                      (0, 0, 255),
                      (75, 0, 130),
                      (148, 0, 211)]
    messages = [None, None, None, None]
    cooldowns = [0, 0, 0, 0]

    state = env.reset()
    done = False
    while True:
        env.render(do_sleep=False)

        pyglet.image.get_buffer_manager().get_color_buffer().save('videos/temp.png')
        im = cv2.imread('videos/temp.png')

        if done:
            out.write(im)
            break

        blockPrint()
        actions = env.act(state)
        enablePrint()

        for i in range(4):
            if type(actions[i]) != tuple:
                continue
            if actions[i][1:] != (0,0):
                print(actions[i][1:])
                messages[i] = actions[i][1:]
                cooldowns[i] = 10

            cooldowns[i] -= 1
            if cooldowns[i] <= 0:
                messages[i] = None

            if messages[i] is not None:
                cx = state[i]['position'][1] * 50 + 20 + 32
                cy = state[i]['position'][0] * 50 + 50 + 12

                cv2.circle(im, (cx, cy), 8, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 16, cy), 8, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.rectangle(im, (cx, cy - 8), (cx + 14, cy + 8), (200, 200, 200), thickness=-1)
                cv2.fillConvexPoly(im, np.int_([(cx - 6, cy), (cx - 8, cy + 20), (cx, cy + 4)]), (200, 200, 200), lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx, cy), 5, MESSAGE_COLORS[messages[i][0]], thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 14, cy), 5, MESSAGE_COLORS[messages[i][1]], thickness=-1, lineType=cv2.LINE_AA, shift=0)

                cx = 600 + 68
                cy = int(i * 147.5) + 36

                cv2.circle(im, (cx, cy), 15, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 30, cy), 15, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.rectangle(im, (cx, cy - 15), (cx + 30, cy + 15), (200, 200, 200), thickness=-1)
                cv2.circle(im, (cx, cy), 10, MESSAGE_COLORS[messages[i][0]], thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 30, cy), 10, MESSAGE_COLORS[messages[i][1]], thickness=-1, lineType=cv2.LINE_AA, shift=0)

        out.write(im)

        state, reward, done, info = env.step(actions)

    out.release()

if __name__ == '__main__':
    PRINTING = False
    DISABLE_MESSAGE = False
    RENDER = True
    DO_SLEEP = False
    SEARCH = False
    OLD = False
    OLD_2 = False

    SAVE_VIDEO = False


    #CHECKPOINT_DIR = 'agents/onyx_population_6_ent_0.005_lr_0.001'
    #CHECKPOINT_DIR = 'agents/onyx_new_start_population_6_ent_0.002_lr_0.001'
    #CHECKPOINT_DIR = 'agents/onyx_start_4M_population_6_ent_0.002_lr_0.001'
    #CHECKPOINT_DIR = 'agents/mustang_population_6_ent_0.007_lr_0.001'
    #CHECKPOINT_DIR = 'agents/distributed_population_10_ent_0.12_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pretrained_distributed_population_10_ent_0.01_lr_0.001'

    #CHECKPOINT_DIR = 'agents/pretrained_2_distributed_population_10_ent_0.0025_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/pretrained_2_distributed_population_10_ent_0.0025_lr_0.001'

    #CHECKPOINT_DIR = 'agents/adversarial_continued_population_10_ent_0.005_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/adversarial_continued_population_10_ent_0.0025_lr_0.001'

    #CHECKPOINT_DIR = 'agents/variable_entropy_cost_population_10_ent_0.005_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pretrained_continued_population_10_ent_0.0005_lr_0.001'

    #CHECKPOINT_DIR = 'agents/pretrained_4_adv_population_10_ent_0.0015_lr_0.001'

    #CHECKPOINT_DIR = 'agents/saved_agent_1'
    #CHECKPOINT_DIR = 'agents/pretrained_variable_ent_cost_population_10_ent_0.0025_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pretrained_ties_variable_ent_cost_population_10_ent_0.002_lr_0.001'
    #CHECKPOINT_DIR = 'agents/adversarial_3_population_10_ent_0.004_lr_0.001'
    #CHECKPOINT_DIR = 'agents/adversarial_3_ties_variable_ent_cost_population_10_ent_0.003_lr_0.001'

    #CHECKPOINT_DIR = 'agents/adversarial_3_population_10_ent_0.003_lr_0.001'

    #CHECKPOINT_DIR = 'agents/pretrained_ties_variable_ent_cost_population_10_ent_0.003_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/pretrained_ties_variable_ent_cost_population_10_ent_0.003_lr_0.001'

    #CHECKPOINT_DIR = 'agents/continued_variable_ent_ties_population_10_ent_0.00255_lr_0.000093'
    #CHECKPOINT_DIR = 'old_agents/continued_variable_ent_ties_population_10_ent_0.003_lr_0.000093'
    #CHECKPOINT_DIR = 'agents/saved_agent_1'

    #CHECKPOINT_DIR = 'agents/continued_variable_ent_ties_population_10_ent_0.002_lr_0.000093'
    #CHECKPOINT_DIR_2 = 'agents/continued_variable_ent_ties_population_10_ent_0.003_lr_0.00005'

    #CHECKPOINT_DIR = 'agents/saved_agent_9'
    #CHECKPOINT_DIR = 'old_agents/continued_variable_ent_ties_population_10_ent_0.001_lr_0.0002'

    CHECKPOINT_DIR = 'agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_short_burnin_threshold_0.15_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_long_intervals_threshold_0.2_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_threshold_0.15_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_no_ties_short_burnin_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_threshold_0.2_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_corrected_state_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_no_ties_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_threshold_0.075_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_intervals_200000000_threshold_0.05_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_big_lstm_multi_ent_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_big_lstm_multi_ent_10_ent_0.12_lr_0.001'
    #CHECKPOINT_DIR = 'agents/pbt_increased_message_ent_cost_population_10_ent_0.15_lr_0.001'

    CHECKPOINT_DIR_2 = 'agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/pbt_short_burnin_threshold_0.15_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/16m_comparison'
    #CHECKPOINT_DIR_2 = 'agents/saved_agent_9'
    #CHECKPOINT_DIR_2 = 'agents/pbt_short_burnin_threshold_0.15_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR_2 = 'agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001'
    #CHECKPOINT_DIR_2 = 'old_agents/continued_variable_ent_ties_population_10_ent_0.001_lr_0.0002'
    #CHECKPOINT_DIR_2 = 'agents/pbt_increased_message_ent_cost_population_10_ent_0.15_lr_0.001'

    AGENT = 9
    AGENT_2 = 7

    if SEARCH:
        search_agents(CHECKPOINT_DIR, 10)
        quit()

    # Create a set of agents (exactly four)
    if False:
        agent_list = [
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=False, disable_message=False, old=OLD_2),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=False, disable_message=False, old=OLD_2),
        ]

        agent_list_2 = [
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=False, disable_message=False, old=OLD_2),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=False, disable_message=False, old=OLD_2),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
        ]
    else:
        agent_list = [
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            #ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            #agents.DockerAgent("scalable_agent", port=10005),
            agents.SimpleAgent(),
            #agents.PlayerAgent(agent_control="arrows"),
            #agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10027),
            #agents.DockerAgent("multiagentlearning/skynet955", port=10003),
            #agents.DockerAgent("multiagentlearning/navocado", port=10051),
            #agents.DockerAgent("multiagentlearning/dypm.1", port=10023),
            #agents.DockerAgent("multiagentlearning/eisenach", port=10021),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            #ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
            #agents.DockerAgent("scalable_agent", port=10006),
            agents.SimpleAgent(),
            #agents.PlayerAgent(agent_control="wasd"),
            #agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10028),
            #agents.DockerAgent('multiagentlearning/skynet955', port=10004),
            #agents.DockerAgent('multiagentlearning/navocado', port=10052),
            #agents.DockerAgent("multiagentlearning/dypm.2", port=10024),
            #agents.DockerAgent("multiagentlearning/eisenach", port=10022),
        ]

        agent_list_2 = [
            agents.SimpleAgent(),
            #agents.PlayerAgent(agent_control="arrows"),
            #agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10029),
            #agents.DockerAgent("multiagentlearning/dypm.1", port=10025),
            #agents.DockerAgent("multiagentlearning/navocado", port=10053),
            #agents.DockerAgent("multiagentlearning/eisenach", port=10023),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING,
                          disable_message=DISABLE_MESSAGE, old=OLD),
            agents.SimpleAgent(),
            #agents.PlayerAgent(agent_control="wasd"),
            #agents.DockerAgent("multiagentlearning/eisenach", port=10024),
            #agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10030),
            #agents.DockerAgent("multiagentlearning/dypm.2", port=10026),
            #agents.DockerAgent("multiagentlearning/navocado", port=10054),
            ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING,
                          disable_message=DISABLE_MESSAGE, old=OLD),
        ]

    # Against skynet955          win percentage: 0.74 tie percentage: 0.26 loss percentage: 0.0
    # Against navocado           win percentage: 0.44 tie percentage: 0.55 loss percentage: 0.01
    # Against dypm.1             win percentage: 0.24 tie percentage: 0.6 loss percentage: 0.16
    # Against eisenach           win percentage: 0.28 tie percentage: 0.1 loss percentage: 0.62
    # Against hakozakijunctions  win percentage: 0.07 tie percentage: 0.41 loss percentage: 0.52


    # saved_agent_9_new
    # eisenach win percentage: 0.38 tie percentage: 0.09 loss percentage: 0.53


    # Make the "Free-For-All" environment using the agent list
    env1 = pommerman.make('PommeRadioCompetition-v2', agent_list)
    env2 = pommerman.make('PommeRadioCompetition-v2', agent_list_2)

    env = env1

    if SAVE_VIDEO:
        save_video(env)
        quit()

    # Run the episodes just like OpenAI Gym
    total_wins = 0
    total_losses = 0
    total_ties = 0
    total_games = 100
    for i_episode in range(total_games):
        env = env1 if i_episode % 2 == 0 else env2

        state = env.reset()
        while True:
            if RENDER:
                env.render(do_sleep=DO_SLEEP)

            if not PRINTING:
                blockPrint()

            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            #time.sleep(0.2)

            if not PRINTING:
                enablePrint()

            if done:
                break

        if i_episode % 2 != 0:
            reward = reward[1:] + [reward[0]]

        print('Episode {} finished. Reward: {}. Other Reward: {}'.format(i_episode, reward[0], reward[1]))

        if reward[0] == 1:
            total_wins += 1
        elif reward[0] == -1 and reward[1] == -1:
            total_ties += 1
        else:
            total_losses += 1

        print('tentative --- win percentage:', total_wins / (i_episode + 1), 'tie percentage:', total_ties / (i_episode + 1),
              'loss percentage:', total_losses / (i_episode + 1))

    print('win percentage:', total_wins/total_games, 'tie percentage:', total_ties/total_games, 'loss percentage:', total_losses/total_games)
    env.close()
