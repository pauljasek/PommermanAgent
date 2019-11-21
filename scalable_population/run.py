from pommerman import agents
from pommerman.runner import DockerAgentRunner

from agent import ScalableAgent

class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self._agent = ScalableAgent(checkpoint_dir='/agent/agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001',
                                           agent_num=9, printing=False, disable_message=False, old=False)

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
