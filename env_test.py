'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents

def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="wasd")
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    print(env.action_space)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            print(state[2]['board'])

        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
