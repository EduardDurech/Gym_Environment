
from env_pkg.envs.fooEnvPY import FooEnv
import numpy as np
from dqn import Agent
from collections import deque

environment = FooEnv(
        n_cars=1, 
        n_acts=5, 
        min_obs=-1.0, 
        max_obs=1.0, 
        n_nodes=1, 
        n_feats=11
) 

agent = Agent(
    alpha=0.001, 
    gamma=0.99, 
    epsilon=1.0, 
    input_shape=5, 
    batch_size=64, 
    n_actions=5
)


win_rate = deque(maxlen=100)

# Train for 300 episodes
for episode in range(300):

    # Initialize episode
    obs = environment.reset()
    terminal = {'__all__': False}
    score = 0.0
    steps = 0
    while not terminal['__all__'] and steps < 8 * (36 + 36) - 1:
        all_actions = [None] * environment._rail_env.get_num_agents()
        for agent_id in range(environment._rail_env.get_num_agents()):
            all_actions[agent_id] = agent.choose_action(obs[agent_id])
        states, reward, terminal, info = environment.step(action=all_actions)
        # print(terminal[0])
        score += sum(reward.values())
        for agent_id in range(len(states)):
            if all_actions[agent_id] is not None or terminal[agent_id]:
                agent.remember(obs[agent_id], all_actions[agent_id], reward[agent_id], states[agent_id], terminal[agent_id])
                agent.learn()
            obs = states
        perc_done = [v for k, v in terminal.items() if k is not '__all__'].count(True)/environment._rail_env.get_num_agents()
        # print(f'{score} ----- {perc_done}')
        # environment.render()
        steps += 1
    win_rate.append(perc_done or 0)
    print(f'Episode: {episode+1} Last 100 win rate: {np.mean(win_rate)}')
    print(win_rate)


environment.close()