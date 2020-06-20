from collections import deque
import numpy as np
from dqn import Agent
from fl_environment import FlatlandEnv


# Keep track of last 100 results
win_rate = deque(maxlen=100)

# Env params
x_dim = 36 # TODO: random sampling
y_dim = 36 #
max_steps = 8 * (x_dim + y_dim) - 1
learn_every = 10

environment = FlatlandEnv(
        x_dim=x_dim,
        y_dim=y_dim,
        n_cars=1, 
        n_acts=5, 
        min_obs=-1.0, 
        max_obs=1.0, 
        n_nodes=1, 
        n_feats=11
) 

agent = Agent(
    alpha=0.0001, 
    gamma=0.99, 
    epsilon=1.0, 
    input_shape=5, 
    batch_size=64, 
    n_actions=5
)


# Train for 300 episodes
for episode in range(300):

    # Initialize episode
    obs, info = environment.reset()
    steps = 0
    all_done = False
    while not all_done and steps < max_steps:

        # Clear action buffer
        all_actions = [None] * environment.n_cars
        
        # Pick action for each agent
        for agent_id in range(environment.n_cars):
            if info['action_required'][agent_id]:
                all_actions[agent_id] = agent.choose_action(obs[agent_id])

        # Perform actions in environment
        states, reward, terminal, info = environment.step(action=all_actions)

        # Learn from taken actions
        for agent_id in range(environment.n_cars):
            
            # If agent took an action or completed
            if all_actions[agent_id] is not None or terminal[agent_id]:
                
                # Add state to memory
                agent.remember(obs[agent_id], all_actions[agent_id], reward[agent_id], states[agent_id], terminal[agent_id])
                
        # Learn every 10 steps
        if steps % learn_every == 0:
            agent.learn()
            
        # Update old states        
        obs = states

        # Calculate percentage complete
        perc_done = [v for k, v in terminal.items() if k is not '__all__'].count(True)/environment.n_cars

        # We done yet?
        all_done = terminal['__all__']
        steps += 1

    win_rate.append(perc_done or 0)
    print(f'Episode: {episode+1} Last 100 win rate: {np.mean(win_rate)}')

environment.close()