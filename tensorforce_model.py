from tensorforce import Agent, Environment, Runner
from tensorforce.agents import DQNAgent
from env_pkg.envs.fooEnvPY import FooEnv
import env_pkg
import numpy as np


# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='foo-v0', max_episode_timesteps=500, visualize=False, n_cars=10
)

# Instantiate a Tensorforce agent
# agent = Agent.create(
#     agent='dqn',
#     environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
#     memory=10000,
#     # update=dict(unit='timesteps', batch_size=64),
#     # optimizer=dict(type='adam', learning_rate=3e-4),
#     # policy=dict(network='auto'),
#     # objective='policy_gradient',
#     # reward_estimation=dict(horizon=20)
# )

agent = Agent.create(
    agent='dqn',
    environment=environment,
    network="auto",
    memory=100000,
    start_updating=64,
    learning_rate=0.001,
    batch_size=64,
    update_frequency=1,
    discount=1.0,
    seed=0
)


# Create the runner
runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r, e):
    print(r)
    print(e)
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=e, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(num_episodes=3000, callback=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)

# # Train for 300 episodes
# for _ in range(300):

#     # Initialize episode
#     obs = environment.reset()
#     terminal = False
#     print('asdasd')
#     while not terminal:
#         # Episode timestep
#         actions = agent.act(states=obs)
#         states, terminal, reward = environment.execute(actions=actions)
#         # print(states, terminal, reward)
#         # print(reward)
#         agent.observe(terminal=terminal, reward=reward)

# agent.close()
# environment.close()