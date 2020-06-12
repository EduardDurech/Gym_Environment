from tensorforce import Agent, Environment
from env_pkg.envs.fooEnvPY import FooEnv
import env_pkg


# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='foo-v0', max_episode_timesteps=500, visualize=True
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

# Train for 300 episodes
for _ in range(300):

    # Initialize episode
    obs = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=obs)
        states, terminal, reward = environment.execute(actions=actions)
        print(reward)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()