Skeleton Gym Environment for Deep Reinforcement Learning


# Dependency
Make sure OpenAI's gym is installed https://gym.openai.com/docs/, e.g.
```
pip install gym
```

# Installation
Navitage to main directory (Gym_Environment)
```
pip install -e .
```

# Implementation
```
import gym
import env_pkg

env = gym.make('fooEnv_ID')
```

Based off of https://github.com/MartinThoma/banana-gym
