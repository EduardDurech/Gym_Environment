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
import Gym_Environment

env = gym.make('fooEnv-v0')
```

Based off of https://github.com/MartinThoma/banana-gym
