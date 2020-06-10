#Basic Implementation using DQN in keras-rl
import gym
import env_pkg # <-- This is our env

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

n_acts = 5

env = gym.make('foo-v0', 
    n_cars=1, 
    n_acts=n_acts, 
    min_obs=-1.0, 
    max_obs=1.0, 
    n_nodes=2, 
    n_feats=11)
env.__init__()


#Architecture, simple feed-forward dense net
inp = Input(shape=(1,231,))
fl1 = Flatten()(inp)
dn1 = Dense(100, activation='relu')(fl1)
dn1 = Dense(100, activation='relu')(dn1)
dn2 = Dense(n_acts, activation='linear')(dn1)

DQNModel = Model(inp, dn2)
DQNModel.summary()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

agentDQN = DQNAgent(model=DQNModel, nb_actions=n_acts, memory=memory, nb_steps_warmup=20,
                    target_model_update=1e-2, policy=policy)
agentDQN.compile(Adam(lr=1e-4), metrics=['mae'])
agentDQN.fit(env, nb_steps=10000, visualize=False, verbose=2)
