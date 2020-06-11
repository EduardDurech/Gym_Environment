#Basic Implementation using DQN in keras-rl

import gym
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Flatten, Dense, Concatenate, Reshape
from keras.optimizers import Adam
from keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

cfg = tf.ConfigProto(allow_soft_placement=True )
cfg.gpu_options.allow_growth = True


n_cars = 10
n_acts = 5
min_obs = -1.0
max_obs = 1.0
n_nodes = 5
n_features = 11

env = gym.make('fooEnv_ID', n_cars, n_acts, min_obs, max_obs, n_nodes, n_features)

car_layer = {}

#Architecture, simple feed-forward dense net
inp = Input(((1,) + env.observation_space.shape))
fl1 = Flatten()(inp)
dn1 = Dense(100, activation='relu')(fl1)
for i in range(n_cars):
  car_layer["dnR"+str(i)] = Reshape([1,n_acts])(Dense(n_acts)(dn1))
modelDQN = Model(input=inp, output=Concatenate(axis=1)(list(car_layer.values()))) #Current output (n_cars x n_acts)
                                                                                           #Can also have n_cars length vector with each position giving the car's action number
                                                                                           #output = Dense(n_cars, activation='linear')(dn1)

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

agentDQN = DQNAgent(model=modelDQN, nb_actions=n_acts, memory=memory, nb_steps_warmup=10,
                    target_model_update=1e-2, policy=policy)
agentDQN.compile(Adam(lr=1e-3), metrics=['mae'])
agentDQN.fit(env, nb_steps=10000, visualize=False)
