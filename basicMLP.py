#Basic Implementation using DQN in keras-rl

import gym
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Flatten, Dense, Concatenate
from keras.optimizers import Adam
from keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

cfg = tf.ConfigProto(allow_soft_placement=True )
cfg.gpu_options.allow_growth = True


env = gym.make('fooEnv_ID')
env.__init__(n_cars, n_acts, min_obs, max_obs, n_nodes, n_features)


#Architecture, simple feed-forward dense net
inp = Input(((1,) + env.observation_space.shape))
fl1 = Flatten()(inp)
dn1 = Dense(100, activation='relu')(fl1)
dn2 = Dense(100, activation='relu')(dn1)
otp = Dense(n_cars, activation='linear')(dn2) #Current output a 
                                              #Can also have n_cars length vector with each position giving the car's action number
                                              #otp = Dense(n_cars, activation='linear')(dn2)
DQNModel = Model(input=inp, output=otp)

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

agentDQN = DQNAgent(model=DQNModel, nb_actions=n_cars, memory=memory, nb_steps_warmup=10,
                    target_model_update=1e-2, policy=policy)
agentDQN.compile(Adam(lr=1e-3), metrics=['mae'])
agentDQN.fit(env, nb_steps=10000, visualize=False)
