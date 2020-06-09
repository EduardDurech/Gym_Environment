#Basic Implementation using DQN in keras-rl

import gym
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

cfg = tf.ConfigProto(allow_soft_placement=True )
cfg.gpu_options.allow_growth = True


env = gym.make('fooEnv_ID')

n_actions = env.action_space.n  #This will depend on the space you use, i.e. an action_space box would be env.action_space.shape[0]
                                #reference https://github.com/openai/gym/tree/master/gym/spaces


#Architecture, simple feed-forward dense net
inp = Input(((1,) + env.observation_space.shape))
fl1 = Flatten()(inp)
dn1 = Dense(100, activation='relu')(fl1)
dn2 = Dense(100, activation='relu')(dn1)
otp = Dense(n_actions, activation='linear')(dn2)
DQNModel = Model(input=inp, output=otp)


memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

random_process = OrnsteinUhlenbeckProcess() #Optional random process, see https://github.com/keras-rl/keras-rl/blob/master/rl/random.py
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

agentDQN = DQNAgent(model=DQNModel, nb_actions=n_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
agentDQN.compile(Adam(lr=1e-3), metrics=['mae'])
agentDQN.fit(env, nb_steps=10000, visualize=False)
