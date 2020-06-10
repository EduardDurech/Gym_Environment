#Basic Implementation using DQN in keras-rl
import sys
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


def build_agent(n_acts=5, n_feats=231):
    #Architecture, simple feed-forward dense net
    inp = Input(shape=(1, n_feats))
    fl1 = Flatten()(inp)
    dn1 = Dense(100, activation='relu')(fl1)
    dn1 = Dense(100, activation='relu')(dn1)
    dn2 = Dense(n_acts, activation='linear')(dn1)
    DQNModel = Model(inp, dn2)
    DQNModel.summary()

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    agentDQN = DQNAgent(model=DQNModel, nb_actions=n_acts, memory=memory, nb_steps_warmup=50,
                        target_model_update=1e-2, policy=policy)
    agentDQN.compile(Adam(lr=1e-2), metrics=['mae'])
    return agentDQN


def build_env(n_acts):
    env = gym.make('foo-v0', 
        n_cars=1, 
        n_acts=n_acts, 
        min_obs=-1.0, 
        max_obs=1.0, 
        n_nodes=2, 
        n_feats=11)
    return env


def train():
    n_acts = 5
    env = build_env(n_acts)

    agentDQN = build_agent(n_acts)
    agentDQN.fit(env, nb_steps=10000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    agentDQN.save_weights('dqn_{}_weights.h5f'.format('flatland'), overwrite=True)
    test(env, agent=agentDQN)


def test(env, path='dqn_flatland_weights.h5f', agent=None):
    if agent is None:
        agent = build_agent()
    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if 'train' in sys.argv[1].lower():
            train()

        elif 'test' in sys.argv[1].lower():
            test(build_env())
    