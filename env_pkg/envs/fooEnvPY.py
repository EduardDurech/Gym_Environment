import gym
from gym import spaces
import numpy as np
import random

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

x_dim = 36
y_dim = 36

# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

n_agents = 10

# Use a the malfunction generator to break agents from time to time
stochastic_data = MalfunctionParameters(malfunction_rate=1/10000,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )

random.seed(1)
np.random.seed(1)

class FooEnv(gym.Env):
    def __init__(self):
        self.minPositionX = 0.0
        self.maxPositionX = 0.0 #Example vars for Observation of x position

        # Define Action Space, see https://github.com/openai/gym/tree/master/gym/spaces for types
        # spaces.Discrete(5) will create 5 discrete possible actions, which will be passed as a
        # an integer to step(...) as the var action in range [0, n-1] with n actions
        self.action_space = spaces.Discrete(5)

        # Define Observation Space using spaces as in Action, in a spaces.Box there must be a [low, high]
        self.observation_space = spaces.Box(self.minPositionX, self.maxPositionX)

        self._rail_env = RailEnv(
            width=x_dim,
            height=y_dim,
            rail_generator=sparse_rail_generator(max_num_cities=3,
                                                # Number of cities in map (where train stations are)
                                                seed=1,  # Random seed
                                                grid_mode=False,
                                                max_rails_between_cities=2,
                                                max_rails_in_city=3),
            schedule_generator=sparse_schedule_generator(speed_ration_map),
            number_of_agents=n_agents,
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            obs_builder_object=TreeObservation)

    def set_action(self, action):
        pass

    def step(self, action):
        """
        ----------
        action : int
        return obs, reward, resetFlag, info
            see https://gym.openai.com/docs/#observations
        """
        
        """
        ... Operations on action, i.e.
        # self.set_action()
        
        ... Operations to determine reward, i.e.
        reward = self.get_reward()
        
        ... Operations to get observations, i.e.
        obs = self.get_state()
        
        ... Operations to determine if reset conditions met, i.e.
        resetFlag = self.check_is_done()
        """

        self.set_action(action)

        return obs, reward, resetFlag, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        return obs: initial observation of the space
        """
        
        """
        ... Operations to reset, i.e.
        self.current_position = 0.0
        
        ... Operations to get observations, i.e.
        obs = self.get_state()
        """
        return self.get_state()
