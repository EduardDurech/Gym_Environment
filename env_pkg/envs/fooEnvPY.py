import gym
from gym import spaces
import numpy as np
import random
import cv2


from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
# from utils.observation_utils import normalize_observation
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

n_agents = 1

# Use a the malfunction generator to break agents from time to time
stochastic_data = MalfunctionParameters(malfunction_rate=1/10000,  # Rate of malfunction occurence
                                        min_duration=15,  # Minimal duration of malfunction
                                        max_duration=50  # Max duration of malfunction
                                        )

random.seed(1)
np.random.seed(1)

class FooEnv(gym.Env):
    def __init__(self, n_cars, n_acts, min_obs, max_obs, n_nodes, n_features):

        # Define Action Space, see https://github.com/openai/gym/tree/master/gym/spaces for types
        # spaces.Discrete(5) will create 5 discrete possible actions, which will be passed as a
        # an integer to step(...) as the var action in range [0, n-1] with n actions
        self.action_space = spaces.Tuple([spaces.Discrete(n_acts)]*n_cars) #(n_cars x n_acts) Discrete vector

        # Define Observation Space using spaces as in Action, in a spaces.Box there must be a [low, high]
        self.observation_space = space.Box(low=min_obs, high=max_obs, shape=(n_cars, n_nodes*n_features), dtype=np.float32) #(n_cars x n_nodes*n_features) Continuous vector [min_obs, max_obs]
                                                                                                                            #If normalizing, set min_obs=-1, max_obs=1

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

        self.action_dict = dict()
        self.action_prob = [0] * 5
        self.info = dict()

    def set_action(self, action):
        # if self.info[0]:
        self.action_dict.update({0: action})

    def step(self, action):
        """
        ----------
        action : int
        return obs, reward, resetFlag, info
            see https://gym.openai.com/docs/#observations
        """
        self.set_action(action)
        next_obs, all_rewards, done, self.info = self._rail_env.step(self.action_dict)
        return next_obs, all_rewards, done, self.info

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        return obs: initial observation of the space
        """
        obs, self.info = self._rail_env.reset(True, True)
        return obs


    def render(self):
        env_renderer = RenderTool(self._rail_env, gl="PILSVG")
        env_renderer.render_env()
        image = env_renderer.get_image()
        cv2.imshow('sdasda', image)



if __name__ == "__main__":
    env = FooEnv()
    env.reset()

    for i in range(1000):
        next_obs, all_rewards, done, info = env.step(np.random.randint(5))

        print(f'Observation: {next_obs[0]}')
        print(f'Rewards: {all_rewards}')

        env.render()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
