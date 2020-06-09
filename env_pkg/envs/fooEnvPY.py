import gym
from gym import spaces

class FooEnv(gym.Env):
    def __init__(self):
        self.minPositionX = 0.0
        self.maxPositionX = 10.0 #Example vars for x position observation bounds

        # Define Action Space, see https://github.com/openai/gym/tree/master/gym/spaces for types
        # spaces.Discrete(5) will create 5 discrete possible actions, which will be passed as a
        # an integer to step(...) as the var action in range [0, n-1] with n actions
        self.action_space = spaces.Discrete(21)

        # Define Observation Space using spaces as in Action, in a spaces.Box there must be a [low, high]
        self.observation_space = spaces.Box(self.minPositionX, self.maxPositionX)

    def step(self, action):
        """
        ----------
        action : int
        return obs, reward, resetFlag, info
            see https://gym.openai.com/docs/#observations
        """
        
        """
        ... Operations on action, i.e.
        self.set_action()
        
        ... Operations to determine reward, i.e.
        reward = self.get_reward()
        
        ... Operations to get observations, i.e.
        obs = self.get_state()
        
        ... Operations to determine if reset conditions met, i.e.
        resetFlag = self.check_is_done()
        """
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
        return obs
