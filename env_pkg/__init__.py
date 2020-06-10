from gym.envs.registration import register

register(id="foo-v0", entry_point="env_pkg.envs:FooEnv")
