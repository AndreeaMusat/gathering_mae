import torch


class MultiAgentBaseEnv:
    def __init__(self, cfg):
        self.no_agents = cfg.no_agents
        self.use_cuda = cfg.use_cuda
        self.max_steps = cfg.env_max_steps_no

        # Recorder
        self.record_data = cfg.record_data.turned_on
        if self.record_data:
            cfg.visualize = True
            self.record_data_prop = cfg.record_data

        self.step_cnt = -1
        self.ep_cnt = -1
        self.prev_observation = None
        self._record_ep = False
        self.recorded_data = None

    def reset(self):
        self._record_ep = False
        self.ep_cnt += 1
        self.step_cnt = 0
        observation = self._reset()
        self.prev_observation = observation
        return observation

    def restart_game(self, record_episode=False):
        no_agents = self.no_agents
        obs = self.reset()
        reward = torch.zeros(no_agents)

        self._record_ep = record_episode
        if record_episode:
            self.recorded_data = self.init_ep_record_data()

        return obs, reward, torch.zeros(no_agents).byte()

    def get_recorded_data(self):
        return self.record_data

    def step(self, action):
        observation, reward, done = self._step(action)
        self.step_cnt += 1
        if self.step_cnt >= self.max_steps:
            done = torch.ones(self.no_agents).byte()

        if self._record_ep:
            self.record_step_data(observation, reward, done)

        self.prev_observation = observation
        return observation, reward, done, {}

    def record_step_data(self, observation, reward, done):
        raise NotImplementedError

    def init_ep_record_data(self):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        return self._render(*args, **kwargs)

    def _render(self, *args, **kwargs):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    @property
    def action_space(self):
        """ Return gym spaces variable """
        raise NotImplementedError

    @property
    def observation_space(self):
        """ Return torch tensor Size """
        raise NotImplementedError

    @property
    def is_cuda(self):
        return self.use_cuda
