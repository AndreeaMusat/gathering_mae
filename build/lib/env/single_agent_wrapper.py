from .gathering_env import GatheringEnv
import torch


class SingleAgentGatheringEnv(GatheringEnv):
    def __init__(self, cfg):
        assert cfg.no_agents == 1, "Config not configured for 1 agent."
        super(SingleAgentGatheringEnv, self).__init__(cfg)

    def reset(self):
        self._record_ep = False
        self.ep_cnt += 1
        self.step_cnt = 0
        observation = self._reset()
        self.prev_observation = observation
        return observation[0]

    def restart_game(self, record_episode=False):
        obs = self.reset()
        reward = 0

        self._record_ep = record_episode
        if record_episode:
            self.recorded_data = self.init_ep_record_data()

        return obs, reward, False

    def get_recorded_data(self):
        return self.record_data

    def step(self, action):
        observation, reward, done = self._step(torch.LongTensor([action]))
        self.step_cnt += 1
        if self.step_cnt >= self.max_steps:
            done = torch.ones(self.no_agents).byte()

        if self._record_ep:
            self.record_step_data(observation, reward, done)

        self.prev_observation = observation
        return observation[0], reward[0].item(), done[0].item(), {}
