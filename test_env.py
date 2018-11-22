import torch
import numpy as np
import time

from gathering_mae import GatheringEnv
from configs import get_config

if __name__ == '__main__':

    # Get default config
    cfg = get_config("default")

    EVAL_STEPS = 100000

    no_agents = cfg.no_agents
    env = GatheringEnv(cfg)
    steps = 0

    def restart_game():
        obs = env.reset()
        reward = torch.zeros(no_agents)
        done = False

        return obs, reward, done

    done = True

    # Force visual
    cfg.visualize = True
    visualize = cfg.visualize

    start_time = time.time()
    while steps < EVAL_STEPS:

        # check if env needs reset
        if done:
            obs, r, done= restart_game()

        actions = np.random.randint(7, size=no_agents)

        obs, r, done, _ = env.step(actions)

        steps += 1

        if steps % 10000 == 0:
            print(time.time() - start_time)
            start_time = time.time()
