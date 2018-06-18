import torch
import cv2
from logbook import Logger, StreamHandler
import time
import sys
import numpy as np

from env.gathering_env import GatheringEnv
from utils.config import generate_configs

np.set_printoptions(threshold=np.nan, linewidth=np.nan, precision=2)


NAMESPACE = 'test_env'
log = Logger(NAMESPACE)

if __name__ == '__main__':
    StreamHandler(sys.stdout).push_application()

    log.info("TEST - Artificial Map")
    log.warn('Logbook is too awesome for most applications')

    # -- Parse config file & generate
    procs_no, arg_list = generate_configs()

    log.info("Starting...")

    # -- Just run first experiment from list
    cfg, run_id, path = arg_list[0]

    EVAL_STEPS = 100000

    no_agents = cfg.env.no_agents
    # Force visual
    cfg.env.visualize = True
    visualize = cfg.env.visualize

    env = GatheringEnv(cfg.env)
    env_step = 0
    ep_r = 0

    done = torch.ones(1).byte()

    agent0_r = []
    start_time = time.time()
    while env_step < EVAL_STEPS:
        # check if env needs reset
        if done.any():
            obs, r, done = env.restart_game()
            if visualize:
                env.render()

            print("Episode finished:")
            print("Return per episode: {}".format(sum(agent0_r)))
            print("Agent0 reward: {}".format(np.mean(agent0_r)))

            agent0_r = []

        actions = np.random.randint(7, size=no_agents)

        if visualize:
            # actions[1] = 4
            actions[0] = -1
            while actions[0] == -1:
                key = cv2.waitKey(0) & 0xFF

                # if the 'ESC' key is pressed, Quit
                if key == 27:
                    quit()
                elif key == 82:   # Arrow up
                    actions[0] = 0
                elif key == 83:  # Arrow right
                    actions[0] = 1
                elif key == 84:  # Arrow down
                    actions[0] = 2
                elif key == 81:  # Arrow left
                    actions[0] = 3
                elif key == 113:  # Null action q
                    actions[0] = 4
                elif key == 119:  # Turn clockwise w
                    actions[0] = 5
                elif key == 101:  # Turn counterclockwise e
                    actions[0] = 6
                elif key == 114:  # Action - laser r
                    actions[0] = 7
                else:
                    print("Unknown key: {}".format(key))

            print(actions[0])
        obs, r, done, _ = env.step(actions)

        env_step += 1
        agent0_r.append(r[0])

        if visualize:
            print("Step: {}".format(env_step))
            print(r)
            env.render()

        if env_step % 10000 == 0:
            print(time.time() - start_time)
            start_time = time.time()