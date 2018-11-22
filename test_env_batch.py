import torch
import torch.multiprocessing as mp
from configs import get_config
from logbook import Logger, StreamHandler
from gathering_mae import GatheringEnv
from gathering_mae.torch_types import TorchTypes
import time
import sys

NAMESPACE = 'test_env'
log = Logger(NAMESPACE)


def play_env(id, cfg, shared_mem, exp_queue, exp_queue_in):
    obs, r, done, action = shared_mem

    env = GatheringEnv(cfg)

    r.zero_()
    done.zero_()

    def restart_game():
        obs.copy_(env.reset(), async=True)

    d = True

    while True:
        if d:
            obs.copy_(env.reset(), async=True)

        exp_queue.put((id, True))

        signal = exp_queue_in.get()

        o, r_, done[0], _ = env.step(action)

        obs.copy_(o, async=True)
        r.copy_(r_, async=True)

        if signal is None:
            exp_queue.put(None)
            return


def init_share_objects(no_envs, no_agents, env_demo, torch_type):
    # Optimize for GPU
    shared_obs = torch_type.FT(torch.Size([no_envs]) + env_demo.observation_space)
    shared_action = torch_type.LT(no_envs, no_agents)
    shared_done = torch_type.LT(no_envs, 1)
    shared_rew = torch_type.FT(no_envs, no_agents)

    shared_obs.share_memory_()
    shared_rew.share_memory_()
    shared_done.share_memory_()
    shared_action.share_memory_()

    return [shared_obs, shared_rew, shared_done, shared_action]


if __name__ == '__main__':
    mp.set_start_method('spawn')

    StreamHandler(sys.stdout).push_application()

    log.info("TEST - Artificial Map")
    log.warn('Logbook is too awesome for most applications')

    # Parse cmdl args for the config file and return config as Namespace
    CFG = get_config("default")
    # print(CFG)
    # log.info("Results_folder: {}".format(CFG.general.results_path))

    EVAL_STEPS = 100000

    no_agents = CFG.no_agents
    no_envs = CFG.no_envs
    test_env = GatheringEnv(CFG)
    torch_type = TorchTypes(cuda=CFG.use_cuda)

    max_action = int(test_env.action_space.nvec.max()-1)
    shared_mem = init_share_objects(no_envs, no_agents, test_env, torch_type)
    shared_obs, shared_r, shared_d, shared_act = shared_mem

    def get_agent_i_mem(i): return list(map(lambda x: x[i], shared_mem))

    # Init envs
    play_procs, exp_queues_out = [], []
    exp_queues_in = mp.Queue(maxsize=100)
    for i in range(no_envs):
        exp_queues_out.append(mp.Queue(maxsize=100))
        play_procs.append(mp.Process(target=play_env, args=(i, CFG, get_agent_i_mem(i),
                                                            exp_queues_in, exp_queues_out[i])))
        play_procs[i].start()

    del test_env

    steps = 0
    no_envs_range = range(no_envs)
    start_time = time.time()
    ids = [None] * no_envs
    while steps < EVAL_STEPS:

        for i_env in no_envs_range:
            ids[i_env], signal = exp_queues_in.get()

        shared_act.random_(0, max_action)
        # shared_obs, shared_r, shared_d, shared_act

        for i_env in no_envs_range:
            exp_queues_out[ids[i_env]].put(True)

        steps += no_envs

        if steps % 1000 == 0:
            print(time.time() - start_time)
            start_time = time.time()
