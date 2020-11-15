import multiprocessing
import math
import time
import gym
import rlbench.gym
import os
import sys

# We need this below line for Python to actually find the utils directory
# Otherwise, the module is not in the Python path!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.torch import *
from utils.replay_memory import Memory

def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, min_batch_size):
    try:
        if isinstance(env, str):
            if render:
                env = gym.make(env, render_mode='human')
            else:
                env = gym.make(env)

        torch.randn(pid)
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        total_c_reward = 0
        min_c_reward = 1e6
        max_c_reward = -1e6
        num_episodes = 0


        while num_steps < min_batch_size:
            state = env.reset()
            reward_episode = 0

            for t in range(600):
                state_var = tensor(state).unsqueeze(0)
                with torch.no_grad():
                    if mean_action:
                        action = policy(state_var)[0][0].numpy()
                    else:
                        action = policy.select_action(state_var)[0].numpy()
                action = int(action) if policy.is_disc_action else action.astype(np.float64)
                next_state, reward, done, _ = env.step(action)
                reward_episode += reward

                if custom_reward is not None:
                    reward = custom_reward(state, action)
                    total_c_reward += reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

                mask = 0 if done else 1

                memory.push(state, action, mask, next_state, reward)

                if render:
                    env.render()
                if done:
                    break

                state = next_state

            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward
        if custom_reward is not None:
            log['total_c_reward'] = total_c_reward
            log['avg_c_reward'] = total_c_reward / num_steps
            log['max_c_reward'] = max_c_reward
            log['min_c_reward'] = min_c_reward

        if queue is not None:
            queue.put([pid, memory, log])
        else:
            return memory, log
    except Exception as e:
        if queue is not None:
            queue.put([pid, memory, log])
        else:
            raise e


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env_name, policy, device, custom_reward=None,
                 mean_action=False, render=False, num_threads=1):
        self.env_name = env_name
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads):
            worker_args = (i, queue, self.env_name, self.policy, self.custom_reward, self.mean_action,
                           False, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        print('compiling workers')
        for _ in workers:
            print('getting worker')
            e = queue.get()
            print('got worker')
            try:
                pid, worker_memory, worker_log = e
                worker_memories[pid - 1] = worker_memory
                worker_logs[pid - 1] = worker_log
            except Exception:
                print(e)
        print('appending memories')
        memory = worker_memories[0]
        for worker_memory in worker_memories[1:]:
            memory.append(worker_memory)
        print('sampling memories')
        if len(memory) ==0:
            raise Exception('Empty memory!!')
        batch = memory.sample()
        print('merging logs')
        if self.num_threads > 1:
            log = merge_log(worker_logs)
        to_device(self.device, self.policy)
        t_end = time.time()
        print('building stats')
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
