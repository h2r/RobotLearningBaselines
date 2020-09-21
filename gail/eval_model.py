import argparse
import gym
import rlbench.gym
import os
import sys
import pickle
import time
from glob import glob
import math
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="hit_ball_with_queue", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--mode', default="state", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--version', default="bc", metavar='G',
                    help='version of weights to load')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
args.env_name = args.env_name# + '-' + args.mode + '-v0'

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=1)

env = gym.make(args.env_name, render_mode='human')

"""seeding"""
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#env.seed(args.seed)

"""define actor and critic"""
policy, _, _ = pickle.load(open(os.path.join(assets_dir(), 'learned_models/{}_{}.p'.format(args.env_name, args.version)), 'rb'))
policy.eval()
def main_loop():
    num_pressed = 0
    with torch.no_grad():
        for i_iter in range(args.max_iter_num):
            seed = torch.randint(0, 1, (1,)).item()
            #np.random.seed(seed)
            state = env.reset()
            done = False
            i_step = 0
            while not done and i_step < 100:
                if args.mode == 'vision':
                    state_var = (tensor(state['state']).unsqueeze(0), tensor(state['front_rgb']).permute(2, 1, 0).unsqueeze(0).type(dtype))
                    action = policy(*state_var)[0][0].numpy()
                else:
                    state_var = tensor(state).unsqueeze(0)
                    action = policy(state_var)[0][0].numpy()

                try:
                    state, _, done, _ = env.step(action)
                except Exception:
                    break
                #env.render()
                i_step += 1
            num_pressed += int(done)
            print('successfully pressed %d of %d' % (num_pressed, i_iter+1), end='\r')
    print('successfully pressed %d of %d      ' % (num_pressed, args.max_iter_num))

main_loop()
