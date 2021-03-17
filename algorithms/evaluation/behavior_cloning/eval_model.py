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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils import *

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="push_button-state-v0", metavar='G',
                    help=("name of the environment to run. IMPORTANT: specify state or" 
                    "vision here; eg. push_button-state-v0 vs push_button-vision-v0"))
parser.add_argument('--mode', default="state", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--weights-location', default="bc", metavar='G',
                    help='absolute folder location of weights to load')
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

env = gym.make(args.env_name, render_mode='human', observation_mode=args.mode, arm_action_mode='delta_ee_pose_world_frame')

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
policy, _, _ = pickle.load(open(args.weights_location, 'rb'))
policy.eval()
def main_loop():
    num_pressed = 0
    with torch.no_grad():
        for i_iter in range(args.max_iter_num):
            # seed = torch.randint(0, 1, (1,)).item()
            #np.random.seed(seed)
            state = env.reset()
            done = False
            i_step = 0
            while not done and i_step < 100:
                if args.mode == 'vision':
                    state_var = (tensor(state['front_rgb']).permute(2, 1, 0).unsqueeze(0).type(dtype))
                    action = policy(state_var)[0][0].numpy()
                else:
                    # Via some experimentation, we've found that trying to do 
                    # Imitation Learning on the full 80-dimensional observation that's
                    # returned by default does not work.
                    # Instead, we only need 3 quantities:
                    # (gripper_open, gripper_pose and task_low_dim_state)
                    state_var = tensor(state) # <- this is 80-dimensional
                    gripper_open = state_var[0].unsqueeze(0) # <- 1 dimensional
                    gripper_pose = state_var[22:29] # <- 7 dimensional
                    task_low_dim_state = state_var[37:] # <- IMPORTANT: for push_button, this is 43 dims, but it may be different for other tasks. Maybe find a way to nicely engineer this in?
                    problem_state = torch.cat((gripper_open, gripper_pose, task_low_dim_state))
                    problem_state = problem_state.unsqueeze(0)
                    action = policy(problem_state)[0][0].numpy()

                try:
                    # print(action)
                    state, _, done, _ = env.step(action)
                except Exception:
                    print('Action output from your model was nonsensical - RIP')
                    break
                #env.render()
                i_step += 1
            num_pressed += int(done)
            print('successfully pressed %d of %d' % (num_pressed, i_iter+1), end='\r')
    print('successfully pressed %d of %d      ' % (num_pressed, args.max_iter_num))
    env.close()

main_loop()

# Sample command:
# python algorithms/evaluation/eval_model.py --env-name push_button-state-v0 --max-iter-num 10 --weights-location push_button-state-v0-state-v0_latest_model_seanparams/60250 --mode state
