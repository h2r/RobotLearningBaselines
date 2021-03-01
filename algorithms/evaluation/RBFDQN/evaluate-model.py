import gym
import rlbench.gym
import sys
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from algorithms.RBFDQN.reaching_Her_RBFDQN_joint import Net
from algorithms.RBFDQN import utils_for_q_learning, buffer_class

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
hyper_parameter_name = sys.argv[1]
alg = 'rbf'
params = utils_for_q_learning.get_hyper_parameters(hyper_parameter_name, alg)
params['hyper_parameters_name'] = hyper_parameter_name

env = gym.make('reach_target-state-v0', render_mode="human"  )
params['env'] = env
params['seed_number'] = int(sys.argv[2])
s0 = env.reset()
#override s to make observation space smaller
# new s is [arm.get_joint_positions(), tip.get_pose(), target.get_position()] 
s0 = [*s0[8:15], *s0[22:29], *s0[-3:]]

# Load the trained agent
Q_object = Net(params,
                   env,
                   state_size=len(s0),
                   action_size=len(env.action_space.low),
                   device=device)
Q_object.load_state_dict(torch.load('trained_weights/RBFDQN/obj_net'))
Q_object.eval()


# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent

num_episodes = 100

for j in range(num_episodes):
    obs = env.reset()
    obs = [*obs[8:15], *obs[22:29], *obs[-3:]]
    for i in range(4000):
        action = Q_object.e_greedy_policy(obs, j + 1, 'test')

        obs, reward, done, info = env.step(action)
        obs = [*obs[8:15], *obs[22:29], *obs[-3:]]
        print("action taken:", action, "finished? ", done)
        
        #env.render()
        if done:
            break
