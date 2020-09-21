import argparse
import gym
import rlbench.gym
import os
import sys
import pickle
import time
import math
from glob import glob
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import VisionPolicy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from core.loss_func import BehaviorCloneLoss

from pyquaternion import Quaternion

def pose_dif(end, start):
    pos_dif = end[:3] - start[:3]

    q_end = Quaternion(np.concatenate([end[-1:], end[3:-1]]))
    q_start = Quaternion(np.concatenate([start[-1:], start[3:-1]])).inverse

    m_end = q_end.rotation_matrix
    m_start = q_start.rotation_matrix
    m_dif = np.matmul(m_end, m_start)

    q_dif = Quaternion(matrix=m_dif)

    dif = np.concatenate([pos_dif, q_dif.imaginary, [q_dif.real]])
    return dif


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="hit_ball_with_queue", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--version', default="bc", metavar='G',
                    help='version of weights to load')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size per update (default: 128)')
parser.add_argument('--max-iter-num', type=int, default=2048, metavar='N',
                    help='maximal number of main iterations (default: 2000)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('-l1', '--lambda_l1', default=1, type=float, help='l1 loss weight')
parser.add_argument('-l2', '--lambda_l2', default=.01, type=float, help='l2 loss weight')
parser.add_argument('-lc', '--lambda_c', default=.0005, type=float, help='c loss weight')
parser.add_argument('-la', '--lambda_aux', default=1, type=float, help='aux loss weight')
args = parser.parse_args()
args.env_name = args.env_name + '-vision-v0'

os.makedirs(os.path.join(assets_dir(), 'learned_models/{}_{}'.format(args.env_name, args.version)))#, exist_ok=True)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

def key(x):
    return int(x[x.rfind('/')+1:x.rfind('.')])

"""environment"""
print('Data Setup')
smoothing = 5

expert_traj = [sorted(glob(paths+'/*'), key=key) for paths in glob(args.expert_traj_path+'/*')]
expert_traj = [[pickle.load(open(path, mode='rb')) for path in paths] for paths in expert_traj]

state_dim = (expert_traj[0][0].get_low_dim_data().shape[0], expert_traj[0][0].front_rgb.shape)
action_dim = np.hstack([expert_traj[0][0].gripper_pose, expert_traj[0][0].gripper_open]).shape[0]

expert_traj = [[[obs.get_low_dim_data(), obs.front_rgb, obs.gripper_pose, obs.gripper_open] for obs in traj] for traj in expert_traj]
expert_traj = [np.hstack([traj[i][0], traj[i][1].flatten(), pose_dif(traj[i+min(smoothing, len(traj)-i-1)][2], traj[i][2]), traj[i+min(smoothing, len(traj)-i-1)][3]]) for traj in expert_traj for i in range(len(traj))]
#expert_traj = [np.hstack([traj[i][0], traj[i+min(3, len(traj)-i-1)][1], traj[i+min(3, len(traj)-i-1)][2]]) for traj in expert_traj for i in range(len(traj))]
expert_traj = np.vstack(expert_traj)
expert_traj = torch.from_numpy(expert_traj).type(dtype)
train_data = expert_traj[:-1*expert_traj.shape[0]//10]
test_data  = expert_traj[-1*expert_traj.shape[0]//10:]
print(train_data.shape[0])
print(test_data.shape[0])

#print(expert_traj.mean(dim=0))
#print(expert_traj.std(dim=0))

print('Data Setup Complete')
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)

"""define actor and critic"""
policy_net = VisionPolicy(state_dim, action_dim)
criterion = BehaviorCloneLoss(args.lambda_l2, args.lambda_l1, args.lambda_c, args.lambda_aux)
to_device(device, policy_net)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)


print(policy_net)
print(args)


print('Setup complete, beginning training.')
n_batches_train = math.ceil(train_data.shape[0] / args.batch_size)
n_batches_test = math.ceil(test_data.shape[0] / args.batch_size)
for i_iter in range(1, args.max_iter_num+1):
    train_loss = 0
    test_loss = 0
    id = torch.randperm(train_data.shape[0])
    train_data = train_data[id]
    policy_net.train()
    for i_batch in range(n_batches_train):
        optimizer_policy.zero_grad()
        batch_inds = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)
        batch = train_data[batch_inds].to(device)
        batch_ins = (batch[:, :state_dim[0]], batch[:, state_dim[0]:-action_dim].view(-1, *state_dim[1]).permute(0, 3, 1, 2))
        batch_outs = batch[:, -action_dim:]
        policy_outs, _, _ = policy_net(*batch_ins)
        loss = criterion(policy_outs, batch_outs)
        loss.backward()
        optimizer_policy.step()
        train_loss += loss.item() * batch.shape[0] / train_data.shape[0]

    policy_net.eval()
    with torch.no_grad():
        for i_batch in range(n_batches_test):
            batch_inds = slice(i_batch*args.batch_size, (i_batch+1)*args.batch_size)
            batch = test_data[batch_inds].to(device)
            batch_ins = (batch[:, :state_dim[0]], batch[:, state_dim[0]:-action_dim].view(-1, *state_dim[1]).permute(0, 3, 1, 2))
            batch_outs = batch[:, -action_dim:]
            policy_outs, _, _ = policy_net(*batch_ins, b_print=i_batch==0)
            loss = criterion(policy_outs, batch_outs)#, i_batch == 0)
            test_loss += loss.item() * batch.shape[0] / test_data.shape[0]

    if i_iter % args.log_interval == 0:
        print('{}\ttrain_loss {:.5f}\ttest_loss {:.5f}'.format(
            i_iter, train_loss, test_loss))

    if args.save_model_interval > 0 and i_iter % args.save_model_interval == 0:
        to_device(torch.device('cpu'), policy_net)
        pickle.dump((policy_net, {}, {}), open(os.path.join(assets_dir(), 'learned_models/{}_{}/{}.p'.format(args.env_name, args.version, i_iter)), 'wb'))
        to_device(device, policy_net)

    """clean up gpu memory"""
    torch.cuda.empty_cache()
