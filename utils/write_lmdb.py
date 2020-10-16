import argparse
import pdb
import sys
sys.path.remove('/home/nishanth/Desktop/RobotLearningBaselines/utils')
import datetime
from pympler import asizeof
import pyarrow as pa
import lmdb
import gym
import rlbench.gym
import os
import pickle
import re
import time
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

# IMPORTANT: Currently, this script is only intended to work for datasets including vision data
# Do not use it on datasets that only take the 51D proprioceptive state.

def serialize_pyarrow(obj):
    return pa.serialize(obj).to_buffer()

def key(x):
    return int(x[x.rfind('/')+1:x.rfind('.')])

def make_lmdb(mode, trajs):
    # Setting the map_size of the lmdb to this database size doesn't work for some reason.
    # No clue why. I picked a large default value to work so that things are fine for now...
    # TODO: Figure out why the following 4 lines provides an UNDERESTIMATE of the lmdb size necessary
    # object_size = asizeof.asizeof(pickle.load(open(expert_traj[0][0], mode='rb')))
    # list_num_pickles = [len(traj) for traj in expert_traj]
    # total_num_pickles = sum(list_num_pickles)
    # database_size = total_num_pickles * (object_size + 10) # k,v pair stores approximately (object_size + 10 bytes)

    lmdb_path = os.path.join(args.save_path, "{}.lmdb".format(mode))
    isdir = os.path.isdir(lmdb_path)
    key_list = []

    with lmdb.open(lmdb_path, subdir=isdir, map_size=1099511627776 * 2,
        readonly=False, meminit=False, map_async=True) as db:
        txn = db.begin(write=True)
        for path in trajs:
            with open(path, mode='rb') as pickle_file:
                expert_traj_data = pickle.load(pickle_file)
                # Since the data collection setup saves files as '.../<trajectory>/<timestep>.pkl' for every trajectory
                # we can recover this information to be used as the key for storing a trajectory data in the lmdb
                m = re.search("(\d*)/(\d*).pkl",path)
                trajectory = int(m.group(1))
                timestep = int(m.group(2))
                lmdb_key = (trajectory, timestep)
                key_list.append(lmdb_key)
                # Put the key,value pair into the lmdb
                txn.put(str(lmdb_key).encode('utf-8'), pickle.dumps(expert_traj_data))
        # Additionally, put the list of keys and its length into the lmdb
        # We will need easy access to this for training/testing via a PyTorch DataLoader
        txn.put(b'__keys__', serialize_pyarrow(key_list))
        txn.put(b'__len__', serialize_pyarrow(len(key_list)))
    db.close()


parser = argparse.ArgumentParser(description='Script to convert a dataset into an LMDB to enable efficient I/O and low memory usage')
parser.add_argument('--raw-data-path', default="/tmp/rlbench_data", metavar='G',
                    help='path to the raw, expert trajectory data')
parser.add_argument('--save-path', default="assets/lmdb_files", metavar='G',
                    help='path at which to save the generated lmdb files')
parser.add_argument('--train-test-split', default='0.9', metavar='G',
                    help='fraction of trajectories to use for training and testing sets')                    
args = parser.parse_args()

# Make a directory at the correct save path
os.makedirs(args.save_path)

expert_traj = [sorted(glob(paths+'/*'), key=key) for paths in glob(args.raw_data_path+'/*')]
flattened_expert_traj = [timestep for trajectory in expert_traj for timestep in trajectory]
total_num_timesteps = len(flattened_expert_traj)
train_timesteps = round(total_num_timesteps * float(args.train_test_split))
train_traj = flattened_expert_traj[:train_timesteps]
test_traj = flattened_expert_traj[train_timesteps:]
make_lmdb("train", train_traj)
make_lmdb("test", test_traj)
