import sys
sys.path.remove('/home/nishanth/Desktop/RobotLearningBaselines/utils')
import torch
from torch.utils.data import Dataset

import lmdb
from multiprocessing import Pool

import rlbench.gym
import os.path as osp
import pyarrow as pa
import pickle
import random

class ImitationLMDBWithVision(Dataset):
    def __init__(self, dest, mode):
        super(ImitationLMDBWithVision, self).__init__()
        lmdb_file = osp.join(dest, mode+".lmdb")
        # Open the LMDB file
        self.env = lmdb.open(lmdb_file, subdir=osp.isdir(lmdb_file),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = self.loads_pyarrow(txn.get(b'__len__'))
            self.keys = self.loads_pyarrow(txn.get(b'__keys__'))

        # We don't need to shuffle here because this will be done by the DataLoader
        self.shuffled = [i for i in range(self.length)]
        # random.shuffle(self.shuffled)

    def loads_pyarrow(self, buf):
        return pa.deserialize(buf)

    def __getitem__(self, idx):
        low_dim_data, front_rgb, gripper_pose, gripper_open = (None, None, None, None)
        
        index = self.shuffled[idx]
        env = self.env
        # Go grab the byte object using the key that maps to this idx
        with env.begin(write=False) as txn:
            byteflow = txn.get(str(self.keys[index]).encode('utf-8'))

        rlbench_obs = pickle.loads(byteflow)

        # load data
        low_dim_data = rlbench_obs.get_low_dim_data()
        front_rgb = rlbench_obs.front_rgb
        gripper_pose = rlbench_obs.gripper_pose
        gripper_open = rlbench_obs.gripper_open

        # These are the data fields the old version of the network
        # (Jonathan era) were using... We may want to extract things like the aux,
        # tau and target from rlbench

        # rgb = torch.from_numpy(unpacked[0]).type(torch.FloatTensor)
        # depth = torch.from_numpy(unpacked[1]).type(torch.FloatTensor)
        # eof = torch.from_numpy(unpacked[2]).type(torch.FloatTensor)
        # tau = torch.from_numpy(unpacked[3]).type(torch.FloatTensor)
        # aux = torch.from_numpy(unpacked[4]).type(torch.FloatTensor)
        # target = torch.from_numpy(unpacked[5]).type(torch.FloatTensor)

        return [low_dim_data, front_rgb, gripper_pose, gripper_open]

    def __len__(self):
        return self.length

    def close(self):
        self.env.close()