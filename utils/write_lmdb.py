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

# IMPORTANT: Currently, this script is only intended to work for datasets including vision data
# Do not use it on datasets that only take the 51D proprioceptive state.
parser = argparse.ArgumentParser(description='Script to convert a dataset into an LMDB to enable efficient I/O and low memory usage')
parser.add_argument('--raw-data-path', default="/tmp/rlbench_data", metavar='G',
                    help='path to the raw, expert trajectory data')
parser.add_argument('--save-path', default="assets/lmdb_files", metavar='G',
                    help='path at which to save the generated lmdb files')                    
args = parser.parse_args()

# Make a directory at the correct save path
os.makedirs(args.save_path)

# TODO: loop through the files in the raw_data_path, then open, unpickle and read each one
# and convert to an LMDB. Reference Jonathan's previous code if you need to