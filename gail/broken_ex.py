import gym
import rlbench.gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

env = gym.make('push_button-state-v0', render_mode='human')
env.reset()
