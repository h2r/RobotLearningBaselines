from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import OpenDrawer
from rlbench.tasks import OpenDoor
from rlbench.tasks import OpenWindow
from rlbench.tasks import ToiletSeatUp
from rlbench.tasks import PushButton
from rlbench.tasks import ReachAndDrag
from rlbench.tasks import PlayJenga
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

#task = env.get_task(ReachTarget)
task = env.get_task(ToiletSeatUp)
#task = env.get_task(ReachAndDrag)

episodes = 5
for _ in range(episodes): #Try the task this many times
    task.reset()

    print("Task waypoints:")

    waypoints = task._task.get_waypoints()
    grasped = False
    for i, point in enumerate(waypoints):
        #i = 0
        #point = waypoints[0]

        point.start_of_path()

        try:
            path = point.get_path()
        except ConfigurationPathError as e:
            raise DemoError(
                'Could not get a path for waypoint %d.' % i,
                self._active_task) from e

        ext = point.get_ext()
        print("Ext:",ext)

        for joint_pose in chunks(path._path_points,7):
            full_joint_pose = np.concatenate([joint_pose, [1.0]], axis=-1) #gripper open
            obs, reward, terminate = task.step(full_joint_pose)

        point.end_of_path()

        if "close_gripper()" in ext:
            full_joint_pose = np.concatenate([joint_pose, [0]], axis=-1) #gripper open
            for _ in range(10):
                obs, reward, terminate = task.step(full_joint_pose)

            break

env.shutdown()