# RobotLearningBaselines
A repository containing algorithms + trained weights for common robot learning baselines. All the code here is intended to work out of the box so you can compare your fancy new method to SOTA right away! This repository is based on the amazing RLBench simulator.

# Quickstart guide
## Collecting a dataset
1. The script to do this can be found under the `RLBench/tools/dataset_generator.py` folder (Note, the `RLBench` folder will exist only if you clone h2r's fork of the RLBench library. The fork can be found [here](https://github.com/h2r/RLBench)).
2. The script takes 3 important arguments:
    1. `--task`: the name of an RLBench task you want to collect data for.
    2. `--processes`: The number of different processes you want to run in parallel (it's a good idea to set this to the number of CPU's available on your machine)
    3. `--episodes_per_task`: For each process, how many episodes of the robot completing the task do you want to collect? (In total, the dataset will contain `processes * episodes` number of trajectories)
    4. [Optional] `--save_path`: the location you want to save your dataset to. By default, `/tmp/rlbench_data/` will be used.
3. Voila! There will now be a folder in your `save_path` containing one subfolder per trajectory.
4. Full example command: `python RLBench/tools/dataset_generator.py --task push_button-state-v0 --processes 8 --episodes_per_task 125`

## Training a model
1. Depending on the learning method you want to use, a different script can (currently) be found under the `gail` folder.
2. Sample command: `python gail/bc_gym.py --env-name push_button-state-v0 --expert-traj-path /tmp/rlbench_data/push_button --save-model-interval 250 --gpu-index 0 --max-iter-num 100000 --l2-reg 0.00005 --version bc_button1000_newloss_16 -lc 0.000 -l2 1`

## Evaluating and visualizing performance of trained weights
1. Currently, the file being used for eval is `gail/eval_model.py`
2. Sample command: `python gail/eval_model.py --env-name push_button-state-v0 --max-iter-num 10 --version state-v0_bc_button1000_newloss_16/3000`