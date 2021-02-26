# RobotLearningBaselines
A repository containing algorithms + trained weights for common robot learning baselines. All the code here is intended to work out of the box so you can compare your fancy new method to SOTA right away! This repository is based on the amazing RLBench simulator.
Note: the `trained_weights` folder contains files stored with [git LFS](https://git-lfs.github.com/), so make sure you install LFS correctly to pull and use these.

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
1. Depending on the learning method you want to use, a different script can be found under sub-folders of the `algorithms` folder.
2. Sample command: `python algorithms/behavior_cloning/bc_gym.py --env-name push_button-state-v0 --expert-traj-path /tmp/rlbench_data/push_button --save-model-interval 250 --gpu-index 0 --max-iter-num 100000 --l2-reg 0.00005 --version bc_button1000_sean_params_lc0.0005_la2 -lc 0.0005 -l2 1 --l2-reg 0 -la 2`

## Evaluating and visualizing performance of trained weights
1. Currently, the file being used for eval is `algorithms/evaluation/eval_model.py`
2. Sample command: `python algorithms/evaluation/eval_model.py --env-name push_button-state-v0 --max-iter-num 10 --version state-v0_bc_button1000_newloss_16/3000`
3. Evaluating RBFDQN by using following command: 'python3 algorithms/evaluation/evaluate-model.py 99 0'

# ToDo's
- Create a `requirements.txt` file at the root of the repo
- Create a test script that tests that there are no import errors with running existing algorithms.

# Contribution Guide
If you are a member of H2R that's contributing to RobotLearningBaselines, remember to **make your own fork** of the repo. When pull-requesting to merge back into the main fork, ensure the following are true:
1. You have updated the `requirements.txt` file at the root of the directory with any packages necessary to run your new algorithm. Additionally, you have checked that these updates don't break any of the tests for previous algorithms
2. Code for your new algorithm lives under the `algorithms/` folder of the repository and can be run correctly from the root of the repository
3. Any trained weights live inside a subfolder under the `trained_weights` folder of the repository.
