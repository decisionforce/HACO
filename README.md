# Human-AI Copilot Optimization (HACO)

[ICLR 22] Official implementation of paper: Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization

[**Webpage**](https://decisionforce.github.io/HACO/) | 
[**Code**](https://github.com/decisionforce/HACO) | 
[**Video**](https://decisionforce.github.io/HACO/#video) |
[**Paper**](https://arxiv.org/pdf/2202.10341.pdf)


## Installation

```bash
# Clone the code to local
git clone https://github.com/decisionforce/HACO.git
cd HACO

# Create virtual environment
conda create -n haco python=3.7
conda activate haco

# Install basic dependency
pip install -e .

conda install cudatoolkit=10.1 cudnn
# Now you can run the training script of HACO in MetaDrive Environment.
```
## Training HACO
HACO is designed for teaching AI to learn a generalizable autonomous driving policy efficiently and safely.
Supported by [MetaDrive](https://github.com/decisionforce/metadrive), the concrete goal of driving tasks is to drive the vehicle to the destination with as fewer collision as possible. 
Also, to prevent driving out of the road which will terminate the episode, please follow the yellow checkpoints indicating navigation information when you are training HACO.

**Note:** we mask the reward signal for HACO agent, so it is a reward-free method.

### Quick Start
Since the main experiment of HACO takes one hour and requires a steering wheel (Logitech G29), we further provide an 
easy task for users to experience HACO.
```bash
cd haco/run_main_exp/
python train_haco_keyboard_easy.py --num-gpus=1
```
In this task, human is authorized to take over the vehicle by pressing **W/A/S/D** and guide or safeguard the agent to 
the destination ("E" can be used to pause simulation). 
Since there is only one map in this task, 10 minutes or 5000 transitions is enough for HACO agent to learn a policy.

### Main Experiment
To reproduce the main experiment reported in paper, run following scripts:
```bash
python train_haco.py --num-gpus=1
```
If steering wheel is not available, set ```controller="keyboard"``` in the script to train HACO agent. After launching this script,
one hour is required for human to assist HACO agent to learn a generalizable driving policy by training in 50 different maps.

### CARLA Experimennt
CARLA used in our experiment is version 0.9.9.4, so pleas follow the instruction in 
[CARLA offical repo](https://github.com/carla-simulator/carla) to install it.
After installation, launch CARLA server by:
```bash
./CarlaUE4.sh -carla-rpc-port=9000 
```

For the interacting with CARLA core, we utilize the CARLA client wrapper implemented in [DI-Drive](https://github.com/opendilab/DI-drive), so new dependencies
is needed. We recommend initializing a **new** conda environment by:
```bash
# Create new virtual environment
conda create -n haco-carla python=3.7
conda activate haco-carla

# Install basic dependency
pip install -e .
# install DI-Engine
pip install di-engine==0.2.0 markupsafe==2.0.1

conda install cudatoolkit=10.1 cudnn
# Now you can run the training script of HACO in CARLA Environment.
```
After all these steps, launch the CARLA experiment through:
```bash
python train_haco_in_carla.py --num-gpus=1
```
Currently, a steering wheel controller is default to reproduce the CARLA Experiment. 
We also provide keyboard interface for controlling vehicles in CARLA, which can be turned on by setting
```keyboard_control:True``` in the training script.
For providing navigation information, there is a status, namely ```command:```, at the upper-left of the interface.

## Training baselines
### RL baselines 
For SAC/PPO/PPO-Lag/SAC-Lag, there is no additional requirement to run the training scripts. 
```bash
# use previous haco environment
conda activate haco  
cd haco/run_baselines
# launch baseline experiment
python train_[ppo/sac/sac_lag/ppo_lag].py --num-gpus=[your_gpu_num]
```

### Human Demonstration
Human demonstration is required to run Imitation Learning (IL). You can collect human demonstration by runing:
```bash
cd haco/utils
python collect_human_data_set.py
```
or you can use the data collected by our human expert [here](https://github.com/decisionforce/HACO/releases/tag/haco-0.0.0)

### CQL/BC
If you wish to run CQL, extra setting is required as follows:
```bash
# ray needs to be updated to 1.2.0
pip install ray==1.2.0
cd haco/run_baselines
# launch baseline experiment
python train_cql.py --num-gpus=0 # do not use gpu
```
For BC training, modify the config ```bc_iter=1000000``` in ```train_cql.py``` to convert the CQL into BC, and re-run this script.  

### Human-in-the-loop baselines and GAIL
To run GAIL/HG-DAgger/IWR, please create a new conda environment and install GPU-version of torch:
```bash
# Create virtual environment
conda create -n haco-torch python=3.7
conda activate haco-torch

# Install basic dependency
pip install -e .

# install torch
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
conda install condatoolkit==9.2
```
Now, IWR/HG-Dagger/GAIL can be trained by:
```bash
cd haco/run_baselines 
python train_[IWR/gail/hg_dagger].py
```




## Reference

```latex
@inproceedings{
    li2022efficient,
    title={Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization},
    author={Quanyi Li and Zhenghao Peng and Bolei Zhou},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=0cgU-BZp2ky}
}
```


