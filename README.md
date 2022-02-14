# Human-AI Copilot Optimization (HACO)

The official implementation of ICLR 2022 paper: Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization

[**Webpage**](https://decisionforce.github.io/HACO/) | 
[**Code**](https://github.com/decisionforce/HACO) | 
[**Video**](https://decisionforce.github.io/HACO/#video) |
[**Paper**](https://openreview.net/pdf?id=0cgU-BZp2ky)


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

# Now you can run the training script of HACO.
```
## Training HACO

```bash
cd haco/training_script/
python train_haco.py
```

## Training baselines

All baselines in ```haco/baselines``` except BC/CQL can be directly run.

If you wish to run  CQL/BC, some extra environmental, extra setting is required as follows:
```bash
# ray needs to be updated to 1.2.0
pip install ray==1.2.0

# To run GAIL/DAgger, please install GPU-version of torch:
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
conda install condatoolkit==9.2
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


