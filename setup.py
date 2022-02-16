# Please don't change the order of following packages!
import sys
from distutils.core import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name="haco",
    install_requires=[
        "yapf==0.30.0",
        "tensorflow==2.3.1",
        "tensorflow-probability==0.11.1",
        "tensorboardX",
        "metadrive-simulator==0.2.3",
        "loguru",
        "imageio",
        "easydict",
        "tensorboardX",
        "pyyaml",
        "gym==0.18.0",
        "ray[all]==1.0.0",
        "stable_baselines3",
    ],
)
