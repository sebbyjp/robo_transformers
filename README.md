[![Code Coverage](https://codecov.io/gh/sebbyjp/dgl_ros/branch/code_cov/graph/badge.svg?token=9225d677-c4f2-4607-a9dd-8c22446f13bc)](https://codecov.io/gh/sebbyjp/dgl_ros)

# Library for Robotic Transformers. RT-1 and RT-X-1.
# Instructions:

- Clone this repo: `https://github.com/sebbyjp/robo_transformers.git --recurse-submodules`
- Run `pip install poetry`
- Run `poetry build .`
- Run inference `python robo_transformers/rt1_inference.py`
## Download RT-1-X model from the Open-X Embodiment paper.
- Install gsutil: `pip install gsutil`
- Run: `gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip ./checkpoints/`
- Unzip: `cd checkpoints && unzip rt_1_x_tf_trained_for_002272480_step.zip`
  
## Optional
To install the checkpoints from the robotics_transformer git repo, you will need git-lfs
- Install git-lfs (use brew or apt if on unix)
- Run `git lfs install`
- Run `git lfs clone https://www.github.com/google-research/robotics_transformer.git `