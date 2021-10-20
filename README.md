# Implicit_Distributional_RL
This is the official repo, which contains Pytorch and Tensorflow implementation of algorithm IDAC, proposed in the paper *Implicit Distributional Reinforcement Learning* (https://arxiv.org/abs/2007.06159)

The pytorch implementation is based on pytorch>=1.8.0, which is easier to use compared to tensorflow version, since the tensorflow code is old and based on tensorflow==1.4. 

### Quick Start
To run the experiment, just use run_idac.py file, for example,

`python run_idac.py
--device "cuda:0"
--env_name "Hopper-v2"`

By simply replacing the `env_name` with other MuJoCo environments, such as `HalfCheetah-v2`, you could train an IDAC agent on the selected tasks. 

### Requirements
* `pytorch >= 1.8.0`
* `mujoco-py<2.1,>=2.0`

### Hyper-parameters
The pytorch-version is a new implementation, and hence the hyper-parameter setting could be a little bit different from the original paper suggestions.
We recommend to use default values in the `run_idac.py` file, only except the `use_automatic_entropy_tuning` parameter. 

| env      | use_automatic_entropy_tuning |
| ----------- | ----------- |
| Hopper-v2      | False; alpha=0.3       |
| Walker2d-v2   | True        |
| HalfCheetah-v2   | True        |
| Ant-v2   | True        |
| Humanoid-v2   | True        |

## Citation
```bibtex
@inproceedings{yue2020implicit,
  title={Implicit Distributional Reinforcement Learning},
  author={Yue, Yuguang and Wang, Zhendong and Zhou, Mingyuan},
  booktitle = {NeurIPS 2020: Advances in Neural Information Processing Systems},
  month={Dec.},
  year = {2020}
}
