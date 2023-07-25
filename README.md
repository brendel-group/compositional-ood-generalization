# Compositional Generalization from First Principles
This repository provides the codes for all experiments shown in the paper [Compositional
Generalization from First Principles](https://arxiv.org/abs/2307.05596).

## Setup
- use Python 3.9
- run `pip install -r requirements.txt`
- if desired, a Dockerfile is provided. You can create the container with `docker build .`

## Running the code
- all experiments where run on a single NVIDIA-RTX-2080Ti
- config files for all experiments can be found in `cfgs/`
- to start an experiment, run `main.py --config path/to/experiment/config.yml`

## Citation
If you find the insights from the paper or our code base useful, please cite
```
@misc{wiedemer2023compositional,
      title={Compositional Generalization from First Principles}, 
      author={Thaddäus Wiedemer and Prasanna Mayilvahanan and Matthias Bethge and Wieland Brendel},
      year={2023},
      eprint={2307.05596},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```