# Outside the Box

This repository contains the implementation and data used in the paper [Outside the Box: Abstraction-Based Monitoring of Neural Networks](https://arxiv.org/abs/1911.09032) to be published at [ECAI 2020](http://ecai2020.eu/). To cite the paper, follow [this link](https://dblp.dagstuhl.de/rec/bibtex/journals/corr/abs-1911-09032) or directly use:

```
@article{outsidethebox19,
  author    = {Thomas A. Henzinger and
               Anna Lukina and
               Christian Schilling},
  title     = {Outside the Box: Abstraction-Based Monitoring of Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1911.09032},
  year      = {2019}
}
```

# Installation

We use Python 3.6 but other Python versions may work as well. The package requirements that need to be installed are found in the file `requirements.txt`.

# Recreation of the results

Below we describe how to obtain the results shown in the paper.

## Data

Due to Gihub's limitation of the file size, one of the datasets needed to be compressed. To use this dataset, you need to manually go to the folder `data/GTSRB/` and unzip the file `train.zip`.

## Models

The repository contains the pretrained models used in the evaluation.
The models can be recomputed using the scripts `run/train_INSTANCE.py` where `INSTANCE` is the name of the model/data combination.

## Evaluation

The scripts to reproduce the figures and tables of the paper are found in the folder `run/`:

- Fig. 2: `plot_toy_model.py`
- Fig. 3: `plot_boxes.py` (Note that this will not produce the exact same figure because we obtained the figure for a network with different training parameters that we forgot to note down.)
- Fig. 4: `plot_explanation_alpha_thresholding.py`
- Table 2: `plot_legend.py`
- Figures 5-8 and Table 3: `run_experiments.py` (This script calls scripts for the individual experiments that can also be run in isolation.)

Intermediate results of the experiments are stored in `.csv` files in the `run/` folder. The final plots are stored in the top folder.
