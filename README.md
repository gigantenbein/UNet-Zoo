# UNet-Zoo
This repository features different U-Net implementations such as vanilla U-Net, reversible U-Net, probabilistic U-Net,
PHiSeg and a reversible version, RevPHiSeg.
The experiments on the LIDC and an in-house prostate dataset with memory savings of up to 30 % have been published at
[UNSURE by MICCAI 2020](https://unsuremiccai.github.io/). The paper can also be found on
[arxiv](https://arxiv.org/abs/2008.06999).

# Setting up the environment

Install PyTorch by following the instructions on pytorch.org. For the pip packages do

```pip install -r requirements.txt```

in your project directory.

# Training the models

If you want to train the models with the LIDC dataset, you can find a preprocessed version of the LIDC dataset
 on the GitHub page of Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

For running the experiments on the LIDC datasets, there are different experiment files which you can
find in the models/experiments folder. To run these experiments, change paths in config/system.py or
config/local_config.py.

The experiment files contain the configuration for the model, training iterations and other parameters.

You can train the models by running train_model.py and passing the desired experiment file and the parameter local with
your script. For example:

```python train_model.py /path/to/the/experiment.py local```

or if you want to run it with the system_config.py configuration, run

```python train_model.py /path/to/the/experiment.py system```

# Acknowledgements

The code for the Probabilistic U-Net has been adapted from [Stefan Knegt's
implementation](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch).
The PHiSeg implementation was based on the Tensorflow implementation of https://github.com/baumgach/PHiSeg-code which
was published at [MICCAI 2019](https://arxiv.org/abs/1906.04045).
For integrating the reversible blocks I took inspiration from
https://github.com/RobinBruegger/PartiallyReversibleUnet which was published at
[MICCAI 2019](https://arxiv.org/abs/1906.06148) aswell.
