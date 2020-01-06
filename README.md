# UNet-Zoo
This repository features different U-Net implementations such as vanilla U-Net, reversible U-Net, probabilistic U-Net, PHiSeg and a reversible variant of these as well.

# Setting up the environment

Install PyTorch by following the instructions on pytorch.org. For the pip packages do

>> pip install -r requirements.txt

in your project directory

# Training the models

If you want to train the models with the LIDC dataset, you can find a preprocessed version of the LIDC dataset on the GitHub page of Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

# Acknowledgements

The code for the Probabilistic U-Net has been adapted from Stefan Knegt's implementation https://github.com/stefanknegt/Probabilistic-Unet-Pytorch. The PHiSeg implementation was based on the Tensorflow implementation of https://github.com/baumgach/PHiSeg-code