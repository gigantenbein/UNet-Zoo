import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

at_biwi = False  # Are you running this code from the ETH Computer Vision Lab (Biwi)?

project_root = '/Users/marcgantenbein/PycharmProjects/UNet-Zoo'
log_root = '/Users/marcgantenbein/PycharmProjects/UNet-Zoo/log'

dummy_data_root = '/Users/marcgantenbein/PycharmProjects/UNet-Zoo/data'

data_root = '/Users/marcgantenbein/scratch/data/data_lidc.pickle'

uzh_root = '/Users/marcgantenbein/scratch/data/prostate_original.mat'

preproc_folder = '/Users/marcgantenbein/scratch/data/preproc'