import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

at_biwi = True  # Are you running this code from the ETH Computer Vision Lab (Biwi)?

project_root = '/scratch_net/airfox/ganmarc/UNet-Zoo'
log_root = '/scratch_net/airfox/ganmarc/log'

data_root = '/scratch_net/airfox/ganmarc/data/'
dummy_data_root = None

data_root = '/scratch_net/airfox/ganmarc/data/data_lidc.pickle'

uzh_segmentations = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/UZH_Prostate_annotations_Christian'
uzh_images = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/Prostate/'

uzh_preproc_folder = '/scratch_net/airfox/ganmarc/data/preproc'