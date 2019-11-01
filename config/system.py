# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

at_biwi = True  # Are you running this code from the ETH Computer Vision Lab (Biwi)?

project_root = '/scratch_net/airfox/ganmarc/UNet-Zoo'
log_root = '/scratch_net/airfox/ganmarc/log'

#data_root = '/scratch_net/airfox/ganmarc/data'
data_root = '/Users/marcgantenbein/scratch/data/'
dummy_data_root = '/Users/marcgantenbein/PycharmProjects/UNet-Zoo/data'

##################################################################################