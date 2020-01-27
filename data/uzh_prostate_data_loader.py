
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape[:2]

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny,...]
    elif x == nx and y == ny:
        slice_cropped = slice
    else:
        cropped_shape = list(slice.shape)
        cropped_shape[0] = nx
        cropped_shape[1] = ny
        slice_cropped = np.zeros(cropped_shape)
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :,...] = slice[:, y_s:y_s + ny,...]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y,...] = slice[x_s:x_s + nx, :,...]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y,...] = slice[:, :,...]

    return slice_cropped

def prepare_data(input_image_folder, input_mask_folder, output_file, size, target_resolution):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")

    expert_list = ['Readings_AH', 'Readings_EK', 'Readings_KC', 'Readings_KS', 'Readings_OD', 'Readings_UM']
    num_annotators = len(expert_list)

    logging.info('Counting files and parsing meta data...')
    patient_id_list = {'test': [], 'train': [], 'validation': []}

    image_file_list = {'test': [], 'train': [], 'validation': []}
    mask_file_list = {'test': [], 'train': [], 'validation': []}

    num_slices = {'test': 0, 'train': 0, 'validation': 0}

    logging.info('Counting files and parsing meta data...')

    for folder in os.listdir(input_image_folder):

        folder_path = os.path.join(input_image_folder, folder)
        if os.path.isdir(folder_path) and folder.startswith('888'):

            patient_id = int(folder.lstrip('888'))

            if patient_id == 9:
                logging.info('WARNING: Skipping case 9, because one annotation has wrong dimensions...')
                continue

            if patient_id % 5 == 0:
                train_test = 'test'
            elif patient_id % 4 == 0:
                train_test = 'validation'
            else:
                train_test = 'train'

            file_path = os.path.join(folder_path, 't2_tse_tra.nii.gz')

            annotator_mask_list = []
            for exp in expert_list:
                mask_folder = os.path.join(input_mask_folder, exp)
                file = glob.glob(os.path.join(mask_folder, '*'+str(patient_id).zfill(4)+'_*.nii.gz'))
                # for ii in range(len(file)):
                #     if 'NCI' in file[ii]:
                #         del file[ii]
                assert len(file) == 1, 'more or less than one file matches the glob pattern %s' % ('*'+str(patient_id).zfill(5)+'*.nii.gz')
                annotator_mask_list.append(file[0])

            mask_file_list[train_test].append(annotator_mask_list)
            image_file_list[train_test].append(file_path)

            patient_id_list[train_test].append(patient_id)

            nifty_img = nib.load(file_path)
            num_slices[train_test] += nifty_img.shape[2]

    # Write the small datasets
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))

    nx, ny = size
    n_test = num_slices['test']
    n_train = num_slices['train']
    n_val = num_slices['validation']

    print('Debug: Check if sets add up to correct value:')
    print(n_train, n_val, n_test, n_train + n_val + n_test)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size) + [num_annotators], dtype=np.uint8)

    mask_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}

    logging.info('Parsing image files')

    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0

        patient_counter = 0
        for img_file, mask_files in zip(image_file_list[train_test], mask_file_list[train_test]):

            patient_counter += 1

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % img_file)

            img_dat = utils.load_nii(img_file)
            img = img_dat[0]

            masks = []
            for mf in mask_files:
                mask_dat = utils.load_nii(mf)
                masks.append(mask_dat[0])
            masks_arr = np.asarray(masks)  # annotator, size_x, size_y, size_z
            masks_arr = masks_arr.transpose((1,2,3,0)) # size_x, size_y, size_z, annotator

            img = utils.normalise_image(img)

            pixel_size = (img_dat[2].structarr['pixdim'][1],
                          img_dat[2].structarr['pixdim'][2],
                          img_dat[2].structarr['pixdim'][3])

            logging.info('Pixel size:')
            logging.info(pixel_size)

            scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

            for zz in range(img.shape[2]):

                slice_img = np.squeeze(img[:, :, zz])
                slice_rescaled = transform.rescale(slice_img,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   multichannel=False,
                                                   mode='constant')

                slice_mask = np.squeeze(masks_arr[:, :, zz,:])
                mask_rescaled = transform.rescale(slice_mask,
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  multichannel=True,
                                                  mode='constant')

                slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                # REMOVE SEMINAL VESICLES
                mask_cropped[mask_cropped==3] = 0

                # DEBUG
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(slice_img)
                #
                # plt.figure()
                # plt.imshow(slice_rescaled)
                #
                # plt.figure()
                # plt.imshow(slice_cropped)
                #
                # plt.show()
                # END DEBUG

                img_list[train_test].append(slice_cropped)
                mask_list[train_test].append(mask_cropped)

                write_buffer += 1

                # Writing needs to happen inside the loop over the slices
                if write_buffer >= MAX_WRITE_BUFFER:
                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0


        # after file loop: Write the remaining data

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)

    # After test train loop:
    hdf5_file.close()

def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr

def _release_tmp_memory(img_list, mask_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    gc.collect()

def load_and_maybe_process_data(input_image_folder,
                                input_mask_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data
    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param mode: Can either be '2D' or '3D'. 2D saves the data slice-by-slice, 3D saves entire volumes
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_size_%s_res_%s_no_seminal_vesicles.hdf5' % (size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_image_folder, input_mask_folder, data_file_path, size, target_resolution)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    input_image_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/Prostate'
    input_mask_folder = '/usr/bmicnas01/data-biwi-01/baumgach/uzh_prostate_annotations'
    # preprocessing_folder = '/srv/glusterfs/baumgach/preproc_data/uzh_prostate'
    preprocessing_folder = '/itet-stor/baumgach/net_scratch/preproc_data/uzh_prostate'

    d = load_and_maybe_process_data(input_image_folder,
                                    input_mask_folder,
                                    preprocessing_folder,
                                    (192, 192),
                                    (0.6, 0.6),
                                    force_overwrite=False)
