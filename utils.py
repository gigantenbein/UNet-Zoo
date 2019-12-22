import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from medpy.metric import jc
import logging

import numpy as np
import os

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:
    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)

    def rotate_image_as_onehot(img, angle, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = rotate_image(convert_to_onehot(img, nlabels=nlabels), angle, interp)
        return np.argmax(onehot_output, axis=-1)

    def resize_image(im, size, interp=cv2.INTER_LINEAR):

        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        return im_resized

    def resize_image_as_onehot(im, size, nlabels, interp=cv2.INTER_LINEAR):

        onehot_output = resize_image(convert_to_onehot(im, nlabels), size, interp=interp)
        return np.argmax(onehot_output, axis=-1)


    def deformation_to_transformation(dx, dy):

        nx, ny = dx.shape

        # grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
        grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # Robin's change to make it work with non-square images

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        return map_x, map_y

    def dense_image_warp(im, dx, dy, interp=cv2.INTER_LINEAR, do_optimisation=True):

        map_x, map_y = deformation_to_transformation(dx, dy)

        # The following command converts the maps to compact fixed point representation
        # this leads to a ~20% increase in speed but could lead to accuracy losses
        # Can be uncommented
        if do_optimisation:
            map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)
        return cv2.remap(im, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT) #borderValue=float(np.min(im)))


    def dense_image_warp_as_onehot(im, dx, dy, nlabels, interp=cv2.INTER_LINEAR, do_optimisation=True):

        onehot_output = dense_image_warp(convert_to_onehot(im, nlabels), dx, dy, interp, do_optimisation=do_optimisation)
        return np.argmax(onehot_output, axis=-1)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s + 1e-6)


def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,...]
        X_white[ii,...] = normalise_image(Xc)

    return X_white.astype(np.float32)


def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)


def generalised_energy_distance(sample_arr, gt_arr, nlabels=1, **kwargs):

    def dist_fct(m1, m2):

        label_range = kwargs.get('label_range', range(nlabels))

        per_label_iou = []
        for lbl in label_range:

            # assert not lbl == 0  # tmp check
            m1_bin = (m1 == lbl)*1
            m2_bin = (m2 == lbl)*1

            if torch.sum(m1_bin) == 0 and torch.sum(m2_bin) == 0:
                per_label_iou.append(1)
            elif torch.sum(m1_bin) > 0 and torch.sum(m2_bin) == 0 or torch.sum(m1_bin) == 0 and torch.sum(m2_bin) > 0:
                per_label_iou.append(0)
            else:
                per_label_iou.append(jc(m1_bin.detach().cpu().numpy(), m2_bin.detach().cpu().numpy()))

        # print(1-(sum(per_label_iou) / nlabels))

        return 1-(sum(per_label_iou) / nlabels)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
            d_sy.append(dist_fct(sample_arr[i,...], gt_arr[j,...]))

    for i in range(N):
        for j in range(N):
            # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
            d_ss.append(dist_fct(sample_arr[i,...], sample_arr[j,...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i,...], gt_arr[j,...]))

    return (2./(N*M))*sum(d_sy) - (1./N**2)*sum(d_ss) - (1./M**2)*sum(d_yy)

def variance_ncc_dist(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):

        log_samples = np.log(m_samp + eps)

        return -1.0*np.sum(m_gt*log_samples, axis=0)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """
    sample_arr = sample_arr.detach().cpu().numpy()
    gt_arr = gt_arr.detach().cpu().numpy()

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[2]
    sY = sample_arr.shape[3]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)


def show_tensor(tensor):
    """Show images with matplotlib for debugging, only for 128,128"""
    with torch.no_grad():
        import matplotlib.pyplot as plt
        try:
            tensor = tensor.detach()
        except:
            pass

        height = tensor.shape[2]
        width = tensor.shape[3]

        batch_size = tensor.shape[0]

        result = tensor[0].view(height, width)
        for i in range(1, batch_size):
            result = torch.cat([result, tensor[i].view(height, width)], dim=1)

        plt.imshow(result, cmap='Greys_r')

# def convert_to_onehot(lblmap, nlabels):
#
#     output = torch.zeros((lblmap.shape[0], nlabels, lblmap.shape[2], lblmap.shape[3]))
#     for ii in range(nlabels):
#         output[:, ii, :, :] = (lblmap == ii).view(-1, lblmap.shape[2], lblmap.shape[3]).long()
#
#     assert output.shape == (lblmap.shape[0], nlabels, lblmap.shape[2], lblmap.shape[3])
#
#     return output
def convert_to_onehot(lblmap, nlabels):

    output = np.zeros((lblmap.shape[0], lblmap.shape[1], nlabels))
    for ii in range(nlabels):
        output[:, :, ii] = (lblmap == ii).astype(np.uint8)
    return output


# needs a torch tensor as input instead of numpy array
# accepts format HW and CHW
def convert_to_onehot_torch(lblmap, nlabels):
    if len(lblmap.shape) == 3:
        # 2D image
        output = torch.zeros((nlabels, lblmap.shape[-2], lblmap.shape[-1]))
        for ii in range(nlabels):
            lbl = (lblmap == ii).view(lblmap.shape[-2], lblmap.shape[-1])
            output[ii, :, :] = lbl
    elif len(lblmap.shape) == 4:
        # 3D images from brats are already one hot encoded
        output = lblmap
    return output.long()



def convert_batch_to_onehot(lblbatch, nlabels):
    out = []
    for ii in range(lblbatch.shape[0]):
        lbl = convert_to_onehot_torch(lblbatch[ii,...], nlabels)
        # TODO: check change
        out.append(lbl.unsqueeze(dim=0))

    result = torch.cat(out, dim=0)
    return result


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def convert_nhwc_to_nchw(tensor):
    result = tensor.transpose(1, 3).transpose(2, 3)
    return result


def convert_nchw_to_nhwc(tensor):
    result = tensor.transpose(1, 3).transpose(1, 2)
    assert torch.equal(tensor, convert_nhwc_to_nchw(result))
    return result

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

