# %%

import os
import glob

import numpy as np
import pandas as pd
import nibabel as nib

# %%


def hist_equalize(img, number_bins=256):        
    # get image histogram
    img_hist, bins = np.histogram(img.flatten(), number_bins, density=True)
    cdf = img_hist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    img_eq = np.interp(img.flatten(), bins[:-1], cdf)

    return img_eq.reshape(img.shape)


def gamma_correction(img, gamma):
    return img**(gamma)


def random_gamma_correction(img):
    gamma = np.random.normal(loc=1, scale=0.05)
    return gamma_correction(img, gamma)


def log_transform(img):
    return np.log(img+1e-18)


def threshold_mask(mask, label_values):

    new_mask = np.zeros_like(mask)

    for val in label_values:
        new_mask[mask == val] = 1

    return new_mask


def normalize(img, norm_type='intensity'):

    if norm_type == 'intensity':
        return (img-img.min())/(img.max()-img.min())
    elif norm_type == 'zscore':
        mu = img[img.nonzero()].mean()
        sigma = img[img.nonzero()].std()
        return (img - mu)/(sigma)
    elif norm_type == 'hist-eq':
        return hist_equalize(img)
    else:
        return img


def pad_to_shape(image, mask=None, input_shape=[256, 256, 1]):

#    print('Original Shape : ', image.shape)
    
    delta_a1 = input_shape[0]-image.shape[1]
    delta_a2 = input_shape[1]-image.shape[2]

    if delta_a1 > 0:
        image = np.pad(image, ((0, 0), (0, delta_a1), (0, 0)), 'constant', constant_values=0)
        if mask is not None:
            mask = np.pad(mask, ((0, 0), (0, delta_a1), (0, 0)), 'constant', constant_values=0)
    elif delta_a1 < 0:
        image = image[:, :delta_a1, :]
        if mask is not None:
            mask = mask[:, :delta_a1, :]

    if delta_a2 > 0:
        image = np.pad(image, ((0, 0), (0, 0), (0, delta_a2)), 'constant', constant_values=0)
        if mask is not None:
            mask = np.pad(mask, ((0, 0), (0, 0), (0, delta_a2)), 'constant', constant_values=0)
    elif delta_a2 < 0:
        image = image[:, :, :delta_a2]
        if mask is not None:
            mask = mask[:, :, :delta_a2]

    return image, mask, (delta_a1, delta_a2)


def remove_padding(mask, delta):

    print('Original Shape : ', mask.shape)
    delta_a1, delta_a2 = delta
    delta_a1 = delta_a1 * -1
    delta_a2 = delta_a2 * -1

    if delta_a1 > 0:
        mask = np.pad(mask, ((0, 0), (0, delta_a1), (0, 0)), 'constant', constant_values=0)
    elif delta_a1 < 0:
        mask = mask[:, :delta_a1, :]

    if delta_a2 > 0:
        mask = np.pad(mask, ((0, 0), (0, 0), (0, delta_a2)), 'constant', constant_values=0)
    elif delta_a2 < 0:
        mask = mask[:, :, :delta_a2]

    return mask


def get_roi(LabelImg, padding=0):
    
    shape = LabelImg.shape
    
    hMin = LabelImg.nonzero()[0].min()
    wMin = LabelImg.nonzero()[1].min()
    dMin = LabelImg.nonzero()[2].min()
    
    hMax = LabelImg.nonzero()[0].max()
    wMax = LabelImg.nonzero()[1].max()
    dMax = LabelImg.nonzero()[2].max()
    
    hMin = max(0, hMin-padding)
    hMax = min(shape[0], hMax+padding)
    wMin = max(0, wMin-padding)
    wMax = min(shape[1], wMax+padding)
    dMin = max(0, dMin-padding)
    dMax = min(shape[2], dMax+padding)

    return hMin, hMax, wMin, wMax, dMin, dMax


def get_subject_list(base_datadir,
                     Mask_format='Mask.nii.gz',
                     Image_format='Image.nii.gz',
                     save_path=None):

    dict_list = []
    cols = ['Name', 'Image', 'Mask']

    if isinstance(base_datadir, (str,)):
        base_datadir = [base_datadir]

    for datadir in base_datadir:
        for sub_path, dirs, fnames in os.walk(datadir, followlinks=True):
            if len(fnames) > 0:
                Image = os.path.join(sub_path, Image_format) if Image_format in fnames else None
                Mask = os.path.join(sub_path, Mask_format) if Mask_format in fnames else None
                if Image is not None:
                    data_dict = {
                        'Name': sub_path,
                        'Image': Image,
                        'Mask': Mask,
                    }
                    dict_list.append(data_dict)

    df = pd.DataFrame(dict_list, columns=cols)
    if save_path is not None:
        df.to_csv(os.path.join(save_path, 'subject_list.csv'), index=False)

    return df
def det_transpose_axes(shape):
    a = shape[0]
    b = shape[1]
    c = shape[2]
    
    if(a <= b <= c):
        return([0, 1, 2])
    elif(b <= a <= c):
        return([1, 0, 2])
    else:
        return([2, 0, 1])

def get_subject(img_path, mask_path, label_values, norm_type='intensity'):
    if img_path is None:
        print('Image Path Invalid')
        img = None
    else:
        img = nib.load(img_path).get_fdata()
        transpose_axes = det_transpose_axes(img.shape)
        img = img.transpose(transpose_axes)
        img = normalize(img, norm_type=norm_type)

    if mask_path is None:
        print('No Mask present')
        msk = None
    else:
        msk = nib.load(mask_path).get_fdata()
        #transpose_axes = det_transpose_axes(img.shape)
        msk = msk.transpose(transpose_axes)
        msk = threshold_mask(msk, label_values)

    return img, msk


def get_all_subjects(df, data_col, mask_col, label_values, transpose_axes=[0, 1, 2], name_col=None, norm_type='intensity'):
    """
        To retrive data of all subjects

        df : dataframe from which the images need to be accessed
        data_col : column name which defines the data images which need to masked
        mask_col : column name which defines the mask images (normally segmented data)
        label_values : list of values to threshold
        transpose_axes : axis along which image needs to be transposed
        norm_type : normalization type
    """
    data_list = []
    mask_list = []
    delta_list = []

    for index, row in df.iterrows():

        if name_col is not None:
            name = row[name_col]
            print(name)
        else:
            print(index)

        img, msk = get_subject(row[data_col], row[mask_col], label_values,norm_type=norm_type)
        
        """transpose_axes=transpose_axes"""

        data_list.append(img)
        mask_list.append(msk)

    return data_list, mask_list

# %%
