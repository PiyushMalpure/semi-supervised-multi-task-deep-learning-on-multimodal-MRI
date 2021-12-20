# %% Import Statements

import os
import random
import numpy as np
import pandas as pd
from numpy import pad as numpy_pad
from tensorflow import keras
import nibabel as nib
from skimage.util import view_as_windows
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


class PatchGenerator(keras.utils.Sequence):

    """
    Instantiates the PatchGenerator.

        # Arguments

        patch_shape: A list, specifying the dimensions of patches to be extracted.
            Default is [32,32,32].
        strides: An integer or list, specifying the stride length of the between patches.
            Default is [1,1,1]
        batch_size: Int, mini-batch size.
            Default is 16
        label_mode: One of 'one-hot', 'categorical','input','image' or None.
            default is 'one-hot'.
            Determines the type of label arrays that are returned:
            - 'one-hot' will be one-hot encoded labels,
            - 'categorical' will be integer labels,
            - 'input' will be images identical to input images except for target preprocessing functions (below).
            Mainly used to work with autoencoders),
            - 'image' will load target images considering y_col as paths to images,
            - 'raw' will be numpy array of y_col data,
            - None, no labels are returned (the generator will only yield batches of image data).
        return_patch_number: Boolean, To return patch number along with patch (i.e get_item returns [x_patch,patch_idx],y ).
            Default is False.
        image_preproc_fn: A Funtion, which accepts a numpy array of input image and returns preprocessed output.
            default is None.
        target_image_preproc_fn: A Funtion, which accepts a numpy array of target image and returns preprocessed output.
            Only used when label_mode is 'input' or 'image'
            default is None.
        patch_preproc_fn: A Funtion, which accepts a numpy array of input patch and returns preprocessed output.
            default is None.
        target_patch_preproc_fn: A Funtion, which accepts a numpy array of target patch and returns preprocessed output.
            default is None.
        pad_zeros: Bool, whether to pad zeros before getting patches.
            Default is False.
        shuffle: Bool, whether to shuffle the patches.
        seed: Seed for shuffling

        # Returns
        A Keras Sequence instance.

        # Raises
        ValueError: for invalid out_mode or mismatch in shapes
    """

    def __init__(self, patch_shape=[32, 32, 32], strides=[1, 1, 1], batch_size=16,
                 label_mode='one-hot', return_patch_number=False,
                 image_preproc_fn=None, target_image_preproc_fn=None,
                 patch_preproc_fn=None, target_patch_preproc_fn=None,
                 pad_zeros=False, shuffle=True, seed=1,
                 ):
        '''
        TO DO :
        - multi_channel (now using expand_dims assuming single modality input)
        - add functionality for condition-joining
        - add padding feature for patch extraction
        - inbuilt augmentation
        - inbuilt normalization and preprocessing
        - include multilabel and mask output
        '''

        self.patch_shape = patch_shape
        self.strides = strides
        self.batch_size = batch_size

        self.label_mode = label_mode
        self.return_patch_number = return_patch_number

        self.x_preproc = image_preproc_fn
        self.y_preproc = target_image_preproc_fn
        self.x_patch_preproc = patch_preproc_fn
        self.y_patch_preproc = target_patch_preproc_fn
        self.pad_zeros = pad_zeros

        self.shuffle = shuffle
        self.seed = seed
        self.indices = None
        self.df = None

    def fit_dataframe(self, df, x_col, y_col, x_condition=None, y_condition=None, join_conditions='inner',
                      use_preprocessing=False):
        """
        Obtains patches that satisy given conditions from the given data.
        Requires data to be stored in a dataframe with input in 'x_col' and target output in 'y_col'.

        # Arguments
        dataframe: dataframe containing input and target labels
        x_col: String, specifying Column name for input(must be a 3D image)
        y_col: String, specifying Column name for target labels
        x_condition: Function, condition to selct extracted patches.
                    Default is None.
                    Note that the function should return indices for given patches.
                    For example -
                        condition_discard_bg(patches):
                            indices = []
                            for i in range(0,patches.shape[0]):
                                if np.sum(patches[i]) > 0:
                                    indices.append[i]

                            return indices
        y_condition: Function, condition to selct extracted patches.
                    Default is None.
                    Note that the function should return indices for given patches.
        join_conditions: One of 'inner', 'full', 'outer'.
                    Default is 'inner'
        use_preprocessing: Boolean, Whether to use self._.preprocessing functions for fit.
                    Default is False.

        # Returns
        df : Dataframe with x, y and useful patches

        """

        print('Analyzing Data :')

        if self.label_mode == 'one-hot':
            self.y_encoder = LabelBinarizer()
            self.y_encoder.fit(df[y_col])

        elif self.label_mode == 'categorical':
            self.y_encoder = LabelEncoder()
            self.y_encoder.fit(df[y_col])

        elif self.label_mode is None or self.label_mode == 'input' or self.label_mode == 'image' or self.label_mode == 'raw':
            self.y_encoder = None

        else:
            self.y_encoder = None
            raise ValueError("Invalid output mode")

        if use_preprocessing:
            x_preproc = self.x_preproc
            y_preproc = self.y_preproc
            x_patch_preproc = self.x_patch_preproc
            y_patch_preproc = self.y_patch_preproc
        else:
            x_preproc = None
            y_preproc = None
            x_patch_preproc = None
            y_patch_preproc = None

        df_list = []

        for idx, row in df.iterrows():

            print(str(idx))
            X = row[x_col]
            img = self.__load_data__(X, x_preproc)
            x_patches = self.__get_useful_patches__(img, self.patch_shape, self.strides, self.pad_zeros, x_condition, x_patch_preproc)

            if self.y_encoder is not None:
                Y = np.squeeze(self.y_encoder.transform(row[y_col]))
                y_patches = x_patches

            elif self.label_mode == 'raw':
                Y = row[y_col]
                y_patches = x_patches

            elif self.label_mode is None:
                Y = None
                y_patches = x_patches

            elif self.label_mode == 'input':
                Y = X
                y_patches = x_patches

            elif self.label_mode == 'image':
                Y = row[y_col]
                y = self.__load_data__(row[y_col], y_preproc)
                assert img.shape == y.shape, "Target output shape doesn't match with input"
                y_patches = self.__get_useful_patches__(y, self.patch_shape, self.strides, self.pad_zeros, y_condition, y_patch_preproc)

            else:
                y_patches = x_patches
                raise ValueError("Invalid output mode")

            final_patches = list(set(x_patches) & set(y_patches))
            for i in final_patches:
                df_list.append({'X': X, 'Y': Y, 'patch_idx': i})

        self.df = pd.DataFrame(df_list, columns=['X', 'Y', 'patch_idx'])
        self.indices = list(self.df.index)

        return self.df

    def __len__(self):
        if self.indices is None or self.df is None:
            raise ValueError("No data provided : please use 'fit_' methods to provide data")
        return int(len(self.indices)/self.batch_size)

    def __getitem__(self, idx):
        if self.indices is None or self.df is None:
            raise ValueError("No data provided : please use 'fit_' methods to provide data")

        batch_indexes = self.indices[(idx*self.batch_size): ((idx+1)*self.batch_size)]
        tmp_df = self.df.loc[batch_indexes, :]

        X, y, patch_nos = self.__data_generation__(tmp_df)

        if self.return_patch_number:
            return [np.expand_dims(X, -1), patch_nos], np.expand_dims(y, -1)
        else:
            return np.expand_dims(X, -1), np.expand_dims(y, -1)

    def __data_generation__(self, df):

        X = []
        Y = []
        patch_nos = []

        if self.label_mode is None:
            for row in df.itertuples():
                _x = self.__load_patch__(row.X, row.patch_idx, self.x_preproc, self.x_patch_preproc)
                _y = None
                X.append(_x)
                patch_nos.append(row.patch_idx)
            return np.array(X), None, patch_nos

        elif self.label_mode == 'input':
            for row in df.itertuples():
                _x = self.__load_patch__(row.X, row.patch_idx, self.x_preproc, self.x_patch_preproc)
                _y = self.__load_patch__(row.X, row.patch_idx, self.y_preproc, self.y_patch_preproc)
                X.append(_x)
                Y.append(_y)
                patch_nos.append(row.patch_idx)
            return np.array(X), np.array(Y), patch_nos

        elif self.label_mode == 'image':
            for row in df.itertuples():
                _x = self.__load_patch__(row.X, row.patch_idx, self.x_preproc, self.x_patch_preproc)
                _y = self.__load_patch__(row.Y, row.patch_idx, self.y_preproc, self.y_patch_preproc)
                X.append(_x)
                Y.append(_y)
                patch_nos.append(row.patch_idx)
            return np.array(X), np.array(Y), patch_nos

        else:
            for row in df.itertuples():
                _x = self.__load_patch__(row.X, row.patch_idx, self.x_preproc, self.x_patch_preproc)
                _y = row.Y
                X.append(_x)
                Y.append(_y)
                patch_nos.append(row.patch_idx)
            return np.array(X), np.array(Y), patch_nos

    def __get_useful_patches__(self, image, patch_shape, strides, pad_zeros=False, condition=None, preproc=None):

        windows = view_as_windows(image, window_shape=patch_shape, step=strides)
        patches = np.reshape(windows, [-1]+patch_shape)
        _indices = np.arange(0, patches.shape[0]).tolist()

        if condition is None:
            return _indices
        else:
            if preproc is not None:
                for i in range(0, patches.shape[0]):
                    patches[i] = preproc(patches[i])

            return condition(patches)

    def __load_data__(self, path, preproc_fn=None):
        if preproc_fn is None:
            data = nib.load(path).get_fdata()
            return np.ascontiguousarray(data)
        else:
            data = preproc_fn(nib.load(path).get_fdata())
            return np.ascontiguousarray(data)

    def __load_patch__(self, img_path, patch_idx, img_preproc=None, patch_preproc=None):

        image = self.__load_data__(img_path, img_preproc)
        patches = view_as_windows(image, window_shape=self.patch_shape, step=self.strides)
        del image
        patches = np.reshape(patches, [-1]+self.patch_shape)
        patch = np.copy(patches[patch_idx])
        del patches

        if patch_preproc is None:
            return patch
        else:
            return patch_preproc(patch)

    def on_epoch_end(self):
        if self.indices is None or self.df is None:
            raise ValueError("No data provided : please use 'fit_' methods to provide data")
        if self.shuffle:
            random.seed = self.seed
            random.shuffle(self.indices)
