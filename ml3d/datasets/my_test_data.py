import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging

from .utils import DataProcessing as DP
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

class MyTestData(BaseDataset):
    
    def __init__(self,
                 dataset_path,
                 name='MyTestData',
                 cache_dir='./logs/cache',
                 use_cache=True,
                 num_points=65536,
                 class_weights=[],
                 test_result_folder='./test',
                 val_files=['Custom.ply'],
                 **kwargs):
         """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about the dataset. This is default to dataset path if nothing is provided.
            name: The name of the dataset (NuScenes in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.

        Returns:
            class: The corresponding class.
        """
         if info_path is None:
            info_path = dataset_path

        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 10
        self.label_to_names = self.get_label_to_names()

        self.train_info = {}
        self.test_info = {}
        self.val_info = {}

        if os.path.exists(join(info_path, 'infos_train.pkl')):
            self.train_info = pickle.load(
                open(join(info_path, 'infos_train.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val.pkl')):
            self.val_info = pickle.load(
                open(join(info_path, 'infos_val.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_test.pkl')):
            self.test_info = pickle.load(
                open(join(info_path, 'infos_test.pkl'), 'rb'))

    
    @staticmethod
    def get_label_to_names():
    
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.

            Names are extracted from Matterport3D's `metadata/category_mapping.tsv`'s
            "ShapeNetCore55" column.
        """
        label_to_names = {
            0: 'ball',
            1: 'cylinder',
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return joblib.load(path)

    @staticmethod
    def read_label(path):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        assert Path(path).exists()
        boxes = joblib.load(path)
        objects = []
        for b in boxes:
            name, img_left, img_top, img_right, img_bottom, center_x, center_y, center_z, l, w, h, yaw = b
            yaw = -np.deg2rad(np.float32(yaw))
            # image_bb = np.array([img_left, img_top, img_right, img_bottom])
            size = np.array([l, h, w],
                            np.float32)  # Weird order is what the BEV box takes
            center = np.array([center_x, center_y, center_z],
                              np.float32)  # Actual center of the box
            objects.append(BEVBox3D(center, size, yaw, name, 1))
        return objects

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return MatterportObjectsSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then resturn the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        pass


class MyTestDataSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('pc', 'boxes').replace('.bin', '.txt')

        pc = self.dataset.read_lidar(pc_path)
        label = self.dataset.read_label(label_path)

        data = {
            'point': pc,
            'calib': {},
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(MyTestData)
