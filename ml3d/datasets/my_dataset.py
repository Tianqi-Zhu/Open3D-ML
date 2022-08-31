from sqlite3 import DatabaseError
import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging
import open3d as o3d
import json
import math

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET
from .utils import BEVBox3D

class MyDataset(BaseDataset):
    def __init__(self,
                dataset_path,
                name='MyDataset',
                cache_dir='./logs/cache',
                use_cache=False,
                test_result_folder='./test',
                **kwargs
                ):
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         test_result_folder=test_result_folder,
                         **kwargs
                         )
        # read file lists.
        
        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = dataset_path
        self.num_classes = 2
        self.label_to_names = self.get_label_to_names()
        
        self.train_folder = cfg.train_folder
        self.val_folder = cfg.val_folder
        self.test_folder = cfg.test_folder
        self.test_result_folder = cfg.test_result_folder
        
        self.train_files = MyDataset.get_path_list_from_folder(self.train_folder)
        self.val_files = MyDataset.get_path_list_from_folder(self.val_folder)
        self.test_files = MyDataset.get_path_list_from_folder(self.test_folder)

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'ball',
            1: 'cylinder'
        }
        return label_to_names

    def get_split(self, split):
        return MyDatasetSplit(self, split=split)

    def get_split_list(self, split):
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            return self.train_files
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    @staticmethod
    def get_path_list_from_folder(folder_path):
        out_path_list = []
        for file in os.listdir(folder_path):
            if file.endswith('.pcd'):
                path = os.path.join(folder_path, file)
                out_path_list.append(path)
        return out_path_list

    @staticmethod
    def get_label(path):
        f = open(path)
        if not f.is_file():
            return []
        data = json.load(f)
        objects = data['objects']
        out_labels = []
        for object in objects:
            class_name = 0
            if object['name'] == 'cylinder':
                class_name = 1
            elif object['name'] == 'ball':
                class_name = 0
            else:
                raise ValueError("Invalid class name {}".format(class_name))
            center = (object['centroid']['x'], object['centroid']['y'], object['centroid']['z'])
            size = (object['dimensions']['width'], object['dimensions']['depth'], object['dimensions']['height'])
            yaw = object['rotations']['z'] / 180 * math.pi
            out_label = Object3d(class_name, center, size, yaw)
            out_labels.append(out_label)
        return out_labels

    def is_tested(self, attr):
        # checks whether attr['name'] is already tested.
        # wtf is this
        pass
    
    # requires modification after seeing output format
    def save_test_result(self, results, attr):
        # save results['predict_labels'] to file.
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.txt')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class MyDatasetSplit():
    def __init__(self, dataset, split='train'):
        self.split = split
        self.path_list = dataset.get_split_list(split)

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path = self.path_list[idx]
        pcd_loaded = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd_loaded.points).astype(np.float32)
        label_path = path.replace('pcds', 'ann')
        label_path = label_path.replace('.pcd', '.json')
        labels = MyDataset.get_label(label_path)
        return {'point': points, 'feat': None, 'label': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1].replace('.pcd', '')
        return {'idx': idx, 'name': name, 'path': path, 'split': self.split}


class Object3d(BEVBox3D):
    """The class stores details that are object-specific, such as bounding box
    coordinates, occlusion and so on.
    """
    def __init__(self, name, center, size, yaw):
        super().__init__(center, size, yaw, name, -1.0)

        self.occlusion = 0.0

DATASET._register_module(MyDataset)
