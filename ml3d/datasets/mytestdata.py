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
    
    super().__init__(dataset_path=dataset_path,
                name=name,
                cache_dir=cache_dir,
                use_cache=use_cache,
                val_split=val_split,
                test_result_folder=test_result_folder,
                **kwargs)