'''
This module is for loading the 3ET dataset
@author: Chengyi Ma
'''

import h5py
import pandas as pd
import numpy as np
import os

class ThreeETDataLoader:
    '''
    ThreeETDataLoader is a class that loads the 3ET dataset
    '''
    def __init__(self, three_et_data_root:str):

        three_et_data_train_dir = os.path.join(three_et_data_root, "train")
        three_et_data_test_dir = os.path.join(three_et_data_root, "test")

        # Temporarily only load one session of the data
        # Get first subdirectory in the train directory
        train_data_list = os.listdir(three_et_data_train_dir)
        print(f"Found {len(train_data_list)} sessions in the train directory")

        # Only load one session of the data for now
        this_train_data_session_id = train_data_list[0]
        
        data_df, labels_df = self._load_data_single_session(this_train_data_session_id, three_et_data_train_dir)
    




    def _load_data_single_session(self, data_session_id:str, data_session_root:str, debug_mode:bool = True):
        '''
        load a single .h5 file and corresponding label file from the 3ET dataset
        Args:
            data_session_id: the id of the data session
            data_session_root: the root directory of the data session
            debug_mode: if True, print debug information
        Returns:
            None
        '''
        if debug_mode:
            print(f"Loading data for session {data_session_id}...")

        data_path_session = os.path.join(data_session_root, data_session_id)
        data_path_h5 = os.path.join(data_path_session, data_session_id + ".h5")
        data_path_label = os.path.join(data_path_session, "label.txt")

        data_file = h5py.File(data_path_h5, "r")

        data_df = pd.DataFrame(data_file['events'])

        if debug_mode:
            print(f"data file keys: {data_file.keys()}")
            print(f"data file events head 5 rows: {data_df[:5]}")


        t_min = data_file['events']['t'].min()
        t_max = data_file['events']['t'].max()
        t_range = t_max - t_min

        if debug_mode:
            print(f"Data start time: {t_min}, Data end time: {t_max}, Data range: {t_range}")


        data_file.close()

        # load label file
        if os.path.exists(data_path_label):

            # 1. Parse the Labels with Numpy Converters
            # The delimiter ',' splits the line into columns 0, 1, and 2.
            # We target column 0 to strip '(' and column 2 to strip ')'
            cleaners = {
                0: lambda s: int(s.replace('(', '')),
                2: lambda s: int(s.replace(')', ''))
            }

            label_date_structure = np.dtype([('x', 'int16'), ('y', 'int16'), ('z', 'int16')])
            labels_np = np.loadtxt(
                data_path_label, 
                delimiter=',', 
                converters=cleaners, 
                dtype=label_date_structure
            )

            labels_df = pd.DataFrame(labels_np)
        
            if debug_mode:
                print(f"Labels head 5 rows: {labels_np[:5]}")
                print(f"Labels count: {labels_np.shape[0]}")


        if debug_mode:
            print(f"Data for session {data_session_id} loaded successfully")

        return data_df, labels_df