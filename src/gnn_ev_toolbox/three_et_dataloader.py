'''
This module is for loading the 3ET dataset
@author: Chengyi Ma
'''

import h5py
import pandas as pd
import numpy as np
import os
import torch

from torch_geometric.data import Dataset, Data
from gnn_ev_toolbox.gnn_tools import GnnBuilder

class ThreeETDataLoader:
    '''
    ThreeETDataLoader is a class that loads the 3ET dataset
    '''
    def __init__(self, three_et_data_root: str, session_id: str | None = None, split: str = "train"):

        split_dir = os.path.join(three_et_data_root, split)

        # Temporarily only load one session of the data
        # Get subdirectories in the split directory
        session_list = sorted(os.listdir(split_dir))
        print(f"Found {len(session_list)} sessions in the {split} directory")

        # Choose which session to load
        if session_id is None or session_id == "auto":
            chosen_session_id = session_list[0]
        else:
            if session_id not in session_list:
                raise FileNotFoundError(
                    f"Session '{session_id}' not found under '{split_dir}'. "
                    f"Example available session: '{session_list[0] if session_list else '<none>'}'"
                )
            chosen_session_id = session_id

        self.session_id = chosen_session_id
        self.split = split
        self.data_df, self.labels_df = self._load_data_single_session(chosen_session_id, split_dir)


    def _load_data_single_session(self, data_session_id:str, data_session_root:str, debug_mode:bool = True):
        '''
        Load a single .h5 file and corresponding label file from the 3ET dataset
        Args:
            data_session_id: the id of the data session
            data_session_root: the root directory of the data session
            debug_mode: if True, print debug information
        Returns:
            data_df: a pandas dataframe with the events data
            labels_df: a pandas dataframe with the labels data
        '''
        if debug_mode:
            print(f"Loading data for session {data_session_id}...")

        data_path_session = os.path.join(data_session_root, data_session_id)
        data_path_h5 = os.path.join(data_path_session, data_session_id + ".h5")
        data_path_label = os.path.join(data_path_session, "label.txt")

        data_file = h5py.File(data_path_h5, "r")

        # Convert HDF5 structured dataset into a columnar DataFrame.
        # `pd.DataFrame(data_file['events'])` can produce a single column with a structured dtype,
        # which then breaks downstream `data_df['t']` / `['x']` access.
        events_np = np.array(data_file["events"])
        data_df = pd.DataFrame.from_records(events_np)

        if debug_mode:
            print(f"data file keys: {data_file.keys()}")
            # Printing the DataFrame directly can crash when HDF5 structured dtypes
            # contain non-numeric fields that trigger Pandas' formatting checks.
            try:
                events_preview = events_np[:5]
                print(f"data file events first 5 raw rows: {events_preview}")
            except Exception as e:
                print(f"Could not preview raw events rows due to: {e}")
            print(f"data_df columns: {list(data_df.columns)}")
            print(f"data_df dtypes: {data_df.dtypes.to_dict()}")


        t_min = data_file['events']['t'].min()
        t_max = data_file['events']['t'].max()
        t_range = t_max - t_min

        if debug_mode:
            print(f"Data start time: {t_min}, Data end time: {t_max}, Data range: {t_range}")


        data_file.close()

        # load label file
        labels_df = pd.DataFrame()
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

class ThreeETDataset(Dataset):
    def __init__(self, three_et_data_root, session_id, spatial_scale=0.125, 
                 graph_radius=10.0, temporal_subsample=5):
        super().__init__(root=three_et_data_root)
        
        # 1. New Parameter: temporal_subsample
        # If 5, we turn 100Hz labels into 20Hz training steps.
        self.temporal_subsample = temporal_subsample
        
        # Instantiate your loader 
        self.session_id = session_id
        self.loader = ThreeETDataLoader(three_et_data_root, session_id=session_id, split="train")
        
        self.spatial_scale = spatial_scale
        self.builder = GnnBuilder() 
        self.graph_radius = graph_radius

    def len(self):
        # 2. Divide total labels by subsample factor.
        # This tells the model how many 20Hz "steps" are in the recording.
        return len(self.loader.labels_df) // self.temporal_subsample

    def get(self, idx):
        # 3. The "Multiplier": Map 20Hz index back to 100Hz label row.
        # Example: idx 1 becomes row 5 (50ms).
        actual_label_idx = idx * self.temporal_subsample
        
        # 4. The "Anchor": Where is the target label in time?
        t_target = actual_label_idx * 10_000 
        
        # 5. The "Constant Beam": Always look 100ms into the past.
        t_start = t_target - 100_000 

        # Slicing from your loader's data_df
        all_t = self.loader.data_df['t'].values
        start_idx = np.searchsorted(all_t, max(0, t_start))
        end_idx = np.searchsorted(all_t, t_target)
        window = self.loader.data_df.iloc[start_idx:end_idx]

        # Process coordinates and build graph
        coords = np.column_stack((
            window['x'].values * self.spatial_scale, 
            window['y'].values * self.spatial_scale, 
            window['t'].values
        ))
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        if coords_tensor.size(0) > 0:
            graph = self.builder.build_radius_graph(coords_tensor, r=self.graph_radius)
        else:
            empty = torch.empty((0, 3), dtype=torch.float32)
            graph = Data(
                x=empty,
                pos=empty,
                edge_index=torch.empty((2, 0), dtype=torch.long),
            )

        # 6. Bind the Scaled Label using the actual_label_idx
        target_x = float(self.loader.labels_df.iloc[actual_label_idx]['x']) * self.spatial_scale
        target_y = float(self.loader.labels_df.iloc[actual_label_idx]['y']) * self.spatial_scale
        graph.y = torch.tensor([[target_x, target_y]], dtype=torch.float32)

        return graph