'''
This module is for loading the 3ET dataset
@author: Chengyi Ma
'''

import random
import h5py
import pandas as pd
import numpy as np
import os
import torch

from torch_geometric.data import Dataset, Data
from gnn_ev_toolbox.gnn_tools import GnnBuilder


def _resolve_session_id(split_dir: str, session_id: str) -> str:
    session_list = sorted(os.listdir(split_dir))
    if not session_list:
        raise FileNotFoundError(f"No session folders found under '{split_dir}'.")
    if session_id not in session_list:
        preview = ", ".join(session_list[:8])
        more = f" (+{len(session_list) - 8} more)" if len(session_list) > 8 else ""
        raise FileNotFoundError(
            f"Session '{session_id}' not found under '{split_dir}'. "
            f"Available: {preview}{more}"
        )
    return session_id


def train_val_session_ids_split(
    session_ids: list[str],
    val_fraction: float,
    *,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Partition unique session IDs into disjoint train and validation lists.

    Each session appears in exactly one list. Suitable for strict session-level
    splits so that no recording leaks across train and validation.

    Args:
        session_ids: Candidate folder names (e.g. under ``train/``).
        val_fraction: Target fraction of sessions for validation, in (0, 1).
        random_seed: Controls shuffling before the cut.

    Returns:
        ``(train_session_ids, val_session_ids)`` with empty set intersection.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must lie in (0, 1).")
    uniq = list(dict.fromkeys(session_ids))
    if len(uniq) < 2:
        raise ValueError("Need at least two distinct session IDs for a train/validation split.")
    rng = random.Random(random_seed)
    order = uniq.copy()
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    n_val = min(n_val, len(order) - 1)
    val_ids = order[:n_val]
    train_ids = order[n_val:]
    assert not (set(train_ids) & set(val_ids))
    return train_ids, val_ids


def load_labels_dataframe(label_path: str) -> pd.DataFrame:
    '''
    Parse 3ET label.txt into a columnar DataFrame (x, y, z).
    Returns an empty DataFrame if the file is missing.
    '''
    if not os.path.exists(label_path):
        return pd.DataFrame()

    cleaners = {
        0: lambda s: int(s.replace('(', '')),
        2: lambda s: int(s.replace(')', '')),
    }
    label_dtype = np.dtype([('x', 'int16'), ('y', 'int16'), ('z', 'int16')])
    labels_np = np.loadtxt(
        label_path,
        delimiter=',',
        converters=cleaners,
        dtype=label_dtype,
    )
    return pd.DataFrame(labels_np)


class ThreeETDataLoader:
    '''
    ThreeETDataLoader is a class that loads the 3ET dataset.
    This class is for general purpose inspection, not for PyG training.
    It eagerly loads one chosen session into memory.

    ``session_id`` must be the folder name of one session under
    ``<three_et_data_root>/<split>/``.
    '''
    def __init__(self, three_et_data_root: str, session_id: str, split: str = "train"):

        split_dir = os.path.join(three_et_data_root, split)

        session_list = sorted(os.listdir(split_dir))
        print(f"Found {len(session_list)} sessions in the {split} directory")

        chosen_session_id = _resolve_session_id(split_dir, session_id)

        self.session_id = chosen_session_id
        self.split = split
        self.data_df, self.labels_df = self._load_data_single_session(chosen_session_id, split_dir)

    def _load_data_single_session(
        self, data_session_id: str, data_session_root: str, debug_mode: bool = True
    ):
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

        events_np = np.array(data_file["events"])
        data_df = pd.DataFrame.from_records(events_np)

        if debug_mode:
            print(f"data file keys: {data_file.keys()}")
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

        labels_df = load_labels_dataframe(data_path_label)

        if debug_mode and len(labels_df) > 0:
            print(f"Labels head 5 rows: {labels_df.head().to_string()}")
            print(f"Labels count: {labels_df.shape[0]}")

        if debug_mode:
            print(f"Data for session {data_session_id} loaded successfully")

        return data_df, labels_df


class ThreeETDataset(Dataset):
    '''
    PyTorch Geometric Dataset for 3ET+ event windows and gaze targets.

    ``session_id`` must be the folder name of one session under
    ``<root>/<split>/`` (there is no automatic session selection).

    Standalone from ThreeETDataLoader: paths are resolved in ``__init__`` but
    heavy data (events table, labels) is loaded lazily on ``len`` / ``get``
    when first needed.
    '''

    def __init__(
        self,
        root: str,
        session_id: str,
        split: str = "train",
        spatial_scale: float = 1.0,
        graph_radius: float = 10.0,
        temporal_subsample: int = 5,
        max_num_neighbors: int = 32,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log=log)
        self.temporal_subsample = temporal_subsample
        self.spatial_scale = spatial_scale
        self.graph_radius = graph_radius
        self.max_num_neighbors = max_num_neighbors
        self.builder = GnnBuilder()
        self.split = split

        split_dir = os.path.join(self.root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        chosen_session_id = _resolve_session_id(split_dir, session_id)
        self.session_id = chosen_session_id
        self._session_dir = os.path.join(split_dir, chosen_session_id)
        self._h5_path = os.path.join(self._session_dir, f"{chosen_session_id}.h5")
        self._label_path = os.path.join(self._session_dir, "label.txt")

        if not os.path.isfile(self._h5_path):
            raise FileNotFoundError(f"Events HDF5 not found: {self._h5_path}")
        if not os.path.isfile(self._label_path):
            raise FileNotFoundError(f"Labels file not found: {self._label_path}")

        self._labels_df: pd.DataFrame | None = None
        self._events_t: np.ndarray | None = None

    def _ensure_labels(self) -> None:
        if self._labels_df is not None:
            return
        self._labels_df = load_labels_dataframe(self._label_path)
        if self._labels_df.empty:
            raise ValueError(f"No labels parsed from {self._label_path}")

    def _ensure_events_t(self) -> None:
        if self._events_t is not None:
            return
        with h5py.File(self._h5_path, "r") as f:
            self._events_t = np.asarray(f["events"]["t"][:], dtype=np.int64)

    def len(self) -> int:
        self._ensure_labels()
        return len(self._labels_df) // self.temporal_subsample

    def get(self, idx: int) -> Data:
        self._ensure_labels()
        self._ensure_events_t()

        n = len(self._labels_df) // self.temporal_subsample
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range for dataset of length {n}")

        actual_label_idx = idx * self.temporal_subsample
        t_target = actual_label_idx * 10_000
        t_start = t_target - 100_000

        all_t = self._events_t
        start_idx = int(np.searchsorted(all_t, max(0, t_start)))
        end_idx = int(np.searchsorted(all_t, t_target))

        with h5py.File(self._h5_path, "r") as f:
            ev_slice = np.asarray(f["events"][start_idx:end_idx])

        if ev_slice.size == 0:
            coords_tensor = torch.empty((0, 3), dtype=torch.float32)
        else:
            window = pd.DataFrame.from_records(ev_slice)
            coords = np.column_stack(
                (
                    window["x"].values * self.spatial_scale,
                    window["y"].values * self.spatial_scale,
                    window["t"].values,
                )
            )
            coords_tensor = torch.tensor(coords, dtype=torch.float32)

        if coords_tensor.size(0) > 0:
            graph = self.builder.build_radius_graph(
                coords_tensor,
                r=self.graph_radius,
                max_num_neighbors=self.max_num_neighbors,
            )
        else:
            empty = torch.empty((0, 3), dtype=torch.float32)
            graph = Data(
                x=empty,
                pos=empty,
                edge_index=torch.empty((2, 0), dtype=torch.long),
            )

        target_x = float(self._labels_df.iloc[actual_label_idx]["x"]) * self.spatial_scale
        target_y = float(self._labels_df.iloc[actual_label_idx]["y"]) * self.spatial_scale
        graph.y = torch.tensor([[target_x, target_y]], dtype=torch.float32)

        return graph
