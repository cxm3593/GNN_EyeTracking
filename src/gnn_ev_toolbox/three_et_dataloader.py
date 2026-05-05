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
    uniq = list[str](dict.fromkeys(session_ids))
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
        label_normalization: str | None = None,
        resolution_width: int | None = None,
        resolution_height: int | None = None,
        time_to_pixel_us: float | None = None,
        graph_radius: float = 10.0,
        temporal_subsample: int = 5,
        max_num_neighbors: int = 32,
        window_us: int = 100_000,
        normalize_input: bool = False,
        include_polarity: bool = False,
        augment_translate: bool = False,
        translate_max_px: float = 0.0,
        augment_seed: int | None = None,
        compute_edge_attr: bool = False,
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
        self.label_normalization = label_normalization  # None | "zscore" | "minmax01" | "resolution"
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        # time_to_pixel_us rescales event timestamps so that 1 unit of time == 1 pixel
        # of space in the graph coordinate system. None => keep raw microseconds.
        self.time_to_pixel_us = time_to_pixel_us
        self.window_us = int(window_us)
        self.normalize_input = bool(normalize_input)
        # When True, append a polarity channel to data.x (mapped raw {0, 1} -> {-1, +1}
        # so the feature is mean-centered). data.pos stays 3D (xyt only) so it can be
        # used for edge-geometry features later without polarity contaminating distances.
        self.include_polarity = bool(include_polarity)

        # Train-only random translation augmentation. Each call to get() draws a
        # single (dx, dy) ~ Uniform(-T, T) in raw pixels and applies it to BOTH the
        # event positions and the pupil-center label, so the (events -> pupil)
        # relationship is preserved exactly while the absolute sensor coordinates
        # change. Off (default) for val/test. The RNG is created from augment_seed
        # for run-to-run reproducibility; pass a different seed per session at the
        # call site if you want decorrelated streams across sessions.
        self.augment_translate = bool(augment_translate)
        self.translate_max_px = float(translate_max_px)
        if self.augment_translate and self.translate_max_px > 0.0:
            self._aug_rng = np.random.default_rng(augment_seed)
        else:
            self._aug_rng = None

        # When True, also compute data.edge_attr = pos[dst] - pos[src] (in the same
        # units as data.pos, i.e. normalized if normalize_input=True). Required by
        # GINEConv (and any conv that uses edge geometry); ignored by SAGEConv.
        self.compute_edge_attr = bool(compute_edge_attr)

        if label_normalization == "resolution":
            if resolution_width is None or resolution_height is None:
                raise ValueError(
                    "label_normalization='resolution' requires resolution_width and "
                    "resolution_height to be set."
                )

        # Per-axis divisor applied to node features (data.x) and positions (data.pos)
        # AFTER the radius graph is built, so the graph topology is unaffected by the
        # rescaling. Keeping all three input channels in roughly [0, 1] makes
        # optimization much more stable, and matches the "resolution" normalization
        # used for the regression target.
        if self.normalize_input:
            if resolution_width is None or resolution_height is None:
                raise ValueError(
                    "normalize_input=True requires resolution_width and resolution_height."
                )
            t_scale = (
                float(self.window_us) / float(self.time_to_pixel_us)
                if self.time_to_pixel_us is not None
                else float(self.window_us)
            )
            self._input_scale = torch.tensor(
                [float(resolution_width), float(resolution_height), t_scale],
                dtype=torch.float32,
            )
        else:
            self._input_scale = None

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
        self._label_center: np.ndarray | None = None  # shape (2,)
        self._label_scale: np.ndarray | None = None   # shape (2,)

    def _ensure_labels(self) -> None:
        if self._labels_df is not None:
            return
        self._labels_df = load_labels_dataframe(self._label_path)
        if self._labels_df.empty:
            raise ValueError(f"No labels parsed from {self._label_path}")

        if self.label_normalization is None:
            self._label_center = np.array([0.0, 0.0], dtype=np.float32)
            self._label_scale = np.array([1.0, 1.0], dtype=np.float32)
            return

        x = self._labels_df["x"].to_numpy(dtype=np.float32) * self.spatial_scale
        y = self._labels_df["y"].to_numpy(dtype=np.float32) * self.spatial_scale

        if self.label_normalization == "zscore":
            cx, cy = float(np.mean(x)), float(np.mean(y))
            sx, sy = float(np.std(x)), float(np.std(y))
            sx = sx if sx > 1e-6 else 1.0
            sy = sy if sy > 1e-6 else 1.0
            self._label_center = np.array([cx, cy], dtype=np.float32)
            self._label_scale = np.array([sx, sy], dtype=np.float32)
            return

        if self.label_normalization == "minmax01":
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            sx = (xmax - xmin) if (xmax - xmin) > 1e-6 else 1.0
            sy = (ymax - ymin) if (ymax - ymin) > 1e-6 else 1.0
            self._label_center = np.array([xmin, ymin], dtype=np.float32)
            self._label_scale = np.array([sx, sy], dtype=np.float32)
            return

        if self.label_normalization == "resolution":
            # Deterministic normalization to [0, 1] based on sensor resolution.
            # Independent of the label distribution, so it is consistent across
            # every split and every session.
            self._label_center = np.array([0.0, 0.0], dtype=np.float32)
            self._label_scale = np.array(
                [float(self.resolution_width), float(self.resolution_height)],
                dtype=np.float32,
            )
            return

        raise ValueError(
            f"Unknown label_normalization='{self.label_normalization}'. "
            "Use None, 'zscore', 'minmax01', or 'resolution'."
        )

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

        # n is the number of temporal subsamples
        n = len(self._labels_df) // self.temporal_subsample
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range for dataset of length {n}")

        actual_label_idx = idx * self.temporal_subsample
        t_target = actual_label_idx * 10_000
        t_start = t_target - self.window_us

        # Sample one (dx, dy) per call. Drawn here (before we know whether the
        # window ends up empty) so the RNG advances by the same amount per get()
        # regardless of data, which keeps augmentation deterministic given seed.
        if self._aug_rng is not None:
            dx = float(self._aug_rng.uniform(-self.translate_max_px, self.translate_max_px))
            dy = float(self._aug_rng.uniform(-self.translate_max_px, self.translate_max_px))
        else:
            dx = 0.0
            dy = 0.0

        all_t = self._events_t
        start_idx = int(np.searchsorted(all_t, max(0, t_start)))
        end_idx = int(np.searchsorted(all_t, t_target))

        with h5py.File(self._h5_path, "r") as f:
            ev_slice = np.asarray(f["events"][start_idx:end_idx])

        if ev_slice.size == 0:
            coords_tensor = torch.empty((0, 3), dtype=torch.float32)
            polarity_tensor = torch.empty((0, 1), dtype=torch.float32)
        else:
            window = pd.DataFrame.from_records(ev_slice)
            # Shift time so every graph starts at 0, then (optionally) rescale so
            # 1 unit of time == 1 pixel of space, making `graph_radius` consistent
            # across the x/y/t axes.
            t_window_ref = max(0, t_start)
            t_rel_us = window["t"].values.astype(np.float64) - float(t_window_ref)
            if self.time_to_pixel_us is not None:
                t_scaled = t_rel_us / float(self.time_to_pixel_us)
            else:
                t_scaled = t_rel_us
            coords = np.column_stack(
                (
                    window["x"].values * self.spatial_scale + dx,
                    window["y"].values * self.spatial_scale + dy,
                    t_scaled,
                )
            )
            coords_tensor = torch.tensor(coords, dtype=torch.float32)
            # Map raw polarity {0, 1} -> {-1, +1} so the channel is mean-centered.
            # We materialize this even when include_polarity=False so the empty/non-empty
            # branches stay structurally identical; it's cheap.
            pol_np = window["p"].values.astype(np.float32) * 2.0 - 1.0
            polarity_tensor = torch.tensor(pol_np, dtype=torch.float32).unsqueeze(1)

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

        # Normalize the geometric positions in data.pos AFTER the radius graph is
        # built, so GRAPH_RADIUS continues to mean what it did before (raw
        # pixel / scaled-time units). data.x is rebuilt below from data.pos so the
        # feature scale stays consistent with the position scale.
        if self._input_scale is not None:
            graph.pos = graph.pos / self._input_scale

        # Build the input feature tensor data.x. Polarity is *not* divided by anything
        # — it already lives in {-1, +1} and is on the same scale as the normalized
        # spatial/temporal channels. data.pos stays 3D (xyt) regardless of polarity
        # so any future edge-geometry feature (e.g. relative position) doesn't get
        # mixed with polarity.
        if self.include_polarity:
            graph.x = torch.cat([graph.pos, polarity_tensor], dim=1)
        else:
            graph.x = graph.pos

        # Edge features = relative position between connected nodes, in the same
        # units as data.pos. For an empty edge_index this returns an empty (0, 3)
        # tensor, so the shape is consistent across batches.
        if self.compute_edge_attr:
            src, dst = graph.edge_index[0], graph.edge_index[1]
            graph.edge_attr = graph.pos[dst] - graph.pos[src]

        # Apply the same (dx, dy) to the target so the event->pupil relationship
        # is preserved exactly under augmentation. y_raw is in raw pixels; the
        # downstream label normalization will then divide by the same scale used
        # for events, keeping things consistent.
        target_x = float(self._labels_df.iloc[actual_label_idx]["x"]) * self.spatial_scale + dx
        target_y = float(self._labels_df.iloc[actual_label_idx]["y"]) * self.spatial_scale + dy
        y_raw = torch.tensor([[target_x, target_y]], dtype=torch.float32)
        graph.y_raw = y_raw

        center = self._label_center if self._label_center is not None else np.array([0.0, 0.0], dtype=np.float32)
        scale = self._label_scale if self._label_scale is not None else np.array([1.0, 1.0], dtype=np.float32)
        graph.y = (y_raw - torch.tensor(center).view(1, 2)) / torch.tensor(scale).view(1, 2)
        graph.y_center = torch.tensor(center).view(1, 2)
        graph.y_scale = torch.tensor(scale).view(1, 2)

        return graph
