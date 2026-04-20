'''
Pytest suite for ThreeETDataset and train/validation session splitting.

Run from repo root:
    uv run --extra dev pytest src/test/test_three_et_dataloader.py -q
'''

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset

from gnn_ev_toolbox.three_et_dataloader import ThreeETDataset, train_val_session_ids_split


EVENT_DTYPE = np.dtype([("t", "uint64"), ("x", "uint64"), ("y", "uint64"), ("p", "uint64")])


def write_label_file(path: Path, rows: list[tuple[int, int, int]]) -> None:
    """Write label.txt in the format expected by load_labels_dataframe."""
    lines = [f"({x},{y},{z})\n" for x, y, z in rows]
    path.write_text("".join(lines), encoding="utf-8")


def write_session_h5(path: Path, events: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("events", data=events)


def make_events_dense_times(t_values: np.ndarray, x0: int = 10, y0: int = 20) -> np.ndarray:
    """Build structured events array with given timestamps (uint64)."""
    n = len(t_values)
    ev = np.empty(n, dtype=EVENT_DTYPE)
    ev["t"] = t_values.astype(np.uint64)
    ev["x"] = np.arange(x0, x0 + n, dtype=np.uint64)
    ev["y"] = np.arange(y0, y0 + n, dtype=np.uint64)
    ev["p"] = np.zeros(n, dtype=np.uint64)
    return ev


@pytest.fixture
def mock_root_dense(tmp_path: Path) -> Path:
    """
    One train session with events in early time so several temporal windows are non-empty.
    25 label rows -> len == 25 // 5 == 5 training steps.
    """
    root = tmp_path / "event_data"
    split_dir = root / "train"
    sid = "sess_dense"
    sess = split_dir / sid
    sess.mkdir(parents=True)

    # For idx>=1, t_target = actual_label_idx * 10_000 >= 5000; include events from t=0..90_000
    t_vals = np.linspace(0, 90_000, num=40, dtype=np.int64)
    events = make_events_dense_times(t_vals)
    write_session_h5(sess / f"{sid}.h5", events)

    labels = [(100 + i, 200 + i, 0) for i in range(25)]
    write_label_file(sess / "label.txt", labels)

    return root


@pytest.fixture
def mock_root_empty_first_window(tmp_path: Path) -> Path:
    """
    All events occur after the first label windows, so idx=0 has zero events in-window.
    """
    root = tmp_path / "event_data"
    split_dir = root / "train"
    sid = "sess_empty0"
    sess = split_dir / sid
    sess.mkdir(parents=True)

    # First windows need t < t_target for small actual_label_idx; push all t very large
    events = make_events_dense_times(np.array([1_000_000 + i for i in range(5)], dtype=np.int64))
    write_session_h5(sess / f"{sid}.h5", events)

    labels = [(10, 20, 0)] * 15
    write_label_file(sess / "label.txt", labels)

    return root


@pytest.fixture
def mock_root_two_sessions(tmp_path: Path) -> tuple[Path, str, str]:
    """Two train sessions with different label counts for ConcatDataset tests."""
    root = tmp_path / "event_data"
    train = root / "train"

    s_a = "sess_a"
    d_a = train / s_a
    d_a.mkdir(parents=True)
    t_a = np.linspace(0, 200_000, num=30, dtype=np.int64)
    write_session_h5(d_a / f"{s_a}.h5", make_events_dense_times(t_a, x0=1, y0=2))
    write_label_file(d_a / "label.txt", [(i, i + 1, 0) for i in range(20)])  # len ds = 20//5=4

    s_b = "sess_b"
    d_b = train / s_b
    d_b.mkdir(parents=True)
    t_b = np.linspace(0, 200_000, num=25, dtype=np.int64)
    write_session_h5(d_b / f"{s_b}.h5", make_events_dense_times(t_b, x0=50, y0=60))
    write_label_file(d_b / "label.txt", [(50 + i, 60 + i, 0) for i in range(15)])  # len = 15//5=3

    return root, s_a, s_b


class TestThreeETDatasetTensorShapes:
    def test_index_pyg_structure_matches_huber_expectations(self, mock_root_dense: Path):
        ds = ThreeETDataset(
            root=str(mock_root_dense),
            session_id="sess_dense",
            split="train",
            spatial_scale=0.125,
            graph_radius=50.0,
            temporal_subsample=5,
            log=False,
        )
        assert len(ds) == 5
        # idx=1 maps to t_target=50_000; mock events cover [0, 90_000] so the window is non-empty.
        idx = 1
        data = ds[idx]
        assert data.num_nodes > 0, "fixture should yield a non-empty graph for idx=1"

        assert data.x.dim() == 2
        assert data.x.dtype == torch.float32
        assert data.x.shape[1] == 3

        assert data.edge_index.dim() == 2
        assert data.edge_index.dtype == torch.long
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == data.num_edges

        assert hasattr(data, "y") and data.y is not None
        assert data.y.dim() == 2
        assert data.y.shape == (1, 2)
        assert data.y.dtype == torch.float32

    def test_random_index_always_valid_pyg_shapes(self, mock_root_dense: Path):
        """Any valid index must return a consistent Data layout (empty or not)."""
        ds = ThreeETDataset(
            root=str(mock_root_dense),
            session_id="sess_dense",
            split="train",
            spatial_scale=0.125,
            graph_radius=50.0,
            temporal_subsample=5,
            log=False,
        )
        rng = random.Random(999)
        for _ in range(8):
            idx = rng.randrange(len(ds))
            data = ds[idx]
            assert data.x.dim() == 2 and data.x.shape[1] == 3 and data.x.dtype == torch.float32
            assert data.edge_index.shape[0] == 2 and data.edge_index.dtype == torch.long
            assert data.edge_index.shape[1] == data.num_edges
            assert data.y is not None and data.y.shape == (1, 2)


class TestEmptyGraphWindow:
    def test_empty_window_returns_zero_nodes(self, mock_root_empty_first_window: Path):
        ds = ThreeETDataset(
            root=str(mock_root_empty_first_window),
            session_id="sess_empty0",
            split="train",
            spatial_scale=1.0,
            graph_radius=10.0,
            temporal_subsample=5,
            log=False,
        )
        assert len(ds) >= 1
        data = ds[0]

        assert data.num_nodes == 0
        assert data.x.shape == (0, 3)
        assert data.edge_index.shape == (2, 0)
        assert data.y.shape == (1, 2)
        assert torch.isfinite(data.y).all()


class TestConcatDatasetLengthAndRouting:
    def test_concat_length_and_second_session_index(self, mock_root_two_sessions: tuple[Path, str, str]):
        root, s_a, s_b = mock_root_two_sessions
        ds_a = ThreeETDataset(root=str(root), session_id=s_a, split="train", graph_radius=50.0, log=False)
        ds_b = ThreeETDataset(root=str(root), session_id=s_b, split="train", graph_radius=50.0, log=False)

        assert len(ds_a) == 4
        assert len(ds_b) == 3

        full = ConcatDataset([ds_a, ds_b])
        assert len(full) == len(ds_a) + len(ds_b)

        first_b = len(ds_a)
        sample = full[first_b]
        ref = ds_b[0]
        assert sample.num_nodes == ref.num_nodes
        assert torch.equal(sample.x, ref.x)
        assert torch.equal(sample.edge_index, ref.edge_index)
        assert torch.allclose(sample.y, ref.y)


class TestTrainValSessionSplitNoLeakage:
    def test_train_and_val_session_ids_are_disjoint(self):
        sessions = [f"s{i}" for i in range(10)]
        train_ids, val_ids = train_val_session_ids_split(
            sessions, val_fraction=0.3, random_seed=7
        )
        assert set(train_ids).isdisjoint(set(val_ids))
        assert set(train_ids) | set(val_ids) == set(sessions)
        assert len(train_ids) >= 1 and len(val_ids) >= 1

    def test_intersection_empty_for_multiple_seeds(self):
        pool = ["alpha", "beta", "gamma", "delta"]
        for seed in range(5):
            tr, va = train_val_session_ids_split(pool, 0.25, random_seed=seed)
            assert set(tr) & set(va) == set()

    def test_invalid_val_fraction_raises(self):
        with pytest.raises(ValueError):
            train_val_session_ids_split(["a", "b"], val_fraction=0.0)
        with pytest.raises(ValueError):
            train_val_session_ids_split(["a", "b"], val_fraction=1.0)

    def test_too_few_sessions_raises(self):
        with pytest.raises(ValueError):
            train_val_session_ids_split(["only_one"], val_fraction=0.2)
