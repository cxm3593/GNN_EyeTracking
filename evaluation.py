"""
Minimal evaluation scaffold (incremental):

Step 1: load model weights
Step 2: load evaluation dataset(s)
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
from torch.utils.data import ConcatDataset

import main as train_cfg
from gnn_ev_toolbox.models import SimplePupilGNN
from gnn_ev_toolbox.three_et_dataloader import ThreeETDataset

from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt


# --- Config / constants ------------------------------------------------------

# Config to edit
CHECKPOINT_PATH = r"C:\Users\cxm3593\Academic\Workspace\GNN_EyeTracking\checkpoints\step5;SAGE_AdamW_wd1e-4_LRplateau(SAGE_LN_InputNorm_Polarity_Radius10_K8_TempSubsample5_TimeToPixel30_HuberDelta0.05)_20260505_004557_LR0.001_BS2_E100\model_best.pt"
SPLIT: str = "test"  # "train" | "test"
SESSIONS: str | list[str] = "2_2"  # "all" | comma-separated | ["sid1", "sid2", ...]



# Constants that DO NOT EDIT
MODEL_NAME = "SimplePupilGNN"

DATA_ROOT_3ET = train_cfg.DATA_ROOT_3ET

# Prefer training defaults from `main.py` so widths/operators stay in sync with how you trained.
MODEL_INPUT_DIM = train_cfg.GNN_INPUT_DIM  # derived from INCLUDE_POLARITY
MODEL_HIDDEN_DIM = 64                      # SimplePupilGNN default in `models.py` (training uses implicit default)
MODEL_CONV_DROPOUT = train_cfg.CONV_DROPOUT
MODEL_CONV_TYPE = train_cfg.CONV_TYPE
MODEL_EDGE_DIM = train_cfg.EDGE_DIM

DEVICE = train_cfg.DEVICE

SENSOR_WIDTH = train_cfg.SPATIAL_RESOLUTION_WIDTH
SENSOR_HEIGHT = train_cfg.SPATIAL_RESOLUTION_HEIGHT
TIME_TO_PIXEL_US = train_cfg.TIME_TO_PIXEL_US
WINDOW_US = train_cfg.WINDOW_US


def _denormalize_pred_to_pixels(pred_norm: torch.Tensor, batch) -> torch.Tensor:
    """
    Match `main.py::_pixel_errors` / training:
        pred_px = pred * y_scale + y_center
    where pred is in the same normalized units as `batch.y`.
    """
    y_scale = batch.y_scale
    y_center = batch.y_center
    return pred_norm * y_scale + y_center


def _events_pos_to_pixels(pos_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert `batch.pos` (post-graph-build coords, possibly normalized) back to pixel x/y and
    a reasonable time axis for visualization.

    Mirrors `ThreeETDataset` scaling logic:
      - if normalize_input: divide by [W, H, t_scale]
      - t_scale is WINDOW_US or WINDOW_US/TIME_TO_PIXEL_US depending on TIME_TO_PIXEL_US.
    """
    w = float(train_cfg.SPATIAL_RESOLUTION_WIDTH)
    h = float(train_cfg.SPATIAL_RESOLUTION_HEIGHT)
    window_us = float(train_cfg.WINDOW_US)

    x = pos_norm[:, 0]
    y = pos_norm[:, 1]
    t = pos_norm[:, 2]
    if bool(train_cfg.NORMALIZE_INPUT):
        x = x * w
        y = y * h
        t = t * WINDOW_US
    return x, y, t


def _pick_device(spec: str) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec == "cuda":
        # Canonicalize to cuda:0 so `.to(device)` and tensor devices compare consistently.
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(spec, str) and spec.startswith("cuda:"):
        return torch.device(spec if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        # Raw state dict
        return {"model_state_dict": ckpt, "epoch": None, "metric": None}
    raise ValueError(f"Unsupported checkpoint format at: {path}")


def _build_model() -> SimplePupilGNN:
    conv_type = str(MODEL_CONV_TYPE).lower()
    if conv_type == "gine" and not bool(train_cfg.COMPUTE_EDGE_ATTR):
        raise ValueError("MODEL_CONV_TYPE='gine' requires COMPUTE_EDGE_ATTR=True in training (`main.py`).")
    return SimplePupilGNN(
        input_dim=int(MODEL_INPUT_DIM),
        hidden_dim=int(MODEL_HIDDEN_DIM),
        output_dim=2,
        conv_dropout=float(MODEL_CONV_DROPOUT),
        conv_type=str(MODEL_CONV_TYPE),
        edge_dim=int(MODEL_EDGE_DIM),
    )


def _list_sessions(data_root: str, split: str) -> list[str]:
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name)))


def _resolve_sessions(data_root: str, split: str, sessions: str | list[str]) -> list[str]:
    available = _list_sessions(data_root, split)
    if isinstance(sessions, str):
        s = sessions.strip()
        if s.lower() in {"all", "*"}:
            return available
        wanted = [p.strip() for p in s.split(",") if p.strip()]
    else:
        wanted = [str(p).strip() for p in sessions if str(p).strip()]

    missing = [s for s in wanted if s not in available]
    if missing:
        preview = ", ".join(available[:10]) + (" ..." if len(available) > 10 else "")
        raise FileNotFoundError(
            f"Sessions not found under '{data_root}/{split}': {missing}. Available: {preview}"
        )
    return wanted


def _build_eval_dataset(session_ids: Iterable[str]) -> ConcatDataset:
    """
    Match `main.py` dataset construction for val/test-like evaluation:
      - no translation augmentation
      - same graph/window/normalization knobs as training constants
    """
    ds_list: list[ThreeETDataset] = []
    for sid in session_ids:
        ds_list.append(
            ThreeETDataset(
                root=DATA_ROOT_3ET,
                session_id=str(sid),
                split=SPLIT,
                spatial_scale=float(train_cfg.SPATIAL_SCALE),
                label_normalization=train_cfg.LABEL_NORMALIZATION,
                resolution_width=int(train_cfg.SPATIAL_RESOLUTION_WIDTH),
                resolution_height=int(train_cfg.SPATIAL_RESOLUTION_HEIGHT),
                time_to_pixel_us=train_cfg.TIME_TO_PIXEL_US,
                graph_radius=float(train_cfg.GRAPH_RADIUS),
                temporal_subsample=int(train_cfg.TEMPORAL_SUBSAMPLE),
                max_num_neighbors=int(train_cfg.MAX_NUM_NEIGHBORS),
                window_us=int(train_cfg.WINDOW_US),
                normalize_input=bool(train_cfg.NORMALIZE_INPUT),
                include_polarity=bool(train_cfg.INCLUDE_POLARITY),
                augment_translate=False,
                translate_max_px=0.0,
                augment_seed=None,
                compute_edge_attr=bool(train_cfg.COMPUTE_EDGE_ATTR),
                log=False,
            )
        )
    return ConcatDataset(ds_list)


@torch.no_grad()
def main() -> int:
    device = _pick_device(str(DEVICE))
    ckpt = _load_checkpoint(CHECKPOINT_PATH)

    model = _build_model()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # Load dataset that we want to evaluate on
    if SPLIT not in {"train", "test"}:
        raise ValueError("SPLIT must be 'train' or 'test'.")

    sessions = _resolve_sessions(DATA_ROOT_3ET, SPLIT, SESSIONS)
    dataset = _build_eval_dataset(sessions)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Handle output directory
    _safe = lambda s: "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))
    _split_dir = os.path.join("evaluation_output", _safe(SPLIT))
    for _sid in sessions:
        os.makedirs(os.path.join(_split_dir, _safe(_sid)), exist_ok=True)
    OUTPUT_DIR = (
        os.path.join(_split_dir, _safe(sessions[0]))
        if len(sessions) == 1
        else _split_dir
    )

    batch_idx = 1
    for batch in data_loader:


        batch = batch.to(device)

        if int(batch.num_nodes) == 0:
            continue

        pred_norm = model(batch)  # [B, 2] normalized targets (same space as batch.y)
        pred_px = _denormalize_pred_to_pixels(pred_norm, batch)  # [B, 2] pixels
        pred_px = pred_px.detach().cpu().numpy()

        # Get events from the batch
        events_norm = batch.pos.detach().cpu().numpy()
        labels_px = batch.y_raw.detach().cpu().numpy()

        # Calculate MAE
        mae = np.linalg.norm(pred_px - labels_px)

        # Denormalize the events
        events_px, events_py, events_t = _events_pos_to_pixels(events_norm)
        events_px = np.clip(events_px, 0, SENSOR_WIDTH).astype(int)
        events_py = np.clip(events_py, 0, SENSOR_HEIGHT).astype(int)
        events_p = batch.x[:, 3].detach().cpu().numpy()
        
        # Determine t range
        events_t = events_t.astype(np.int64)

        actual_label_idx = batch_idx * train_cfg.TEMPORAL_SUBSAMPLE
        t_target = actual_label_idx * 10_000
        t_start = max(0, t_target - WINDOW_US)

        events_t_min = np.min(events_t) 
        events_t_min = events_t_min.astype(np.int64) + t_start
        events_t_max = np.max(events_t) 
        events_t_max = events_t_max.astype(np.int64) + t_start
        events_t_min = events_t_min.astype(np.int64)
        events_t_max = events_t_max.astype(np.int64)

        # --- Data Processing for Visualization ---

        # Bin events into frames
        positive_frame = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH))
        negative_frame = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH))

        # masks
        positive_mask = events_p > 0
        negative_mask = events_p < 0

        # count events in frame
        np.add.at(positive_frame, (events_py[positive_mask], events_px[positive_mask]), 1)
        np.add.at(negative_frame, (events_py[negative_mask], events_px[negative_mask]), 1)

        # Total count in frame
        total_count_frame = positive_frame + negative_frame
        safe_total = np.where(total_count_frame == 0, 1.0, total_count_frame)

        # Color the frame by ratio
        positive_channel = positive_frame / safe_total
        negative_channel = negative_frame / safe_total

        # Find the count larger than 90% of the total count
        max_val = np.percentile(total_count_frame[total_count_frame > 0], 75)

        # Intensity frame
        intensity_frame = total_count_frame / max_val
        intensity_frame = np.power(intensity_frame, 0.5) # Gamma correction

        # Color frame by ratio
        color_frame = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH, 3))
        color_frame[:, :, 0] = negative_channel * 255
        color_frame[:, :, 1] = 0
        color_frame[:, :, 2] = positive_channel * 255

        color_frame = color_frame * intensity_frame[:, :, np.newaxis]
        color_frame = np.clip(color_frame, 0, 255).astype(np.uint8)

    
        # --- Visualization ---
        plt.figure(figsize=(10, 10))
        plt.imshow(color_frame)
        plt.tight_layout()
        plt.title(f"Event frame with prediction and label. Time window: {t_start} - {t_target} us (Timestamps:{events_t_min} - {events_t_max} us)")

        # Draw prediction and label
        plt.scatter(pred_px[0, 0], pred_px[0, 1], color='orange', s=60, marker='x', label='Prediction')
        plt.scatter(labels_px[0, 0], labels_px[0, 1], color='green', s=60, marker='o', label='Label')

        # Draw a line between prediction and label
        plt.plot([pred_px[0, 0], labels_px[0, 0]], [pred_px[0, 1], labels_px[0, 1]], color='cyan', linewidth=2)


        
        plt.legend()


        figure_name = f"{SPLIT}_{SESSIONS}_{batch_idx}_MAE{mae:.4f}_{t_start}_{t_target}.png"
        figure_path = os.path.join(OUTPUT_DIR, figure_name)
        plt.savefig(figure_path)

        batch_idx += 1


        

        

        


if __name__ == "__main__":
    raise SystemExit(main())
