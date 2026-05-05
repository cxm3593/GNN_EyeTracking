'''
Main file for the gnn-eyetracking project
@author: Chengyi Ma
'''

from gnn_ev_toolbox.data_tools import DataManager
from gnn_ev_toolbox.gnn_tools import GnnBuilder
from gnn_ev_toolbox.three_et_dataloader import (
    ThreeETDataset,
    train_val_session_ids_split,
)
import torch
import os
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from gnn_ev_toolbox.models import SimplePupilGNN
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import yaml

# --- Constants ---
DATA_PATH = None  # Set this if you use `sample_data_processing()`.

DATA_ROOT_3ET = r"C:\Users\cxm3593\Academic\Workspace\Data\3ET+\3ET+ dataset\event_data" # The path in lab machine
# DATA_ROOT_3ET = r"C:\Users\VirgilMA\Academics\Research\Data\3ET+ dataset\event_data" # The path in home machine

EXAMPLE_WINDOW_SIZE = 100_000 # 100 ms

VISUALIZATION_MODE = False
TENSORBOARD_ROOT = "runs"
CHECKPOINT_ROOT = "checkpoints"

# Dataset parameters shared across every session in the concat datasets.
SPATIAL_SCALE = 1.0
GRAPH_RADIUS = 10.0
TEMPORAL_SUBSAMPLE = 5
MAX_NUM_NEIGHBORS = 8  # Hard cap on edges per node (radius_graph). Lower = less GPU memory.
LABEL_NORMALIZATION = "resolution"  # None | "zscore" | "minmax01" | "resolution"
SPATIAL_RESOLUTION_WIDTH = 640
SPATIAL_RESOLUTION_HEIGHT = 480

# Time unit for graph coordinates: µs per 1 pixel of space.
# A value here rescales t so that `graph_radius` has the same meaning on x/y/t.
# Set to None to keTIME_TO_PIXEL_USep raw µs.
TIME_TO_PIXEL_US: float | None = 30.0

# Window of events (in µs) feeding each prediction; matches the hardcoded value
# previously used in ThreeETDataset.get(). Also drives input normalization on t.
WINDOW_US = 100_000

# When True, divide node features and positions by [W, H, t_max_in_window] AFTER
# the radius graph is built. Keeps GRAPH_RADIUS semantics intact while putting
# all three input channels in roughly [0, 1] for stable optimization.
NORMALIZE_INPUT = True

# When True, append polarity as a 4th node feature on data.x (mapped {0, 1} -> {-1, +1}).
# data.pos stays 3D (xyt) so any future edge-geometry features stay clean.
INCLUDE_POLARITY = True

# Train-only random translation augmentation. Each train window's events AND its
# pupil-center label are jointly shifted by (dx, dy) ~ Uniform(-T, T) raw pixels.
# Targets generalization, not capacity: same local pattern, different absolute
# sensor coordinates -> the model can't lean on subject-specific position priors.
AUGMENT_TRANSLATE = True
TRANSLATE_MAX_PX = 50.0
AUGMENT_SEED = 12345  # Reproducibility for the dx/dy stream across runs.

# Graph convolution operator. "sage" trains faster and reached the same test MAE
# as "gine" on this dataset; reverting to "sage" as the production default. The
# "gine" path remains available as an option (uses relative edge geometry).
CONV_TYPE = "sage"  # "sage" | "gine"
COMPUTE_EDGE_ATTR = (CONV_TYPE == "gine")
EDGE_DIM = 3  # (dx, dy, dt) — only used when CONV_TYPE == "gine".

VAL_FRACTION = 0.2
SPLIT_SEED = 1996 # Used 42,


# --- Global Variables ---
RUN_NOTE = "step5;SAGE_AdamW_wd1e-4_LRplateau(SAGE_LN_InputNorm_Polarity_Radius10_K8_TempSubsample5_TimeToPixel30_HuberDelta0.05)"

LEARNING_RATE = 1e-3
BATCH_SIZE = 2
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Huber delta in label units. With "resolution" normalization, labels are in [0, 1]
# so delta=0.05 corresponds to ~5% of the sensor (roughly 25-30 px), a reasonable
# threshold between the quadratic (small, precise) and linear (outlier-robust) regimes.
HUBER_DELTA = 0.05 if LABEL_NORMALIZATION is not None else 5.0

# AdamW weight decay. 1e-4 is the canonical default for AdamW on vision/graph tasks
# (it's also what Hugging Face / timm / PyTorch examples use). Stronger (1e-3)
# can over-regularize a small model; gentler (1e-5) is barely distinguishable
# from no decay.
WEIGHT_DECAY = 1e-4

# ReduceLROnPlateau: when val loss plateaus for LR_PATIENCE epochs, multiply LR
# by LR_FACTOR. Stops at LR_MIN_LR. LR_PATIENCE must be strictly less than
# EARLY_STOP_PATIENCE so the scheduler gets several chances to reduce LR before
# early stop kicks in. Recommended ratio is 3-4x; we have 5 vs 20 (4x).
LR_FACTOR = 0.5
LR_PATIENCE = 5
LR_MIN_LR = 1e-5

# Number of input channels into the GNN: 3 (x, y, t) or 4 (+ polarity). Derived
# from INCLUDE_POLARITY so the model and dataset stay in sync.
GNN_INPUT_DIM = 4 if INCLUDE_POLARITY else 3

# Per-node dropout applied after each conv -> norm -> relu block in the encoder.
# Mild regularization on the encoder side; 0.0 disables it.
CONV_DROPOUT = 0.1

# Early-stopping: halt training when validation loss has not improved by at least
# EARLY_STOP_MIN_DELTA for EARLY_STOP_PATIENCE consecutive epochs. Saves wall-clock
# time without changing the trained model's quality (best checkpoint is unchanged).
# Loosened in step 2: previous (10, 1e-4) cut runs short before the slow late-epoch
# val improvement we saw in the baseline could materialize.
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.0

# --- Helper Functions ---

def sample_data_processing():
    '''
    The sample code used for testing, not used for now.
    '''
    print("--------------------------------")
    print("Program started")
    print("--------------------------------")
    # Load data
    print("Loading data...")
    dm = DataManager()
    sample_data = dm.load_dataset_EvEye_raw(DATA_PATH)
    print(sample_data.head())
    print(sample_data.info())

    ## Evaluate and remove hot pixels
    print("Evaluating hot pixels...")
    hot_pixels = dm.evaluate_hotpixels_event_rate(sample_data)
    print(f"Hot pixels: {hot_pixels}")
    sample_data = dm.remove_hot_pixels(sample_data, hot_pixels)
    print(f"Sample data after removing hot pixels: {sample_data.head()}")
    print(f"Sample data after removing hot pixels: {sample_data.info()}")

    # Convert events to points
    t_start = sample_data["timestamp"].iloc[0]
    t_end   = t_start + EXAMPLE_WINDOW_SIZE
    points = dm.events_to_points_window(sample_data, [t_start, t_end], time_conversion_factor=1e-3)


    pt = torch.tensor(points[["x", "y", "t"]].values, dtype=torch.float32)
    builder = GnnBuilder()
    graph = builder.build_radius_graph(pt, r=10.0)
    print(f"\nGraph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    builder.visualize_graph_3d(graph)

def _list_sessions(data_root: str, split: str) -> list[str]:
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )


def _build_session_datasets(
    data_root: str,
    split: str,
    session_ids: list[str],
    time_to_pixel_us: float | None,
    augment: bool = False,
    augment_seed_base: int | None = None,
) -> list[ThreeETDataset]:
    '''
    Build one ThreeETDataset per session id.

    Args:
        augment: If True, enable train-time augmentation (currently random
            translation). Should only be True for the train split.
        augment_seed_base: Base RNG seed for augmentation. Each session derives
            its own seed as ``augment_seed_base + session_index`` so per-session
            (dx, dy) streams are decorrelated. Ignored when ``augment`` is False.
    '''
    datasets: list[ThreeETDataset] = []
    for i, sid in enumerate(session_ids):
        if augment and AUGMENT_TRANSLATE:
            per_session_seed = (
                augment_seed_base + i if augment_seed_base is not None else None
            )
            translate_px = TRANSLATE_MAX_PX
            do_aug = True
        else:
            per_session_seed = None
            translate_px = 0.0
            do_aug = False

        datasets.append(
            ThreeETDataset(
                root=data_root,
                session_id=sid,
                split=split,
                spatial_scale=SPATIAL_SCALE,
                label_normalization=LABEL_NORMALIZATION,
                resolution_width=SPATIAL_RESOLUTION_WIDTH,
                resolution_height=SPATIAL_RESOLUTION_HEIGHT,
                time_to_pixel_us=time_to_pixel_us,
                graph_radius=GRAPH_RADIUS,
                temporal_subsample=TEMPORAL_SUBSAMPLE,
                max_num_neighbors=MAX_NUM_NEIGHBORS,
                window_us=WINDOW_US,
                normalize_input=NORMALIZE_INPUT,
                include_polarity=INCLUDE_POLARITY,
                augment_translate=do_aug,
                translate_max_px=translate_px,
                augment_seed=per_session_seed,
                compute_edge_attr=COMPUTE_EDGE_ATTR,
                log=False,
            )
        )
    return datasets


def build_3et_datasets(
    data_root: str,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[ConcatDataset, ConcatDataset, ConcatDataset, dict]:
    '''
    Enumerate every session under ``<data_root>/train`` and ``<data_root>/test``,
    split the training sessions into train/validation by session id (no leakage),
    and return concatenated datasets ready for a PyG DataLoader.

    Returns:
        (train_ds, val_ds, test_ds, info) where ``info`` contains the session
        ids used for each split.
    '''
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"3ET+ data root not found: {data_root}\n"
            "Update DATA_ROOT_3ET in main.py to point at the dataset root that contains 'train/' and 'test/'."
        )

    all_train_sessions = _list_sessions(data_root, "train")
    all_test_sessions = _list_sessions(data_root, "test")
    print(f"Train sessions found: {len(all_train_sessions)}")
    print(f"Test sessions found : {len(all_test_sessions)}")

    train_sessions, val_sessions = train_val_session_ids_split(
        all_train_sessions, val_fraction=val_fraction, random_seed=seed
    )
    print(
        f"Split train/val by session: {len(train_sessions)} train / "
        f"{len(val_sessions)} val (seed={seed}, val_fraction={val_fraction})"
    )

    print(
        f"Graph t-scaling: time_to_pixel_us="
        f"{'raw µs' if TIME_TO_PIXEL_US is None else f'{float(TIME_TO_PIXEL_US):.2f} µs/px'}"
    )

    train_list = _build_session_datasets(
        data_root, "train", train_sessions, TIME_TO_PIXEL_US,
        augment=True, augment_seed_base=AUGMENT_SEED,
    )
    val_list = _build_session_datasets(
        data_root, "train", val_sessions, TIME_TO_PIXEL_US,
        augment=False,
    )
    test_list = _build_session_datasets(
        data_root, "test", all_test_sessions, TIME_TO_PIXEL_US,
        augment=False,
    )

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)
    test_ds = ConcatDataset(test_list)

    print(
        f"Concat sizes (steps): train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)}"
    )

    info = {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": all_test_sessions,
    }
    return train_ds, val_ds, test_ds, info


def _gpu_mem_mb(device) -> dict[str, float]:
    '''Return GPU memory stats in MiB; all zeros on CPU.

    Keys:
      alloc      : currently allocated tensor bytes
      peak_alloc : peak allocated since last reset
      reserved   : total pool currently held by PyTorch allocator
      peak_reserved: peak pool held since last reset
    '''
    if isinstance(device, str):
        on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    else:
        on_cuda = getattr(device, "type", "") == "cuda" and torch.cuda.is_available()
    if not on_cuda:
        return {"alloc": 0.0, "peak_alloc": 0.0, "reserved": 0.0, "peak_reserved": 0.0}
    mb = 1024 ** 2
    return {
        "alloc": torch.cuda.memory_allocated() / mb,
        "peak_alloc": torch.cuda.max_memory_allocated() / mb,
        "reserved": torch.cuda.memory_reserved() / mb,
        "peak_reserved": torch.cuda.max_memory_reserved() / mb,
    }


def _pixel_errors(pred: torch.Tensor, batch) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Denormalize predictions and targets back to pixel units and return:
        (per-sample MAE over x,y coords, per-sample Euclidean distance)
    Both tensors have shape [B].
    '''
    pred_px = pred * batch.y_scale + batch.y_center          # [B, 2]
    y_px = batch.y_raw                                       # [B, 2]
    abs_err = (pred_px - y_px).abs()                         # [B, 2]
    mae_per_sample = abs_err.mean(dim=1)                     # [B]
    l2_per_sample = torch.linalg.norm(pred_px - y_px, dim=1) # [B]
    return mae_per_sample, l2_per_sample


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> dict[str, float]:
    '''
    Returns a dict with:
        loss     : mean Huber loss in normalized units (what we optimize)
        mae_px   : mean absolute error per coordinate, in pixels
        l2_px    : mean Euclidean distance between prediction and target, in pixels
    '''
    model.eval()
    total_loss = 0.0
    n_steps = 0
    sum_mae_px = 0.0
    sum_l2_px = 0.0
    n_samples = 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_nodes == 0:
            continue
        pred = model(batch)
        loss = criterion(pred, batch.y)
        total_loss += float(loss.item())
        n_steps += 1

        mae_per_sample, l2_per_sample = _pixel_errors(pred, batch)
        sum_mae_px += float(mae_per_sample.sum().item())
        sum_l2_px += float(l2_per_sample.sum().item())
        n_samples += int(mae_per_sample.numel())
    model.train()
    return {
        "loss": total_loss / max(n_steps, 1),
        "mae_px": sum_mae_px / max(n_samples, 1),
        "l2_px": sum_l2_px / max(n_samples, 1),
    }


def _save_checkpoint(path: str, model, *, epoch: int, metric: float | None, extra: dict | None = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metric": metric,
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def model_training(
    train_dataset,
    val_dataset=None,
    device="cpu",
    epochs=10,
    learning_rate=0.01,
    batch_size=4,
    tensorboard_root=TENSORBOARD_ROOT,
    checkpoint_root=CHECKPOINT_ROOT,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = EARLY_STOP_MIN_DELTA,
):
    model = SimplePupilGNN(
        input_dim=GNN_INPUT_DIM,
        conv_dropout=CONV_DROPOUT,
        conv_type=CONV_TYPE,
        edge_dim=EDGE_DIM,
    ).to(device)
    # AdamW = Adam with decoupled weight decay. Without decoupling, classical Adam's
    # adaptive learning rates effectively cancel out the L2 penalty. This is the
    # default optimizer recommended by the original "Decoupled Weight Decay" paper
    # (Loshchilov & Hutter, 2019).
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    # Halve LR when val loss plateaus. Driven by val loss in the train loop below.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN_LR,
    )
    # delta is in the same units as graph.y (normalized units if LABEL_NORMALIZATION != None)
    criterion = torch.nn.HuberLoss(delta=HUBER_DELTA)

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{RUN_NOTE}_{run_name}_LR{learning_rate}_BS{batch_size}_E{epochs}"

    
    log_dir = os.path.join(tensorboard_root, run_name)
    ckpt_dir = os.path.join(checkpoint_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Dump the full hyperparameter snapshot to a YAML file inside the run dir.
    # Source-of-truth for what each run actually was, independent of the (often
    # human-edited) directory name. Also mirrored to TensorBoard's "Text" tab so
    # it travels with the events file.
    run_config = {
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "augment_seed": AUGMENT_SEED,
        "augment_translate": AUGMENT_TRANSLATE,
        "translate_max_px": TRANSLATE_MAX_PX,
        "spatial_scale": SPATIAL_SCALE,
        "graph_radius": GRAPH_RADIUS,
        "max_num_neighbors": MAX_NUM_NEIGHBORS,
        "temporal_subsample": TEMPORAL_SUBSAMPLE,
        "window_us": WINDOW_US,
        "time_to_pixel_us": TIME_TO_PIXEL_US,
        "label_normalization": LABEL_NORMALIZATION,
        "normalize_input": NORMALIZE_INPUT,
        "include_polarity": INCLUDE_POLARITY,
        "compute_edge_attr": COMPUTE_EDGE_ATTR,
        "conv_type": CONV_TYPE,
        "edge_dim": EDGE_DIM,
        "gnn_input_dim": GNN_INPUT_DIM,
        "conv_dropout": CONV_DROPOUT,
        "huber_delta": HUBER_DELTA,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "lr_factor": LR_FACTOR,
        "lr_patience": LR_PATIENCE,
        "lr_min_lr": LR_MIN_LR,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "batch_size": batch_size,
        "epochs": epochs,
        "device": str(device),
        "run_note": RUN_NOTE,
    }
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, default_flow_style=False)
    writer.add_text("config", "```yaml\n" + yaml.safe_dump(run_config, sort_keys=False) + "```")
    global_step = 0
    best_val = float("inf")
    epochs_since_improvement = 0
    best_path = os.path.join(ckpt_dir, "model_best.pt")
    last_path = os.path.join(ckpt_dir, "model_last.pt")

    print(f"\n--- Starting Training (Huber Loss) ---")
    print(f"TensorBoard log directory: {log_dir}")
    print(f"Checkpoint directory     : {ckpt_dir}")
    print(f"View with: tensorboard --logdir={tensorboard_root}")
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        print(f"CUDA device              : {torch.cuda.get_device_name(0)}")
    model.train()

    on_cuda = isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available()

    for epoch in range(1, epochs + 1):
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()

        total_loss = 0.0
        n_steps = 0
        sum_mae_px = 0.0
        sum_l2_px = 0.0
        n_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            if batch.num_nodes == 0:
                pbar.set_postfix(loss="skip(empty)")
                continue

            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            n_steps += 1

            with torch.no_grad():
                mae_per_sample, l2_per_sample = _pixel_errors(pred, batch)
                batch_mae_px = float(mae_per_sample.mean().item())
                batch_l2_px = float(l2_per_sample.mean().item())
                sum_mae_px += float(mae_per_sample.sum().item())
                sum_l2_px += float(l2_per_sample.sum().item())
                n_samples += int(mae_per_sample.numel())

            mem = _gpu_mem_mb(device)
            writer.add_scalar("HuberLoss/train_step", loss_val, global_step)
            writer.add_scalar("PixelMAE/train_step", batch_mae_px, global_step)
            writer.add_scalar("PixelL2/train_step", batch_l2_px, global_step)
            writer.add_scalar("Graph/num_nodes", int(batch.num_nodes), global_step)
            writer.add_scalar("Graph/num_edges", int(batch.num_edges), global_step)
            if on_cuda:
                writer.add_scalar("GPU/mem_alloc_MB", mem["alloc"], global_step)
                writer.add_scalar("GPU/mem_peak_MB", mem["peak_alloc"], global_step)
                writer.add_scalar("GPU/mem_reserved_MB", mem["reserved"], global_step)
                writer.add_scalar("GPU/mem_peak_reserved_MB", mem["peak_reserved"], global_step)
            global_step += 1

            postfix = {
                "loss": f"{loss_val:.4f}",
                "mae_px": f"{batch_mae_px:.2f}",
                "l2_px": f"{batch_l2_px:.2f}",
                "N": int(batch.num_nodes),
                "E": int(batch.num_edges),
            }
            if on_cuda:
                postfix["gpu"] = f"{mem['alloc']:.0f}/{mem['peak_alloc']:.0f}MB"
                postfix["rsv"] = f"{mem['peak_reserved']:.0f}MB"
            pbar.set_postfix(postfix)

        train_avg = total_loss / max(n_steps, 1)
        train_mae_px = sum_mae_px / max(n_samples, 1)
        train_l2_px = sum_l2_px / max(n_samples, 1)
        writer.add_scalar("HuberLoss/train_epoch_mean", train_avg, epoch)
        writer.add_scalar("PixelMAE/train_epoch_mean", train_mae_px, epoch)
        writer.add_scalar("PixelL2/train_epoch_mean", train_l2_px, epoch)
        if on_cuda:
            epoch_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            epoch_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            writer.add_scalar("GPU/epoch_peak_MB", epoch_peak_alloc, epoch)
            writer.add_scalar("GPU/epoch_peak_reserved_MB", epoch_peak_reserved, epoch)
            print(
                f"  GPU this epoch: peak alloc={epoch_peak_alloc:.0f} MB, "
                f"peak reserved={epoch_peak_reserved:.0f} MB"
            )

        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            writer.add_scalar("HuberLoss/val_epoch_mean", val_loss, epoch)
            writer.add_scalar("PixelMAE/val_epoch_mean", val_metrics["mae_px"], epoch)
            writer.add_scalar("PixelL2/val_epoch_mean", val_metrics["l2_px"], epoch)
            # Step the LR scheduler off val loss. This may halve the LR if val has
            # plateaued for LR_PATIENCE epochs. Log the resulting LR so we can see
            # in TensorBoard exactly when the scheduler kicked in.
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Optim/lr", current_lr, epoch)
            print(
                f"Epoch {epoch:02d} | "
                f"train loss={train_avg:.4f} mae={train_mae_px:.2f}px l2={train_l2_px:.2f}px | "
                f"val loss={val_loss:.4f} mae={val_metrics['mae_px']:.2f}px "
                f"l2={val_metrics['l2_px']:.2f}px | lr={current_lr:.2e}"
            )
            if val_loss < best_val - early_stop_min_delta:
                best_val = val_loss
                epochs_since_improvement = 0
                _save_checkpoint(best_path, model, epoch=epoch, metric=val_loss,
                                 extra={"selection": "best_val"})
                print(f"  Saved best checkpoint -> {best_path} (val loss={val_loss:.4f})")
            else:
                epochs_since_improvement += 1
                print(
                    f"  No val improvement for {epochs_since_improvement}/"
                    f"{early_stop_patience} epoch(s) (best={best_val:.4f})"
                )
        else:
            print(
                f"Epoch {epoch:02d} | "
                f"train loss={train_avg:.4f} mae={train_mae_px:.2f}px l2={train_l2_px:.2f}px"
            )

        _save_checkpoint(last_path, model, epoch=epoch,
                         metric=val_loss if val_loader is not None else train_avg,
                         extra={"selection": "last"})

        # Early stop only when we are actually tracking a validation metric.
        if (
            val_loader is not None
            and early_stop_patience is not None
            and early_stop_patience > 0
            and epochs_since_improvement >= early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: no val improvement "
                f"for {early_stop_patience} consecutive epochs (best val={best_val:.4f})."
            )
            break

    writer.close()
    print(f"Final weights: {last_path}")
    if val_loader is not None and best_val != float("inf"):
        print(f"Best val weights: {best_path} (val loss={best_val:.4f})")
    return model


def model_testing(model, test_dataset, device="cpu", batch_size=4) -> dict[str, float]:
    if test_dataset is None or len(test_dataset) == 0:
        print("No test samples available; skipping final evaluation.")
        return {"loss": float("nan"), "mae_px": float("nan"), "l2_px": float("nan")}
    loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.HuberLoss(delta=HUBER_DELTA)
    metrics = _evaluate(model, loader, criterion, device)
    print(
        f"Test | Huber loss={metrics['loss']:.4f} | "
        f"pixel MAE={metrics['mae_px']:.2f}px | "
        f"pixel L2={metrics['l2_px']:.2f}px"
    )
    return metrics


# --- Main Function ---
def main():
    train_ds, val_ds, test_ds, info = build_3et_datasets(DATA_ROOT_3ET)
    print(f"Train sessions: {info['train_sessions']}")
    print(f"Val   sessions: {info['val_sessions']}")
    print(f"Test  sessions: {info['test_sessions']}")

    model = model_training(
        train_ds,
        val_dataset=val_ds,
        device=DEVICE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    model_testing(model, test_ds, device=DEVICE, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    main()
