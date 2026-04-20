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
LABEL_NORMALIZATION = "minmax01"  # None | "zscore" | "minmax01"

VAL_FRACTION = 0.2
SPLIT_SEED = 42

# --- Global Variables ---
LEARNING_RATE = 0.01
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUBER_DELTA = 0.02 if LABEL_NORMALIZATION is not None else 1.0  # in label units

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


def _build_session_datasets(data_root: str, split: str, session_ids: list[str]) -> list[ThreeETDataset]:
    datasets: list[ThreeETDataset] = []
    for sid in session_ids:
        datasets.append(
            ThreeETDataset(
                root=data_root,
                session_id=sid,
                split=split,
                spatial_scale=SPATIAL_SCALE,
                label_normalization=LABEL_NORMALIZATION,
                graph_radius=GRAPH_RADIUS,
                temporal_subsample=TEMPORAL_SUBSAMPLE,
                max_num_neighbors=MAX_NUM_NEIGHBORS,
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

    train_list = _build_session_datasets(data_root, "train", train_sessions)
    val_list = _build_session_datasets(data_root, "train", val_sessions)
    test_list = _build_session_datasets(data_root, "test", all_test_sessions)

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


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    n_steps = 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_nodes == 0:
            continue
        pred = model(batch)
        loss = criterion(pred, batch.y)
        total_loss += float(loss.item())
        n_steps += 1
    model.train()
    return total_loss / max(n_steps, 1)


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
):
    model = SimplePupilGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # delta is in the same units as graph.y (normalized units if LABEL_NORMALIZATION != None)
    criterion = torch.nn.HuberLoss(delta=HUBER_DELTA)

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(tensorboard_root, run_name)
    ckpt_dir = os.path.join(checkpoint_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    best_val = float("inf")
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

            mem = _gpu_mem_mb(device)
            writer.add_scalar("HuberLoss/train_step", loss_val, global_step)
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
                "N": int(batch.num_nodes),
                "E": int(batch.num_edges),
            }
            if on_cuda:
                postfix["gpu"] = f"{mem['alloc']:.0f}/{mem['peak_alloc']:.0f}MB"
                postfix["rsv"] = f"{mem['peak_reserved']:.0f}MB"
            pbar.set_postfix(postfix)

        train_avg = total_loss / max(n_steps, 1)
        writer.add_scalar("HuberLoss/train_epoch_mean", train_avg, epoch)
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
            val_avg = _evaluate(model, val_loader, criterion, device)
            writer.add_scalar("HuberLoss/val_epoch_mean", val_avg, epoch)
            print(f"Epoch {epoch:02d} | train={train_avg:.4f} | val={val_avg:.4f}")
            if val_avg < best_val:
                best_val = val_avg
                _save_checkpoint(best_path, model, epoch=epoch, metric=val_avg,
                                 extra={"selection": "best_val"})
                print(f"  Saved best checkpoint -> {best_path} (val={val_avg:.4f})")
        else:
            print(f"Epoch {epoch:02d} | train={train_avg:.4f}")

        _save_checkpoint(last_path, model, epoch=epoch,
                         metric=val_avg if val_loader is not None else train_avg,
                         extra={"selection": "last"})

    writer.close()
    print(f"Final weights: {last_path}")
    if val_loader is not None and best_val != float("inf"):
        print(f"Best val weights: {best_path} (val={best_val:.4f})")
    return model


def model_testing(model, test_dataset, device="cpu", batch_size=4) -> float:
    if test_dataset is None or len(test_dataset) == 0:
        print("No test samples available; skipping final evaluation.")
        return float("nan")
    loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.HuberLoss(delta=HUBER_DELTA)
    loss = _evaluate(model, loader, criterion, device)
    print(f"Test Huber loss: {loss:.4f}")
    return loss


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
