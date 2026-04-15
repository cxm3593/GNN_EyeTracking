'''
Main file for the gnn-eyetracking project
@author: Chengyi Ma
'''

from gnn_ev_toolbox.data_tools import DataManager
from gnn_ev_toolbox.gnn_tools import GnnBuilder
from gnn_ev_toolbox.three_et_dataloader import ThreeETDataLoader, ThreeETDataset
import torch
import pandas as pd
import os
from torch_geometric.loader import DataLoader as PyGDataLoader
from gnn_ev_toolbox.models import SimplePupilGNN
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --- Constants ---
DATA_PATH = None  # Set this if you use `sample_data_processing()`.

# DATA_ROOT_3ET = r"C:\Users\cxm3593\Academic\Workspace\Data\3ET+\3ET+ dataset\event_data" # The path in lab machine
DATA_ROOT_3ET = r"C:\Users\VirgilMA\Academics\Research\Data\3ET+ dataset\event_data" # The path in home machine

EXAMPLE_WINDOW_SIZE = 100_000 # 100 ms

VISUALIZATION_MODE = False
PLOT_LOSS_CURVE = True
LIVE_LOSS_PLOT = True

# --- Global Variables ---
LEARNING_RATE = 0.01
BATCH_SIZE = 4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def three_et_data_processing():
    
    if not os.path.isdir(DATA_ROOT_3ET):
        raise FileNotFoundError(
            f"3ET+ data root not found: {DATA_ROOT_3ET}\n"
            "Update DATA_ROOT_3ET in main.py to point at the dataset root that contains 'train/' and 'test/'."
        )

    print("--------------------------------")
    print("3ET+ smoke test started")
    print("--------------------------------")

    # Minimal: ensure the raw loader works.
    loader = ThreeETDataLoader(three_et_data_root=DATA_ROOT_3ET)
    print(f"Events DF shape: {loader.data_df.shape}")
    print(f"Labels DF shape: {loader.labels_df.shape}")

    # Minimal: ensure the PyG dataset builds at least one graph.
    dataset = ThreeETDataset(
        three_et_data_root=DATA_ROOT_3ET,
        session_id="auto",
        spatial_scale=0.125,
        graph_radius=10.0,
        temporal_subsample=5,
    )
    print(f"Dataset length (steps): {len(dataset)}")

    # --- Visualization for testing ---
    if VISUALIZATION_MODE:
        # Find a non-empty sample (early indices can be empty depending on label time).
        sample = None
        sample_idx = None
        for i in range(min(len(dataset), 50)):
            g = dataset[i]
            if g.num_nodes > 0:
                sample = g
                sample_idx = i
                break

        if sample is None:
            sample = dataset[0]
            sample_idx = 0

        print(
            f"Sample graph (idx={sample_idx}): "
            f"nodes={sample.num_nodes}, edges={sample.num_edges}, "
            f"x_shape={tuple(sample.x.shape)}, y_shape={tuple(sample.y.shape)}"
        )

        # Visualize the sample graph (3D).
        if sample.num_nodes > 0 and sample.num_edges > 0:
            GnnBuilder().visualize_graph_3d(sample, title=f"3ET+ sample graph (idx={sample_idx})")

    

    #--- Visualization end ---

    # Optional: verify batching works (no plotting).
    dl = PyGDataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dl))
    print(
        "Batch graph: "
        f"graphs={batch.num_graphs}, nodes={batch.num_nodes}, edges={batch.num_edges}, "
        f"x_shape={tuple(batch.x.shape)}, y_shape={tuple(batch.y.shape)}"
    )

    return dataset


def model_training(dataset, device="cpu", epochs=10, learning_rate=0.01, batch_size=4):
    # 1. Initialize Model
    model = SimplePupilGNN().to(device)

    # 2. Optimizer (The Coach)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Huber Loss (The Scorecard)
    # delta=1.0 means errors larger than 1 pixel are penalized linearly
    criterion = torch.nn.HuberLoss(delta=1.0)
    
    # Prepare DataLoader
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\n--- Starting Training Test (Huber Loss) ---")
    model.train()

    loss_history = []
    step_loss_history = []

    live_fig = None
    live_ax = None
    live_line = None
    if LIVE_LOSS_PLOT:
        plt.ion()
        live_fig, live_ax = plt.subplots(figsize=(8, 4))
        (live_line,) = live_ax.plot([], [], linewidth=1.5)
        live_ax.set_xlabel("Step")
        live_ax.set_ylabel("Huber Loss")
        live_ax.set_title("Training Loss (per step)")
        live_ax.grid(True, alpha=0.3)
        live_fig.tight_layout()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Some windows contain no events -> empty graphs (num_nodes=0).
            # Those can break graph-level pooling and cause pred/y shape mismatch.
            if batch.num_nodes == 0:
                pbar.set_postfix(loss="skip(empty)")
                continue
            
            # Forward pass: Generate predictions
            pred = model(batch)

            # # If a batch contains trailing empty graphs (no nodes),
            # # `global_mean_pool` returns fewer rows than `batch.num_graphs`.
            # # In that case, the missing graphs are always at the end.
            # if pred.size(0) != batch.y.size(0):
            #     y = batch.y[: pred.size(0)]
            # else:
            #     y = batch.y

            y = batch.y
            
            # Compute Huber Loss
            # Ensure batch.y is the correct shape [batch_size, 2]
            loss = criterion(pred, y)
            
            # Backward pass: Calculate gradients and update weights
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss.item())
            total_loss += loss_val
            step_loss_history.append(loss_val)

            # Update tqdm bar every step with current loss
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            # Optional: live loss curve update (per step)
            if LIVE_LOSS_PLOT and live_ax is not None and live_line is not None:
                xs = range(1, len(step_loss_history) + 1)
                live_line.set_data(list(xs), step_loss_history)
                live_ax.relim()
                live_ax.autoscale_view()
                live_fig.canvas.draw_idle()
                live_fig.canvas.flush_events()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch:02d} | Average Huber Loss: {avg_loss:.4f}")

    print("--- Test Run Complete ---")

    if LIVE_LOSS_PLOT:
        plt.ioff()

    if PLOT_LOSS_CURVE and len(loss_history) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Average Huber Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        plt.show()


# --- Main Function ---
def main():

    dataset = three_et_data_processing()
    model_training(dataset, device=DEVICE, epochs=10, learning_rate=0.01, batch_size=4)


if __name__ == "__main__":
    main()
