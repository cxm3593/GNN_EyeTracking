'''
Main file for the gnn-eyetracking project
@author: Chengyi Ma
'''

from gnn_ev_toolbox.data_tools import DataManager
from gnn_ev_toolbox.gnn_tools import GnnBuilder
from gnn_ev_toolbox.three_et_dataloader import ThreeETDataLoader
import torch
import pandas as pd

# --- Constants ---
# DATA_PATH = r"C:\Users\cxm3593\Academic\Workspace\Data\ev_eye\raw_data\Data_davis\user1\left\session_1_0_1\events\events.txt"

DATA_ROOT_3ET = r"C:\Users\cxm3593\Academic\Workspace\Data\3ET+\3ET+ dataset\event_data"


EXAMPLE_WINDOW_SIZE = 100_000 # 100 ms

# --- Global Variables ---

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
    ThreeETDataLoader(three_et_data_root = DATA_ROOT_3ET)

# --- Main Function ---
def main():

    three_et_data_processing()


if __name__ == "__main__":
    main()
