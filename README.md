# GNN_EyeTracking

An experimental framework for event-based eye tracking using Graph Neural Networks (GNNs). 

## Project Summary

This project explores the application of Graph Neural Networks to neuromorphic event-based vision data (such as the EvEye dataset) for continuous gaze and pupil position regression. Event cameras produce asynchronous streams of data with microsecond resolution, making them ideal for ultra-low latency eye tracking. 

Unlike traditional frame-based CNN approaches, this project models the event stream as a spatio-temporal graph. The goal is to evaluate, optimize, and innovate on graph construction and processing methods to achieve high-accuracy, low-power continuous coordinate regression.

## How to Run

This project uses [`uv`](https://docs.astral.sh/uv/) as its ultra-fast Python package and environment manager.

### Prerequisites
Make sure you have `uv` installed on your system.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/cxm3593/GNN_EyeTracking.git
cd GNN_EyeTracking
```
2. Sync the project and install dependencies:
```
uv sync
```
3. Run the code
```
uv run main.py
```

This project is implemented with Ev-Eye dataset. We will try to support other dataset in future. 

## Project Description and Roadmap

### Basic study: Static GNNs & Optical Flow Distance Metric (Current)

The initial phase focuses on establishing a baseline by building static graphs from time-windows of events and predicting continuous (x,y) pupil coordinates via regression (e.g., using Huber Loss).

**Optical Flow Distance Metric**: Traditional event graphs connect nodes based on a simple spatio-temporal radius (Euclidean distance) and arbitrary or tuned weight parameters. In this project, we are introducing a novel dynamic distance metric. We estimate the optical flow velocity of the events. To measure the true relationship between two events occurring at different times, we use their velocities to forward-project the older event and backward-project the newer event to a shared temporal plane and then calculate the sum of distances at the two timestamps. This method aims to improve the result of radius based graph building method for events with different motion speeds with robustness. 

### Improvement 1: Asynchronous GNNs (Inspired by AEGNN)
Static batches destroy the asynchronous nature of event cameras. In the second phase, we will adopt and evaluate methodologies from AEGNN (Asynchronous Event-based Graph Neural Networks).

* Goal: Instead of reprocessing the entire graph for every new window, we will implement asynchronous updates where only the nodes affected by newly arriving events trigger network activations.

* Technique: Integrating spatial subsampling and event-driven message passing to drastically reduce redundant computations while maintaining regression accuracy.

### Improvement 2: Directed Dynamic Graphs (Inspired by EvGNN)

To further optimize the performance of GNN on event data, we are evaluating the method proposed by EvGNN which adopts a dynamic graph with directed edges only from older nodes(events) to newer nodes.

* Goal: Implement a strictly causal, directed graph architecture.
* Technique: Edges will only be permitted to point from past events to future events. We will implement dynamic graph construction using fixed-size, pixel-specific memory queues. This naturally evicts stale events and limits the spatial search space, resulting in a lightweight, dynamically evolving graph optimized for edge-computing deployment.

*The may choose to evaluate and adopt one of the improvements*

## References

[1] Y. Yang, A. Kneip, and C. Frenkel, “EvGNN: An Event-Driven Graph Neural Network Accelerator for Edge Vision,” IEEE Transactions on Circuits and Systems for Artificial Intelligence, vol. 2, no. 1, pp. 37–50, Mar. 2025, doi: 10.1109/TCASAI.2024.3520905.

[2] S. Schaefer, D. Gehrig, and D. Scaramuzza, “AEGNN: Asynchronous Event-Based Graph Neural Networks,” presented at the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12371–12381. Accessed: May 15, 2025. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2022/html/Schaefer_AEGNN_Asynchronous_Event-Based_Graph_Neural_Networks_CVPR_2022_paper.html

[3] G. Dong et al., “Graph Neural Networks in IoT: A Survey,” ACM Trans. Sen. Netw., vol. 19, no. 2, p. 47:1-47:50, Apr. 2023, doi: 10.1145/3565973.

[4] Z. A. Sahili and M. Awad, “Spatio-Temporal Graph Neural Networks: A Survey,” Feb. 11, 2023, arXiv: arXiv:2301.10569. doi: 10.48550/arXiv.2301.10569.

[5] N. Bandara, T. Kandappu, A. Sen, I. Gokarn, and A. Misra, “EyeGraph: Modularity-aware Spatio Temporal Graph Clustering for Continuous Event-based Eye Tracking”.

[6] G. Zhao et al., “EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras”.

[7] T. Dalgaty, T. Mesquida, D. Joubert, A. Sironi, P. Vivet, and C. Posch, “HUGNet: Hemi-Spherical Update Graph Neural Network applied to low-latency event-based optical flow,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Jun. 2023, pp. 3953–3962. doi: 10.1109/CVPRW59228.2023.00411.

[8] M. Fey and J. E. Lenssen, “Fast Graph Representation Learning with PyTorch Geometric,” Apr. 25, 2019, arXiv: arXiv:1903.02428. doi: 10.48550/arXiv.1903.02428.

[9] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu, “A Comprehensive Survey on Graph Neural Networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 1, pp. 4–24, Jan. 2021, doi: 10.1109/TNNLS.2020.2978386.

[10] Y. Bi, A. Chadha, A. Abbas, E. Bourtsoulatze, and Y. Andreopoulos, “Graph-Based Spatio-Temporal Feature Learning for Neuromorphic Vision Sensing,” IEEE Transactions on Image Processing, vol. 29, pp. 9084–9098, 2020, doi: 10.1109/TIP.2020.3023597.










