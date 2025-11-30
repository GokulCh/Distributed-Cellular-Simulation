# Distributed Cellular Simulation with Dynamic Load Balancing

A high-performance distributed simulation framework for cellular automata (specifically wildfire propagation) implemented in **Python** (for prototyping) and **C++** (for maximum performance) using MPI. This project demonstrates the application of **Dynamic Load Balancing** to address the challenge of uneven computational workloads in spatial simulations.

## 📖 Abstract

Cellular simulations, such as wildfire spread, epidemic diffusion, and fluid flow, often suffer from uneven workload distributions over time. A static division of the grid (Static Domain Decomposition) leads to inefficiency: some processors sit idle while others are overwhelmed by active regions.

This project implements a **Distributed Cellular Automata** framework that:
1.  Decomposes a global grid into 1D horizontal strips across multiple MPI ranks.
2.  Simulates a probabilistic wildfire model.
3.  **Dynamically balances the load** at runtime by migrating grid rows between neighboring processors based on activity levels ("burning" cells).

## 🚀 Key Features

*   **Parallel Execution**: Utilizes `mpi4py` (Python) and `MS-MPI` (C++) for message passing interface communication.
*   **High-Performance C++ Engine**: Includes a C++ implementation that achieves a **13x speedup** over the Python version by minimizing serialization overhead.
*   **1D Domain Decomposition**: Splits the global grid into horizontal strips, allowing scalable processing across multiple nodes.
*   **Dynamic Load Balancing**: Implements a diffusive load balancing algorithm. Processors exchange workload metrics and migrate rows to neighbors to equalize computational intensity.
*   **Non-blocking Communication**: Uses `Isend`/`Irecv` for ghost cell exchanges, enabling potential overlap of communication and computation.
*   **Visualization**: Includes tools to generate 2D heatmaps of the simulation state, visualizing both the fire spread and the changing grid partitions.
*   **Benchmarking**: Automated scripts to compare the scaling performance of Static vs. Dynamic partitioning.

## 🛠️ System Architecture

### 1. Domain Decomposition
The global grid ($N \times M$) is divided along the row dimension.
*   **Rank $i$** manages a subgrid of size $R_i \times M$.
*   $\sum R_i = N$ (Total rows).
*   In **Static Mode**, $R_i$ is constant ($N / P$).
*   In **Dynamic Mode**, $R_i$ changes over time as rows are migrated.

### 2. Ghost Cell Exchange
To simulate the cellular automata rules correctly across boundaries, each rank maintains "ghost rows" (padding) at the top and bottom of its local grid.
*   **Step Start**: Rank $i$ sends its top row to Rank $i-1$ and bottom row to Rank $i+1$.
*   **Step End**: Rank $i$ receives these rows into its ghost buffers.
*   This is handled asynchronously using `MPI_Isend` and `MPI_Irecv`.

### 3. Load Balancing Algorithm
The system uses a **Diffusive Load Balancing** scheme:
1.  **Measurement**: Each rank calculates its *Load* ($L_i$), defined as the number of active "burning" cells.
2.  **Exchange**: Ranks exchange $L_i$ with neighbors ($i-1, i+1$).
3.  **Decision**:
    *   If $L_{self} > L_{neighbor} + Threshold$: Migrate a row to the neighbor.
    *   If $L_{neighbor} > L_{self} + Threshold$: Request a row from the neighbor.
4.  **Migration**: The actual row data is transferred, and the local grid size ($R_i$) is updated.

## 📦 Installation

### Prerequisites
*   **Python 3.12.6 (Most versions should work)**
*   **MPI Implementation**:
    *   **Windows**: [Microsoft MPI (MS-MPI)](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (Required for parallel execution).
*   **C++ Build Tools (Optional)**:
    *   Visual Studio 2019 (or later) with "Desktop development with C++" workload.
    *   MS-MPI SDK (usually included with MS-MPI installer).

### Setup
1.  Clone the repository.
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Compilation (C++)
To compile the high-performance C++ engine:
1.  Open a terminal (Command Prompt or PowerShell).
2.  Run the compilation script:
    ```cmd
    .\compile.bat
    ```
    This will create `simulation.exe` in the root directory.

## 💻 Usage

### Running the Simulation
Use `mpiexec` to launch the simulation on multiple processes.

#### Python (Prototype)
**Basic Command:**
```powershell
# Run on 4 processes
mpiexec -n 4 python main.py --rows 200 --cols 200 --steps 100 --balance --save
```

**Arguments:**
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--rows` | 100 | Total number of rows in the global grid. |
| `--cols` | 100 | Total number of columns in the global grid. |
| `--steps` | 100 | Number of simulation time steps. |
| `--balance` | `False` | **Flag**: Enable Dynamic Load Balancing. If omitted, uses Static decomposition. |
| `--save` | `False` | **Flag**: Save grid snapshots to `results/logs/` for visualization. |
| `--fire-pos` | `center` | Initial fire location: `center` (middle of grid) or `top` (Rank 0). |

#### C++ (High Performance)
**Basic Command:**
```powershell
mpiexec -n 4 simulation.exe --rows 1000 --cols 1000 --steps 200 --balance --fire-pos center
```

**Arguments:**
*   `--rows <N>`: Total rows (default: 1000)
*   `--cols <N>`: Total columns (default: 1000)
*   `--steps <N>`: Simulation steps (default: 200)
*   `--balance`: Enable Dynamic Load Balancing (flag)
*   `--heavy`: Enable artificial computational load (flag)
*   `--fire-pos <pos>`: `center`, `top`, or `corner`

### Visualization
After running with `--save`, generate images from the logs:

```bash
python scripts/visualize.py
```
*   **Input**: `.npy` files in `results/logs/`
*   **Output**: `.png` heatmaps in `results/plots/`

### Benchmarking
To evaluate performance and scaling:

```bash
python scripts/benchmark.py
```
This script runs the simulation with 1, 2, and 4 processes in both Static and Dynamic modes, generating a scaling plot at `results/scaling.png`.

## 📂 Project Structure

```text
Distributed-Cellular-Simulation/
├── docs/               # Documentation and papers
├── results/            # Simulation outputs
│   ├── logs/           # Raw .npy grid snapshots
│   └── plots/          # Generated visualizations
├── scripts/
│   ├── benchmark.py    # Performance testing script
│   └── visualize.py    # Image generation script
├── src/
│   ├── cpp/            # C++ implementation
│   │   └── simulation.cpp
│   ├── config.py       # Constants (FUEL, BURNING, etc.)
│   ├── grid.py         # Grid data structure management
│   ├── load_balancer.py# Dynamic load balancing logic
│   ├── main.py         # Entry point and simulation loop
│   ├── mpi_comm.py     # MPI communication wrapper
│   └── wildfire.py     # Cellular automata rules
├── tests/              # Unit tests
├── compile.bat         # C++ compilation script
├── simulation.exe      # Compiled C++ executable
├── main.py             # CLI Entry point
└── requirements.txt    # Python dependencies
```
