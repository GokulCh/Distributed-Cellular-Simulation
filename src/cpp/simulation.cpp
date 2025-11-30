#include <mpi.h>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <string>

const int FUEL = 0;
const int BURNING = 1;
const int BURNT = 2;
const double P_IGNITE = 0.01;
const double P_SPREAD = 0.5;

const int TAG_BAL = 11;
const int TAG_CMD = 12;
const int LB_THRESHOLD = 5;

struct Grid {
    int rows;
    int cols;
    std::vector<int> data;

    Grid(int r, int c) : rows(r), cols(c), data(r * c, FUEL) {}

    int& at(int r, int c) {
        return data[r * cols + c];
    }
    
    const int& at(int r, int c) const {
        return data[r * cols + c];
    }
};

void set_fire(Grid& grid, int r, int c) {
    if (r >= 0 && r < grid.rows && c >= 0 && c < grid.cols) {
        grid.at(r, c) = BURNING;
    }
}

void update_grid(Grid& grid, Grid& next_grid, int rank, int size, bool heavy_load) {
    int burning_count = 0;
    
    for (int r = 0; r < grid.rows; ++r) {
        for (int c = 0; c < grid.cols; ++c) {
            int state = grid.at(r, c);
            if (state == BURNING) {
                next_grid.at(r, c) = BURNT;
                burning_count++;
                
                if (c + 1 < grid.cols && grid.at(r, c+1) == FUEL) next_grid.at(r, c+1) = BURNING;
                if (c - 1 >= 0 && grid.at(r, c-1) == FUEL) next_grid.at(r, c-1) = BURNING;
                if (r + 1 < grid.rows && grid.at(r+1, c) == FUEL) next_grid.at(r+1, c) = BURNING;
                if (r - 1 >= 0 && grid.at(r-1, c) == FUEL) next_grid.at(r-1, c) = BURNING;
            } else if (state == BURNT) {
                next_grid.at(r, c) = BURNT;
            } else {
                if (next_grid.at(r, c) != BURNING) next_grid.at(r, c) = FUEL;
            }
        }
    }
    
    if (heavy_load && burning_count > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        auto target = start + std::chrono::microseconds(static_cast<long long>(burning_count * 50));
        while (std::chrono::high_resolution_clock::now() < target);
    }
    
    grid.data = next_grid.data;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 1000;
    int cols = 1000;
    int steps = 200;
    bool balance = false;
    bool heavy = false;
    std::string fire_pos = "center";

    // Parse args manually for simplicity
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rows") rows = std::stoi(argv[++i]);
        else if (arg == "--cols") cols = std::stoi(argv[++i]);
        else if (arg == "--steps") steps = std::stoi(argv[++i]);
        else if (arg == "--balance") balance = true;
        else if (arg == "--heavy") heavy = true;
        else if (arg == "--fire-pos") fire_pos = argv[++i];
    }

    // Domain Decomposition
    int rows_per_rank = rows / size;
    int remainder = rows % size;
    int local_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    int offset = rank * rows_per_rank + std::min(rank, remainder);

    Grid grid(local_rows, cols);
    Grid next_grid(local_rows, cols);

    // Init Fire
    if (fire_pos == "top") {
        if (rank == 0) set_fire(grid, 0, cols / 2);
    } else if (fire_pos == "center") {
         int global_center = rows / 2;
         if (offset <= global_center && global_center < offset + local_rows) {
             set_fire(grid, global_center - offset, cols / 2);
         }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step < steps; ++step) {
        update_grid(grid, next_grid, rank, size, heavy);
        
        // Aggressive: Balance every 5 steps
        if (balance && step % 5 == 0) {
            // Check load
            int local_load = 0;
            for(int x : grid.data) if(x == BURNING) local_load++;
            
            // Aggressive migration
            int rows_to_move = 5;
            int items_to_move = rows_to_move * grid.cols;
            
            // Phase 1: Even ranks talk to Down (i -> i+1)
            if (rank % 2 == 0 && rank + 1 < size) {
                int neighbor_load;
                MPI_Sendrecv(&local_load, 1, MPI_INT, rank + 1, TAG_BAL,
                             &neighbor_load, 1, MPI_INT, rank + 1, TAG_BAL,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (local_load > neighbor_load + LB_THRESHOLD && grid.rows > rows_to_move + 1) {
                    // Send bottom rows to neighbor
                    std::vector<int> row_to_send(grid.data.end() - items_to_move, grid.data.end());
                    
                    // Send rows
                    MPI_Send(row_to_send.data(), items_to_move, MPI_INT, rank + 1, TAG_CMD, MPI_COMM_WORLD);
                    
                    // Remove rows
                    grid.data.resize(grid.data.size() - items_to_move);
                    grid.rows -= rows_to_move;
                    next_grid.data.resize(grid.data.size()); 
                    next_grid.rows -= rows_to_move;
                } else if (neighbor_load > local_load + LB_THRESHOLD) {
                     // Receive rows from neighbor
                     std::vector<int> recv_row(items_to_move);
                     MPI_Recv(recv_row.data(), items_to_move, MPI_INT, rank + 1, TAG_CMD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     
                     // Append to bottom
                     grid.data.insert(grid.data.end(), recv_row.begin(), recv_row.end());
                     grid.rows += rows_to_move;
                     next_grid.data.resize(grid.data.size());
                     next_grid.rows += rows_to_move;
                }
            } else if (rank % 2 == 1) {
                // Odd rank (i) is the "Down" neighbor of Even rank (i-1)
                int neighbor_load;
                MPI_Sendrecv(&local_load, 1, MPI_INT, rank - 1, TAG_BAL,
                             &neighbor_load, 1, MPI_INT, rank - 1, TAG_BAL,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                             
                if (neighbor_load > local_load + LB_THRESHOLD) {
                    // Receive rows from Up (i-1)
                    std::vector<int> recv_row(items_to_move);
                    MPI_Recv(recv_row.data(), items_to_move, MPI_INT, rank - 1, TAG_CMD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    grid.data.insert(grid.data.begin(), recv_row.begin(), recv_row.end());
                    grid.rows += rows_to_move;
                    next_grid.data.resize(grid.data.size());
                    next_grid.rows += rows_to_move;
                } else if (local_load > neighbor_load + LB_THRESHOLD && grid.rows > rows_to_move + 1) {
                    // Send top rows to Up (i-1)
                    std::vector<int> row_to_send(grid.data.begin(), grid.data.begin() + items_to_move);
                    
                    MPI_Send(row_to_send.data(), items_to_move, MPI_INT, rank - 1, TAG_CMD, MPI_COMM_WORLD);
                    
                    grid.data.erase(grid.data.begin(), grid.data.begin() + items_to_move);
                    grid.rows -= rows_to_move;
                    next_grid.data.resize(grid.data.size());
                    next_grid.rows -= rows_to_move;
                }
            }
            
            MPI_Barrier(MPI_COMM_WORLD); 
            
             if (rank % 2 == 1 && rank + 1 < size) {
                int neighbor_load;
                MPI_Sendrecv(&local_load, 1, MPI_INT, rank + 1, TAG_BAL,
                             &neighbor_load, 1, MPI_INT, rank + 1, TAG_BAL,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (local_load > neighbor_load + LB_THRESHOLD && grid.rows > rows_to_move + 1) {
                    std::vector<int> row_to_send(grid.data.end() - items_to_move, grid.data.end());
                    MPI_Send(row_to_send.data(), items_to_move, MPI_INT, rank + 1, TAG_CMD, MPI_COMM_WORLD);
                    grid.data.resize(grid.data.size() - items_to_move);
                    grid.rows -= rows_to_move;
                    next_grid.data.resize(grid.data.size());
                    next_grid.rows -= rows_to_move;
                } else if (neighbor_load > local_load + LB_THRESHOLD) {
                     std::vector<int> recv_row(items_to_move);
                     MPI_Recv(recv_row.data(), items_to_move, MPI_INT, rank + 1, TAG_CMD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     grid.data.insert(grid.data.end(), recv_row.begin(), recv_row.end());
                     grid.rows += rows_to_move;
                     next_grid.data.resize(grid.data.size());
                     next_grid.rows += rows_to_move;
                }
            } else if (rank % 2 == 0 && rank > 0) {
                int neighbor_load;
                MPI_Sendrecv(&local_load, 1, MPI_INT, rank - 1, TAG_BAL,
                             &neighbor_load, 1, MPI_INT, rank - 1, TAG_BAL,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                             
                if (neighbor_load > local_load + LB_THRESHOLD) {
                    std::vector<int> recv_row(items_to_move);
                    MPI_Recv(recv_row.data(), items_to_move, MPI_INT, rank - 1, TAG_CMD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    grid.data.insert(grid.data.begin(), recv_row.begin(), recv_row.end());
                    grid.rows += rows_to_move;
                    next_grid.data.resize(grid.data.size());
                    next_grid.rows += rows_to_move;
                } else if (local_load > neighbor_load + LB_THRESHOLD && grid.rows > rows_to_move + 1) {
                    std::vector<int> row_to_send(grid.data.begin(), grid.data.begin() + items_to_move);
                    MPI_Send(row_to_send.data(), items_to_move, MPI_INT, rank - 1, TAG_CMD, MPI_COMM_WORLD);
                    grid.data.erase(grid.data.begin(), grid.data.begin() + items_to_move);
                    grid.rows -= rows_to_move;
                    next_grid.data.resize(grid.data.size());
                    next_grid.rows -= rows_to_move;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "C++ Simulation (" << (balance ? "Dynamic" : "Static") << ") finished in " 
                  << (end_time - start_time) << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
