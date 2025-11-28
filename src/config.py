# Simulation Constants
FUEL = 0
BURNING = 1
BURNT = 2

# Probabilities
P_IGNITE = 0.01  # Probability of spontaneous ignition
P_SPREAD = 0.5  # Probability of fire spreading to neighbor

# Visualization Colors
COLOR_MAP = {
    FUEL: [0, 1, 0],  # Green
    BURNING: [1, 0, 0],  # Red
    BURNT: [0, 0, 0],  # Black
}
