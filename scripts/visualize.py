import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

def plot_heatmap(npy_file, output_file):
    """Plot heatmap for given .npy file."""

    # Load data from .npy file
    data = np.load(npy_file)

    # Create figure
    plt.figure(figsize=(10, 10))

    # Create colormap    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['green', 'red', 'black'])
    
    # Plot heatmap
    plt.imshow(data, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    plt.title(f"Simulation State: {os.path.basename(npy_file)}")
    plt.colorbar(ticks=[0, 1, 2], label='State (0=Fuel, 1=Burning, 2=Burnt)')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {output_file}")

def main():
    """Main function to run visualizations."""

    # Find all .npy files    
    files = sorted(glob.glob("results/logs/step_*.npy"))
    
    # Check if any .npy files were found
    if not files:
        print("No .npy files found in results/logs/. Run main.py with --save.")
        return

    print(f"Found {len(files)} snapshots. Generating images...")

    # Generate images    
    for f in files:
        step_match = re.search(r"step_(\d+)", f)
        step = step_match.group(1) if step_match else "unknown"
        output = f"results/plots/heatmap_step_{step}.png"
        plot_heatmap(f, output)
        
    print("Done.")

if __name__ == "__main__":
    main()
