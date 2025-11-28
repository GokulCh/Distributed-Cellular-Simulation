import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

def plot_heatmap(npy_file, output_file):

    data = np.load(npy_file)

    plt.figure(figsize=(10, 10))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['green', 'red', 'black'])
    
    plt.imshow(data, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    plt.title(f"Simulation State: {os.path.basename(npy_file)}")
    plt.colorbar(ticks=[0, 1, 2], label='State (0=Fuel, 1=Burning, 2=Burnt)')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {output_file}")

def main():

    files = sorted(glob.glob("results/logs/step_*.npy"))
    if not files:
        print("No .npy files found in results/logs/. Run main.py with --save.")
        return

    print(f"Found {len(files)} snapshots. Generating images...")
    
    for f in files:
        step_match = re.search(r"step_(\d+)", f)
        step = step_match.group(1) if step_match else "unknown"
        output = f"results/plots/heatmap_step_{step}.png"
        plot_heatmap(f, output)
        
    print("Done.")

if __name__ == "__main__":
    main()
