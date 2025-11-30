import json
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_FILE = "results/final_results.json"
OUTPUT_FILE = "results/comprehensive_plots.png"

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return []
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

def plot_all(results):
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Distributed Cellular Simulation: Comprehensive Performance Analysis', fontsize=16)

    # --- 1. C++ Scalability (Center Start) ---
    ax = axs[0, 0]
    sizes = ["Small", "Medium", "Large"]
    colors = ['blue', 'green', 'red']
    
    for i, size in enumerate(sizes):
        procs = []
        times = []
        for r in results:
            cfg = r["config"]
            if cfg["cpp"] and cfg["fire_pos"] == "center" and size in cfg["name"]:
                procs.append(cfg["procs"])
                times.append(r["time_static"]) # Use Static as baseline
        
        # Sort by procs
        if procs:
            p_t = sorted(zip(procs, times))
            procs, times = zip(*p_t)
            ax.plot(procs, times, marker='o', label=f"{size} Grid", color=colors[i])

    ax.set_title('C++ Scalability (Static, Center Start)')
    ax.set_xlabel('Processors')
    ax.set_ylabel('Runtime (s)')
    ax.set_xticks([2, 4, 8])
    ax.legend()
    ax.grid(True)

    # --- 2. Python Scalability (Center Start) ---
    ax = axs[0, 1]
    for i, size in enumerate(sizes):
        procs = []
        times = []
        for r in results:
            cfg = r["config"]
            if not cfg["cpp"] and cfg["fire_pos"] == "center" and size in cfg["name"]:
                procs.append(cfg["procs"])
                times.append(r["time_static"])
        
        if procs:
            p_t = sorted(zip(procs, times))
            procs, times = zip(*p_t)
            ax.plot(procs, times, marker='o', linestyle='--', label=f"{size} Grid", color=colors[i])

    ax.set_title('Python Scalability (Static, Center Start)')
    ax.set_xlabel('Processors')
    ax.set_ylabel('Runtime (s)')
    ax.set_xticks([2, 4, 8])
    ax.legend()
    ax.grid(True)

    # --- 3. Dynamic vs Static Impact (C++ Large Grid, 4 Procs) ---
    ax = axs[1, 0]
    positions = ["center", "top", "corner"]
    static_times = []
    dynamic_times = []
    
    for pos in positions:
        found = False
        for r in results:
            cfg = r["config"]
            if cfg["cpp"] and cfg["name"] == f"Cpp_Large_P4_{pos}":
                static_times.append(r["time_static"])
                dynamic_times.append(r["time_dynamic"])
                found = True
                break
        if not found:
            static_times.append(0)
            dynamic_times.append(0)

    x = np.arange(len(positions))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, static_times, width, label='Static', color='gray')
    rects2 = ax.bar(x + width/2, dynamic_times, width, label='Dynamic', color='orange')

    ax.set_title('Load Balancing Impact (C++ Large Grid, 4 Procs)')
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in positions])
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    
    # Add speedup labels
    for i, (s, d) in enumerate(zip(static_times, dynamic_times)):
        if s > 0:
            speedup = (s - d) / s * 100
            color = 'green' if speedup > 0 else 'red'
            ax.text(i, max(s, d) + 0.1, f"{speedup:+.1f}%", ha='center', color=color, fontweight='bold')

    # --- 4. Python vs C++ Comparison (Large Grid, 4 Procs, Center) ---
    ax = axs[1, 1]
    
    py_time = 0
    cpp_time = 0
    
    for r in results:
        if r["name"] == "Py_Large_P4_center_Static": # Check naming convention from script
             py_time = r["time_static"]
        if r["name"] == "Cpp_Large_P4_center": # Check naming convention
             cpp_time = r["time_static"]
             
    # Fallback search if exact name match fails (due to script logic)
    if py_time == 0:
        for r in results:
            if r["config"]["name"] == "Py_Large_P4_center": py_time = r["time_static"]
    if cpp_time == 0:
        for r in results:
            if r["config"]["name"] == "Cpp_Large_P4_center": cpp_time = r["time_static"]

    langs = ['Python', 'C++']
    times = [py_time, cpp_time]
    colors = ['#3776ab', '#00599c'] # Python Blue, C++ Blue
    
    bars = ax.bar(langs, times, color=colors)
    ax.set_title('Language Comparison (Large Grid, 4 Procs)')
    ax.set_ylabel('Runtime (s)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
                
    if cpp_time > 0:
        speedup = py_time / cpp_time
        ax.text(0.5, py_time/2, f"{speedup:.1f}x Faster", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_FILE)
    print(f"Comprehensive plots saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    data = load_results()
    if data:
        plot_all(data)
