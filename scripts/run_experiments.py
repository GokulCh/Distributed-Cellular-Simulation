import subprocess
import time
import json
import os
import matplotlib.pyplot as plt

EXPERIMENTS = [
    {"name": "Small", "rows": 100, "cols": 100, "steps": 50, "procs": 2, "fire_pos": "center"},
    {"name": "Medium", "rows": 500, "cols": 500, "steps": 100, "procs": 4, "fire_pos": "center"},
    {"name": "Large", "rows": 1000, "cols": 1000, "steps": 200, "procs": 4, "fire_pos": "center"},
    {"name": "Uneven_Top", "rows": 1000, "cols": 1000, "steps": 200, "procs": 4, "fire_pos": "top"},
    {"name": "Heavy_Uneven", "rows": 1000, "cols": 1000, "steps": 200, "procs": 4, "fire_pos": "top", "heavy": True},
    {"name": "Super_Heavy_Optimized", "rows": 1000, "cols": 1000, "steps": 200, "procs": 4, "fire_pos": "top", "heavy": True, "balance_freq": 20}
]

RESULTS_FILE = "results/experiment_results.json"

def run_simulation(name, rows, cols, steps, procs, fire_pos, heavy=False, balance_freq=10, balance=True):
    print(f"Running {name} experiment ({rows}x{cols}, {steps} steps, {procs} procs, Pos={fire_pos}, Heavy={heavy}, Freq={balance_freq}, Balance={balance})...")
    cmd = [
        "mpiexec", "-n", str(procs), "python", "main.py",
        "--rows", str(rows), "--cols", str(cols),
        "--steps", str(steps), "--fire-pos", fire_pos,
        "--balance-freq", str(balance_freq)
    ]
    if heavy:
        cmd.append("--heavy")
    if balance:
        cmd.append("--balance")
    
    # Save logs for the largest run to visualize later
    if name == "Large":
        cmd.append("--save")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}: {e}")
        return None
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"  -> Finished in {duration:.4f}s")
    return duration

def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    results = []

    for exp in EXPERIMENTS:
        heavy = exp.get("heavy", False)
        freq = exp.get("balance_freq", 10)
        # Run Static
        time_static = run_simulation(exp["name"], exp["rows"], exp["cols"], exp["steps"], exp["procs"], exp["fire_pos"], heavy=heavy, balance_freq=freq, balance=False)
        
        # Run Dynamic
        time_dynamic = run_simulation(exp["name"], exp["rows"], exp["cols"], exp["steps"], exp["procs"], exp["fire_pos"], heavy=heavy, balance_freq=freq, balance=True)
        
        results.append({
            "name": exp["name"],
            "config": exp,
            "time_static": time_static,
            "time_dynamic": time_dynamic
        })

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {RESULTS_FILE}")
    
    # Plotting
    names = [r["name"] for r in results]
    t_static = [r["time_static"] for r in results]
    t_dynamic = [r["time_dynamic"] for r in results]
    
    x = range(len(names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], t_static, width, label='Static')
    plt.bar([i + width/2 for i in x], t_dynamic, width, label='Dynamic')
    
    plt.xlabel('Experiment Scale')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison: Static vs Dynamic Load Balancing')
    plt.xticks(x, names)
    plt.legend()
    plt.grid(axis='y')
    
    plt.savefig("results/experiment_plot.png")
    print("Plot saved to results/experiment_plot.png")

if __name__ == "__main__":
    main()
