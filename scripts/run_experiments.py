import subprocess
import time
import json
import os
import argparse
import matplotlib.pyplot as plt

RESULTS_FILE = "results/final_results.json"

def run_simulation(name, rows, cols, steps, procs, fire_pos, heavy=False, balance_freq=10, balance=True, cpp=False):
    print(f"Running {name} ({rows}x{cols}, {steps} steps, {procs} procs, Pos={fire_pos}, CPP={cpp})...")
    
    if cpp:
        cmd = [
            "mpiexec", "-n", str(procs), "simulation.exe",
            "--rows", str(rows), "--cols", str(cols),
            "--steps", str(steps), "--fire-pos", fire_pos
        ]
        if balance:
            cmd.append("--balance")
        if heavy:
            cmd.append("--heavy")
    else:
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

def generate_suite(suite_type):
    experiments = []
    
    sizes = [
        {"label": "Small", "rows": 200, "cols": 200, "steps": 100},
        {"label": "Medium", "rows": 500, "cols": 500, "steps": 150},
        {"label": "Large", "rows": 1000, "cols": 1000, "steps": 200}
    ]
    
    procs_list = [2, 4, 8]
    positions = ["center", "top", "corner"]
    
    for size in sizes:
        for procs in procs_list:
            for pos in positions:
                # Construct name
                base_name = f"{suite_type}_{size['label']}_P{procs}_{pos}"
                
                exp = {
                    "name": base_name,
                    "rows": size["rows"],
                    "cols": size["cols"],
                    "steps": size["steps"],
                    "procs": procs,
                    "fire_pos": pos,
                    "heavy": True, # Always heavy for meaningful results
                    "cpp": (suite_type == "Cpp")
                }
                experiments.append(exp)
    return experiments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["python", "cpp", "all", "test"], default="test")
    args = parser.parse_args()

    if not os.path.exists("results"):
        os.makedirs("results")

    experiments = []
    if args.suite == "test":
        experiments = [
            {"name": "Cpp_Large_P4_top_Fixed", "rows": 1000, "cols": 1000, "steps": 200, "procs": 4, "fire_pos": "top", "heavy": True, "cpp": True}
        ]
    elif args.suite == "cpp":
        experiments = generate_suite("Cpp")
    elif args.suite == "python":
        experiments = generate_suite("Py")
    elif args.suite == "all":
        experiments = generate_suite("Cpp") + generate_suite("Py")

    results = []
    
    for exp in experiments:
        # Run Static
        time_static = run_simulation(exp["name"] + "_Static", exp["rows"], exp["cols"], exp["steps"], exp["procs"], exp["fire_pos"], heavy=exp["heavy"], balance=False, cpp=exp["cpp"])
        
        # Run Dynamic
        time_dynamic = run_simulation(exp["name"] + "_Dynamic", exp["rows"], exp["cols"], exp["steps"], exp["procs"], exp["fire_pos"], heavy=exp["heavy"], balance=True, cpp=exp["cpp"])
        
        results.append({
            "name": exp["name"],
            "config": exp,
            "time_static": time_static,
            "time_dynamic": time_dynamic
        })

    # Save results
    final_data = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                final_data = json.load(f)
        except:
            pass
            
    # Update or append
    for new_r in results:
        found = False
        for i, old_r in enumerate(final_data):
            if old_r["name"] == new_r["name"]:
                final_data[i] = new_r
                found = True
                break
        if not found:
            final_data.append(new_r)
            
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_data, f, indent=4)
    
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
