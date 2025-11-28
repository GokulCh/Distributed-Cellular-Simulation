import subprocess
import time
import matplotlib.pyplot as plt
import os

def run_benchmark(procs, steps=100, size=1000, balance=False):
    """Run benchmark for given number of processes."""

    cmd = [r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe", "-n", str(procs), "python", "main.py", 
           "--rows", str(size), "--cols", str(size), 
           "--steps", str(steps), "--fire-pos", "top"]
    if balance:
        cmd.append("--balance")
        
    start = time.time()
    # Capture output to avoid clutter
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to run with {procs} procs: {e}")
        return None
            
    end = time.time()
    return end - start

def main():
    """Main function to run benchmarks."""

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # List of processes to test
    procs_list = [1, 2, 4] 
    times_static = []
    times_dynamic = []
    
    print("Running Benchmarks...")
    
    # Run benchmarks for each process count
    for p in procs_list:
        print(f"Testing with {p} processes (Static)...")
        t = run_benchmark(p, balance=False)
        times_static.append(t)
        
        print(f"Testing with {p} processes (Dynamic)...")
        t = run_benchmark(p, balance=True)
        times_dynamic.append(t)
        
    # Plotting
    plt.figure()
    plt.plot(procs_list, times_static, 'o-', label='Static')
    plt.plot(procs_list, times_dynamic, 's-', label='Dynamic')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (s)')
    plt.title('Scaling Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/scaling.png')
    print("Benchmark saved to results/scaling.png")

if __name__ == "__main__":
    main()
