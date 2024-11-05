import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        proc_count = None
        for line in lines:
            line = line.strip()
            if line.startswith("procs:"):
                proc_count = int(line.split(":")[1].split()[0])
                data[proc_count] = {'vsize': [], 'time': []}
            elif line and not line.startswith("vsize"):
                vsize, time_usec = line.split()
                data[proc_count]['vsize'].append(int(vsize))
                data[proc_count]['time'].append(float(time_usec))
    return data

def plot_reduction_times(data, output_filename_linear="reduction_times_linear.png", output_filename_log="reduction_times_log.png"):
    process_counts = [2, 3, 6, 13, 32]

    # Linear y-scale plot
    plt.figure(figsize=(10, 6))
    for procs in process_counts:
        if procs in data:
            vsize = data[procs]['vsize']
            time = data[procs]['time']
            plt.plot(vsize, time, marker='o', label=f"{procs} processes")

    plt.xscale('log', base=2)
    plt.xlabel("Vector Size")
    plt.ylabel("Reduction Time (µs)")
    plt.title("Reduction Time vs Vector Size for Different Process Counts (Linear Y-Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_filename_linear)
    plt.close()

    # Logarithmic y-scale plot
    plt.figure(figsize=(10, 6))
    for procs in process_counts:
        if procs in data:
            vsize = data[procs]['vsize']
            time = data[procs]['time']
            plt.plot(vsize, time, marker='o', label=f"{procs} processes")

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel("Vector Size")
    plt.ylabel("Reduction Time (µs)")
    plt.title("Reduction Time vs Vector Size for Different Process Counts (Logarithmic Y-Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_filename_log)
    plt.close()

# Load data and plot
filename = "run_log.txt"  # Replace with your actual log file name
data = parse_log_file(filename)
plot_reduction_times(data, output_filename_linear="reduction_times_linear.png", output_filename_log="reduction_times_log.png")
