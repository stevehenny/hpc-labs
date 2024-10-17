import matplotlib.pyplot as plt

# Updated data from the log (threads vs compute time in ms)
threads = [1, 2, 4, 8, 12, 16, 20]
compute_time = [7089.837, 3025.496, 1577.860, 912.493, 924.124, 1129.802, 1246.717]

# Calculate speedup
T_serial = compute_time[0]  # Time for 1 thread
speedup = [T_serial / t for t in compute_time]

# Calculate efficiency
efficiency = [s / n for s, n in zip(speedup, threads)]

# Plot Compute Time vs Threads
plt.figure(figsize=(10, 5))

# Plot Compute Time vs Threads
plt.subplot(1, 2, 1)
plt.plot(threads, compute_time, marker='o', color='b')
plt.title('Compute Time vs Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Compute Time (ms)')
plt.grid(True)

# Plot Speedup vs Threads
plt.subplot(1, 2, 2)
plt.plot(threads, speedup, marker='o', color='g')
plt.title('Speedup vs Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.grid(True)

plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("threads_vs_compute_time.png")

# Calculate efficiency for 20 threads
efficiency_20_threads = efficiency[-1]
print("Efficiency for 20 threads:", efficiency_20_threads)
