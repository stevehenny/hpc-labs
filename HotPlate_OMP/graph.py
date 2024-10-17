import matplotlib.pyplot as plt

# Updated data from the log (threads vs compute time in ms)
threads = [1, 2, 4, 8, 12, 16, 20]
compute_time = [2795.526, 1477.801, 759.490, 413.040, 394.203, 381.019, 373.999]

# Calculate speedup
T_serial = compute_time[0]  # Time for 1 thread
speedup = [T_serial / t for t in compute_time]

# Calculate efficiency
efficiency = [s / n for s, n in zip(speedup, threads)]

# Define tick positions (increments of 2)
tick_positions = list(range(0, 22, 2))  # From 0 to 20 in increments of 2

# Plot Compute Time vs Threads
plt.figure(figsize=(10, 5))

# Plot Compute Time vs Threads
plt.subplot(1, 2, 1)
plt.plot(threads, compute_time, marker='o', color='b')
plt.title('Hot Plate Simulation: OpenMP')
plt.xlabel('Number of Threads')
plt.ylabel('Compute Time (ms)')
plt.grid(True)
plt.xticks(tick_positions)

# Plot Speedup vs Threads
plt.subplot(1, 2, 2)
plt.plot(threads, speedup, marker='o', color='g')
plt.title('Hot Plate Simulation: OpenMP')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.grid(True)
plt.xticks(tick_positions)

plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("dynamic_8_scheduling.png")

# Calculate efficiency for 20 threads
efficiency_20_threads = efficiency[-1]
print("Efficiency for 20 threads:", efficiency_20_threads)