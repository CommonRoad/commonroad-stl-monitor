import numpy as np

# Original data points
x_data = np.array([0, 2, 4, 6, 8, 10, 12, 14])
y_data = np.array([0.5, 0.7, 0.0, -0.6, -0.7, 0.3, 0.5, 0.6])

# Fit a 6th-degree polynomial (which exactly interpolates 7 points)
p = np.polyfit(x_data, y_data, 6)

# Create x values from 0 to 12 in steps of 0.1
x_interp = np.arange(0, 14.01, 0.2)
y_interp = np.polyval(p, x_interp)

# Save the results to a file (with two columns: x and y)
with open("smooth_data_trace.dat", "w") as f:
    for x, y in zip(x_interp, y_interp):
        f.write(f"{x:.1f} {y:.4f} \\\\ \n")


y_min = [min(y_interp[max(0, k-10): k+1]) for k in range(len(y_interp))]
with open("smooth_data_min.dat", "a") as f:
    for x, y in zip(x_interp, y_min):
        f.write(f"{x:.1f} {y:.4f} \\\\ \n")

y_d = [min(y_interp[max(0, k-10): k+1]) if all([y >=0 for y in y_interp[max(0, k-10): k+1]]) else -sum([y < 0 for y in y_interp[max(0, k-10): k+1]]) / 11 for k in range(len(y_interp))]
with open("smooth_data_d.dat", "a") as f:
    for x, y in zip(x_interp, y_d):
        f.write(f"{x:.1f} {y:.4f} \\\\ \n")

y_ds = [min(y_interp[max(0, k-10): k+1]) if all([y >=0 for y in y_interp[max(0, k-10): k+1]]) else sum([y if y < 0 else 0 for y in y_interp[max(0, k-10): k+1]]) / 11 for k in range(len(y_interp))]
with open("smooth_data_ds.dat", "a") as f:
    for x, y in zip(x_interp, y_ds):
        f.write(f"{x:.1f} {y:.4f} \\\\ \n")