import csv
import numpy as np

def write_to_csv(filename, content):
    # Write to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing the numbers as a single row
        writer.writerow(content)


def replace_inf_with_nearest_2d(arr):
    arr = np.array(arr)
    for row in arr:
        if np.isinf(row).any():
            finite_indices = np.where(np.isfinite(row))[0]
            finite_values = row[finite_indices]

            for i in np.where(np.isinf(row))[0]:
                nearest_idx = np.argmin(np.abs(finite_indices - i))
                row[i] = finite_values[nearest_idx]

    return arr