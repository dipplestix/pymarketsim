import csv

def write_to_csv(filename, content):
    # Write to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Writing the numbers as a single row
        writer.writerow(content)


