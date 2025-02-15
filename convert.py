import json
import csv

json_filename = "loan_approval_dataset.json"
csv_filename = "loan_approval_dataset.csv"

with open(json_filename, "r") as json_file:
    data = json.load(json_file)

columns = list(data.keys())

row_indices = sorted(next(iter(data.values())).keys(), key=lambda x: int(x))

with open(csv_filename, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(columns)
    
    for idx in row_indices:
        row = [data[col][idx] for col in columns]
        writer.writerow(row)

print(f"Conversion complete! CSV saved as '{csv_filename}'.")
