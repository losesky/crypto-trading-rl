import csv

input_csv_file = '/Users/melihkarakose/Desktop/EC 581/btc_hourly_data_2022_2025.csv'
output_csv_file = '/Users/melihkarakose/Desktop/EC 581/btc_hourly_data_filtered_from_existing.csv'
columns_to_keep = ['timestamp', 'close', 'volume']

print(f"Filtering data from {input_csv_file} to {output_csv_file}")

try:
    with open(input_csv_file, 'r', newline='') as infile, \
         open(output_csv_file, 'w', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write the header with only the selected columns
        writer.writerow(columns_to_keep)
        
        # Write the filtered data rows
        for row in reader:
            filtered_row = [row[col] for col in columns_to_keep]
            writer.writerow(filtered_row)
            
    print("Filtering complete.")

except FileNotFoundError:
    print(f"Error: The file {input_csv_file} was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
