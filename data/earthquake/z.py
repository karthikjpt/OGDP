import os
import csv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

# Define the path to the earthquake data file
earthquake_data_file_path = '/home/ogdp/ogdp/data/earthquake/earthquake_energy_data.csv'

def create_csv_file_if_not_exists(file_path, fieldnames):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def create_ranges_csv(input_file_path, output_file_path, column_name, ranges):
    try:
        fieldnames = ['date'] + [f'{column_name}{r}' for r in ranges]

        with open(input_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            date_sums = {}

            for row in reader:
                value_str = row[column_name]
                energy_str = row['energy']

                try:
                    date = datetime.strptime(row['time'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
                except ValueError:
                    logging.warning(f"Skipping invalid date value: {row['time']}")
                    continue

                if date not in date_sums:
                    date_sums[date] = {f'{column_name}{r}': 0.0 for r in ranges}

                for r in ranges:
                    if float(value_str) <= r:
                        date_sums[date][f'{column_name}{r}'] += float(energy_str)
                        break

        with open(output_file_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for date, sums in date_sums.items():
                sums['date'] = date
                writer.writerow(sums)

        logging.info(f'Successfully created {column_name} sums CSV: {output_file_path}')
    except Exception as e:
        logging.error(f"Error creating {column_name} sums CSV: {str(e)}")

# Specify output directory
output_directory = '/home/ogdp/ogdp/data/earthquake'

# Create depth sums CSV
depth_sums_csv_path = os.path.join(output_directory, 'depth_energy.csv')
create_ranges_csv(earthquake_data_file_path, depth_sums_csv_path, 'depth', [-600, -400, -240, -180, -120, -60, -30, -20, -10, -5, 0])

# Create magnitude sums CSV
magnitude_sums_csv_path = os.path.join(output_directory, 'magnitude_energy.csv')
create_ranges_csv(earthquake_data_file_path, magnitude_sums_csv_path, 'magnitude', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create latitude sums CSV
latitude_sums_csv_path = os.path.join(output_directory, 'latitude_energy.csv')
create_ranges_csv(earthquake_data_file_path, latitude_sums_csv_path, 'latitude', [-80, -70, -60, -50, -40, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90])

# Create longitude sums CSV
longitude_sums_csv_path = os.path.join(output_directory, 'longitude_energy.csv')
create_ranges_csv(earthquake_data_file_path, longitude_sums_csv_path, 'longitude', [-170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])

