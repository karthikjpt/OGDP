import os
import csv
from datetime import datetime

# Define the path to the earthquake data file
earthquake_data_file_path = '/home/ogdp/ogdp/data/earthquake/earthquake_data.csv'

def create_csv_file_if_not_exists(file_path, fieldnames):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def create_depth_ranges_csv(input_file_path, output_file_path):
    try:
        depth_ranges = [-600, -400, -240, -180, -120, -60, -30, -20, -10, -5, 0]
        fieldnames = ['date'] + [f'{r}' for r in depth_ranges]

        with open(input_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Initialize dictionary to store earthquake counts for each date
            date_counts = {}

            for row in reader:
                date_str = row['time']
                depth_str = row['depth']

                # Convert all positive depth values to negative
                try:
                    depth = float(depth_str)
                    if depth > 0:
                        depth = -depth
                except ValueError:
                    print(f"Skipping invalid depth value: {depth_str}")
                    continue

                # Parse the 'time' column into a date
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
                except ValueError:
                    print(f"Skipping invalid date value: {date_str}")
                    continue

                # Initialize counters for each depth range for the current date
                if date not in date_counts:
                    date_counts[date] = {f'{r}': 0 for r in depth_ranges}

                # Find the appropriate depth range and increment the counter
                for r in depth_ranges:
                    if depth <= r:
                        date_counts[date][f'{r}'] += 1
                        break

            # Write data to the output CSV file
            with open(output_file_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for date, counts in date_counts.items():
                    counts['date'] = date
                    writer.writerow(counts)

            print(f'Successfully created depth ranges CSV: {output_file_path}')
    except Exception as e:
        print(f"Error creating depth ranges CSV: {str(e)}")

def create_magnitude_ranges_csv(input_file_path, output_file_path):
    try:
        magnitude_ranges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        fieldnames = ['date'] + [f'M{r}' for r in magnitude_ranges]

        with open(input_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Initialize dictionary to store earthquake counts for each date
            date_counts = {}

            for row in reader:
                magnitude_str = row['magnitude']

                # Parse the 'time' column into a date
                try:
                    date = datetime.strptime(row['time'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
                except ValueError:
                    print(f"Skipping invalid date value: {row['time']}")
                    continue

                # Initialize counters for each magnitude range for the current date
                if date not in date_counts:
                    date_counts[date] = {f'M{r}': 0 for r in magnitude_ranges}

                # Find the appropriate magnitude range and increment the counter
                for r in magnitude_ranges:
                    if float(magnitude_str) <= r:
                        date_counts[date][f'M{r}'] += 1
                        break

            # Write data to the output CSV file
            with open(output_file_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for date, counts in date_counts.items():
                    counts['date'] = date
                    writer.writerow(counts)

            print(f'Successfully created magnitude ranges CSV: {output_file_path}')
    except Exception as e:
        print(f"Error creating magnitude ranges CSV: {str(e)}")


def create_latitude_ranges_csv(input_file_path, output_file_path):
    try:
        latitude_ranges = [-80, -70, -60, -50, -40, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
        fieldnames = ['date'] + [f'{r}' for r in latitude_ranges]

        with open(input_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Initialize dictionary to store earthquake counts for each date
            date_counts = {}

            for row in reader:
                date_str = row['time']
                latitude_str = row['latitude']

                # Parse the 'time' column into a date
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
                except ValueError:
                    print(f"Skipping invalid date value: {date_str}")
                    continue

                # Initialize counters for each latitude range for the current date
                if date not in date_counts:
                    date_counts[date] = {f'{r}': 0 for r in latitude_ranges}

                # Find the appropriate latitude range and increment the counter
                for r in latitude_ranges:
                    if float(latitude_str) <= r:
                        date_counts[date][f'{r}'] += 1
                        break

            # Write data to the output CSV file
            with open(output_file_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for date, counts in date_counts.items():
                    counts['date'] = date
                    writer.writerow(counts)

            print(f'Successfully created latitude ranges CSV: {output_file_path}')
    except Exception as e:
        print(f"Error creating latitude ranges CSV: {str(e)}")

def create_longitude_ranges_csv(input_file_path, output_file_path):
    try:
        longitude_ranges = [-170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        fieldnames = ['date'] + [f'{r}' for r in longitude_ranges]

        with open(input_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Initialize dictionary to store earthquake counts for each date
            date_counts = {}

            for row in reader:
                date_str = row['time']
                longitude_str = row['longitude']

                # Parse the 'time' column into a date
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
                except ValueError:
                    print(f"Skipping invalid date value: {date_str}")
                    continue

                # Initialize counters for each longitude range for the current date
                if date not in date_counts:
                    date_counts[date] = {f'{r}': 0 for r in longitude_ranges}

                # Find the appropriate longitude range and increment the counter
                for r in longitude_ranges:
                    if float(longitude_str) <= r:
                        date_counts[date][f'{r}'] += 1
                        break

            # Write data to the output CSV file
            with open(output_file_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for date, counts in date_counts.items():
                    counts['date'] = date
                    writer.writerow(counts)

            print(f'Successfully created longitude ranges CSV: {output_file_path}')
    except Exception as e:
        print(f"Error creating longitude ranges CSV: {str(e)}")


# Create depth ranges CSV
depth_ranges_csv_path = os.path.join('/home/ogdp/ogdp/data/earthquake/', 'depth_ranges.csv')
create_depth_ranges_csv(earthquake_data_file_path, depth_ranges_csv_path)
print(f'Successfully created depth ranges CSV: {depth_ranges_csv_path}')

# Create magnitude ranges CSV
magnitude_ranges_csv_path = os.path.join('/home/ogdp/ogdp/data/earthquake/', 'magnitude_ranges.csv')
create_magnitude_ranges_csv(earthquake_data_file_path, magnitude_ranges_csv_path)
print(f'Successfully created magnitude ranges CSV: {magnitude_ranges_csv_path}')

# Create latitude ranges CSV
latitude_ranges_csv_path = os.path.join('/home/ogdp/ogdp/data/earthquake/', 'latitude_ranges.csv')
create_latitude_ranges_csv(earthquake_data_file_path, latitude_ranges_csv_path)
print(f'Successfully created latitude ranges CSV: {latitude_ranges_csv_path}')

# Create longitude ranges CSV
longitude_ranges_csv_path = os.path.join('/home/ogdp/ogdp/data/earthquake/', 'longitude_ranges.csv')
create_longitude_ranges_csv(earthquake_data_file_path, longitude_ranges_csv_path)
print(f'Successfully created longitude ranges CSV: {longitude_ranges_csv_path}')

