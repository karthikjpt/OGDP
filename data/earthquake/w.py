import os
import csv

def calculate_energy(magnitude):
    return 10**(1.5 * float(magnitude) + 4.4)

def create_energy_data_csv(input_file_path, output_file_path):
    try:
        # Read all columns from the input file
        with open(input_file_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames + ['energy']

            # Write data to the output CSV file
            with open(output_file_path, mode='w', newline='') as csvfile_out:
                writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    magnitude_str = row['magnitude']
                    energy = calculate_energy(magnitude_str)

                    # Add the energy to the row data
                    row['energy'] = energy
                    writer.writerow(row)

        print(f'Successfully created earthquake energy data CSV: {output_file_path}')

    except Exception as e:
        print(f"Error creating earthquake energy data CSV: {str(e)}")

# Create earthquake energy data CSV
earthquake_data_file_path = '/home/ogdp/ogdp/data/earthquake/earthquake_data.csv'
earthquake_energy_data_file_path = os.path.join('/home/ogdp/ogdp/data/earthquake/', 'earthquake_energy_data.csv')
create_energy_data_csv(earthquake_data_file_path, earthquake_energy_data_file_path)
print(f'Successfully created earthquake energy data CSV: {earthquake_energy_data_file_path}')

