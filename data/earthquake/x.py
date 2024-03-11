import os
import requests
import csv
from datetime import datetime, timedelta
import time

def calculate_date_range(days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59")
    return start_date_str, end_date_str

def retry_request(func, max_attempts=3, delay_seconds=10, *args, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error during attempt {attempt}: {str(e)}")
            if attempt < max_attempts:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("Max attempts reached. Giving up.")
                raise

def earthquake_data_speu(mindepth=None, minmagnitude=None):
    start_date_str, end_date_str = calculate_date_range(30)

    params = {
        'format': 'json',
        'starttime': start_date_str,
        'endtime': end_date_str,
        'mindepth': mindepth,
        'minmagnitude': minmagnitude
    }

    url = f"https://www.seismicportal.eu/fdsnws/event/1/query"

    response = retry_request(requests.get, url=url, params=params)
    response.raise_for_status()

    if 'InvalidChunkLength' in response.text:
        raise requests.exceptions.RequestException('InvalidChunkLength error')

    return response.json()

def save_earthquake_data_to_csv(file_path, earthquake_data):
    fieldnames = ['id', 'source_id', 'source_catalog', 'lastupdate', 'time', 'flynn_region',
                  'latitude', 'longitude', 'depth', 'evtype', 'auth', 'magnitude', 'magtype', 'unid']

    try:
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for feature in earthquake_data.get('features', []):
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                coordinates = geometry.get('coordinates', [])

                writer.writerow({
                    'id': properties.get('id', None),
                    'source_id': properties.get('source_id', None),
                    'source_catalog': properties.get('source_catalog', None),
                    'lastupdate': properties.get('lastupdate', None),
                    'time': properties.get('time', None),
                    'flynn_region': properties.get('flynn_region', None),
                    'latitude': coordinates[1] if len(coordinates) > 1 else None,
                    'longitude': coordinates[0] if len(coordinates) > 0 else None,
                    'depth': coordinates[2] if len(coordinates) > 2 else None,
                    'evtype': properties.get('evtype', None),
                    'auth': properties.get('auth', None),
                    'magnitude': properties.get('mag', None),
                    'magtype': properties.get('magtype', None),
                    'unid': properties.get('unid', None)
                })

        return True
    except Exception as e:
        print(f"Error saving earthquake data to CSV: {str(e)}")
        return False

# Define the path to store earthquake_data.csv
desktop_path = '/home/ogdp/ogdp/data/earthquake/'
earthquake_data_file_path = os.path.join(desktop_path, 'earthquake_data.csv')


# Fetch earthquake data and update CSV if already exists
if os.path.exists(earthquake_data_file_path):
    # Fetch new earthquake data
    new_earthquake_data = retry_request(earthquake_data_speu, minmagnitude=0)

    if new_earthquake_data:
        # Save new data to CSV, overwriting the existing file
        if save_earthquake_data_to_csv(earthquake_data_file_path, new_earthquake_data):
            print(f'Earthquake data updated and saved to CSV successfully at: {earthquake_data_file_path}')
        else:
            print('Failed to save updated earthquake data to CSV.')
    else:
        print('Failed to fetch new earthquake data.')
else:
    print(f'Earthquake data CSV does not exist at: {earthquake_data_file_path}')

