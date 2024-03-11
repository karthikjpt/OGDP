import csv
import gzip
import io
import math
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from math import log10
import feedparser
import netCDF4
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from flask import Flask, render_template, make_response, request, send_file, send_from_directory, redirect
from werkzeug.utils import secure_filename
from flask_sitemap import Sitemap
import zipfile
from tempfile import mkdtemp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
import numpy as np
from requests.exceptions import SSLError

app = Flask(__name__, static_folder='static')

BASE_DIR = '/home/ogdp/ogdp'

data_directory = os.path.join(BASE_DIR, 'data', 'CIP')

TEMP_FOLDER = mkdtemp()

def add_watermark(ax):
    watermark_text = 'www.ogdp.in'  # Watermark text
    ax.text(0.95, 0.05, watermark_text, color='gray', alpha=0.5,
            fontsize=20, rotation=0, ha='right', va='bottom', transform=ax.transAxes)

# Function to read data from CSV files
def read_csv(file_path):
    return pd.read_csv(file_path)

def dynamic_earthquake_count_download_csv(filename):
    directory = os.path.join(app.root_path, 'data', 'earthquake')
    return send_from_directory(directory, filename)

#followings codes are related to pole coordinates, delta-t and gravity; data source paris observatory

# Get the pole coordinates and UT1-UTC from Paris Observatory
def eop_pc():
    try:
        url = 'https://hpiers.obspm.fr/eop-pc/index.php'
        response = requests.get(url, timeout=10)  # Specify a timeout in seconds
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        span_elements = soup.find_all("span")
        desired_text = None

        for span in span_elements:
            if "Latest C04 values for pole coordinates" in span.get_text():
                desired_text = span.get_text()
                break

        if desired_text:
            # Find the index of " ms" in the text and save until that point
            end_index = desired_text.index(" ms") + 3  # Include the " ms" characters
            desired_text = desired_text[:end_index]

            # Process the date as you have done
            month_mapping = {
                'Janvier': 'January',
                'Fevrier': 'February',
                'Mars': 'March',
                'Avril': 'April',
                'Mai': 'May',
                'Juin': 'June',
                'Juillet': 'July',
                'Aout': 'August',
                'Septembre': 'September',
                'Octobre': 'October',
                'Novembre': 'November',
                'Decembre': 'December'
            }
            date_pattern = r'(\d{1,2} [A-Za-zéûû]{3,} \d{4})'

            match = re.search(date_pattern, desired_text)

            if match:
                extracted_date_str = match.group(0)
                for french_month, english_month in month_mapping.items():
                    extracted_date_str = extracted_date_str.replace(french_month, english_month)
                extracted_date = datetime.strptime(extracted_date_str, '%d %B %Y')
                formatted_date = extracted_date.strftime('%Y-%m-%d')

                # Load the latest data (if needed)
                existing_data = load_latest_data()

                # Compare existing data with the scraped data and save if different
                if existing_data != desired_text:
                    data_file_name = f"data/CIP/{formatted_date}.txt"
                    with open(data_file_name, 'w') as data_file:
                        data_file.write(desired_text)
                    print(f"Data saved to {data_file_name}")
        else:
            print("Desired text not found on the webpage.")
    except requests.exceptions.RequestException as e:
        print(f"Error during web scraping: {e}")

# Function to load the latest data
def load_latest_data():
    data_files = os.listdir("data/CIP")
    data_files.sort(reverse=True)
    if data_files:
        latest_data_file = data_files[0]
        with open(f"data/CIP/{latest_data_file}", 'r') as file:
            latest_data = file.read()
        return latest_data
    return None

# Function to load and sort data from the "data" folder
def load_data():
    data_files = os.listdir("data/CIP")
    data_files.sort(reverse=True)
    data_dict = {}
    for data_file in data_files:
        date = data_file[:-4]
        with open(f"data/CIP/{data_file}", 'r') as file:
            data = file.read()
        data_dict[date] = data
    return data_dict

# Call the function to scrape and save the data
eop_pc()

def update_data_for_eop_pc():
    # Call the function to scrape and save the data
    eop_pc()

#creating tables and charts that was saved from paris observatory
def poledailygraphdataprocess():
    data_folder = 'data/CIP'
    file_list = os.listdir(data_folder)

    dates = []
    x_values = []
    y_values = []
    ut1_utc_values = []

    for file_name in file_list:
        if file_name.endswith('.txt'):
            with open(os.path.join(data_folder, file_name), 'r') as file:
                content = file.read()
                # Extracting date from file name
                date = file_name.split('.')[0]
                dates.append(date)

                # Extracting x and y values from file content using regular expressions
                match_x = re.search(r'x\s*=\s*([0-9.]+)\s*mas', content)
                match_y = re.search(r'y\s*=\s*([0-9.]+)\s*mas', content)
                match_ut1_utc = re.search(r'UT1-UTC\s*=\s*(-?[0-9.]+(?:\.[0-9]+)?)\s*ms', content)

                x_value = float(match_x.group(1)) if match_x else None
                y_value = float(match_y.group(1)) if match_y else None
                ut1_utc_value = float(match_ut1_utc.group(1)) if match_ut1_utc else None

                x_values.append(x_value)
                y_values.append(y_value)
                ut1_utc_values.append(ut1_utc_value)

    # Creating a Pandas DataFrame
    data = {
        'Date': dates,
        'X': x_values,
        'Y': y_values,
        'UT1-UTC': ut1_utc_values
    }
    df = pd.DataFrame(data)

    # Dropping rows with missing values
    df.dropna(inplace=True)

    # Convert 'Date' column to datetime format and sort by date (remove 'infer_datetime_format')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date')

    # Remove rows with NaT (not a valid date)
    df = df.dropna(subset=['Date'])

    # Calculate X and Y and ut1-utc variations
    df['X_Variation'] = df['X'].diff().fillna(0)  # Calculate difference between consecutive X values
    df['Y_Variation'] = df['Y'].diff().fillna(0)  # Calculate difference between consecutive Y values
    df['Delta_T'] = df['UT1-UTC'].diff().fillna(0)

    # Calculate X+Y variation (X variation + Y variation)
    df['X+Y_Variation'] = (df['X_Variation'].abs() + df['Y_Variation'].abs())

    # Sort DataFrame by date in descending order to get the most recent records first
    df = df.sort_values(by='Date', ascending=False)

    # Select the last 30 records or dates
    df_last_30 = df.head(30)

    # Plotting the graphs and saving them separately using the filtered data (last 30 records)
    with app.app_context():
        plt.figure(figsize=(8, 6))

    # Plotting the graph with 'UT1-UTC' values
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(df_last_30['Date'], df_last_30['UT1-UTC'], marker='o', linestyle='-', label='UT1-UTC')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('UT1-UTC')
    ax1.set_title('UT1-UTC (Last 30 records)')
    ax1.tick_params(axis='x', rotation=45)
    add_watermark(ax1)  # Call the function to add watermark
    ax1.legend()  # Show legend for the line plot

    # Get the indices for dates at seven-day intervals
    date_indices_plot_1 = np.arange(0, len(df_last_30), 7)
    date_labels_plot_1 = df_last_30['Date'].iloc[date_indices_plot_1].dt.strftime('%Y-%m-%d')

    # Annotate data points with dates at seven-day intervals
    for i, label in zip(date_indices_plot_1, date_labels_plot_1):
        plt.annotate(label, (df_last_30['Date'].iloc[i], df_last_30['UT1-UTC'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig('static/images/ut1_utc_plot_1.png')  # Update plot file name
    plt.close()

    # Create a copy of the DataFrame for the specific plot
    df_plot_2 = df.copy()

    # Plot 2: Line graph for X and converted Y values - Last 30 records
    fig, ax2 = plt.subplots(figsize=(8, 6))  # Adjust dimensions as needed

    # Select only the last 30 records for Plot 2 based on the 'Date' column
    df_last_30_plot_2 = df_plot_2.sort_values('Date').tail(30)

    # Ensure at least 5 data points for the line plot
    num_points_for_line = max(5, len(df_last_30_plot_2))
    date_indices_for_line = np.linspace(0, len(df_last_30_plot_2) - 1, num_points_for_line, dtype=int)

    ax2.plot(df_last_30_plot_2['X'], df_last_30_plot_2['Y'], marker='o', linestyle='-', label='CIP: X, Y', markersize=8)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Pole Coordinates (x, y) - Last 30 records')

    # Invert the Y axis to display negative values upward
    ax2.invert_yaxis()

    # Adjust margins to use available space more efficiently
    plt.margins(x=0.05, y=0.05)

    # Set ticks with consistent intervals for both x and y axes for the last 30 records
    tick_interval_plot_2 = 5  # Change this to your desired interval
    ax2.set_yticks(np.arange(min(df_last_30_plot_2['Y']), max(df_last_30_plot_2['Y']) + 1, tick_interval_plot_2))

    # Annotate data points with dates at 2-day intervals
    for idx in range(0, len(df_last_30_plot_2), 3):
        plt.annotate(df_last_30_plot_2['Date'].iloc[idx].strftime('%Y-%m-%d'),
                     (df_last_30_plot_2['X'].iloc[idx], df_last_30_plot_2['Y'].iloc[idx]),
                     textcoords="offset points", xytext=(0,10), ha='center')

    # Assuming add_watermark is defined
    add_watermark(ax2)

    ax2.legend()  # Show legend for the line plot
    plt.tight_layout()
    plt.savefig('static/images/pole_coordinates_plot_2.png')
    plt.close()

    # Plot 3: Line graph for X+Y Variation
    fig, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(df_last_30['Date'], df_last_30['X+Y_Variation'], marker='o', linestyle='-', label='CIP: X, Y variation')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('X+Y Variation')
    ax3.set_title('X+Y Variation (X + Y) - Last 30 records')
    ax3.tick_params(axis='x', rotation=45)
    add_watermark(ax3)  # Call the function to add watermark
    ax3.legend()  # Show legend for the line plot

    # Get the indices for dates at seven-day intervals
    date_indices_plot_3 = np.arange(0, len(df_last_30), 7)
    date_labels_plot_3 = df_last_30['Date'].iloc[date_indices_plot_3].dt.strftime('%Y-%m-%d')

    # Annotate data points with dates at seven-day intervals
    for i, label in zip(date_indices_plot_3, date_labels_plot_3):
        plt.annotate(label, (df_last_30['Date'].iloc[i], df_last_30['X+Y_Variation'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig('static/images/xy_variations_plot_3.png')
    plt.close()

    # Plotting the graph with 'UT1-UTC' values and their variations
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(df_last_30['Date'], df_last_30['Delta_T'], marker='o', linestyle='-', label='Delta_T (ms)')
    ax1.set_xlabel('Date')
    ax1.set_title('Delta_T (UT1 minus UTC or TAI daily variation) (Last 30 records)')
    ax1.tick_params(axis='x', rotation=45)
    add_watermark(ax1)  # Call the function to add watermark
    ax1.legend()  # Show legend for the line plot

    # Get the indices for dates at seven-day intervals
    date_indices_plot_1 = np.arange(0, len(df_last_30), 7)
    date_labels_plot_1 = df_last_30['Date'].iloc[date_indices_plot_1].dt.strftime('%Y-%m-%d')

    # Annotate data points with dates at seven-day intervals
    for i, label in zip(date_indices_plot_1, date_labels_plot_1):
        ax1.annotate(label, (df_last_30['Date'].iloc[i], df_last_30['Delta_T'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig('static/images/ut1_utc_variation_plot.png')  # Update plot file name
    plt.close()

    # Generate HTML table from updated DataFrame
    html_table = df.to_html(classes='data', index=False)

    # Save all available data as a .csv file in the 'data/gravity' folder (overwrite mode)
    file_path = 'data/gravity/recent_data.csv'
    df.to_csv(file_path, index=False)

    return render_template('poledailygraph.html', tables=[html_table],
                          graph_urls=['static/ut1_utc_plot_1.png', 'static/pole_coordinates_plot_2.png', 'static/xy_variations_plot_3.png', 'static/ut1_utc_variation_plot.png'])

def update_poledailygraph():
    #update_call
    poledailygraphdataprocess()

#merging daily paris observatory data with historical data and processed data

def merge_gravity_files():
    try:
        # Generate date range from 1982-01-01 to today
        start_date = datetime(1972, 1, 1)
        end_date = datetime.now().date()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Create DataFrame with the date range
        merged_df = pd.DataFrame({'Date': date_range})

        # Read 'Date' and 'UT1-UTC_Variation' columns from from_1982.csv
        from_1972 = pd.read_csv('data/gravity/from_1972.csv', usecols=['Date', 'Delta_T', 'sun_deldot', 'moon_deldot', 'zero'])

        # Convert 'Date' columns to datetime if they are not already in datetime format
        from_1972['Date'] = pd.to_datetime(from_1972['Date'])

        # Merge the dataframes on 'Date' column
        merged_df = pd.merge(merged_df, from_1972, on='Date', how='left')

        # Read 'Date' and 'UT1-UTC_Variation' columns from recent_data.csv
        recent_data = pd.read_csv('data/gravity/recent_data.csv', usecols=['Date', 'Delta_T'])

        # Convert 'Date' columns to datetime if they are not already in datetime format
        recent_data['Date'] = pd.to_datetime(recent_data['Date'])

        # Update 'UT1-UTC_Variation' values for the available recent dates in merged_df from recent_data
        for index, row in recent_data.iterrows():
            date = row['Date']
            Delta_T = row['Delta_T']
            # Only update rows with NaN values in the 'Delta_T' column
            if pd.isna(merged_df.loc[merged_df['Date'] == date, 'Delta_T']).any():
                merged_df.loc[merged_df['Date'] == date, 'Delta_T'] = Delta_T

        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv('data/gravity/data_for_gravity_graph.csv', index=False, columns=['Date', 'Delta_T', 'moon_deldot', 'sun_deldot', 'zero'])

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Call the function to merge and save the CSV files
merge_gravity_files()

# Create the 'static/images' directory if it doesn't exist
image_directory = os.path.join(app.root_path, 'static/images')
os.makedirs(image_directory, exist_ok=True)

# Load the data from the updated CSV file
def load_gravity_data():
    merge_gravity_files()  # Update the data before loading
    return pd.read_csv('data/gravity/data_for_gravity_graph.csv')

def generate_default_plot():

    gravity_data = load_gravity_data()

    # Calculate the default date range (last 30 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)

    # Filter data based on the default date range
    default_data = gravity_data[(gravity_data['Date'] >= start_date.strftime('%Y-%m-%d')) & (gravity_data['Date'] <= end_date.strftime('%Y-%m-%d'))]

    # Parse 'Date' column to datetime objects
    default_data.loc[:, 'Date'] = pd.to_datetime(default_data['Date'])

    # Sort data by date
    default_data = default_data.sort_values(by='Date')

    # Create Matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.plot(default_data['Date'], default_data['zero'], color='grey', marker='o')
    plt.plot(default_data['Date'], default_data['moon_deldot'], color='#EDB120', label='Moon deldot (au)')
    plt.plot(default_data['Date'], default_data['sun_deldot'], color='#D95319', label='Sun deldot (au)')
    plt.plot(default_data['Date'], default_data['Delta_T'], color='#0072BD', label='Earth Delta T (ms)')
    plt.xlabel('Date')
    plt.ylabel('Delta T and deldot')
    plt.title('Variation in daily Earth rotational speed and rate of change in distance between Earth and celestial bodies')
    plt.legend()  # Show legend

    # Set the locator and formatter for x-axis ticks
    tick_interval = len(default_data) // 5  # Adjust the divisor for desired tick count
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    add_watermark(plt.gca())

    # Save the plot as an image file in 'static/images'
    default_plot_path = os.path.join(image_directory, 'gravity_plot.png')
    plt.savefig(default_plot_path)

    plt.close()

def update_generate_default_plot():
    # Call the merge function to update data_for_gravity_graph.csv
    generate_default_plot()

#Following codes are for tools

#merge netcdf csv file and extract solar wind related files
def merge_and_process_csv_files_netcdf():
    uploaded_files = request.files.getlist('file[]')
    dfs = []

    try:
        # Read and store uploaded CSV files as DataFrames
        for file in uploaded_files:
            if file.filename.endswith('.csv'):
                # Read CSV file using BytesIO to handle binary mode
                file_content = io.BytesIO()
                file.save(file_content)
                file_content.seek(0)
                df = pd.read_csv(file_content, delimiter='\t')  # Assuming tab-separated values, modify delimiter if needed
                dfs.append(df)

        if not dfs:
            return "No valid CSV files uploaded"

        # Merge DataFrames if multiple CSV files are uploaded
        merged_df = pd.concat(dfs, ignore_index=True)

        # Keep only specified columns
        columns_to_keep = ['data-time', 'sample_count', 'proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']
        merged_df = merged_df[columns_to_keep]

        # Create a buffer to store the CSV data
        buffer = io.StringIO()
        merged_df.to_csv(buffer, index=False)

        # Set up the buffer for download as a CSV file
        buffer.seek(0)
        return send_file(io.BytesIO(buffer.getvalue().encode()), as_attachment=True, download_name='mergeandselectedcolumn.csv', mimetype='text/csv')

    except Exception as e:
        return f"Error processing files: {str(e)}"


#tool for date fill
# Function to process uploaded file and perform operations
def process_file(uploaded_file, temp_folder):
    file_path = os.path.join(temp_folder, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Read uploaded file
    df = pd.read_csv(file_path)  # For CSV, use pd.read_excel() for Excel files

    # Look for specified date-related headers
    date_columns = [col for col in df.columns if col.lower() in ['date', 'year', 'month', 'time']]

    if len(date_columns) == 0:
        return "No date-related columns found"

    # Find oldest and newest dates or values
    min_date = df[date_columns].min().min()
    max_date = df[date_columns].max().max()

    # Check if 'date' column exists in the DataFrame
    if 'date' in df.columns:
        # Fill missing date value rows
        df['date'] = pd.to_datetime(df['date'])
        idx = pd.date_range(start=min_date, end=max_date)
        df = df.set_index('date').reindex(idx).reset_index()

    # Convert DataFrame back to CSV file format
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return output

def datefiller_process():
    if 'file' not in request.files:
        return "No file part"

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return "No selected file"

    if uploaded_file:
        try:
            modified_file = process_file(uploaded_file, TEMP_FOLDER)

            # Return modified file for download
            return send_file(modified_file, as_attachment=True, download_name='date_filled_file.csv', mimetype='text/csv')
        except Exception as e:
            return f"Error processing file: {str(e)}"

    return "Error processing file"

#netcdf to csv converter tool
def Unidata_NetCDF_to_CSV():
    uploaded_files = request.files.getlist('file[]')
    processed_files = []

    with BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zipf:
            for file in uploaded_files:
                file_data = file.read()
                processed_data = process_single_data(file_data)
                if processed_data:
                    zipf.writestr(file.filename[:-3] + '_processed.csv', processed_data)

        zip_buffer.seek(0)
        temp_file_path = os.path.join(TEMP_FOLDER, 'processed_data.zip')
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(zip_buffer.getvalue())

        return send_file(temp_file_path, as_attachment=True, mimetype='application/zip', download_name='Unidata_NetCDF_to_CSV.zip')

def process_single_data(file_data):
    try:
        with BytesIO(file_data) as file_buffer:
            with gzip.GzipFile(fileobj=file_buffer, mode='rb') as gz:
                with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as ncdata:
                    data_dict = {}
                    for var in ncdata.variables.keys():
                        if var == 'time':
                            time_var = ncdata.variables['time']
                            dtime = netCDF4.num2date(time_var[:], time_var.units)
                            data_dict['data-time'] = dtime
                        else:
                            data_dict[var] = ncdata.variables[var][:]

                    df = pd.DataFrame(data_dict)
                    csv_output = df.to_csv(sep='\t', index=False).encode()  # Convert DataFrame to CSV bytes
                    return csv_output
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

#quake summary tool
def read_csv_with_delimiter(file, delimiter):
    df = pd.read_csv(file, delimiter=delimiter)
    return df

def process_uploaded_file(file):
    try:
        file.seek(0)  # Reset file pointer to the beginning
        file_content = file.stream.read().decode('utf-8')

        if ';' in file_content:
            df = read_csv_with_delimiter(io.StringIO(file_content), ';')
        else:
            df = read_csv_with_delimiter(io.StringIO(file_content), ',')

        return df

    except Exception as e:
        return None  # Return None if there's an issue processing the file

def upload_csvfile_to_quake_summary():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Retrieve multiple files
        if files:
            try:
                dfs = []  # List to hold individual DataFrames from multiple files

                for file in files:
                    if file:
                        # Process each uploaded file
                        processed_df = process_uploaded_file(file)

                        if processed_df is None:
                            return f"Error processing file: {secure_filename(file.filename)}. Please check the file format."

                        # Append the processed DataFrame to the list
                        dfs.append(processed_df)

                if not dfs:
                    return "No valid CSV files uploaded."

                # Check for various 'date' column name variations before concatenating
                combined_df = pd.concat(dfs, ignore_index=True)

                date_column_names = ['Date', 'DATE', 'date', 'time']
                date_column = next((col for col in combined_df.columns if col in date_column_names), None)

                mag_column_names = ['mag', 'Mag', 'Magnitude', 'magnitude', 'MAGNITUDE', 'richter_scale_magnitude', 'richter scale magnitude', 'richter scale']
                mag_column = next((col for col in combined_df.columns if col in mag_column_names), None)

                if not date_column:
                    return "Date column not found in the CSV file."

                if not mag_column:
                    return "Magnitude column not found in the CSV file."

                # Handling 'YYYY-MM-DDTHH:MM:SS.sssZ' format for the 'date' column
                combined_df[date_column] = pd.to_datetime(combined_df[date_column], errors='coerce')
                combined_df[date_column] = combined_df[date_column].dt.strftime('%Y-%m-%d')

                # Group by 'date' column and count occurrences
                date_counts = combined_df[date_column].value_counts().to_dict()

                # Calculate total energy based on 'magnitude' column
                combined_df['total_energy'] = round(10**(1.5 * combined_df[mag_column] + 4.4), 2)

                # Group by 'date' and sum 'total_energy'
                total_energy_by_date = combined_df.groupby(date_column)['total_energy'].sum().to_dict()

                # Prepare data for CSV file
                data_for_csv = []
                for date in date_counts:
                    total_energy_value = total_energy_by_date.get(date, 0)
                    log10_energy_value = round(math.log10(total_energy_value), 2) if total_energy_value > 0 else 0
                    richter_scale_magnitude = round((1.0 / 1.5) * (math.log10(total_energy_value) - 4.4), 2) if total_energy_value > 0 else 0
                    data_for_csv.append({
                        "date": date,
                        "count": date_counts[date],
                        "total_energy": total_energy_value,
                        "richter_scale_magnitude": richter_scale_magnitude,
                    })

                # Create a DataFrame
                sorted_data = pd.DataFrame(data_for_csv)

                # Sort by 'date' column in ascending order
                sorted_data['date'] = pd.to_datetime(sorted_data['date'])
                sorted_data = sorted_data.sort_values(by='date')

                # Assign rank based on total energy
                sorted_data['rank'] = sorted_data['total_energy'].rank(ascending=False, method='dense').astype(int)

                # Create a CSV file in memory
                output = BytesIO()
                sorted_data.to_csv(output, index=False)
                output.seek(0)

                # Save the CSV file in the temporary folder
                temp_file_path = os.path.join(TEMP_FOLDER, 'earthquake_data_processed.csv')
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(output.getvalue())

                # Send the temporary file for download
                return send_file(temp_file_path, as_attachment=True, mimetype='text/csv', download_name='earthquake_summary.csv')

            except Exception as e:
                return f"Error processing file: {str(e)}"

    return render_template('csvtotable.html')

#following codes are get data from various source and display
# Function to scrape volcanic eruption reports with retry mechanism
def get_all_volcanic_reports():
    rss_url = "https://volcano.si.edu/news/WeeklyVolcanoRSS.xml"
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            feed = feedparser.parse(rss_url)
            reports = []
            for entry in feed.entries:
                title = entry.title
                description = entry.description
                reports.append({"title": title, "description": description})
            return reports  # Return the reports if successful
        except Exception as e:
            # Log the error or print it for debugging purposes
            print(f"Error occurred: {e}")

        # If an error occurred, retry after delay
        time.sleep(retry_delay)

    # If all retries failed, return an empty list or handle the failure accordingly
    return []

def fetch_noaa_swpc_sgarf():
    NOAA_URL = "https://services.swpc.noaa.gov/text/sgarf.txt"
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(NOAA_URL)
            if response.status_code == 200:
                return response.text
        except SSLError as e:
            # Log the error or print it for debugging purposes
            print(f"SSL Error occurred: {e}")

        # If SSL error or status code != 200, retry after delay
        time.sleep(retry_delay)

    # If all retries failed or encountered SSL errors, return None
    return None

# Define a function to fetch data from NOAA with retry mechanism for SSL errors
def fetch_noaa_swpc_weekly():
    url = "https://services.swpc.noaa.gov/text/weekly.txt"
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
        except SSLError as e:
            # Log the error or print it for debugging purposes
            print(f"SSL Error occurred: {e}")

        # If SSL error or status code != 200, retry after delay
        time.sleep(retry_delay)

    # If all retries failed or encountered SSL errors, return None
    return None


#Earthquake summary process from usgs and emsc data
def calculate_date_range(days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59")
    return start_date_str, end_date_str

# Custom Jinja2 filter to format magnitude with one decimal place
def format_magnitude(value):
    return f"{value:.1f}"

app.jinja_env.filters['format_magnitude'] = format_magnitude

def usgsquake_summary_process(earthquake_data):
    # Shared logic involving the if statement
    processed_data = []

    if earthquake_data:
        # Group earthquake data by day and count earthquakes for each day
        earthquake_counts = {}
        total_energy_by_date = {}  # Store total energy for each date
        non_zero_energy_dates = []  # Store dates with non-zero Total Energy
        rank = 0

        for feature in earthquake_data['features']:
            timestamp = feature['properties']['time'] / 1000  # Convert milliseconds to seconds
            date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
            earthquake_counts[date] = earthquake_counts.get(date, 0) + 1
            feature['properties']['formatted_time'] = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')

            # Calculate Total Energy using corrected Richter scale formula and add it to the date's total energy
            magnitude = feature['properties']['mag']
            total_energy = round(10**(1.5 * magnitude + 4.4), 2)
            total_energy_by_date[date] = total_energy_by_date.get(date, 0) + total_energy

            if date not in non_zero_energy_dates and total_energy > 0:
                non_zero_energy_dates.append(date)

        # Calculate the sum of total energy for every date
        sum_total_energy_by_date = {}
        for date, energy in total_energy_by_date.items():
            if energy > 0:
                sum_total_energy_by_date[date] = energy

        # Sort the non-zero energy dates by Total Energy
        sorted_dates = sorted(non_zero_energy_dates, key=lambda date: total_energy_by_date[date], reverse=True)

        # Assign ranks to dates with non-zero Total Energy, considering ties
        ranked_dates = {}
        rank = 1
        prev_energy = None

        for date in sorted_dates:
            total_energy = total_energy_by_date[date]

            # Check if the total energy has changed
            if total_energy != prev_energy:
                ranked_dates[date] = rank
                rank += 1
            else:
                ranked_dates[date] = rank - 1

            prev_energy = total_energy

        # Calculate the date range (last 30 days) with the correct starting rank
        end_date = datetime.now()
        start_date = end_date - timedelta(days=29)
        date_range = [start_date + timedelta(days=i) for i in range(30)]

        # Prepare the data for the template and add Richter scale magnitudes
        data_for_template = []
        for date in reversed(date_range):
            total_energy_value = total_energy_by_date.get(date.strftime('%Y-%m-%d'), 0)
            log10_energy_value = round(math.log10(total_energy_value), 2) if total_energy_value > 0 else 0

            # Calculate the Richter Scale Magnitude for each earthquake based on total energy
            richter_scale_magnitude = round((1.0 / 1.5) * (math.log10(total_energy_value) - 4.4), 2) if total_energy_value > 0 else 0

            # Format total_energy_value to scientific notation
            formatted_total_energy = f"{total_energy_value:.10e}" if total_energy_value > 0 else '0.00e+00'

            data_for_template.append(
                {
                    "date": date.strftime('%Y-%m-%d'),
                    "count": earthquake_counts.get(date.strftime('%Y-%m-%d'), 0),
                    "total_energy": formatted_total_energy,  # Add total energy in scientific notation
                    "richter_scale_magnitude": richter_scale_magnitude,  # Add the Richter Scale Magnitude
                    "rank": ranked_dates.get(date.strftime('%Y-%m-%d'), 0),
                }
            )

        processed_data = {
            'earthquake_counts': data_for_template,
            'earthquake_data': earthquake_data,  # Pass the earthquake data to the template
        }

    return processed_data

def speuquake_summary_process(earthquake_data):
    # Shared logic involving the if statement
    processed_data = []

    if earthquake_data:
        # Group earthquake data by day and count earthquakes for each day
        earthquake_counts = {}
        total_energy_by_date = {}  # Store total energy for each date
        non_zero_energy_dates = []  # Store dates with non-zero Total Energy
        rank = 0

        for feature in earthquake_data['features']:
            timestamp = feature['properties']['time']  # Timestamp as a string
            date_time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')  # Convert to datetime
            date = date_time.strftime('%Y-%m-%d')
            earthquake_counts[date] = earthquake_counts.get(date, 0) + 1
            feature['properties']['formatted_time'] = date_time.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Calculate Total Energy using corrected Richter scale formula and add it to the date's total energy
            magnitude = feature['properties']['mag']
            total_energy = round(10**(1.5 * magnitude + 4.4), 2)
            total_energy_by_date[date] = total_energy_by_date.get(date, 0) + total_energy

            if date not in non_zero_energy_dates and total_energy > 0:
                non_zero_energy_dates.append(date)

        # Calculate the sum of total energy for every date
        sum_total_energy_by_date = {date: energy for date, energy in total_energy_by_date.items() if energy > 0}

        # Sort the non-zero energy dates by Total Energy
        sorted_dates = sorted(non_zero_energy_dates, key=lambda date: total_energy_by_date[date], reverse=True)

        # Assign ranks to dates with non-zero Total Energy, considering ties
        ranked_dates = {}
        rank = 1
        prev_energy = None

        for date in sorted_dates:
            total_energy = total_energy_by_date[date]

            # Check if the total energy has changed
            if total_energy != prev_energy:
                ranked_dates[date] = rank
                rank += 1
            else:
                ranked_dates[date] = rank - 1

            prev_energy = total_energy

        # Calculate the date range (last 30 days) with the correct starting rank
        end_date = datetime.now()
        start_date = end_date - timedelta(days=29)
        date_range = [start_date + timedelta(days=i) for i in range(30)]

        # Prepare the data for the template and add Richter scale magnitudes
        data_for_template = []
        for date in reversed(date_range):
            total_energy_value = total_energy_by_date.get(date.strftime('%Y-%m-%d'), 0)
            log10_energy_value = round(math.log10(total_energy_value), 2) if total_energy_value > 0 else 0

            # Calculate the Richter Scale Magnitude for each earthquake based on total energy
            richter_scale_magnitude = round((1.0 / 1.5) * (math.log10(total_energy_value) - 4.4), 2) if total_energy_value > 0 else 0

            # Format total_energy_value to scientific notation
            formatted_total_energy = f"{total_energy_value:.10e}" if total_energy_value > 0 else '0.00e+00'

            data_for_template.append(
                {
                    "date": date.strftime('%Y-%m-%d'),
                    "count": earthquake_counts.get(date.strftime('%Y-%m-%d'), 0),
                    "total_energy": formatted_total_energy,  # Add total energy in scientific notation
                    "richter_scale_magnitude": richter_scale_magnitude,  # Add the Richter Scale Magnitude
                    "rank": ranked_dates.get(date.strftime('%Y-%m-%d'), 0),
                }
            )

        processed_data = {
            'earthquake_counts': data_for_template,
            'earthquake_data': earthquake_data,  # Pass the earthquake data to the template
        }

    return processed_data

# Function to scrape earthquake data with user-defined parameters
def get_usgsquake_data(min_depth=None, max_depth=None, min_magnitude=None, max_magnitude=None,
                        min_latitude=None, max_latitude=None, min_longitude=None, max_longitude=None, latitude=None, longitude=None, maxradius=None, max_retries=3, retry_delay=10):
    start_date_str, end_date_str = calculate_date_range(29)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
    params = {
        "starttime": start_date_str,
        "endtime": end_date_str
    }

    # Add user-defined parameters to the URL if provided
    if min_depth is not None:
        params["mindepth"] = min_depth
    if max_depth is not None:
        params["maxdepth"] = max_depth
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude
    if min_latitude is not None:
        params["minlatitude"] = min_latitude
    if max_latitude is not None:
        params["maxlatitude"] = max_latitude
    if min_longitude is not None:
        params["minlongitude"] = min_longitude
    if max_longitude is not None:
        params["maxlongitude"] = max_longitude
    if latitude is not None:
        params["latitude"] = latitude
    if longitude is not None:
        params["longitude"] = longitude
    if maxradius is not None:
        params["maxradius"] = maxradius

    response = None  # Initialize response to None
    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            time.sleep(retry_delay)
            retries += 1

    # If all retries fail, return None or handle the failure as needed in your production environment
    return None

# Function to scrape earthquake data with user-defined parameters
def get_speuquake_data(min_depth=None, max_depth=None, min_magnitude=None, max_magnitude=None,
                        min_latitude=None, max_latitude=None, min_longitude=None, max_longitude=None, latitude=None, longitude=None, maxradius=None, max_retries=3, retry_delay=10):
    start_date_str, end_date_str = calculate_date_range(29)

    url = "https://www.seismicportal.eu/fdsnws/event/1/query?format=json"
    params = {
        "starttime": start_date_str,
        "endtime": end_date_str
    }

    # Add user-defined parameters to the URL if provided
    if min_depth is not None:
        params["mindepth"] = min_depth
    if max_depth is not None:
        params["maxdepth"] = max_depth
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude
    if min_latitude is not None:
        params["minlatitude"] = min_latitude
    if max_latitude is not None:
        params["maxlatitude"] = max_latitude
    if min_longitude is not None:
        params["minlongitude"] = min_longitude
    if max_longitude is not None:
        params["maxlongitude"] = max_longitude
    if latitude is not None:
        params["latitude"] = latitude
    if longitude is not None:
        params["longitude"] = longitude
    if maxradius is not None:
        params["maxradius"] = maxradius

    response = None  # Initialize response to None
    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            time.sleep(retry_delay)
            retries += 1

    # If all retries fail, return None or handle the failure as needed in your production environment
    return None

@app.before_request
def redirect_non_www():
    if not request.host.startswith('localhost') and \
       not request.host.startswith('127.') and \
       not request.host.startswith('0.') and \
       not request.host.startswith('blog.') and \
       request.host != 'www.ogdp.in':

        # Perform redirection only for non-local and non-www requests
        return redirect('https://www.ogdp.in' + request.path, code=301)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/refresh')
def refresh():
    update_data_for_eop_pc()  # Update data for something related to 'eop_pc'
    update_poledailygraph()   # Update something related to 'poledailygraph'
    update_generate_default_plot()  # Update to generate a default plot
    return "Data refreshed manually"

@app.route('/dynamic-earthquake-counts')
def dynamiccounts():
    # Reading data from CSV files
    magnitude_data = read_csv('data/earthquake/magnitude_ranges.csv')
    depth_data = read_csv('data/earthquake/depth_ranges.csv')
    latitude_data = read_csv('data/earthquake/latitude_ranges.csv')
    longitude_data = read_csv('data/earthquake/longitude_ranges.csv')

    # Rendering HTML page with tables
    return render_template('dynamiccounts.html', magnitude_data=magnitude_data, depth_data=depth_data, latitude_data=latitude_data, longitude_data=longitude_data)

@app.route('/dynamic-earthquake-energy')
def dynamicenergy():
    # Reading data from CSV files
    magnitude_data = read_csv('data/earthquake/magnitude_energy.csv')
    depth_data = read_csv('data/earthquake/depth_energy.csv')
    latitude_data = read_csv('data/earthquake/latitude_energy.csv')
    longitude_data = read_csv('data/earthquake/longitude_energy.csv')

    # Rendering HTML page with tables
    return render_template('dynamicenergy.html', magnitude_data=magnitude_data, depth_data=depth_data, latitude_data=latitude_data, longitude_data=longitude_data)


@app.route('/download/earthquake_data')
def download_earthquake_data():
    return dynamic_earthquake_count_download_csv('earthquake_data.csv')

@app.route('/download/depth_ranges')
def download_depth_ranges():
    return dynamic_earthquake_count_download_csv('depth_ranges.csv')

@app.route('/download/magnitude_ranges')
def download_magnitude_ranges():
    return dynamic_earthquake_count_download_csv('magnitude_ranges.csv')

@app.route('/download/latitude_ranges')
def download_latitude_ranges():
    return dynamic_earthquake_count_download_csv('latitude_ranges.csv')

@app.route('/download/longitude_ranges')
def download_longitude_ranges():
    return dynamic_earthquake_count_download_csv('longitude_ranges.csv')

@app.route('/download/earthquake_energy_data')
def download_earthquake_energy_data():
    return dynamic_earthquake_count_download_csv('earthquake_energy_data.csv')

@app.route('/download/depth_energy')
def download_depth_energy():
    return dynamic_earthquake_count_download_csv('depth_energy.csv')

@app.route('/download/magnitude_energy')
def download_magnitude_energy():
    return dynamic_earthquake_count_download_csv('magnitude_energy.csv')

@app.route('/download/latitude_energy')
def download_latitude_energy():
    return dynamic_earthquake_count_download_csv('latitude_energy.csv')

@app.route('/download/longitude_energy')
def download_longitude_energy():
    return dynamic_earthquake_count_download_csv('longitude_energy.csv')

@app.route('/delta-t', methods=['POST'])
def deltat():

    gravity_data = load_gravity_data()

    # Get user-selected date range from the form
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Filter data based on user input or predefined options
    selected_data = gravity_data[(gravity_data['Date'] >= start_date) & (gravity_data['Date'] <= end_date)]

    # Parse 'Date' column to datetime objects
    selected_data.loc[:, 'Date'] = pd.to_datetime(selected_data['Date'])

    # Sort data by date
    selected_data = selected_data.sort_values(by='Date')

    # Create Matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data['Date'], selected_data['zero'], color='grey')
    plt.plot(selected_data['Date'], selected_data['moon_deldot'], color='#EDB120', label='Moon deldot (au)')
    plt.plot(selected_data['Date'], selected_data['sun_deldot'], color='#D95319', label='Sun deldot (au)')
    plt.plot(selected_data['Date'], selected_data['Delta_T'], color='#0072BD', label='Earth Delta T (ms)')
    plt.xlabel('Date')
    plt.ylabel('Delta T and deldot')
    plt.title('Variation in daily Earth rotational speed and rate of change in distance between Earth and celestial bodies')
    plt.legend()  # Show legend

    # Set the locator and formatter for x-axis ticks
    tick_interval = max(len(selected_data) // 5, 1)  # Ensure the tick interval is at least 1
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    add_watermark(plt.gca())

    # Save the plot as an image in BytesIO object
    img_bytesio = BytesIO()
    plt.savefig(img_bytesio, format='png')
    plt.close()

    # Encode the BytesIO object to base64
    img_str = base64.b64encode(img_bytesio.getvalue()).decode('utf-8')

    # Pass the base64-encoded image to the template
    return render_template('deltat.html', img_str=img_str)

@app.route('/usgs_quake_data_process')
def process_usgs_quake_data():
    # Retrieve parameters from the request's query parameters
    min_depth = request.args.get('min_depth')
    max_depth = request.args.get('max_depth')
    min_magnitude = request.args.get('min_magnitude')
    max_magnitude = request.args.get('max_magnitude')
    min_latitude = request.args.get('min_latitude')
    max_latitude = request.args.get('max_latitude')
    min_longitude = request.args.get('min_longitude')
    max_longitude = request.args.get('max_longitude')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    maxradius = request.args.get('maxradius')


    # Fetch earthquake data using the defined parameters
    earthquake_data = get_usgsquake_data(
        min_depth=min_depth, max_depth=max_depth,
        min_magnitude=min_magnitude, max_magnitude=max_magnitude,
        min_latitude=min_latitude, max_latitude=max_latitude,
        min_longitude=min_longitude, max_longitude=max_longitude,
        latitude=latitude, longitude=longitude, maxradius=maxradius
    )

    if earthquake_data:
        # Process earthquake data as needed
        processed_data = usgsquake_summary_process(earthquake_data)
        return render_template('usgssummary.html', **processed_data)
    else:
        return "Failed to fetch earthquake data. Refine your query or try a different database (EMSC/USGS)."

@app.route('/speu_quake_data_process')
def process_speu_quake_data():
    # Retrieve parameters from the request's query parameters
    min_depth = request.args.get('min_depth')
    max_depth = request.args.get('max_depth')
    min_magnitude = request.args.get('min_magnitude')
    max_magnitude = request.args.get('max_magnitude')
    min_latitude = request.args.get('min_latitude')
    max_latitude = request.args.get('max_latitude')
    min_longitude = request.args.get('min_longitude')
    max_longitude = request.args.get('max_longitude')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')
    maxradius = request.args.get('maxradius')


    # Fetch earthquake data using the defined parameters
    earthquake_data = get_speuquake_data(
        min_depth=min_depth, max_depth=max_depth,
        min_magnitude=min_magnitude, max_magnitude=max_magnitude,
        min_latitude=min_latitude, max_latitude=max_latitude,
        min_longitude=min_longitude, max_longitude=max_longitude,
        latitude=latitude, longitude=longitude, maxradius=maxradius
    )

    if earthquake_data:
        # Process earthquake data as needed
        processed_data = speuquake_summary_process(earthquake_data)
        return render_template('speusummary.html', **processed_data)
    else:
        return "Failed to fetch earthquake data. Refine your query or try a different database (EMSC/USGS)."

@app.route('/volcanic_reports')
def volcanic_reports():
    reports = get_all_volcanic_reports()
    return render_template('page3.html', reports=reports)

@app.route('/earth_orientation_parameters_pole_coordinates_x_y')
def earth_orientation_parameters_pole_coordinates_x_y():
    eop_pc()
    data_dict = load_data()
    return render_template('page4.html', data=data_dict)

@app.route('/noaa_swpc_sgarf')
def noaa_swpc_sgarf():
    sgarf_data = fetch_noaa_swpc_sgarf()
    return render_template('page5.html', sgarf_data=sgarf_data)

@app.route('/noaa_swpc_weekly')
def noaa_swpc_weekly():
    noaa_data = fetch_noaa_swpc_weekly()
    return render_template('page6.html', noaa_data=noaa_data)

@app.route('/quakemap')
def quakemap():
    return render_template('quakemap.html')

@app.route('/volcanomap')
def volcanomap():
    return render_template('volcanomap.html')

@app.route('/pvegraph')
def pvegraph():
    return render_template('pve.html')

@app.route('/minimum-depth-wise-earthquake-count-graph')
def mindepthquakecount():
    return render_template('mindepth.html')

@app.route('/polegraph')
def polegraph():
    return render_template('pole.html')

@app.route('/netCDFtoCSV')
def upload_netcdffile():
    return render_template('netCDFtoCSV.html')

@app.route('/datefill')
def datefill():
    return render_template('datefill.html')

@app.route('/delta-t')
def deltathtml():
    return render_template('deltat.html')

@app.route('/netcdfcsvmerge')
def netcdfcsv():
    return render_template('netcdfcsvmerge.html')

@app.route('/mquake')
def mquake():
    return render_template('mquake.html')

@app.route('/pole_coordinates_dailygraph')
def poledailygraph():
    return poledailygraphdataprocess()

@app.route('/merge_and_process_csv', methods=['POST'])
def merge_and_process_csv():
    return merge_and_process_csv_files_netcdf()

@app.route('/datefiller', methods=['POST'])
def datefiller():
    return datefiller_process()

@app.route('/csv_to_quake_summary', methods=['GET', 'POST'])
def csv_to_quake_summary():
    return upload_csvfile_to_quake_summary()

@app.route('/processnetcdf', methods=['POST'])
def processnetcdf():
    return Unidata_NetCDF_to_CSV()

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/robots.txt')
def serve_robots_txt():
    return send_from_directory(app.static_folder, 'robots.txt')

# Generate sitemap route
@app.route('/sitemap.xml', methods=['GET'])
def sitemap_xml():
    try:
        routes = [
            '/',
            '/earth_orientation_parameters_pole_coordinates_x_y',
            '/pole_coordinates_dailygraph',
            '/dynamic-earthquake-counts',
            '/dynamic-earthquake-energy',
            '/quakemap',
            '/volcanomap',
            '/pvegraph',
            '/minimum-depth-wise-earthquake-count-graph',
            '/polegraph',
            '/volcanic_reports',
            '/noaa_swpc_sgarf',
            '/noaa_swpc_weekly',
            '/csv_to_quake_summary',
            '/netCDFtoCSV',
            '/datefill',
            '/netcdfcsvmerge',
            '/mquake',
            '/privacy-policy'
        ]

        # Generate sitemap XML content
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'

        for route in routes:
            xml_content += f'\t<url>\n\t\t<loc>https://ogdp.in{route}</loc>\n\t</url>\n'

        xml_content += '</urlset>'

        response = make_response(xml_content)
        response.headers["Content-Type"] = "application/xml"

        return response

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=False)
