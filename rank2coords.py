import os
import pandas as pd

def read_coordinates_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        coordinates = []
        for line in lines[1:]:  # Пропускаємо перший рядок
            try:
                lon, lat = map(float, line.strip().split(','))
                coordinates.append((lat, lon))
            except ValueError:
                continue
        return coordinates

def convert_to_dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return d, m, sd

def format_coordinates(lat, lon):
    lat_d, lat_m, lat_s = convert_to_dms(lat)
    lon_d, lon_m, lon_s = convert_to_dms(lon)
    
    lat_direction = 'N' if lat >= 0 else 'S'
    lon_direction = 'E' if lon >= 0 else 'W'
    
    formatted_lat = f"{abs(lat_d)}° {lat_m}' {lat_s:.4f}\" {lat_direction}"
    formatted_lon = f"{abs(lon_d)}° {lon_m}' {lon_s:.4f}\" {lon_direction}"
    
    return formatted_lat, formatted_lon

def calculate_center(coordinates):
    avg_lat = sum(lat for lat, lon in coordinates) / len(coordinates)
    avg_lon = sum(lon for lat, lon in coordinates) / len(coordinates)
    return avg_lat, avg_lon

def rank2coords(csv_path, img_folder_path):
    # Зчитуємо файл CSV
    df = pd.read_csv(csv_path)

    # Перетворюємо стовпець score у числовий формат (без %)
    df['score'] = df['score'].str.rstrip('%').astype(float)

    # Знаходимо рядок із максимальним значенням score
    max_row = df.loc[df['score'].idxmax()]
    max_image = max_row['image']

    # Формуємо шлях до файлу .dat
    dat_file_path = os.path.join(img_folder_path, f"{max_image}.dat")

    # Перевіряємо, чи існує файл
    if os.path.exists(dat_file_path):
        # Читаємо вміст файлу .dat
        coordinates = read_coordinates_from_file(dat_file_path)
        center_lat, center_lon = calculate_center(coordinates)
        formatted_lat, formatted_lon = format_coordinates(center_lat, center_lon)
        return formatted_lat, formatted_lon
    else:
        print("Файл не знайдено за вказаним шляхом.")