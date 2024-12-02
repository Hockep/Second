import os
import pandas as pd

def read_coordinates_from_file(filename):
    # Відкриваємо файл для читання
    with open(filename, 'r') as file:
        lines = file.readlines()
        coordinates = []
        # Пропускаємо перший рядок (заголовок) і обробляємо кожен наступний рядок
        for line in lines[1:]:
            try:
                # Розділяємо рядок на довготу і широту та перетворюємо їх у числа з плаваючою комою
                lon, lat = map(float, line.strip().split(','))
                # Додаємо координати до списку
                coordinates.append((lat, lon))
            except ValueError:
                # Пропускаємо рядки, які не вдалося перетворити
                continue
        return coordinates

def convert_to_dms(deg):
    # Розбиваємо значення на градуси, хвилини та секунди
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return d, m, sd

def format_coordinates(lat, lon):
    # Форматуємо широту та довготу у формат DMS (градуси, хвилини, секунди)
    lat_d, lat_m, lat_s = convert_to_dms(lat)
    lon_d, lon_m, lon_s = convert_to_dms(lon)
    
    # Визначаємо напрямок (північ/південь для широти, схід/захід для довготи)
    lat_direction = 'N' if lat >= 0 else 'S'
    lon_direction = 'E' if lon >= 0 else 'W'
    
    # Форматуємо координати у рядки
    formatted_lat = f"{abs(lat_d)}° {lat_m}' {lat_s:.4f}\" {lat_direction}"
    formatted_lon = f"{abs(lon_d)}° {lon_m}' {lon_s:.4f}\" {lon_direction}"
    
    return formatted_lat, formatted_lon

def calculate_center(coordinates):
    # Обчислюємо середнє значення широти та довготи
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
    
    dat_file_path = os.path.join(img_folder_path, f"{max_image}.dat")

    # Перевіряємо, чи існує файл
    if os.path.exists(dat_file_path):
        # Читаємо вміст файлу .dat
        coordinates = read_coordinates_from_file(dat_file_path)
        # Обчислюємо центр координат
        center_lat, center_lon = calculate_center(coordinates)
        # Форматуємо координати у формат DMS
        formatted_lat, formatted_lon = format_coordinates(center_lat, center_lon)
        return formatted_lat, formatted_lon
    else:
        # Виводимо повідомлення, якщо файл не знайдено
        print("Файл не знайдено за вказаним шляхом.")