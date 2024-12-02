import os
import pandas as pd
import shutil

def summary4csv(main_directory, output_csv):
    # Шлях до файлу summary_filtered_sorted.csv
    output_csv_path = os.path.join(main_directory, output_csv)

    # Перевірка чи існує файл summary_filtered_sorted.csv, якщо так то видаляємо
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    summary_data = {}

    # Перевірка кожної підпапки
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            # Пошук CSV файлу у підпапці
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            if len(csv_files) != 1:
                continue
            
            # Читання CSV файлу
            csv_path = os.path.join(subdir_path, csv_files[0])
            data = pd.read_csv(csv_path)
            
            # Перетворення стовпця `score` у числовий формат
            data['score'] = data['score'].str.rstrip('%').astype(float)
            
            # Сумування значень стовпця `score` за іменами зображень
            for _, row in data.iterrows():
                image = row['image']
                score = row['score']
                if image in summary_data:
                    summary_data[image] += score
                else:
                    summary_data[image] = score

    # Перетворення результатів у DataFrame
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['image', 'score'])

    # Відфільтрувати лише ті рядки, де `image` не є числом
    summary_df = summary_df[~summary_df['image'].str.isnumeric()]

    # Сортування за зменшенням значення `score`
    summary_df = summary_df.sort_values(by='score', ascending=False)

    # Додавання символу `%` до значень `score`
    summary_df['score'] = summary_df['score'].apply(lambda x: f'{x:.3f}%')

    # Збереження результатів у новий CSV файл
    summary_df.to_csv(output_csv_path, index=False)

    # Видалення усіх каталогів у папці main_directory
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path):
            shutil.rmtree(subdir_path)