import crop
from pathlib import Path

# Функція photo2coords виконує весь процес від обрізання фотографії до отримання координат
def photo2coords(frames_dir, superpoints_dir, rank_dir, csv_name, map_dir):
    import img2superpoint
    import superpoints2rank
    import summary4csv
    import rank2coords

    # Створюємо файл з ключовими точками для кожного кадру
    img2superpoint.img2superpoint(frames_dir, superpoints_dir)

    # Створюємо файл з рангами для кожного кадру
    superpoints2rank.superpoints2rank(superpoints_dir, rank_dir)

    # Створюємо файл з сумою рангів кожної групи
    summary4csv.summary4csv(rank_dir, csv_name)

    # Отримуємо координати тайлу з найвищим рангом співпадіння
    formatted_lat, formatted_lon = rank2coords.rank2coords(str(rank_dir)+ "/" +csv_name, map_dir)

    return formatted_lat, formatted_lon

# Список фото з дрона для тестування
images = [
    './camera/Bing1.png',
    './camera/Bing2.png',
    './camera/Google1.png',
    './camera/Google2.png',
    './camera/Here1.png',
    './camera/Here2.png'
]

# Директорії для зберігання проміжних результатів
frames_dir = './img/frames'
superpoints_dir = './data/frame_superpoints'
rank_dir = './data/ranks'
map_dir = './img'

# Обробка кожного зображення
for idx, image in enumerate(images, 1): 
    print(f'\t[{idx}/{len(images)}] - Processing {Path(image).stem}')

    # Визначення імені файлу CSV
    csv_name = Path(image).stem + '.csv'

    # Розбиття фото на кадри та їх первинна обробка
    crop.crop(image, frames_dir)

    # Отримання координат
    x, y = photo2coords(frames_dir, superpoints_dir, rank_dir, csv_name, map_dir)
    print(f'\tcoords of {Path(image).stem}: {x}, {y}')

    