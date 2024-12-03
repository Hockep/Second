import filter_map
import img2superpoint


# Директоріх для зберігання файлів
map_dir = './img'
map_superpoints_dir = './data'

# Створюємо файли з ключовими точками для кожного кадру
filter_map.filter_map(map_dir)
img2superpoint.img2superpoint(map_dir, map_superpoints_dir)
