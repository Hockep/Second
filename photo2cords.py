import crop
import time
from pathlib import Path

def photo2coords(frames_dir, superpoints_dir, rank_dir, csv_name, map_dir):
    import img2superpoint
    import superpoints2rank
    import summary4csv
    import rank2coords

    img2superpoint.img2superpoint(frames_dir, superpoints_dir)
    superpoints2rank.superpoints2rank(superpoints_dir, rank_dir)
    summary4csv.summary4csv(rank_dir, csv_name)
    formatted_lat, formatted_lon = rank2coords.rank2coords(str(rank_dir)+ "/" +csv_name, map_dir)
    return formatted_lat, formatted_lon

program_start_time = time.time()
print(f'[DEBUG] Start time: {time.ctime()}\n')
images = [
    './camera/Bing1.png',
    './camera/Bing2.png',
    './camera/Google1.png',
    './camera/Google2.png',
    './camera/Here1.png',
    './camera/Here2.png'
]
frames_dir = './img/frames'
superpoints_dir = './data/frame_superpoints'
rank_dir = './data/ranks'
map_dir = './img'


for idx, image in enumerate(images, 1):  # idx починається з 1
    print(f'\t[DEBUG] {idx}/{len(images)} - Processing {Path(image).stem}')
    loop_time = time.time()
    csv_name = Path(image).stem + '.csv'
    crop.crop(image, frames_dir)
    x, y = photo2coords(frames_dir, superpoints_dir, rank_dir, csv_name, map_dir)
    print(f'\tcoords of {Path(image).stem}: {x}, {y}')
    print(f'\tprocessed in {time.time() - loop_time} sec\n')

print(f'[DEBUG] Total execution time: {time.time() - program_start_time} sec')