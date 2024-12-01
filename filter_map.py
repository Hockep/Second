import os
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2

def is_black_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return np.all(image_array == 0)

def has_transparent_pixels(image_path):
    image = Image.open(image_path).convert("RGBA")
    image_array = np.array(image)
    transparent_pixels = np.sum(image_array[:, :, 3] < 255)
    total_pixels = image_array.shape[0] * image_array.shape[1]
    return transparent_pixels / total_pixels

def process_image(filename, directory):
    print(f"Checking: {filename}")
    if filename.endswith('.png'):
        png_path = os.path.join(directory, filename)
        dat_path = os.path.join(directory, filename.replace('.png', '.dat'))

        if is_black_image(png_path) or has_transparent_pixels(png_path) > 0.33:
            if os.path.exists(png_path):
                os.remove(png_path)
            if os.path.exists(dat_path):
                os.remove(dat_path)
            print(f"Deleted: {png_path} and {dat_path}")
        else:
            # Перевести зображення в ЧБ
            image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            
            # Видалити шум
            image = cv2.medianBlur(image, 3)
            
            # Відрегулювати яскравість та контрастність
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
            
            # Зберегти результат
            cv2.imwrite(png_path, image)
            print(f"Processed and saved: {png_path}")

def delete_black_images_and_pairs(directory):
    with ThreadPoolExecutor() as executor:
        filenames = os.listdir(directory)
        futures = [executor.submit(process_image, filename, directory) for filename in filenames]
        for future in futures:
            future.result()

delete_black_images_and_pairs('./img')
