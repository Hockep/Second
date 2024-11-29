import os
from PIL import Image

def crop(input_path, output_dir):
    # Очистити вихідну директорію
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Відкрити вхідне зображення
    with Image.open(input_path) as img:
        width, height = img.size
        smaller_side = min(width, height)  # Менша сторона
        larger_side = max(width, height)

        # Визначити кількість зображень (3)
        step = (larger_side - smaller_side) // 2  # Відстань для зміщення

        for i, offset in enumerate([0, step, larger_side - smaller_side]):
            if width > height:
                # Зміщення вздовж ширини
                left = offset
                upper = 0
            else:
                # Зміщення вздовж висоти
                left = 0
                upper = offset

            right = left + smaller_side
            lower = upper + smaller_side

            # Вирізати квадрат
            cropped_img = img.crop((left, upper, right, lower))

            # Перетворити на чорно-біле
            cropped_img = cropped_img.convert('L')

            # Зберегти квадрат як окремий файл
            output_file = os.path.join(output_dir, f"{i + 1}.png")
            cropped_img.save(output_file)