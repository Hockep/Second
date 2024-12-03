import os
import cv2

def crop(input_path, output_dir):
    # Перевіряємо, чи існує вихідний каталог, якщо ні - створюємо його
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Якщо каталог існує, видаляємо всі файли в ньому
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Завантажуємо зображення у відтінках сірого
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    smaller_side = min(width, height)  # Знаходимо меншу сторону зображення
    larger_side = max(width, height)  # Знаходимо більшу сторону зображення

    # Обчислюємо крок для обрізки
    step = (larger_side - smaller_side) // 2  

    # Обрізаємо зображення на три частини
    for i, offset in enumerate([0, step, larger_side - smaller_side]):
        if width > height:
            left = offset
            upper = 0
        else:
            left = 0
            upper = offset

        right = left + smaller_side
        lower = upper + smaller_side

        # Обрізаємо зображення
        cropped_img = img[upper:lower, left:right]

        # Видаляємо шум за допомогою медіанного фільтра
        cropped_img = cv2.medianBlur(cropped_img, 3)

        # Регулюємо яскравість та контрастність
        cropped_img = cv2.convertScaleAbs(cropped_img, alpha=1.5, beta=20)

        # Зберігаємо обрізане зображення у вихідний каталог
        output_file = os.path.join(output_dir, f"{i + 1}.png")
        cv2.imwrite(output_file, cropped_img)

crop('./camera/600.png', './c')  # Виклик функції crop для тестування