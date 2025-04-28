import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_image_with_histogram(image_path):
    """Загружает изображение и отображает его с гистограммой"""
    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден!")
        return
    
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
    # Создание фигуры
    plt.figure(figsize=(12, 6))
    
    # Отображение изображения
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Изображение: {os.path.basename(image_path)}")
    plt.axis('off')
    
    # Вычисление и отображение гистограммы
    plt.subplot(1, 2, 2)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.bar(range(256), hist.ravel(), width=1.0)
    plt.title("Гистограмма")
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.xlim([0, 255])
    
    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Путь к обработанному изображению (измените на свой)
    image_path = "OI_LR2\First.jpg"
    
    # Запуск визуализации
    load_and_plot_image_with_histogram(image_path)
    
    # Для удобства можно добавить интерактивный ввод:
    user_path = input("Введите путь к изображению (или нажмите Enter для стандартного пути): ")
    if user_path.strip():
        load_and_plot_image_with_histogram(user_path)