import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageProcessor:
    def __init__(self, image_path, gamma):
        self.image_path = image_path
        self.gamma = gamma
        self.name = os.path.splitext(os.path.basename(image_path))[0]
        self.output_dir = os.path.join("C:/VisualCode/OI_EA/results", self.name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.original = self._load_image()
        self.gray = self._to_grayscale(self.original)

        self.equalized = self.equalize_histogram(self.gray)
        self.clahe = self.apply_clahe(self.gray)
        self.gamma_corrected = self.gamma_correction(self.gray, self.gamma)

    def _load_image(self):
        img = mpimg.imread(self.image_path)
        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).astype(np.uint8)
        return img

    def _to_grayscale(self, img):
        if len(img.shape) == 3:
            return np.mean(img, axis=-1).astype(np.uint8)
        return img.astype(np.uint8)

    def generate_histogram(self, img):
        hist = np.zeros(256, dtype=int)
        for pixel in img.flatten():
            hist[pixel] += 1
        return hist

    def equalize_histogram(self, img):
        hist = self.generate_histogram(img)
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        return cdf_normalized[img].astype(np.uint8)

    def apply_clahe(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    def gamma_correction(self, img, gamma):
        norm_img = img / 255.0
        corrected = np.power(norm_img, gamma)
        return (corrected * 255).astype(np.uint8)

    def save_image(self, img, name):
        path = os.path.join(self.output_dir, name)
        cv2.imwrite(path, img)

    def save_histogram_data(self, hist, filename):
        """Сохраняет гистограмму в текстовый файл."""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            for i, count in enumerate(hist):
                f.write(f"{i}\t{count}\n")

    def analyze_image(self):
        # Анализ гистограммы оригинального изображения
        hist = self.generate_histogram(self.gray)
        total_pixels = self.gray.size
        mean_brightness = np.mean(self.gray)
        std_dev = np.std(self.gray)  # Мера контраста

        # Определяем перекос гистограммы
        skew = np.sum((np.arange(256) - mean_brightness)**3 * hist) / (total_pixels * std_dev**3)
        
        # Оценка типа изображения
        if mean_brightness < 60:
            brightness_comment = "Тёмное (недоэкспонированное)"
        elif mean_brightness > 200:
            brightness_comment = "Светлое (переэкспонированное)"
        else:
            brightness_comment = "Нормальная экспозиция"

        if std_dev < 40:
            contrast_comment = "Низкий контраст"
        elif std_dev > 80:
            contrast_comment = "Высокий контраст"
        else:
            contrast_comment = "Умеренный контраст"

        if skew > 1:
            skew_comment = "Перекос в тени"
        elif skew < -1:
            skew_comment = "Перекос в светлые области"
        else:
            skew_comment = "Сбалансированная гистограмма"

        # Сохраняем гистограмму в файл
        self.save_histogram_data(hist, "histogram_original.txt")

        # Формируем отчёт
        report = f"""
        === Анализ изображения {self.name} ===
        - Средняя яркость: {mean_brightness:.1f} ({brightness_comment})
        - Стандартное отклонение (контраст): {std_dev:.1f} ({contrast_comment})
        - Перекос гистограммы: {skew:.2f} ({skew_comment})
        - Рекомендуемые методы:
            * Текущая гамма-коррекция (γ={self.gamma}): {'затемнение' if self.gamma > 1 else 'осветление'}
            * {'Гистограммная эквализация (улучшит тени)' if skew > 0.5 else 'CLAHE (если есть локальные перепады)'}
            * {'Логарифмическое преобразование (если очень тёмное)' if mean_brightness < 50 else ''}
        """
        print(report)
        return report

    def plot_results(self):
        images = [
            ("Оригинал", self.gray, "original.png"),
            ("Эквализация", self.equalized, "equalized.png"),
            ("CLAHE", self.clahe, "clahe.png"),
            ("Гамма-коррекция", self.gamma_corrected, "gamma_corrected.png"),
        ]

        plt.figure(figsize=(16, 10))

        # Отображение изображений и гистограмм
        for i, (title, img, filename) in enumerate(images):
            plt.subplot(3, 4, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
            self.save_image(img, filename)

            hist = self.generate_histogram(img)
            plt.subplot(3, 4, i + 5)
            plt.bar(range(256), hist, width=1.0)
            plt.title(f"Гистограмма: {title}")
            plt.xlim([0, 255])

            # Сохраняем гистограмму в файл
            self.save_histogram_data(hist, f"histogram_{filename.split('.')[0]}.txt")

        # Градационные кривые: до и после гамма-коррекции
        x = np.arange(256)
        y_original = x
        y_gamma = np.power(x / 255.0, self.gamma) * 255

        plt.subplot(3, 2, 6)
        plt.plot(x, y_original, label="Исходная (линейная)", color='blue')
        plt.plot(x, y_gamma, label=f"Гамма = {self.gamma}", color='red')
        plt.title("Градационная кривая (гамма-коррекция)")
        plt.xlabel("Входное значение")
        plt.ylabel("Выходное значение")
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Обработка: {self.name}")
        plt.tight_layout()
        plt.show()
        
        self.analyze_image()


# === Обработка всех изображений по очереди ===
image_info = [
    ("OI_EA/First.jpg", 1.5),
    ("OI_EA/Second.jpg", 0.8),
    ("OI_EA/Third.jpg", 0.8),
    ("OI_EA/Fourth.jpg", 0.8),
]

for img_path, gamma in image_info:
    processor = ImageProcessor(img_path, gamma)
    
    # Специальная обработка для разных изображений
    if "First.jpg" in img_path:
        # Обработка для первого изображения
        clahe_img = processor.apply_clahe(processor.gray, clip_limit=1.5)
        combined_img = processor.gamma_correction(clahe_img, 0.7)
        processor.save_image(combined_img, "Final.png")
        processor.save_image(clahe_img, "intermediate_clahe.png")
        
    elif "Second.jpg" in img_path:
        # Оптимизированная обработка для второго изображения
        # 1. Более мягкое уменьшение яркости (0.9 вместо 0.85)
        adjusted = np.clip(processor.gray.astype('float32') * 0.9, 0, 255).astype('uint8')

        # 2. Менее агрессивное CLAHE (clip_limit=3.0 вместо 4.0)
        clahe_img = processor.apply_clahe(adjusted, clip_limit=3.0, tile_grid_size=(16,16))

        # 3. Лёгкая гамма-коррекция (0.8 вместо 0.6)
        corrected = processor.gamma_correction(clahe_img, 0.8)
        processor.save_image(corrected, "Final.png")
        
    elif "Third.jpg" in img_path:
        # 1. Мягкое осветление через гамма-коррекцию (вместо логарифма)
        gamma_corrected = processor.gamma_correction(processor.gray, 0.7)  # Сильное осветление
    
        # 2. Адаптивное выравнивание гистограммы с ограничением
        clahe_img = processor.apply_clahe(
            gamma_corrected,
            clip_limit=1.0,  # Очень мягкое ограничение
            tile_grid_size=(24,24)  # Крупные тайлы для плавности
        )
    
        processor.save_image(clahe_img, "Final.png")
    
    elif "Fourth.jpg" in img_path:
        # 1. Применим CLAHE с обычными параметрами
        clahe_img = processor.apply_clahe(
            processor.gray,
            clip_limit=2.0,
            tile_grid_size=(16, 16)
        )

        # 2. Применим осветляющую гамма-коррекцию
        gamma_corrected = processor.gamma_correction(clahe_img, 0.7)

        # 3. Сохраняем результат
        processor.save_image(gamma_corrected, "Final.png")
    
    # Стандартная обработка для всех изображений
    processor.plot_results()