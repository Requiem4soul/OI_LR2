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
        self.output_dir = os.path.join("C:/Users/Ilya/Documents/GitHub/OI_LR2", self.name)
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
        """Анализ изображения с тремя новыми метриками"""
        hist = self.generate_histogram(self.gray)
        total_pixels = self.gray.size
        
        # 1. Метрика: Соотношение информации по тоновым зонам
        zone_size = 256 // 3
        zone1 = np.sum(hist[:zone_size])  # Тёмные тона (0-85)
        zone2 = np.sum(hist[zone_size:2*zone_size])  # Средние тона (85-170)
        zone3 = np.sum(hist[2*zone_size:])  # Светлые тона (170-255)
        
        # Нормируем значения и вычисляем соотношения
        zone_ratios = [
            zone1 / total_pixels,
            zone2 / total_pixels,
            zone3 / total_pixels
        ]
        
        # Оценка баланса (чем ближе к 1, тем равномернее распределение)
        balance_metric = min(zone_ratios) / max(zone_ratios)
        
        # 2. Метод оценки контраста (дельта L = Lmax - Lmin)
        # Игнорируем крайние значения с малым количеством пикселей
        threshold = total_pixels * 0.001  # 0.1% от общего числа пикселей
        l_min = np.where(hist > threshold)[0][0] if np.any(hist > threshold) else 0
        l_max = np.where(hist > threshold)[0][-1] if np.any(hist > threshold) else 255
        delta_L = l_max - l_min
        
        # 3. Дисперсия (мера разброса яркостей)
        mean_brightness = np.mean(self.gray)
        variance = np.var(self.gray)  # Собственно дисперсия
        std_dev = np.sqrt(variance)   # Стандартное отклонение
        
        # Дополнительная метрика: асимметрия распределения
        skewness = np.sum((np.arange(256) - mean_brightness)**3 * hist) / (total_pixels * std_dev**3)
        
        # Формируем отчёт
        report = f"""
        === Анализ изображения {self.name} ===
        [1] Соотношение тоновых зон:
            • Тёмные (0-85): {zone_ratios[0]:.2%}
            • Средние (85-170): {zone_ratios[1]:.2%}
            • Светлые (170-255): {zone_ratios[2]:.2%}
            • Баланс: {balance_metric:.2f} (1 = идеально)
        
        [2] Оценка контраста:
            • Lmin: {l_min} (первый значимый уровень)
            • Lmax: {l_max} (последний значимый уровень)
            • ΔL (диапазон): {delta_L} (чем больше, тем лучше)
        
        [3] Дисперсия и распределение:
            • Дисперсия: {variance:.1f}
            • Стандартное отклонение: {std_dev:.1f}
            • Асимметрия: {skewness:.2f} (>0 - перекос в тени, <0 - в светах, 0 - идеал)
        """
        
        # Сохраняем данные в файл
        metrics = {
            "zone_ratios": zone_ratios,
            "balance_metric": balance_metric,
            "delta_L": delta_L,
            "variance": variance,
            "skewness": skewness
        }
        
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
    ("OI_LR2/First.jpg", 1.5),
    ("OI_LR2/Second.jpg", 0.8),
    ("OI_LR2/Third.jpg", 0.8),
    ("OI_LR2/Fourth.jpg", 0.8),
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

        # 1. Менее агрессивное CLAHE (clip_limit=3.0 вместо 4.0)
        clahe_img = processor.apply_clahe(processor.gray, clip_limit=3.0, tile_grid_size=(16,16))

        # 2. Лёгкая гамма-коррекция (0.8 вместо 0.6)
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