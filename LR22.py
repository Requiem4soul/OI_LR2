import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageProcessor:
    def __init__(self, image_path, gamma=1.0, clip_limit=2.0, tile_grid_size=(8,8)):
        self.image_path = image_path
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
        self.name = os.path.splitext(os.path.basename(image_path))[0]
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "results", self.name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.original = self._load_image()
        self.gray = self._to_grayscale(self.original)

        # Обработанные изображения
        self.equalized = self.equalize_histogram(self.gray)
        self.stretched = self.contrast_stretching(self.gray)
        self.gamma_corrected = self.gamma_correction(self.gray, self.gamma)
        self.clahe_corrected = self.clahe_enhancement(self.gray)

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
    
    def contrast_stretching(self, img):
        procent = 5
        low = np.percentile(img, procent)
        high = np.percentile(img, 100 - procent)
        if high == low:
            return img.copy()
        stretched = np.clip(img, low, high)
        stretched = ((stretched - low) / (high - low) * 255).astype(np.uint8)
        return stretched

    def gamma_correction(self, img, gamma):
        norm_img = img / 255.0
        corrected = np.power(norm_img, gamma)
        return (corrected * 255).astype(np.uint8)

    def clahe_enhancement(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(img)

    def save_image(self, img, name):
        path = os.path.join(self.output_dir, name)
        cv2.imwrite(path, img)
        
    def analyze_image(self, img):
        hist = self.generate_histogram(img)
        total_pixels = img.size

        zone_size = 256 // 3
        zone1 = np.sum(hist[:zone_size])
        zone2 = np.sum(hist[zone_size:2*zone_size])
        zone3 = np.sum(hist[2*zone_size:])

        zone_ratios = [
            zone1 / total_pixels,
            zone2 / total_pixels,
            zone3 / total_pixels
        ]
        balance_metric = min(zone_ratios) / max(zone_ratios)

        threshold = total_pixels * 0.001
        l_min = np.where(hist > threshold)[0][0] if np.any(hist > threshold) else 0
        l_max = np.where(hist > threshold)[0][-1] if np.any(hist > threshold) else 255
        delta_L = l_max - l_min

        levels = np.arange(256)
        mean = np.sum(levels * hist) / total_pixels
        variance = np.sum(((levels - mean) ** 2) * hist) / total_pixels
        std_dev = np.sqrt(variance)
        skewness = np.sum(((levels - mean) ** 3) * hist) / (total_pixels * std_dev**3)

        return {
            "zone_ratios": zone_ratios,
            "balance_metric": balance_metric,
            "delta_L": delta_L,
            "variance": variance,
            "skewness": skewness
        }

    def print_metrics(self, metrics, original=None):
        zone_ratios = metrics["zone_ratios"]
        print(f"  • Тёмные (0-85): {zone_ratios[0]:.2%}")
        print(f"  • Средние (85-170): {zone_ratios[1]:.2%}")
        print(f"  • Светлые (170-255): {zone_ratios[2]:.2%}")
        
        def format_diff(label, value, base, better_higher=True):
            diff = value - base
            
            # Особый случай для skewness: сравниваем абсолютные значения
            if label == "Асимметрия":
                abs_base = abs(base)
                abs_value = abs(value)
                if abs_value < abs_base:
                    return f"{label}: {value:.2f} — улучшилось на {abs_base - abs_value:.2f}"
                elif abs_value > abs_base:
                    return f"{label}: {value:.2f} — ухудшилось на {abs_value - abs_base:.2f}"
                else:
                    return f"{label}: {value:.2f} — не изменилось"
            
            # Обычная логика для других метрик
            if abs(diff) < 1e-2:
                return f"{label}: {value:.2f} — не изменилось"
            trend = "улучшилось" if (diff > 0) == better_higher else "ухудшилось"
            return f"{label}: {value:.2f} — {trend} на {abs(diff):.2f}"

        if original:
            print("  • " + format_diff("Баланс", metrics["balance_metric"], original["balance_metric"], better_higher=True))
            print("  • " + format_diff("ΔL", metrics["delta_L"], original["delta_L"], better_higher=True) + " (диапазон контраста)")
            print("  • " + format_diff("Асимметрия", metrics["skewness"], original["skewness"], better_higher=False))
        else:
            print(f"  • Баланс: {metrics['balance_metric']:.2f} (1 = идеально)")
            print(f"  • ΔL: {metrics['delta_L']} (диапазон контраста)")
            print(f"  • Асимметрия: {metrics['skewness']:.2f}")

    def track_metric_changes(self):
        print(f"\n=== Метрики для изображения: {self.name} ===\n")
        original_metrics = self.analyze_image(self.gray)
        print(f"Оригинал: {self.name}")
        self.print_metrics(original_metrics)

        processed_images = [
            ("Эквализация гистограммы", self.equalized),
            ("Растяжение контраста", self.stretched),
            ("Гамма-коррекция", self.gamma_corrected),
            ("CLAHE", self.clahe_corrected)
        ]

        for title, img in processed_images:
            print(f"\n{title}:")
            metrics = self.analyze_image(img)
            self.print_metrics(metrics, original_metrics)

    def plot_results(self):
        self.track_metric_changes()
        images = [
            ("Эквализация гистограммы", self.equalized, "equalized.png"),
            ("Растяжение контраста", self.stretched, "stretched.png"),
            ("Гамма-коррекция", self.gamma_corrected, "gamma_corrected.png"),
            ("CLAHE", self.clahe_corrected, "clahe_corrected.png"),
        ]

        plt.figure(figsize=(16, 10))

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

        x = np.arange(256)
        y_original = x
        y_gamma = np.power(x / 255.0, self.gamma) * 255

        plt.subplot(3, 2, 6)
        plt.plot(x, y_original, label="Исходная (линейная)", color='blue')
        plt.plot(x, y_gamma, label=f"Гамма = {self.gamma}", color='red')
        plt.title("Градационные кривые (гамма-коррекция)")
        plt.xlabel("Входное значение")
        plt.ylabel("Выходное значение")
        plt.legend()
        plt.grid(True)

        plt.suptitle(f"Обработка: {self.name}")
        plt.tight_layout()
        plt.show()


# === ОБРАБОТКА ВСЕХ ИЗОБРАЖЕНИЙ ===
image_info = [
    ("OI_LR2/First.jpg", 1.2, 2.0, (8,8)),
    ("OI_LR2/Second.jpg", 1.05, 2.0, (8,8)),
    ("OI_LR2/Third.jpg", 0.8, 2.0, (8,8)),
    ("OI_LR2/Fourth.jpg", 0.8, 2.0, (8,8)),
]

for img_path, gamma, clip_limit, tile_grid_size in image_info:
    processor = ImageProcessor(img_path, gamma=gamma, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    processor.plot_results()
