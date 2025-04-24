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
        self.output_dir = os.path.join("results", self.name)
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


# === Обработка всех изображений по очереди ===
image_info = [
    ("OI_EA/First.jpg", 1.5),
    ("OI_EA/Second.jpg", 0.8),
    ("OI_EA/Third.jpg", 0.8),
    ("OI_EA/Fourth.jpg", 0.8),
]

# Обрабатываем каждое изображение и показываем результаты
for img_path, gamma in image_info:
    processor = ImageProcessor(img_path, gamma)
    processor.plot_results()