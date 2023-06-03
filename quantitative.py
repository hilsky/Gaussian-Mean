import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def resize_image(image, size):
    return cv2.resize(image, (size, size))

def evaluate_quantitative(original_image, filtered_image):
    mse = mean_squared_error(original_image, filtered_image)
    psnr = peak_signal_noise_ratio(original_image, filtered_image)
    ssim = structural_similarity(original_image, filtered_image, multichannel=True)
    return mse, psnr, ssim

# Load original image
original_image_path = "Dataset/Image5.jpg"
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print(f"Failed to read original image: {original_image_path}")
else:
    # Load filtered images (Gaussian and Mean)
    filtered_image_gaussian_path = "Dataset/Image5.jpg"
    filtered_image_gaussian = cv2.imread(filtered_image_gaussian_path, cv2.IMREAD_GRAYSCALE)

    filtered_image_mean_path = "Dataset/Image5.jpg"
    filtered_image_mean = cv2.imread(filtered_image_mean_path, cv2.IMREAD_GRAYSCALE)

    if filtered_image_gaussian is None:
        print(f"Failed to read filtered image (Gaussian): {filtered_image_gaussian_path}")

    if filtered_image_mean is None:
        print(f"Failed to read filtered image (Mean): {filtered_image_mean_path}")

    if original_image is not None and filtered_image_gaussian is not None and filtered_image_mean is not None:
        # Perform quantitative evaluation for Gaussian filter
        mse_gaussian, psnr_gaussian, ssim_gaussian = evaluate_quantitative(original_image, filtered_image_gaussian)

        # Perform quantitative evaluation for Mean filter
        mse_mean, psnr_mean, ssim_mean = evaluate_quantitative(original_image, filtered_image_mean)

        # Create a bar plot
        metrics = ['MSE', 'PSNR', 'SSIM']
        gaussian_values = [mse_gaussian, psnr_gaussian, ssim_gaussian]
        mean_values = [mse_mean, psnr_mean, ssim_mean]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, gaussian_values, width, label='Gaussian Filter')
        rects2 = ax.bar(x + width/2, mean_values, width, label='Mean Filter')

        ax.set_ylabel('Metric Value')
        ax.set_title('Comparison of Quantitative Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Add the metric values as annotations on top of the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.show()
    else:
        print("Failed to read one or more filtered images.")
