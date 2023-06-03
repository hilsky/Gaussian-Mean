import cv2
import os

# Fungsi untuk evaluasi kualitatif
def evaluate_qualitative(original_image, filtered_image_gaussian, filtered_image_mean, image_index):
    output_folder = "Result"
    os.makedirs(output_folder, exist_ok=True)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Filtered Image (Gaussian)", filtered_image_gaussian)
    cv2.imshow("Filtered Image (Mean)", filtered_image_mean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Menyimpan gambar hasil filter Gaussian
    gaussian_image_path = os.path.join(output_folder, f"GaussianImage_{image_index}.jpg")
    cv2.imwrite(gaussian_image_path, filtered_image_gaussian)
    print(f"Gambar hasil filter Gaussian {image_index} disimpan di: {gaussian_image_path}")

    # Menyimpan gambar hasil filter Mean
    mean_image_path = os.path.join(output_folder, f"MeanImage_{image_index}.jpg")
    cv2.imwrite(mean_image_path, filtered_image_mean)
    print(f"Gambar hasil filter Mean {image_index} disimpan di: {mean_image_path}")

# Path ke direktori gambar
image_dir = "Dataset"

# List semua file gambar dalam direktori
image_files = [file for file in os.listdir(image_dir) if file.endswith((".jpg", ".png", ".jpeg"))]

# Loop melalui setiap file gambar
for index, file in enumerate(image_files):
    # Path lengkap file gambar
    image_path = os.path.join(image_dir, file)

    # Baca gambar menggunakan imread
    original_image = cv2.imread(image_path)

    # Periksa apakah gambar berhasil dibaca
    if original_image is not None:
        # Lakukan pengolahan gambar dengan metode filter Gaussian
        filtered_image_gaussian = cv2.GaussianBlur(original_image, (5, 5), 0)

        # Lakukan pengolahan gambar dengan metode filter mean
        filtered_image_mean = cv2.blur(original_image, (5, 5))

        # Evaluasi kualitatif
        evaluate_qualitative(original_image, filtered_image_gaussian, filtered_image_mean, index)
    else:
        print(f"Gagal membaca gambar: {image_path}")


