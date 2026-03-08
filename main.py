import numpy as np
from PIL import Image

def extract_pixles(image_path):

    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    height, width, _ = image_array.shape
    x_coods, y_coods = np.meshgrid(np.arange(width), np.arange(height))

    rgb_flat = image_array.reshape(-1, 3)
    x_flat = x_coods.reshape(-1,1)
    y_flat = y_coods.reshape(-1,1)

    print(rgb_flat.shape, x_flat.shape, y_flat.shape)

    pixel_data = np.hstack((x_flat, y_flat, rgb_flat))

    return pixel_data

def main():

    src  = "sample/src.jpg"
    dest = "sample/dest.jpg"

    src_pixels = extract_pixles(src)
    dest_pixels = extract_pixles(dest)

    print("Source Pixels:")
    print(src_pixels)
    print("\nDestination Pixels:")
    print(dest_pixels)


if __name__ == "__main__":
    main()
