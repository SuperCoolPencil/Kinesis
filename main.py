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
    
    rgb_normalized = pixel_data[:, 2:5] / 255.0
    weights = np.array([0.2126, 0.7152, 0.0722])

    luminance = np.dot(rgb_normalized, weights)

    pixel_data = np.column_stack((pixel_data, luminance))

    sort_indices = np.argsort(pixel_data[:, 5])

    return pixel_data, sort_indices

def main():

    src  = "sample/src.jpg"
    dest = "sample/dest.jpg"

    src_pixels, src_sorted = extract_pixles(src)
    dest_pixels, dest_sorted = extract_pixles(dest)

    sorted_src = src_pixels[src_sorted]
    sorted_dest = dest_pixels[dest_sorted]

    # save new image 
    # Grab max X and Y
    dest_width = int(np.max(dest_pixels[:, 0])) + 1
    dest_height = int(np.max(dest_pixels[:, 1])) + 1
    

    # blank
    output_canvas = np.zeros((dest_height, dest_width, 3), dtype=np.uint8)

    dest_x = sorted_dest[:, 0].astype(int)
    dest_y = sorted_dest[:, 1].astype(int)
    
    src_rgb = sorted_src[:, 2:5].astype(np.uint8)

    output_canvas[dest_y, dest_x] = src_rgb

    final_image = Image.fromarray(output_canvas)
    final_image.save("sample/rearranged_output.jpg")
    print("Done! Image saved as rearranged_output.jpg")

if __name__ == "__main__":
    main()