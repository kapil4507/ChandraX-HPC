import os
import sys
import numpy as np
from PIL import Image

# Disable PIL decompression bomb warning since we are handling very large images
Image.MAX_IMAGE_PIXELS = None

def read_pgm(filename):
    """Reads binary PGM (P5) image into numpy array."""
    with open(filename, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header != 'P5':
            raise ValueError("Only PGM binary format (P5) is supported")
        
        # Skip comments
        line = f.readline().decode('utf-8').strip()
        while line.startswith('#'):
            line = f.readline().decode('utf-8').strip()
            
        width, height = map(int, line.split())
        max_val = int(f.readline().decode('utf-8').strip())
        
        data = np.fromfile(f, dtype=np.uint8)
        data = data[:width * height]
        return data.reshape((height, width))

def main():
    pgm_path = "data/dfsar_focused_lunar_surface.pgm"
    if len(sys.argv) > 1:
        pgm_path = sys.argv[1]
        
    if not os.path.exists(pgm_path):
        print(f"Error: {pgm_path} not found.")
        return

    print(f"Reading massive image: {pgm_path}...")
    try:
        img_data = read_pgm(pgm_path)
    except Exception as e:
        print(f"Failed to read PGM: {e}")
        return

    height, width = img_data.shape
    print(f"Original dimensions: {width} wide x {height} tall")
    
    # 1. Downsample (azimuth decimation)
    # Taking every 32nd row brings the height down to ~15,934 pixels,
    # which fits within GPU texture limits (typically 16k or 32k max).
    decimation = 32
    print(f"Downsampling along height (azimuth) by taking every {decimation}nd line...")
    downsampled_data = img_data[::decimation, :]
    ds_height, ds_width = downsampled_data.shape
    print(f"Downsampled dimensions: {ds_width}x{ds_height}")
    
    try:
        ds_img = Image.fromarray(downsampled_data)
        ds_out = "data/dfsar_lunar_downsampled.png"
        ds_img.save(ds_out)
        print(f"Successfully saved downsampled overview: {ds_out}")
    except Exception as e:
        print(f"Failed to save downsampled image: {e}")

    # 2. Crop a 1024x1024 focused segment
    # This keeps full resolution of a smaller area so you can inspect craters closely.
    crop_size = 1024
    if height > crop_size:
        start_y = height // 2
        end_y = start_y + crop_size
        print(f"Cropping a {crop_size}x{crop_size} segment from row {start_y} to {end_y}...")
        cropped_data = img_data[start_y:end_y, :]
        
        try:
            crop_img = Image.fromarray(cropped_data)
            crop_out = "data/dfsar_lunar_crop.png"
            crop_img.save(crop_out)
            print(f"Successfully saved full-resolution crop: {crop_out}")
        except Exception as e:
            print(f"Failed to save cropped image: {e}")

if __name__ == "__main__":
    main()
