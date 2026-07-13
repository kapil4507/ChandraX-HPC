import sys
import os

def crop_patch(pgm_path, out_path, size=512):
    if not os.path.exists(pgm_path):
        print(f"Error: {pgm_path} does not exist.")
        return
        
    print(f"Reading {pgm_path}...")
    with open(pgm_path, 'rb') as f:
        # Parse PGM header
        header = f.readline().decode('utf-8').strip()
        if header != 'P5':
            print("Error: Input must be a binary PGM (P5) image.")
            return
            
        # Skip comments
        line = f.readline().decode('utf-8').strip()
        while line.startswith('#'):
            line = f.readline().decode('utf-8').strip()
            
        width, height = map(int, line.split())
        max_val = int(f.readline().decode('utf-8').strip())
        
        print(f"PGM Dimensions: {width} x {height}, Max val: {max_val}")
        
        # Read pixel data
        pixels = f.read()
        if len(pixels) != width * height:
            print(f"Error: Pixel data size mismatch. Expected {width * height}, got {len(pixels)}.")
            return
            
    # Crop a patch from the center of the image
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    
    # Keep boundaries safe
    start_x = max(0, min(start_x, width - size))
    start_y = max(0, min(start_y, height - size))
    
    print(f"Cropping a {size}x{size} patch starting at (x={start_x}, y={start_y})...")
    
    patch_pixels = bytearray()
    for y in range(start_y, start_y + size):
        row_offset = y * width
        patch_pixels.extend(pixels[row_offset + start_x : row_offset + start_x + size])
        
    # Write output PGM patch
    with open(out_path, 'wb') as f:
        f.write(f"P5\n{size} {size}\n255\n".encode('utf-8'))
        f.write(patch_pixels)
        
    print(f"Successfully saved cropped patch to {out_path}")

if __name__ == '__main__':
    inp = "data/dfsar_focused_lunar_surface.pgm"
    out = "data/dfsar_lunar_patch.pgm"
    if len(sys.argv) > 1:
        inp = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    crop_patch(inp, out)
