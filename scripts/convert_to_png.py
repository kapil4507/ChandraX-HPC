import sys
from PIL import Image

def main():
    pgm_path = "data/dfsar_focused_lunar_surface.pgm"
    png_path = "data/dfsar_focused_lunar_surface.png"
    
    if len(sys.argv) > 1:
        pgm_path = sys.argv[1]
    if len(sys.argv) > 2:
        png_path = sys.argv[2]
        
    try:
        print(f"Opening PGM file: {pgm_path}...")
        img = Image.open(pgm_path)
        print(f"Saving as PNG file: {png_path}...")
        img.save(png_path)
        print("Conversion successful!")
    except ImportError:
        print("Error: PIL (Pillow) is not installed. Run 'pip install Pillow' first.")
    except Exception as e:
        print(f"Error converting image: {e}")

if __name__ == "__main__":
    main()
