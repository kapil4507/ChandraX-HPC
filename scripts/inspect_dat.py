import os
import numpy as np

def main():
    dat_path = "data/ch2_sar_nrxl_20251106t221014810_d_r0b_xx_cp_xx_d18.dat"
    if not os.path.exists(dat_path):
        print(f"Error: {dat_path} not found.")
        return

    total_line_elements = 4885
    num_lines_to_read = 10
    
    with open(dat_path, 'rb') as f:
        raw_data = f.read(total_line_elements * num_lines_to_read)
        
    # Read as signed int8
    data_signed = np.frombuffer(raw_data, dtype=np.int8).reshape((num_lines_to_read, total_line_elements))
    # Read as unsigned uint8 and subtract 128 (offset binary correction)
    data_unsigned = np.frombuffer(raw_data, dtype=np.uint8).reshape((num_lines_to_read, total_line_elements))
    data_offset_corrected = data_unsigned.astype(np.float32) - 128.0

    print("--- COMPARING SIGNED VS. OFFSET-BINARY (UNSIGNED - 128) ---")
    for i in range(5):
        line_s = data_signed[i]
        line_o = data_offset_corrected[i]
        
        print(f"\nLine {i:02d}:")
        print(f"  [Direct Signed int8]:")
        print(f"    Min: {line_s.min()}, Max: {line_s.max()}, Std Dev: {line_s.std():.2f}")
        print(f"  [Offset Binary (uint8 - 128)]:")
        print(f"    Min: {line_o.min()}, Max: {line_o.max()}, Std Dev: {line_o.std():.2f}")

if __name__ == "__main__":
    main()
