import os
import numpy as np

def main():
    dat_path = "data/ch2_sar_nrxl_20251106t221014810_d_r0b_xx_cp_xx_d18.dat"
    if not os.path.exists(dat_path):
        print(f"Error: {dat_path} not found.")
        return

    print(f"Opening binary file: {dat_path}...")
    file_size = os.path.getsize(dat_path)
    print(f"File size: {file_size} bytes")

    # Read the first 10 lines
    total_line_elements = 4885
    num_lines_to_read = 50
    
    with open(dat_path, 'rb') as f:
        raw_data = f.read(total_line_elements * num_lines_to_read)
        
    data = np.frombuffer(raw_data, dtype=np.int8)
    data = data.reshape((num_lines_to_read, total_line_elements))
    
    print("\n--- Inspecting first 50 lines ---")
    for i in range(10):
        line = data[i]
        non_zero_indices = np.where(line != 0)[0]
        print(f"Line {i:02d}: Non-zero elements: {len(non_zero_indices)} / {total_line_elements}")
        if len(non_zero_indices) > 0:
            print(f"        First non-zero index: {non_zero_indices[0]}, Last non-zero index: {non_zero_indices[-1]}")
            # Print mean and std of non-zero parts
            print(f"        Max value: {line.max()}, Min value: {line.min()}, Std dev: {line.std():.2f}")
            
    # Let's inspect rows around the middle of the file
    print("\n--- Inspecting middle of the file ---")
    middle_line = 509909 // 2
    with open(dat_path, 'rb') as f:
        f.seek(middle_line * total_line_elements)
        raw_data = f.read(total_line_elements * 10)
    
    data_mid = np.frombuffer(raw_data, dtype=np.int8).reshape((10, total_line_elements))
    for i in range(10):
        line = data_mid[i]
        non_zero_indices = np.where(line != 0)[0]
        print(f"Line {middle_line + i}: Non-zero elements: {len(non_zero_indices)} / {total_line_elements}")
        if len(non_zero_indices) > 0:
            print(f"        First non-zero index: {non_zero_indices[0]}, Last non-zero index: {non_zero_indices[-1]}")
            print(f"        Max value: {line.max()}, Min value: {line.min()}, Std dev: {line.std():.2f}")

if __name__ == "__main__":
    main()
