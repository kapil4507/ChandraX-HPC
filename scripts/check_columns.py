import os
import numpy as np

def main():
    dat_path = "data/ch2_sar_nrxl_20251106t221014810_d_r0b_xx_cp_xx_d18.dat"
    if not os.path.exists(dat_path):
        print(f"Error: {dat_path} not found.")
        return

    total_line_elements = 4885
    lines_to_check = [0, 1000, 10000, 100000, 250000]
    
    print("--- DETAILED RANGE-COLUMN INSPECTION (LH Channel: bytes 789 to 2836) ---")
    
    with open(dat_path, 'rb') as f:
        for line_idx in lines_to_check:
            f.seek(line_idx * total_line_elements)
            raw_bytes = f.read(total_line_elements)
            
            data = np.frombuffer(raw_bytes, dtype=np.uint8)
            # LH channel is from byte 789 to 2836 (2048 bytes)
            lh_bytes = data[789:2837]
            
            print(f"\nLine {line_idx}:")
            # Divide the 2048 bytes (1024 complex samples) into 8 blocks of 128 complex samples (256 bytes)
            for block in range(8):
                start_byte = block * 256
                end_byte = start_byte + 256
                block_data = lh_bytes[start_byte:end_byte].astype(np.float32) - 128.0
                
                std = block_data.std()
                mean = block_data.mean()
                print(f"  Block {block} (Samples {block*128:03d} to {(block+1)*128:03d}): Mean = {mean:+.2f}, Std Dev = {std:.2f}")

if __name__ == "__main__":
    main()
