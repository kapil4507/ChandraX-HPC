import os
import sys
import numpy as np

def analyze_power(xml_path):
    # Parse XML to find parameters
    import re
    with open(xml_path, 'r') as f:
        content = f.read()
        
    file_name = re.search(r'<file_name>([^<]+)</file_name>', content).group(1)
    lines = int(re.search(r'<axis_name>Line</axis_name>\s*<elements>([^<]+)</elements>', content).group(1))
    total_line_elements = int(re.search(r'<axis_name>Sample</axis_name>\s*<elements>([^<]+)</elements>', content).group(1))
    
    bin_path = os.path.join(os.path.dirname(xml_path), file_name)
    if not os.path.exists(bin_path):
        # Fallback to check inside data/ folder directly if path relative
        bin_path = os.path.join("data", file_name)
        if not os.path.exists(bin_path):
            print(f"Error: Binary file not found at {bin_path}")
            return
        
    print(f"Analyzing {bin_path}...")
    print(f"Total lines: {lines}, Line size: {total_line_elements} bytes")
    
    # Subsample lines to analyze quickly
    step = max(1, lines // 1000)
    indices = list(range(0, lines, step))
    powers = []
    
    # Header size is 789, we read the LH polarisation (first 2048 bytes of payload)
    header_bytes = total_line_elements - 2 * 1024 * 2 # 789
    
    with open(bin_path, 'rb') as f:
        for idx in indices:
            f.seek(idx * total_line_elements + header_bytes)
            data = f.read(2048) # 1024 samples * 2 bytes
            if len(data) < 2048:
                break
            # Convert to numpy array of uint8
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            # Subtract 128 offset-binary
            arr = arr - 128.0
            # Calculate mean absolute magnitude
            i_ch = arr[0::2]
            q_ch = arr[1::2]
            mags = np.sqrt(i_ch**2 + q_ch**2)
            powers.append((idx, np.mean(mags)))
            
    # Print profile segments
    print("\nLine Index | Mean Magnitude")
    print("---------------------------")
    for idx, p in powers[::len(powers)//20]: # print ~20 samples
        print(f"{idx:10d} | {p:.2f}")
        
    # Analyze transitions to find noise/replica/imaging boundaries
    print("\nSummary of Power Profile:")
    prev_status = None
    for i, (idx, p) in enumerate(powers):
        if p < 5.0:
            status = "Noise / Space calibration"
        elif p > 40.0:
            status = "Replica Calibration (chirp loop)"
        else:
            status = "Imaging data (Surface echo)"
            
        if i == 0 or status != prev_status:
            print(f"Line {idx:06d}: Starts {status} (power ~ {p:.1f})")
        prev_status = status
        
if __name__ == '__main__':
    xml = "data/real_label.xml"
    if len(sys.argv) > 1:
        xml = sys.argv[1]
    analyze_power(xml)
