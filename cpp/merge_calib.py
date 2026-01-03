import sys
import struct
import os

def hex_to_double(h):
    return struct.unpack('>d', bytes.fromhex(h))[0]

def double_to_hex(d):
    return struct.pack('>d', d).hex()

def hex_to_float(h):
    return struct.unpack('>f', bytes.fromhex(h))[0]

def float_to_hex(f):
    return struct.pack('>f', f).hex()

def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Parse header simply as the first line
    header_line = lines[0].strip()
    data = {}
    keys = []
    for line in lines[1:]:
        parts = line.strip().split(':')
        if len(parts) == 2:
            key = parts[0].strip()
            val = parts[1].strip()
            if key not in data:
                keys.append(key)
            data[key] = val
    return header_line, data, keys

def main():
    base_dir = r"g:\code\KataGomo_fork\cpp\build\Debug\KataGoData\trtcache"
    
    if len(sys.argv) < 4:
        file1 = os.path.join(base_dir, "minmax.bin")
        file2 = os.path.join(base_dir, "minmax2.bin")
        outfile = os.path.join(base_dir, "merged2.bin")
        
        if len(sys.argv) >= 2: file1 = sys.argv[1]
        if len(sys.argv) >= 3: file2 = sys.argv[2]
        if len(sys.argv) >= 4: outfile = sys.argv[3]
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        outfile = sys.argv[3]
        
    print(f"Merging:\n  {file1}\n  {file2}\nTo:\n  {outfile}")
    
    h1, d1, keys1 = read_file(file1)
    h2, d2, keys2 = read_file(file2)
    
    # Use the header from the second file
    new_header = h2
    
    # Merge data
    all_keys = keys1[:]
    for k in keys2:
        if k not in d1:
            all_keys.append(k)
            
    with open(outfile, 'w') as f:
        f.write(new_header + '\n')
        
        for k in all_keys:
            v1 = d1.get(k)
            v2 = d2.get(k)
            
            final_val_hex = v1 # Default to v1 if v2 missing
            
            if v1 and v2:
                # Assuming float (8 hex chars)
                # If length is 16, it's double
                if len(v1) == 8:
                    f1 = hex_to_float(v1)
                    f2 = hex_to_float(v2)
                    avg = (f1 + f2) / 2
                    print(k,f1, f2, avg)
                    final_val_hex = float_to_hex(avg)
                elif len(v1) == 16:
                    assert False, f"Key {k} has double precision, not supported"
                    d1_val = hex_to_double(v1)
                    d2_val = hex_to_double(v2)
                    avg = (d1_val + d2_val) / 2
                    final_val_hex = double_to_hex(avg)
                else:
                    # Unknown length, just keep one
                    final_val_hex = v1
            elif v2:
                assert False, f"Key {k} only found in second file"
                final_val_hex = v2
            
            f.write(f"{k}: {final_val_hex}\n")
    
    print("Done.")

if __name__ == "__main__":
    main()
