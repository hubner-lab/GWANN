import os
import numpy as np

def load_npy_from_dir(directory):
    """Load all .npy files in a directory as numpy arrays (list of tuples: (filename, array))."""
    arrays = []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath) and fname.endswith(".npy"):
            try:
                arr = np.load(fpath)
                arrays.append((fname, arr))
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return arrays

def compare_arrays(ref_arr, other_arrays):
    """Check if ref_arr is identical to any array in other_arrays."""
    for fname, arr in other_arrays:
        if np.allclose(ref_arr, arr, atol=1e-6):  # float comparison
            return fname
    return None

def compare_across_dirs(ref_dir, other_dirs):
    """Compare arrays from ref_dir against each directory in other_dirs separately."""
    ref_arrays = load_npy_from_dir(ref_dir)

    for d in other_dirs:
        other_arrays = load_npy_from_dir(d)
        print(f"\nComparing {len(ref_arrays)} reference arrays against {len(other_arrays)} arrays in {d}...")
        i = 1 
        for ref_name, ref_arr in ref_arrays:
            match = compare_arrays(ref_arr, other_arrays)
            if match:
                print(f"{i}:{ref_name} exists in {d} (matched {match}).")
                i+=1
            else:
                print(f"{ref_name} not found in {d}.")

if __name__ == "__main__":
    ref_dir = "/mnt/data/amir/GWANN-TEST/GWANN/snap/dataSet"  # reference dir
    other_dirs = [
        "/mnt/data/amir/GWANN-TEST/GWANN/snap/simloader",
        "/mnt/data/amir/GWANN-TEST/GWANN/snap/train"
    ]
    compare_across_dirs(ref_dir, other_dirs)
