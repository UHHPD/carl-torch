import numpy as np
import h5py
import os

# Define directories and file parameters
legend = "HDamp"  # Update as needed
n = "7000000"  # Example event count
data_folder = f"/data/dust/user/griesinl/carl-torch/{legend}/NoBootstrappingV3"
hdf5_file_path = os.path.join(data_folder, f"{legend}_data_{n}.h5")

# Create HDF5 file
with h5py.File(hdf5_file_path, "w") as h5f:
    for i in range(100):  # Loop over dataset indices
        for split in ["train", "val"]:  # Handle both training and validation sets
            group = h5f.require_group(split)  # Create group for train/val
            
            for key in ["X0", "w0", "w_CARL", "X1", "w1", "X", "w", "y"]:
                npy_path = os.path.join(data_folder, f"{legend}_{i}_{key}_{split}_{n}.npy")
                if os.path.exists(npy_path):
                    print(f"Converting {npy_path} to HDF5...")
                    data = np.load(npy_path, mmap_mode="r")  # Load efficiently
                    group.create_dataset(f"{i}/{key}", data=data, compression="gzip")  # Store in HDF5
                else:
                    print(f"Warning: {npy_path} not found.")

print(f" Conversion complete: {hdf5_file_path}")
