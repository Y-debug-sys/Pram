import numpy as np


def save_dms_to_csv(dms: np.ndarray, dm_fname: str):
    """
    Save a numpy array dms (shape: num_samples x num_features) to a CSV file.

    Args:
        dms (np.ndarray): The array to save, shape = (num_samples, num_features).
        dm_fname (str): Path of the CSV file to save to.
    """
    if dms.ndim != 2:
        raise ValueError(f"dms must be 2D, got shape {dms.shape}")

    np.savetxt(dm_fname, dms, delimiter=",", fmt="%.6f")

    print(f"✅ Saved dms with shape {dms.shape} to {dm_fname}")