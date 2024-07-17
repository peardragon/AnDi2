import numpy as np
import torch
from andi_datasets.utils_challenge import segment_assignment

def traj_preprocessing(traj):
    """
    traj; shape (2,traj_len)
    """
    # make origin
    traj = traj-traj[:, 0][:, np.newaxis]
    # zero padding
    padded_len = 200 - traj.shape[-1]
    traj = np.pad(traj, ((0,0), (padded_len,0)))
    # scaling
    traj_min = traj.min(axis=-1, keepdims=True)
    traj_max = traj.max(axis=-1, keepdims=True)
    scaled_traj = (traj - traj_min) / (traj_max-traj_min)
    
    return scaled_traj


def traj_preprocessing_nonpad(traj):
    """
    traj; shape (2,traj_len)
    """
    # make origin
    traj = traj-traj[:, 0][:, np.newaxis]
    # zero padding
    padded_len = 0
    traj = np.pad(traj, ((0,0), (padded_len,0)))
    # scaling
    traj_min = traj.min(axis=-1, keepdims=True)
    traj_max = traj.max(axis=-1, keepdims=True)
    scaled_traj = (traj - traj_min) / (traj_max-traj_min)
    
    return scaled_traj



def traj_preprocessing_torch(traj):
    """
    Preprocess the trajectory data using PyTorch.
    
    Parameters:
    - traj: torch.Tensor with shape (2, 1, traj_len)
    
    Returns:
    - scaled_traj: Processed tensor with shape (2, 1, 200), scaled to [0, 1]
    """
    # Translate trajectory to make the origin (0, 0)
    traj = traj - traj[:, :, 0:1]  # Subtract the first column from all columns
    
    # Zero padding to length 200
    padded_len = 200 - traj.shape[-1]
    if padded_len > 0:
        padding = torch.nn.ConstantPad1d((0, padded_len), 0)  # Pad on the last dimension
        traj = padding(traj)
    
    # Min-Max scaling
    traj_min = torch.min(traj, dim=-1, keepdim=True)[0]
    traj_max = torch.max(traj, dim=-1, keepdim=True)[0]
    diff = traj_max - traj_min
    diff[diff == 0] = 1  # Avoid division by zero
    scaled_traj = (traj - traj_min) / diff

    return scaled_traj

####################################################

def gaussian_distribution(length, mu, sigma=5/3):
    x = np.arange(length)
    gauss = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return gauss / np.max(gauss)  # Normalize to peak at 1

def multiple_gaussian(length, mus, sigma=5/3):
    array = np.zeros(length)
    for mu in mus:
        array += gaussian_distribution(length, mu, sigma)
    return np.clip(array, 0, 1)  # Ensure values don't exceed 1

#####################################################

def find_variable_peaks_optimized(data, num_peaks_per_batch, min_distance=10):
    batch_size, _, _, width = data.shape
    data = data.view(batch_size, width)  # Flatten the data to [batch_size, 200]
    
    # Precompute minimum values for each batch
    min_values = data.min(dim=1, keepdim=True).values

    # Prepare a list to hold peak indices for each batch
    batch_peak_indices = []
    
    for b in range(batch_size):
        num_peaks = int(num_peaks_per_batch[b].item())  # Convert to Python scalar
        temp_data = data[b].clone()
        peaks_found = []
        
        for _ in range(num_peaks):
            peak_idx = torch.argmax(temp_data).item()
            if temp_data[peak_idx] == min_values[b] - 1:
                break  # Exit if only masked values are left

            peaks_found.append(peak_idx)
            
            # Mask out the neighborhood of the found peak
            from_idx = max(0, peak_idx - min_distance)
            to_idx = min(width, peak_idx + min_distance + 1)
            temp_data[from_idx:to_idx] = min_values[b] - 1  # Set to a lower value to mask
        peaks_found.append(200)
        batch_peak_indices.append(peaks_found)
        

    return batch_peak_indices


def split_data_at_indices_4d(data, indices):
    # Holds the segmented data for each batch
    segmented_data = []

    # Process each sequence in the batch
    for batch_idx, batch_indices in enumerate(indices):
        segments = []
        last_index = 0

        # Sort indices to ensure proper slicing
        sorted_indices = sorted(batch_indices)

        # Add the slices for each found index
        for idx in sorted_indices:
            # Append segment from the last index to the current index
            if idx < data.size(-1):  # Ensure the index is within bounds of the last dimension
                segments.append(data[batch_idx, :, :, last_index:idx + 1])
                last_index = idx

        # Append the remaining part of the data if any
        if last_index < data.size(-1):
            segments.append(data[batch_idx, :, :, last_index:data.size(-1)])

        # Add batch's segments to the list
        segmented_data.append(segments)

    return segmented_data

#######################################################

def check_no_changepoints(GT_cp, GT_alpha, GT_D, GT_s,
                          preds_cp, preds_alpha, preds_D, preds_s,
                          T:bool|int = None):


    if isinstance(GT_cp, int) or isinstance(GT_cp, float):
        GT_cp = [GT_cp]
    if isinstance(preds_cp, int) or isinstance(preds_cp, float):
        preds_cp = [preds_cp]
        
    no_GT_cp = False; no_preds_cp = False
    # CP always contain the final point of the trajectory, hence minimal length is one
    if len(GT_cp) == 1: no_GT_cp = True
    if len(preds_cp) == 1: no_preds_cp = True       
        

    if no_GT_cp + no_preds_cp == 0:
        return False, None, None, None
    
    else:

        [row_ind, col_ind], _ = segment_assignment(GT_cp, preds_cp, T)   

        if no_GT_cp and not no_preds_cp:
            paired_alpha = [[GT_alpha[0], preds_alpha[col_ind[0]]]]
            paired_D = [[GT_D[0], preds_D[col_ind[0]]]]
            paired_s =[[GT_s[0], preds_s[col_ind[0]]]]

        if no_preds_cp and not no_GT_cp:
            row_position = np.argwhere(col_ind == 0).flatten()[0]           
            paired_alpha = [[GT_alpha[row_position], preds_alpha[col_ind[row_position]]]]
            paired_D = [[GT_D[row_position], preds_D[col_ind[row_position]]]]
            paired_s = [[GT_s[row_position], preds_s[col_ind[row_position]]]]
            
        if no_preds_cp and no_GT_cp: 
            paired_alpha = [[GT_alpha[0], preds_alpha[0]]]
            paired_D = [[GT_D[0], preds_D[0]]]
            paired_s = [[GT_s[0], preds_s[0]]]
            

        return True, paired_alpha, paired_D, paired_s