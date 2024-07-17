import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

def separate_label_values(pred):
    '''
    Given a prediction over trjaectory segments, extracts the predictions for each segment property
    as well as the changepoint values.
    '''        
    Ds = pred[1::4]
    alphas = pred[2::4]
    states = pred[3::4]
    cp = pred[4::4]    
    return Ds, alphas, states, cp

def read_and_process_data(base_dir):
    model_dict = {'single_state':0, 'multi_state':1, 'immobile_traps':2, 'dimerization':3, 'confinement':4}

    all_trajectories = []
    all_traj_labels = []
    all_labels = []
    for root, dirs, files in tqdm(os.walk(base_dir)):
        for file in files:
            pattern = re.compile(r"trajs_fov_\d+\.csv$")
            if pattern.match(file):
                traj_file_path = os.path.join(root, file)
                label_file_path = os.path.join(root,"traj_labs_fov_" + traj_file_path[-5:-4] + ".txt")
                model_file_path = os.path.join(root,"ens_labs_fov_" + traj_file_path[-5:-4] + ".txt")
                
                ###############################################
                df = pd.read_csv(traj_file_path)
                
                # Filter and transform the data
                df_filtered = df[['traj_idx', 'x', 'y']]
                traj_groups = df_filtered.groupby('traj_idx').apply(
                    lambda g: g[['x', 'y']].to_numpy().T, include_groups=False)
                traj_list = list(traj_groups)

                for traj in traj_list:
                    traj = traj_preprocessing(traj)
                    # Append the list of trajectories from this file to the main list
                    all_trajectories.append(traj)

                df_filtered_label = df[['traj_idx', 'alpha', 'D', 'state']]
                traj_label_groups = df_filtered_label.groupby('traj_idx').apply(
                    lambda g: g[['alpha', 'D', 'state']].to_numpy().T, include_groups=False)
                traj_list = list(traj_label_groups)

                for traj in traj_list:
                    # Append the list of trajectories from this file to the main list
                    traj = traj_label_preprocessing(traj)
                    
                    all_traj_labels.append(traj)

                ##################################################
                with open(model_file_path, "r") as f:
                    first_line = f.readline()  # Read the first line of the file
                    parts = first_line.split(';')  # Split the line on semicolon
                        
                    # Iterate through parts to find the model information
                    for part in parts:
                        if 'model' in part:
                            # Extract the model name following the pattern "model: value"
                            model_type = part.split(':')[1].strip()
                            model = model_dict[model_type]
                #################################################
                with open(label_file_path, "r") as f:
                    label_lines = f.read().splitlines()
                
                
                columns = ['traj_idx', 'model', 'Ds', 'alphas', 'states', 'changepoints']                
                for line in label_lines:

                        
                    # Extract values with comma separator and transform to float
                    label_traj = line.split(',')
                    label = [float(i) for i in label_traj]
                    
                    D, alpha, state, cp = separate_label_values(label)
                    all_labels.append([model, D, alpha, state, cp])

                #################################################
                if len(all_trajectories) != len(all_labels):
                    print(traj_file_path)
                
    all_trajectories = np.expand_dims(all_trajectories, axis=2)
    all_traj_labels = np.expand_dims(all_traj_labels, axis=2)

    return all_traj_labels, all_trajectories, all_labels


def traj_preprocessing(traj):
    """
    traj; shape (2,traj_len)
    """

    # zero padding
    padded_len = 200 - traj.shape[-1]
    # traj = np.pad(traj, ((0,0), (padded_len,0)))
    # scaling
    traj_min = traj.min(axis=-1, keepdims=True)
    traj_max = traj.max(axis=-1, keepdims=True)
    scaled_traj = (traj - traj_min) / (traj_max-traj_min)
    # make origin
    scaled_traj = scaled_traj-scaled_traj[:, 0][:, np.newaxis]
    scaled_traj = np.pad(scaled_traj, ((0,0), (padded_len,0)))
    
    return scaled_traj


def traj_label_preprocessing(traj_label):
    """
    traj; shape (#,traj_len)
    """
    # zero padding
    padded_len = 200 - traj_label.shape[-1]
    traj_label = np.pad(traj_label, ((0,0), (padded_len,0)), constant_values=-1)
    
    return traj_label


def traj_preprocessing_sep(traj, label):
    """
    traj; shape (2,traj_len)
    """
    traj_list = []
    label_list = []
    padding_list = []
    if len(label[-1])==1:
        traj = np.squeeze(traj)
        raw_len = label[-1][-1]
        traj = traj_preprocessing(traj)
        traj_list.append(traj)
        label_list.append([label[0], label[1][0], label[2][0], label[3][0], int(label[4][0]+200-raw_len)])  # k ,a, s, cp
        padding_list.append(200 - raw_len)
    else:
        traj = np.squeeze(traj)
        raw_len = label[-1][-1]
        bef_cp =int(200-raw_len)

        for index, raw_cp in enumerate(label[-1]):
            if bef_cp==200:
                continue
            cp = int(raw_cp+200-raw_len)
            sub_traj = traj[:,bef_cp:]
            padding_list.append(int(bef_cp))

            sub_traj = traj_preprocessing(sub_traj)
            traj_list.append(sub_traj)
            label_list.append([label[0], label[1][index], label[2][index], label[3][index], int(label[4][index]+200-raw_len)])

            bef_cp = cp
    return traj_list, label_list, padding_list
