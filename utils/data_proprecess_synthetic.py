import numpy as np
import pandas as pd
import json
from collections import OrderedDict

import os
import glob
import shutil

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def look_at(camera, target, up_):
    # camera : NeRF Synthetic convention. Camera position.
    # target : NeRF Synthetic convention. Object center.
    
    dir_ = camera - target # z-dir
    dir_ = dir_/np.linalg.norm(dir_)
    
    left = np.cross(up_, dir_) # x-dir
    left = left/np.linalg.norm(left)
      
    new_up = np.cross(dir_, left) # y-dir
    
    lookat_mat = np.array([[left[0], new_up[0], dir_[0], camera[0]],
                           [left[1], new_up[1], dir_[1], camera[1]],
                           [left[2], new_up[2], dir_[2], camera[2]],
                           [0, 0, 0, 1]                              ])
    
    return lookat_mat



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess data to make json files for metadata.')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--divide_data', type=bool, default=False)
    
    args = parser.parse_args()
    
    cameras = pd.read_csv(args.csv_path) # Mitsuba2 convention
    cameras = np.array([cameras.iloc[i] for i in range(len(cameras))])
    if args.mode == 'ajar':
        target = np.array([0, 1.5, -3.2]) # ajar
    
    elif args.mode in ['cbox_dragon', 'cbox_sphere', 'pool', 'shoes']:
        target = np.array([0, 0, 0])
    
    else:
        raise ValueError("Choose the correct mode")
    
    ###################################################################
    # Change convention from Mitsuba to OpenGL (or NeRF Synthetic)
    # [x_M, y_M, z_M] = [x_GL, z_GL, -y_GL]
    # [x_GL, y_GL, z_GL] = [x_M, -z_M, y_M]
    
    tmp = np.zeros_like(cameras)
    tmp[...,0], tmp[...,1], tmp[...,2] = cameras[...,0], -cameras[...,2], cameras[...,1]
    cameras = tmp
    tmp = np.zeros_like(target)
    tmp[...,0], tmp[...,1], tmp[...,2] = target[...,0], -target[...,2], target[...,1]
    target = tmp    
    ###################################################################
    
    # up_ = np.array([0, 1, 0])
    up_ = np.array([0, 0, 1])
    
    look_at_mat = []
    for i in range(len(cameras)):
        camera = cameras[i,:]
        look_at_mat.append(look_at(camera, target, up_))

    if args.mode == 'ajar':
        camera_angle = 60 * np.pi / 180 # ajar
    
    elif args.mode == 'cbox_dragon':
        camera_angle = 50 * np.pi / 180 # cbox_dragon
    
    elif args.mode in 'cbox_sphere':
        camera_angle = 39.3077 * np.pi / 180 # cbox_sphere
        
    # elif args.mode in 'pool':
    #     camera_angle = 56 * np.pi / 180 # pool
    
    # elif args.mode in 'shoes':
    #     camera_angle = 53.997748 * np.pi / 180 # shoes
    
    else:
        raise ValueError("Choose the correct mode")
    
    ##################################################################
    # Make json file
    tmp = 0.012566370614359171
    file_data_train = OrderedDict()
    file_data_train["camera_angle_x"] = camera_angle
    file_data_train["frames"] = []
    
    file_data_val = OrderedDict()
    file_data_val["camera_angle_x"] = camera_angle
    file_data_val["frames"] = []
    
    file_data_test = OrderedDict()
    file_data_test["camera_angle_x"] = camera_angle
    file_data_test["frames"] = []
        
    # 1 validation per 8 train data
    for ii, mat in enumerate(look_at_mat):
        tmp_dict = {}
        if ii % 8 == 0: # val
            filename = f"./val/r_{ii}"
        else:
            filename = f"./train/r_{ii}"
        
        tmp_dict["file_path"] = filename
        tmp_dict["rotation"] = tmp
        tmp_dict["transform_matrix"] = mat.tolist()
        
        if ii % 8 == 0: # val
            file_data_val["frames"].append(tmp_dict)
        else:
            file_data_train["frames"].append(tmp_dict)
        file_data_test["frames"].append(tmp_dict)
        
    # Save json files
    json_train_path = os.path.join(Path(args.csv_path).parent, args.mode, "transforms_train.json")
    json_val_path = os.path.join(Path(args.csv_path).parent, args.mode, "transforms_val.json")
    json_test_path = os.path.join(Path(args.csv_path).parent, args.mode, "transforms_test.json")
    
    with open(json_train_path, 'w', encoding='utf-8') as make_file:
        json.dump(file_data_train, make_file, ensure_ascii=False, indent="\t")
        
    with open(json_val_path, 'w', encoding='utf-8') as make_file:
        json.dump(file_data_val, make_file, ensure_ascii=False, indent="\t")
        
    with open(json_test_path, 'w', encoding='utf-8') as make_file:
        json.dump(file_data_test, make_file, ensure_ascii=False, indent="\t")                
    
    # Divide files for train/val sets. Operate once.
    if args.divide_data == True:
        import natsort
        
        path = os.path.join(Path(args.csv_path).parent, args.mode)
        paths = os.listdir(path)
        paths = [os.path.join(path,p) for p in paths if os.path.isdir(os.path.join(path,p))]
        
        for p in paths:
            file_path = natsort.natsorted(glob.glob(os.path.join(p, "*.exr")))
            
            train_path = os.path.join(p, "train")
            val_path = os.path.join(p, "val")
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            
            for i, f in enumerate(file_path):
                filename = f.split("/")[-1]
                pre_path = f
                
                if i % 8 ==0:
                    post_path = os.path.join(val_path, filename)
                else:
                    post_path = os.path.join(train_path, filename)
                
                # Move files..
                shutil.move(pre_path, post_path)