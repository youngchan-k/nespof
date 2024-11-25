import numpy as np
import cv2
import os
import pandas as pd
import glob
import sys
import torch

import OpenEXR
import Imath

from scipy.interpolate import splrep, splev
import argparse
from poses.pose_utils import gen_poses
from tqdm import tqdm


def geometric_calib_params(checkerboard_path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    wc = 7
    hc = 5

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((wc * hc, 3), np.float32)
    objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
        
    images = glob.glob(f"{checkerboard_path}/*.png")
            
    for fname in images:
        if os.path.splitext(fname)[1] not in ['.jpg', '.png', '.jpeg']:
            continue
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (wc, hc), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (10,10), (-1,-1), criteria)
            imgpoints.append(corners2)
            '''
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (wc, hc), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            '''
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)  
        
    return mtx, dist, newcameramtx, roi


def image_undistortion(imgpath, savepath, mtx, dist, newcameramtx, roi):
    img = cv2.imread(imgpath, -1)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    cv2.imwrite(savepath, dst)
 
 
def spec_calib(wavelength):
    calib = pd.read_csv("./datasheet/trasmission_data.csv")
    
    # Filter transmission at wavelength
    calib_1 = np.array(calib[["wave1", "FT"]])
    waves_1, FT = calib_1[:, 0][~np.isnan(calib_1[:, 0])], calib_1[:, 1][~np.isnan(calib_1[:, 1])]
    
    # Quantum efficiency & Polarized transmission at wavelength
    calib_2 = np.array(calib[["wave2", "QE", "PT"]])
    waves_2, QE, PT = calib_2[:, 0][~np.isnan(calib_2[:, 0])], calib_2[:, 1][~np.isnan(calib_2[:, 1])], calib_2[:, 2][~np.isnan(calib_2[:, 2])]
    
    spl_FT = splrep(waves_1, FT)
    spl_QE = splrep(waves_2, QE)
    spl_PT = splrep(waves_2, PT)
    
    FT_w = splev(wavelength, spl_FT)
    QE_w = splev(wavelength, spl_QE)
    PT_w = splev(wavelength, spl_PT)    
    
    return FT_w/100, QE_w/100, PT_w/100


def spec_to_rgb(img_path, save_path, waves, view):
    if os.path.exists(os.path.join(save_path, f"{view}.png")):
        return
    imgs = []
    
    ''' Remove the spec_coefficient as it has already been incorporated into other coefficients. '''
    # coefficient = np.load("./calibration/spec_coefficient.npy")
    
    for w in waves:
        image = os.path.join(img_path, str(w), "-90", view)
        img = cv2.imread(image, -1) / (2**16 - 1)
        
        # idx = int((w - 450)/10)
        # coeff_w = coefficient[idx]
        
        QE_w, PT_w, FT_w = spec_calib(w)
        # img /= (QE_w * PT_w * FT_w / coeff_w)
        img /= (QE_w * PT_w * FT_w)
        
        imgs.append(img)
    
    imgs = np.array(imgs).transpose((1, 2, 0))
    H, W, c = imgs.shape
    imgs = imgs.reshape(H*W, c)
    
    cmf_table = pd.read_csv('./datasheet/spec_to_XYZ.csv')  
    cmf_array = np.array(cmf_table[(cmf_table["wavelength"]%10 == 0) & (cmf_table["wavelength"] >= 450) & (cmf_table["wavelength"] <= 650)])
    cmf = cmf_array[:, 1:]
    
    img_rgb = []
    
    for i in range(H*W):
        xyz = np.sum(imgs[i][:, np.newaxis] * cmf, axis=0)
        
        # https://en.wikipedia.org/wiki/SRGB
        srgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]]) @ xyz
        gamma_correct = np.vectorize(lambda x: 12.92*x if x < 0.0031308 else 1.055*(x**(1.0/2.4))-0.055)
        rgb = gamma_correct(srgb)
        
        img_rgb.append(rgb)
            
    imgs_rgb = np.array(img_rgb).reshape(H, W, 3)[...,::-1] * 100    # BGR -> RGB  +  Intensity 
    
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, f"{view}.png"), imgs_rgb)


# def mueller_QWP(theta):
#     theta = theta * np.pi/180
    
#     # Mueller matrix 
#     mat_QWP = np.array([
#                         [1, 0, 0, 0],
#                         [0, np.cos(2*theta)**2, np.sin(2*theta)*np.cos(2*theta), -np.sin(2*theta)],
#                         [0, np.sin(2*theta)*np.cos(2*theta), np.sin(2*theta)**2, np.cos(2*theta)],
#                         [0, np.sin(2*theta), -np.cos(2*theta), 0]
#                        ])
        
#     return mat_QWP


retardance = [0.2479,0.2468,0.2457,0.2447,0.2439,0.2431,0.2425,0.2419,0.2415,0.2411,0.2409,0.2407,0.2406,0.2406,0.2407,0.2408,0.2410,0.2413,0.2417,0.2421,0.2425]

def mueller_WP(theta, retardance):
    theta = theta * np.pi/180
    retardance = retardance * 2* np.pi
    
    # Mueller matrix 
    mat_WP = np.array([
                        [1, 0, 0, 0],
                        [0, np.cos(2*theta)**2 + (np.sin(2*theta)**2) * np.cos(retardance), np.sin(2*theta)*np.cos(2*theta)*(1-np.cos(retardance)), np.sin(2*theta)*np.sin(retardance)],
                        [0, np.sin(2*theta)*np.cos(2*theta)*(1-np.cos(retardance)), np.sin(2*theta)**2 + (np.cos(2*theta)**2) * np.cos(retardance), -np.cos(2*theta)*np.sin(retardance)],
                        [0, -np.sin(2*theta)*np.sin(retardance), np.cos(2*theta)*np.sin(retardance), np.cos(retardance)]
                      ])
        
    return mat_WP


##################################################


def spec_polar_stokes_recon_real(I_measure, mat_calib): 
    ''' 
    ## stokes_est 
    
    I_meausre: Nx4x1
    mat_calib: Nx4x4
    stokes_GT: Nx4x1
    
    (I_measure - M @ A_theta @ stokes_est)^2 
    => "minimize (AX - B)^2" form => Least square problem ''' 
    
    s_recon = torch.linalg.lstsq(mat_calib, I_measure).solution         # (512*612, 4, 1)     
    
    I_recon = torch.bmm(mat_calib, s_recon)
    error = torch.abs(I_recon - I_measure).mean()
    
    return s_recon, error, I_recon


def polar_to_exr(img_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MAT_CALIB = torch.from_numpy(np.load("./calibration/mat_LCTF.npy"))
    
    waves = os.listdir(img_path)
        
    for w in tqdm(waves):
        ind_w = (int(w) - 450) / 10
        mat_calib = MAT_CALIB[int(ind_w), ...]      # (512*612, 1, 4)
        # mat_calib = np.tile(0.5 * np.array([[1., 1., 0., 0.]])[np.newaxis, ...], (512*612, 1, 1))
        
        # coeff_w = spec_coefficient[int(index)]
        QE, FT, PT = spec_calib(int(w))
        
        angles = os.listdir(os.path.join(img_path, w))
        views = os.listdir(os.path.join(img_path, w, angles[0]))
        
        for view in views:
            imgs_recon_list, A_recon_theta_list = [], []
            
            for angle in angles:
                img_png = os.path.join(img_path, w, angle, view)
                img_arr = cv2.imread(img_png, -1) / (2**16 - 1)
                
                imgs_recon_list.append(img_arr)
                
                # Q_1 = mueller_QWP(int(angle))
                Q_1 = mueller_WP(int(angle), retardance[ind_w])

                A_recon_theta = QE * FT * mat_calib @ Q_1
                A_recon_theta_list.append(A_recon_theta.detach().numpy())
                # A_recon_theta_list.append(A_recon_theta)
            
            I_recon_meas = torch.from_numpy(np.array(imgs_recon_list).squeeze().reshape(len(angles), -1, 1)).to(device)
            I_recon_meas = I_recon_meas.permute((1,0,2))
            
            A_recon_theta = torch.from_numpy(np.array(A_recon_theta_list).squeeze()).to(device)    # (8*1, 1, 4)
            A_recon_theta = A_recon_theta.permute((1,0,2))
            
            stokes_recon, _, _ = spec_polar_stokes_recon_real(I_recon_meas, A_recon_theta)
            stokes_recon = stokes_recon.detach().cpu().numpy().reshape(512, 612, 4).astype(np.float32)
            
            # Save Stokes vector to exr format
            os.makedirs(os.path.join(save_path, view.rstrip('.png')), exist_ok=True)
            fname = os.path.join(save_path, view.rstrip('.png'), w + ".exr")
            
            s0, s1, s2, s3 = stokes_recon[..., 0], stokes_recon[..., 1], stokes_recon[..., 2], stokes_recon[..., 3]
            
            h = OpenEXR.Header(s0.shape[1], s0.shape[0])
            ctype = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
            h["channels"] = {"s0":ctype, "s1":ctype, "s2":ctype, "s3":ctype}
            
            exr = OpenEXR.OutputFile(fname, h)
            exr.writePixels({'s0': s0.tostring(), 's1': s1.tostring(), 's2': s2.tostring(),  's3':s3.tostring()})
            exr.close()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess_real_datasets.')
    parser.add_argument('--checkerboard_path', type=str, required=False, default="./checkerboard")
    parser.add_argument('--imgfolder', type=str, required=True)
    parser.add_argument('--savebase', type=str, required=False, default=None)
    parser.add_argument('--only_run_colmap', type=bool, required=False, default=False)
    parser.add_argument('--match_type', type=str, 
	    				default='exhaustive_matcher', help='type of matcher used.  Valid options: \
		    			exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    
    args = parser.parse_args()
    
    if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
        print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
        sys.exit()
    
    
    mtx, dist, newcameramtx, roi = geometric_calib_params(args.checkerboard_path)
    
    imgfolder = args.imgfolder
    
    if args.savebase is None:
        args.savebase = args.imgfolder
    save_dir = args.savebase
    data_name = imgfolder.split("/")[-1]
    if data_name == '':
        data_name = imgfolder.split("/")[-2]
    
    savefolder = os.path.join(save_dir, "undistort")
    
    # if not args.only_run_colmap:
    #-------------------------------------------------------------------
    # IMAGE UNDISTORTION
    waves = range(450, 650+1, 10)
    print("1.Image undistortion start")
    
    for w in tqdm(waves):
        angles = ['-90', '-45', '30', '60']
        
        for angle in angles:
            folder = os.path.join(imgfolder, str(w), angle)
            
            for imgs in os.listdir(folder):
                imgpath = os.path.join(folder, imgs)
                if os.path.splitext(imgpath)[1] not in ['.jpg', '.png', '.jpeg']:
                    continue
                
                savepath = os.path.join(savefolder, str(w), angle)
                os.makedirs(savepath, exist_ok=True)
                
                if os.path.exists(os.path.join(savepath, imgs)):
                    continue
                image_undistortion(imgpath, os.path.join(savepath, imgs), mtx, dist, newcameramtx, roi)
    
    print("DONE...")
    
    #-------------------------------------------------------------------
    # SPECTRAL CALIBRATION
    print("2.Spectral Calibration start")
    img_path = savefolder
    rgb_path = os.path.join(save_dir, "images_re")
    waves = range(450, 650+1, 10)
    views = os.listdir(os.path.join(img_path, "450/30"))
    
    for v in tqdm(views):
        if os.path.splitext(v)[1] not in ['.jpg', '.png', '.jpeg']:
            continue
        spec_to_rgb(img_path, rgb_path, waves, v)
    
    print("DONE...")
    
    #-------------------------------------------------------------------
    # POLARIMETRIC CALIBRATION
    print("3.Polarimetric Calibration start")
    img_path = savefolder
    exr_path = os.path.join(save_dir, "exr")
    polar_to_exr(img_path, exr_path)
    
    print("DONE...")
    
    #-------------------------------------------------------------------
    # COLMAP
    rgb_path = os.path.join(save_dir, "images")
    print("4.COLMAP start")
    scene_dir = rgb_path
    gen_poses(save_dir, args.match_type)
    
    print("DONE...")
    
    
