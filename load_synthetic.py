import torch
import os
import numpy as np
import json
import cv2
import time

# conda install -c conda-forge openexr
# conda install -c conda-forge openexr-python
# pip install imath
import OpenEXR
import Imath

from run_nespof_helpers import list2map
from utils.stokes_basis_rotate import *
from utils.stokes_params_convert import stokes_to_params


def poses_avg(poses):

    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def poses_median(poses):
    # based on x-axis
    trans = poses[:, :3, 3]
    tmp = trans[...,0]
    idx = np.argsort(tmp)[len(tmp)//2]
    c2w = poses[idx]
    
    return c2w

def poses_median2(poses):
    # based on y-axis
    trans = poses[:, :3, 3]
    tmp = trans[...,1] # 0: x? / 1: y? / 2: z?
    idx = np.argsort(tmp)[len(tmp)//2]
    c2w = poses[idx]
    
    return c2w

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    # tmp = np.stack([vec0,vec1,vec2], 0)
    # m = np.concatenate([tmp.T, pos[np.newaxis]], 0).T
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        
    return render_poses

def render_path_spiral_synthetic(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
        
    return render_poses

def pose_spherical(theta, phi, radius):
    trans_t = lambda t : torch.Tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    
    return c2w


def read_exr_as_np(fn):
    f = OpenEXR.InputFile(fn)
    channels = f.header()['channels']
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    ch_names = []

    image = np.zeros((size[1], size[0], len(channels)))
    for i, ch_name in enumerate(channels):
        ch_names.append(ch_name)
        ch_dtype = channels[ch_name].type
        ch_str = f.channel(ch_name, ch_dtype)
        
        if ch_dtype == Imath.PixelType(Imath.PixelType.FLOAT):
            np_dtype = np.float32
        elif ch_dtype == Imath.PixelType(Imath.PixelType.HALF):
            np_dtype = np.half
            
        image_ch = np.fromstring(ch_str, dtype=np_dtype)
        image_ch.shape = (size[1], size[0])
        image[:,:,i] = image_ch

    return image, ch_names

def mean_var_per_channel(x):
    '''
        Args.
        - x is image of each, [B,H,W,4]
    '''
    return x.mean((0,1,2)), x.var((0,1,2))

def mean_var_per_channel_with_alpha(x):
    '''
        Args.
        - x is image of each stoke and alpha, [B,H,W,5]
    '''
    mask = (x[...,4:5]>0) # [B,H,W,1]
    stokes = x[...,:4]
    
    mu = stokes.sum((0,1,2))/mask.sum()
    var = (mask*((stokes-mu)**2)).sum((0,1,2))/mask.sum()
    return mu, var

def load_synthetic_data(basedir, half_res=False, testskip=1, rotate_stoke=True, target_wavelength=550, debug=False, render_only=False, scale_rads=0.1):        
    splits = ['train', 'val', 'test']
    if debug: # Load small amounts!
        splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    waves = []
    counts = [0]
    
    start, stop, interval = 450, 650, 10
    wavelength = range(start, stop+1, interval)
    if render_only: # Load small amounts!
        print(f"Load exr data for wavelength {target_wavelength} & render_only! (only poses)")
        wavelength = [target_wavelength]
    
    
    for s in splits:
        if s == 'test':
            skip = 1000
            testskip = 1000
            # No TEST sets
        for w in wavelength:
            meta = metas[s]
            imgs = []
            poses = []
            
            if debug == True:
                skip = 100
            elif s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
            
            for frame in meta['frames'][::skip]:
                _, set_name, file_name = frame['file_path'].split("/")                
                fname = os.path.join(basedir, str(w), frame['file_path']+'.exr')                 
                image_stokes, _ = read_exr_as_np(fname)    # (H, W, 4) or (H, W, 5) # [s0, s1, s2, s3, Alpha]
                                    
                imgs.append(image_stokes)     # (H, W, 4) or (H, W, 5)
                poses.append(np.array(frame['transform_matrix']))
                waves.append(w)
                
            imgs = np.array(imgs).astype(np.float32)
            poses = np.array(poses).astype(np.float32)
            
            all_imgs.append(imgs)
            all_poses.append(poses)            
           
        counts.append(counts[-1] + imgs.shape[0] * len(wavelength))
        
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0) 
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # Rotate Stoke Vector!!!
    # convert stokes following measurement to global basis
    use_alpha = imgs.shape[-1] > 4
    if rotate_stoke:
        stokes, alpha = imgs[...,:4], imgs[...,-1:] # separate stokes & alpha
        
        start = time.time()
        
        cam_kwargs={}
        cam_kwargs['pixel_H'] = H
        cam_kwargs['pixel_W'] = W
        stokes_rotated = np.stack([meas2tgt(stokes[i], poses[i], cam_kwargs) for i in range(imgs.shape[0])], 0) # [H,W,4]
        
        if use_alpha:
            imgs = np.concatenate([stokes_rotated, alpha], -1)
        else:
            imgs = stokes_rotated
            
        print(f"Elapsed time for rotating stoke vector : {time.time()-start}")
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    ###################################################################################
    spiral_render = True
    path_zflat = False
    
    if spiral_render:
        if render_only:
            c2w = poses_median(poses)
            print('recentered', c2w.shape)
            print(c2w[:3,:4])
        else:
            c2w = poses_avg(poses)
            print('recentered', c2w.shape)
            print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        # close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        close_depth, inf_depth = 2, 6
        # dt = .75
        # mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        # focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        # rads = np.percentile(np.abs(tt), 45, 0)
        rads = c2w[:3,3]
        rads = np.array([scale_rads, scale_rads, scale_rads])
        
        c2w_path = c2w
        N_views = 60
        N_rots = 1
        zrate = 1.0
        if path_zflat:
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral_synthetic(c2w_path, up, rads, focal, zdelta, zrate=zrate, rots=N_rots, N=N_views)
        render_poses = np.array(render_poses).astype(np.float32)
    ###################################################################################
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    # Repeat wavelengths
    waves = np.stack(waves, 0) # [B]
    waves = list2map(waves, H, W) # [B,H,W,1]
    
    # Loss weight scales for each channel of stoke vectors
    if use_alpha: 
        # Get masked mean and variance using alpha
        gt_mean, gt_var = mean_var_per_channel_with_alpha(imgs)
    else:
        gt_mean, gt_var = mean_var_per_channel(imgs[...,:4])
    
    gt_std = gt_var ** 0.5
    gt_std_inv = np.reciprocal(gt_std)
    
    w_scale = gt_std_inv/(gt_std_inv[...,0]+1e-6) # shape: [4,]
    
    return waves, imgs, poses, render_poses, [H, W, focal], i_split, w_scale