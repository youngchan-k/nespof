import numpy as np
import os, imageio
import cv2

import OpenEXR
import Imath

from load_synthetic import *
from run_nespof_helpers import list2map
from utils.stokes_basis_rotate import meas2tgt
from utils.stokes_params_convert import stokes_to_params


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original
            
def mean_var_per_channel(x):
    '''
        Args.
        - x is image of each, [B,H,W,4]
    '''
    return x.mean((0,1,2)), x.var((0,1,2))

def _load_data(basedir, scale_factor=1., width=None, height=None, load_imgs=True, render_only=False, target_wavelength=550, debug=False):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
        
    imgdir = os.path.join(basedir, 'exr')
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    view_num = len(os.listdir(imgdir))
    imgfiles = [os.path.join(imgdir, f, exr) for f in sorted(os.listdir(imgdir)) for exr in sorted(os.listdir(os.path.join(imgdir, f)))]
    
    waves = [int(exr.split('/')[-1].rstrip('.exr')) for exr in imgfiles]
    if debug:
        imgfiles = imgfiles[:21]
        waves = waves[:21]
        poses = poses[...,:1]
        bds = bds[...,:1]
        
    if render_only == True and debug == True:
        raise ValueError("Not allowed for config with (render only == True) and (debug == True)")
    elif render_only == True:
        wave_num = 21
        wave_start = int((target_wavelength - 450) // 10) # allowed : 0~20
        assert wave_start >= 0 and wave_start <= 20, "Choose wave_start from 0 to 20"
        
        imgfiles = imgfiles[wave_start::wave_num]
        waves = waves[wave_start::wave_num]

    assert os.path.splitext(imgfiles[0])[1] == '.exr', "Support only exr files!"
    
    sh = read_exr_as_np(imgfiles[0])[0].shape
    
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :]
    
    if not load_imgs:
        return poses, bds
    
    imgs = imgs = [read_exr_as_np(f)[0] for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = imgs * scale_factor
    print(f"Scale factor for real scene : {scale_factor}")
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    
    return waves, poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        
    return render_poses
    

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds


def load_real_data(basedir, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, rotate_stoke=True, scale_factor=1., render_only=False, target_wavelength=550, debug=False, scale_rads=0.1):
    waves, poses, bds, imgs = _load_data(basedir, scale_factor, debug=debug) # factor=8 downsamples original imgs by 8x
    
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rotate Stoke Vector!!!
    # convert stokes following measurement to global basis
    if rotate_stoke:
        import time
        start = time.time()
        poses_tmp = np.repeat(poses, len(set(waves)), axis=0)
        images = np.stack([meas2tgt(images[i], poses_tmp[i]) for i in range(images.shape[0])], 0)
        print(f"Elapsed time for rotating stoke vector : {time.time()-start}")
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        # 1) FORWARD FACING - codes from EG3D.
        if render_only:
            c2w = poses_median(poses)
            print('recentered', c2w.shape)
            print(c2w[:3,:4])
            
        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 45, 0)
        rads = rads * np.array([scale_rads, scale_rads, scale_rads])
        rads = np.array([scale_rads, scale_rads, scale_rads])
        
        c2w_path = c2w
        N_views = 60
        N_rots = 1
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            # zloc = -close_depth * .1
            zloc = close_depth * .2
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            # N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
    render_poses = np.array(render_poses).astype(np.float32)
    
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    
    # render_poses = poses
    waves_num = len(set(waves))
    poses = np.repeat(poses, waves_num, axis=0)
    bds = np.repeat(bds, waves_num, axis=0)
    
    # Repeat wavelengths
    H, W = images[0].shape[:2]
    waves = np.stack(waves, 0) # [B]
    waves = list2map(waves, H, W) # [B,H,W,1]
    
    # Loss weight scales for each channel of stoke vectors
    gt_mean, gt_var = mean_var_per_channel(images[...,:4])
    gt_std = gt_var ** 0.5
    gt_std_inv = np.reciprocal(gt_std)
    
    w_scale = gt_std_inv/(gt_std_inv[...,0]+1e-6) # shape: [4,]
    
    return waves, images, poses, bds, render_poses, i_test, w_scale