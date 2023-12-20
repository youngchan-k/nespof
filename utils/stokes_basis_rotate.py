import torch
import os
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F

import OpenEXR
import Imath

"""
1. Calculate the basis of Stokes vector
"""
def mulsign(v1, v2):
    """
    multiplies v1 by the sign of v2
    Code reference: https://github.com/mitsuba-renderer/drjit/blob/8b3cc94a6e638d8dfcc60065cd13fcc561386938/include/drjit/array_router.h
    Args:
        v1: tensor 
        v2: tensor
    Returns:
        v: tensor
    """

    v = torch.zeros_like(v1)
    v[:] = -v1
    v[v2>=0] = v1[v2>=0]
    
    return v


def mulsign_neg(v1, v2):
    """
    multiplies v1 by the sign of -v2
    Code reference: https://github.com/mitsuba-renderer/drjit/blob/8b3cc94a6e638d8dfcc60065cd13fcc561386938/include/drjit/array_router.h
    Args:
        v1: tensor 
        v2: tensor
    Returns:
        v: tensor
    """
    v = torch.zeros_like(v1)
    v[:] = v1
    v[v2>=0] = -v1[v2>=0]
    
    return v


def coordinate_system(z):
    """
    Compute the x and y coordinates of the input unit vectors z. 
    x, y, z consistute the orthonormal basis
    Code reference: https://github.com/mitsuba-renderer/mitsuba3/blob/152352f87b5baea985511b2a80d9f91c3c945a90/include/mitsuba/core/vector.h
    /* Based on "Building an Orthonormal Basis, Revisited" by
       Tom Duff, James Burgess, Per Christensen,
       Christophe Hery, Andrew Kensler, Max Liani,
       and Ryusuke Villemin (JCGT Vol 6, No 1, 2017) */
    Args:
        z: [N,3] unit vectors
    Returns:
        xy: [N,6] unit vectors
            x = xy[:,:3], y = xy[:,3:]
    """

    sign = torch.sign(z[:,2])
    a = -1/(sign + z[:,2])
    b = z[:,0] * z[:,1] * a

    x = torch.stack( [mulsign(((z[:,0])**2) * a, z[:,2]) + 1, mulsign(b, z[:,2]), mulsign_neg(z[:,0], z[:,2])], 1)
    y = torch.stack( [b, z[:,1]*(z[:,1]*a) + sign, -z[:,1]], 1)
    ret = torch.cat([x, y], 1)

    return ret


def stokes_basis(z):
    # https://github.com/mitsuba-renderer/mitsuba3/blob/9bd47c7600b9b2e16c8bc0b5b2266e5915587345/docs/src/key_topics/polarization.rst
    xy = coordinate_system(z)
    x = xy[:, :3]

    return x



"""
2. Compute the Stokes basis in the camera and world coordinate
"""
def stokes_basis_cam(pixel_size_H=2.74e-6, pixel_size_W=2.74e-6, pixel_H=512, pixel_W=612, focal=35e-3):
    """
    Stokes basis in the camera coordinate
    """
    # Camera parameters (For real camera lucid)
    # pixel_size = 2.74e-6
    # pixel_H, pixel_W = 2048, 2448
    # focal = 35e-3
    
    # min_y, max_y = -pixel_size * pixel_H / (4*2), pixel_size * pixel_H / (4*2)
    # min_x, max_x = -pixel_size * pixel_W / (4*2), pixel_size * pixel_W / (4*2)
    min_y, max_y = -pixel_size_H * pixel_H / 2, pixel_size_H * pixel_H / 2
    min_x, max_x = -pixel_size_W * pixel_W / 2, pixel_size_W * pixel_W / 2
    
    
    y, x = torch.meshgrid(torch.arange(min_y, max_y, pixel_size_H), torch.arange(min_x, max_x, pixel_size_W), indexing='ij')
    R, C = x.shape
    
    z = torch.zeros((R, C))
    z[:] = focal
    
    # per-pixel unit vectors on the camera coordinate
    v = torch.stack([x.flatten(), y.flatten(), z.flatten()], 1)
    v = F.normalize(v, p=2, dim=1)
    v *= -1
    
    
    # compute Stokes coordinates
    xy = coordinate_system(v)
    xx = xy[:,:3]
    yy = xy[:,3:]
    
    return xx, yy, v

def stokes_basis_cam_synthetic(pixel_H=512, pixel_W=512):
    """
    Stokes basis in the camera coordinate
    Just use viewing direction of camera for synthetic dataset. (-z direction, [0,0,-1])
    """
    
    v = torch.tensor([[0., 0., -1.]]).repeat(pixel_H*pixel_W,1)

    # compute Stokes coordinates
    xy = coordinate_system(v)
    xx = xy[:,:3]
    yy = xy[:,3:]
    
    return xx, yy, v

def stokes_basis_cam_to_world(cam2world, v, xx):
    """
    Convert stokes basis from camera coordinate to world coordinate
    
    Args:
        cam2world: 4x4 conversion matrix from camera coordinate to world coordinate
        v: per-pixel unit vectors on the camera coordinate
        xx: Stokes basis in camera coordinate
    Returns:
        xx_world:
        xx_world_target: 
    """
    
    # compute the vz in the world coordinate
    v_world = (cam2world @ v.T).T
    xx_world = (cam2world @ xx.T).T
    
    # compute the Stokes coordinate in the world coordinate
    xx_world_target = stokes_basis(v_world)
    
    return xx_world, xx_world_target


"""
3. Rotate the Stokes vector
"""
def unit_angle(a, b):
    dot_uv = (a*b).sum(axis=1)
    temp = 2*torch.asin(0.5 * torch.norm(b - mulsign(a, dot_uv), p=2, dim=1))
    ret = temp
    ret[dot_uv<0] = torch.pi - temp[dot_uv<0]

    return ret[:,None]


def rotator(theta):
    N = theta.shape[0]
    s, c = torch.sin(2*theta.flatten()), torch.cos(2*theta.flatten())
    rot = torch.zeros((N,4,4))
    
    rot[:,0,0] = 1
    rot[:,3,3] = 1

    rot[:,1,1] = c
    rot[:,1,2] = s
    rot[:,2,1] = -s
    rot[:,2,2] = c

    return rot


def rotate_mat_stokes_basis(forward, basis_current, basis_target):
    c = F.normalize(basis_current, p=2, dim=1)
    t = F.normalize(basis_target, p=2, dim=1)

    theta = unit_angle(c, t)
    mask = (forward*torch.cross(basis_current, basis_target)).sum(axis=-1) < 0
    theta[mask] *= -1

    return rotator(theta)


def rotate_stokes_vector(M_rot, stoke_current):
    """
    Rotate stokes vector using rotation matrix
    
    Args:
        M_rot: (pixel_H x pixel_W) x 4 x 4 rotation matrix
        stoke_current: Stokes vector from 'exr' files - (c=4, H, W) shape
    Returns:
        stoke_current: Rotated stokes vector - (c=4, H, W) shape
    """
    H, W, c = stoke_current.shape
    N = M_rot.shape[0]
    
    stoke_current = stoke_current.transpose(2, 0, 1).reshape(c, H*W)
    stoke_target = np.zeros(stoke_current.shape)
    
    M_rot = M_rot.cpu()
    
    stoke_target = np.einsum('ijk,ik->ij',M_rot, stoke_current.transpose(1,0)).transpose(1,0) # [c=4, pixel_H x pixel_W]
    stoke_target = stoke_target.reshape(c, H, W).astype(np.float32)
    
    return stoke_target


def meas2tgt(stokes_meas, poses, cam_kwargs={}):
    """
    Convert stoke vectors following measurement basis to global basis
    
    Args:
        stokes_meas: measured stoke vector, [H,W,4]
        poses: cam2world matrix, [4,4]
    Returns:
        stokes_target: Rotated stokes vector [H,W,4]
    """
    assert stokes_meas.ndim == 3 and poses.ndim == 2, f"dimensions of input stoke and pose are not 3 and 2, but {stokes_meas.ndim} and {poses.ndim}"
    
    cam2world = torch.tensor(poses[:3,:3]) # rotation matrix.
    
    x_cam, y_cam, v_cam = stokes_basis_cam(**cam_kwargs)
    x_world, x_world_target = stokes_basis_cam_to_world(cam2world, v_cam, x_cam)
    
    rot_mat = rotate_mat_stokes_basis(v_cam, x_world, x_world_target)
    stokes_target = rotate_stokes_vector(rot_mat, stokes_meas)

    return stokes_target.transpose(1,2,0) # [C,H,W] -> [H,W,C]


def meas2tgt_synthetic(stokes_meas, poses, cam_kwargs={}):
    """
    Convert stoke vectors following measurement basis to global basis
    
    Args:
        stokes_meas: measured stoke vector, [H,W,4]
        poses: cam2world matrix, [4,4]
    Returns:
        stokes_target: Rotated stokes vector [H,W,4]
    """
    assert stokes_meas.ndim == 3 and poses.ndim == 2, f"dimensions of input stoke and pose are not 3 and 2, but {stokes_meas.ndim} and {poses.ndim}"
    
    cam2world = torch.tensor(poses[:3,:3]) # rotation matrix.
    
    x_cam, y_cam, v_cam = stokes_basis_cam_synthetic(**cam_kwargs) # default resolution : 512 x 512
    x_world, x_world_target = stokes_basis_cam_to_world(cam2world, v_cam, x_cam)
    
    rot_mat = rotate_mat_stokes_basis(v_cam, x_world, x_world_target)
    stokes_target = rotate_stokes_vector(rot_mat, stokes_meas)

    return stokes_target.transpose(1,2,0) # [C,H,W] -> [H,W,C]


def tgt2meas(stokes_global, poses, cam_kwargs={}):
    """
    Convert stoke vectors following measurement global to local basis
    
    Args:
        stokes_meas: Stokes vector in global coordinate, [H,W,4]
        poses: cam2world matrix, [4,4]
    Returns:
        stokes_target: Stokes vector in local coordinate [H,W,4]
    """
    assert stokes_global.ndim == 3 and poses.ndim == 2, f"dimensions of input stoke and pose are not 3 and 2, but {stokes_meas.ndim} and {poses.ndim}"
    
    cam2world = torch.tensor(poses[:3,:3]) # rotation matrix.
    
    x_cam, y_cam, v_cam = stokes_basis_cam(**cam_kwargs)
    x_world, x_world_target = stokes_basis_cam_to_world(cam2world, v_cam, x_cam)
    
    rot_mat = rotate_mat_stokes_basis(v_cam, x_world, x_world_target)
    rot_mat_inv = torch.linalg.inv(rot_mat)
    
    stokes_local = rotate_stokes_vector(rot_mat_inv, stokes_global)

    return stokes_local.transpose(1,2,0) # [C,H,W] -> [H,W,C]


def tgt2meas_synthetic(stokes_global, poses, cam_kwargs={}):
    """
    Convert stoke vectors following global basis to local basis
    
    Args:
        stokes_global: Stokes vector in global coordinate, [H,W,4]
        poses: cam2world matrix, [4,4]
    Returns:
        stokes_local: Stokes vector in local coordinate, [H,W,4]
    """
    assert stokes_global.ndim == 3 and poses.ndim == 2, f"dimensions of input stoke and pose are not 3 and 2, but {stokes_meas.ndim} and {poses.ndim}"
    
    cam2world = torch.tensor(poses[:3,:3]) # rotation matrix.
    
    x_cam, y_cam, v_cam = stokes_basis_cam_synthetic(**cam_kwargs) # default resolution : 512 x 512
    x_world, x_world_target = stokes_basis_cam_to_world(cam2world, v_cam, x_cam)
    
    rot_mat = rotate_mat_stokes_basis(v_cam, x_world, x_world_target)
    rot_mat_inv = torch.linalg.inv(rot_mat)
    
    stokes_local = rotate_stokes_vector(rot_mat_inv, stokes_global)

    return stokes_local.transpose(1,2,0) # [C,H,W] -> [H,W,C]