import numpy as np
import torch

def stokes_to_params(stokes, eps=1e-7):
    '''
        Args.
        - stokes (torch.Tensor): [N,H,W,4]
        - eps = 1e-10      # Prevent that the denominator becomes zero
        
        Return.
        - stokes_param (torch.Tensor): [N,H,W,4]
    '''
    N, H, W = stokes.shape[:3]
    
    stokes = stokes.reshape(N, H*W, 4)
    stokes[..., 0] = np.clip(stokes[..., 0], 0, 1)     # Clip the s0 value in range [0, 1]
    stokes[..., 1:4] = np.clip(stokes[..., 1:4], -1, 1)     # Clip the s1~s3 value in range [-1, 1]

    stokes_param = []
    
    for i in range(N):
        for j in range(H*W):
            s0, s1, s2, s3 = stokes[i, j, 0]+eps, stokes[i, j, 1]+eps, stokes[i, j, 2], stokes[i, j, 3]
            
            dop = np.sqrt(s1**2 + s2**2 + s3**2) / s0
            dop = np.clip(dop, 0, 1) + eps
            
            chi = 0.5 * np.arctan(s3 / np.sqrt(s1**2+s2**2))
            psi = 0.5 * np.arctan(s2 / s1)

            stokes_param.append(np.array([s0, dop, chi, psi]))    # (H, W)
        
    stokes_param = np.array(stokes_param).reshape(N, H, W, 4).astype(np.float32)
    stokes_param = torch.from_numpy(stokes_param)
    
    return stokes_param   # (N,H,W,4)


def params_to_stokes(stokes_params):
    '''
        Args.
        - stokes_param (torch.Tensor): [N_rays,N_samples,4]
        
        Return.
        - stokes (torch.Tensor): [N_rays,N_samples,4]
    '''
    
    s0, dop, chi, psi = stokes_params[..., 0], stokes_params[..., 1], stokes_params[..., 2], stokes_params[..., 3]
    
    s1 = s0 * dop * torch.cos(2*chi) * torch.cos(2*psi)
    s2 = s0 * dop * torch.cos(2*chi) * torch.sin(2*psi)
    s3 = s0 * dop * torch.sin(2*chi)
    
    stokes = torch.stack([s0, s1, s2, s3], dim=-1)      # (N_rays,N_samples,4)
    
    return stokes