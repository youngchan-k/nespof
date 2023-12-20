''' Reference
https://github.com/mitsuba-renderer/mitsuba2/issues/470
https://dl.acm.org/doi/pdf/10.1145/1925059.1925070      '''

import numpy as np
import numpy.matlib as mlib
import matplotlib.pyplot as plt
import os


# Calculate the polarimetric information
def polar_inf(s):
    R, C, _ = s.shape
    eps = 1e-10
    
    # Compute DoP
    def compute_DoP(s):
        DoP = np.sqrt(s[:, 1]**2 + s[:, 2]**2 + s[:, 3]**2) / (s[:, 0] + eps)
        
        return DoP
    
    # Compute ToP
    def compute_ToP(s):
        rDoPL = np.sqrt(s[:, 1]**2 + s[:, 2]**2) / (np.sqrt(s[:, 1]**2 + s[:, 2]**2 + s[:, 3]**2) + eps)
        rDoPC = abs(s[:, 3]) / (np.sqrt(s[:, 1]**2 + s[:, 2]**2 + s[:, 3]**2) + eps)
        
        return rDoPL, rDoPC
    
    # intensity
    intensity = np.squeeze(s[:, :, 0])
    
    # DoP
    s = s.reshape(R*C, 4)
    DoP = compute_DoP(s)    
    dop = np.tile(DoP[:, np.newaxis], (1,3)) * mlib.repmat([1,0,0], R*C, 1)
    
    # ToP
    rDoPL, rDoPC = compute_ToP(s)
    top = np.tile(rDoPL[:, np.newaxis], (1,3)) * mlib.repmat([0,1,1], R*C, 1) + np.tile(rDoPC[:, np.newaxis], (1,3)) * mlib.repmat([1,1,0], R*C, 1)
    top = top*np.tile(DoP[:, np.newaxis], (1,3))
                
    # AoLP
    sn = s / (np.tile(s[:, 0][:, np.newaxis], (1,4)) + eps)
    
    s2 = np.zeros((R*C, 3))
    s2[:, 0] = sn[:, 1]
    for i in range(sn[:, 1].shape[0]):
        if sn[i, 1] >= 0:
            s2[i, 0] = 0 
            s2[i, 1] = sn[i, 1]     # postivie value as green            
        else:
            s2[i, 0] = -sn[i, 1]    # negative value as red
            s2[i, 1] = 0
    
    s3 = np.zeros((R*C, 3))
    s3[:, 2] = sn[:, 2]
    for j in range(sn[:, 2].shape[0]):
        if sn[j, 2] >= 0:
            s3[j, 0:2] = mlib.repmat(sn[j, 2], 1, 2)    # positive value as yellow
            s3[j, 2] = 0 
        else:
            s3[j, 0:2] = 0
            s3[j, 2] = -sn[j, 2]    # negative value as blue 
    
    s4 = (s2 + s3) / 2
    s2 = s2.reshape(R, C, 3)
    s3 = s3.reshape(R, C, 3)
    aolp = s4*np.tile(rDoPL[:, np.newaxis], (1,3))
    
    # CoP
    cop = np.zeros((R*C, 3))
    for k in range(sn[:, 3].shape[0]):
        if sn[k, 3] >= 0:
            cop[k, 2] = sn[k, 3]    # positive value as blue
        
        else:
            cop[k, 0:2] = -mlib.repmat(sn[k, 3], 1, 2)    # negative value as yellow
    
    # Set range from 0 to 1
    pol_inf = [dop, top, aolp, cop]
    
    for index, p in enumerate(pol_inf):
        p = np.clip(p, 0, 1) # p ~ [0,1]                    
        pol_inf[index] = p
        
    dop, top, aolp, cop = pol_inf    
        
    # Mask
    mask = (np.isnan(dop) + np.isinf(dop) + np.isnan(top) + np.isinf(top) + np.isnan(aolp) + np.isinf(aolp) + np.isnan(cop) + np.isinf(cop)) == 0
    
    return intensity, dop, top, aolp, cop, mask


# Polarimetric visualization
def polar_vis(s, path, gamma=2.2):
    intensity, dop, top, aolp, cop, _ = polar_inf(s)
    
    R, C, _ = s.shape
    
    pol_inf = [dop, top, aolp, cop]
    for index, p in enumerate(pol_inf):
        
        pol_inf[index] = p.reshape(R, C, 3)
        
    dop, top, aolp, cop = pol_inf
    
    plt.imsave(f"{path}/intensity.png", intensity, cmap="gray")
    plt.imsave(f"{path}/DoP.png", (dop**(1/gamma)).real)
    plt.imsave(f"{path}/ToP.png", (top**(1/gamma)).real)
    plt.imsave(f"{path}/AoLP.png", (aolp**(1/gamma)).real)
    plt.imsave(f"{path}/CoP.png", cop**(1/gamma))
    
    return True