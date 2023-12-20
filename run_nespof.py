import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
import cv2

import torch.nn.functional as F
from tqdm import tqdm, trange

import OpenEXR
import Imath
import datetime

from run_nespof_helpers import *
from utils.stokes_basis_rotate import *
from utils.stokes_params_convert import *
from load_real import *
from load_synthetic import *
from stokes_visualization import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, wavelength, fn, embed_fn, embeddirs_fn, embedwaves_fn, netchunk=1024*64, wave_both=True):
    """Prepares inputs and applies network 'fn'.
    
    # inputs: [N_rays, N_samples, 3]
    # viewdirs: [N_rays, 3]
    # wavelength: [N_rays, 1]
    # wave_both: bool. If True, concatenate wavelength input in both input_pts and input_views
    """
    N_rays, N_samples, _ = inputs.shape
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    
    if wave_both == True:
        pwaves = (wavelength[:,None].expand([N_rays, N_samples, 1]) - 550) / 100
        pwaves_flat = torch.reshape(pwaves, [-1, pwaves.shape[-1]])
        
        embedded_pwaves = embedwaves_fn(pwaves_flat)
        embedded = torch.cat([embedded, embedded_pwaves], -1)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        
        vwaves = (wavelength[:,None].expand([N_rays, N_samples, 1]) - 550) / 100
        vwaves_flat = torch.reshape(vwaves, [-1, vwaves.shape[-1]])
        
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded_vwaves = embedwaves_fn(vwaves_flat)
        
        embedded_dirs = torch.cat([embedded_dirs, embedded_vwaves], -1)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    
    return outputs


def batchify_rays(wavelength, rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(wavelength[i:i+chunk], rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    
    return all_ret


def render(wavelength, H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      wavelength: array of shape [B*H*W, 1] or [B,H,W,1]. Wavelength map.
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
       
    Returns:
      stoke_map: [batch_size, 4]. Predicted stoke values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    assert wavelength.dim() == 2 or wavelength.dim() == 4, "Check the wavelength's dimension!!! (should be [B,H,W,1] or [B*H*W,1])"
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    
    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray & wave batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    wavelength = torch.reshape(wavelength, [-1,1]).float() # [B*H*W,1]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    
    # Render and reshape
    all_ret = batchify_rays(wavelength, rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    
    k_extract = ['stoke_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    
    return ret_list + [ret_dict]


def render_path(wavelength, render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    # wavelength: int
    assert isinstance(wavelength, int), "Input integer wavelength for 'render_path' function!!!"
    
    H, W, focal = hwf
    
    if render_factor!=0:            
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    
    wavelength = torch.tensor(list2map(np.array([wavelength]), H, W)).to(device) # int:wave -> []:wave -> [1,H,W,1]:wave
        
    stokes = []
    disps = []
    
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        
        if wavelength.shape[0] == 1:
            stoke, disp, acc, _ = render(wavelength, H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        else:
            stoke, disp, acc, _ = render(wavelength[i:i+1], H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        
        cam_kwargs={}
        cam_kwargs['pixel_H'] = H
        cam_kwargs['pixel_W'] = W
        
        stoke = tgt2meas(stoke.cpu().numpy(), c2w, cam_kwargs)    # Global coordinate  ->  Local coordinate
        
        stokes.append(stoke)
        disps.append(disp.cpu().numpy())
        
        if i==0:
            print(stoke.shape, disp.shape)
        
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        
        if savedir is not None:
            fname = os.path.join(savedir, 'exr', f'{i}.exr')
            os.makedirs(os.path.join(savedir, 'exr'), exist_ok=True)
            os.makedirs(os.path.join(savedir, 'disp'), exist_ok=True)
            
            s0, s1, s2, s3 = stokes[-1][:, :, 0], stokes[-1][:, :, 1], stokes[-1][:, :, 2], stokes[-1][:, :, 3]
            
            h = OpenEXR.Header(s0.shape[1], s0.shape[0])
            ctype = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
            h["channels"] = {"s0":ctype, "s1":ctype, "s2":ctype, "s3":ctype}
            exr = OpenEXR.OutputFile(fname, h)
            exr.writePixels({'s0': s0.tostring(), 's1': s1.tostring(), 's2': s2.tostring(),  's3': s3.tostring()})
            exr.close()

            fname = os.path.join(savedir, 'disp', f'{i}.png')
            fig = plt.figure(figsize = (5,5)) # Your image (W)idth and (H)eight
            
            # Stretch image to full figure, removing "grey region"
            plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
            im = plt.imshow(disps[-1], cmap='magma') # Show the image
            pos = fig.add_axes([0.8,0.1,0.02,0.35]) # Set colorbar position in fig
            fig.colorbar(im, cax=pos) # Create the colorbar
            plt.savefig(fname)
            plt.close(fig)

    stokes = np.stack(stokes, 0)    
    disps = np.stack(disps, 0)
    
    return stokes, disps


def create_nespof(args):
    """Instantiate NeSpoF's MLP model.
    """
    embed_fn, input_ch = get_embedder_pts(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder_views(args.multires_views, args.i_embed)
        
    embedwaves_fn, input_ch_waves = get_embedder_waves(args.multires_waves, args.i_embed)
    
    output_ch = 6 if args.N_importance > 0 else 5
    skips = [4]
    
    if args.wave_both == True:
        input_ch_ = input_ch+input_ch_waves
    else:
        input_ch_ = input_ch
    
    input_ch_views_ = input_ch_views+input_ch_waves
    
    model = NeSpoF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch_, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views_, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeSpoF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch_, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views_, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, wavelength, network_fn : run_network(inputs, viewdirs, wavelength, network_fn,
                                                                                    embed_fn=embed_fn,
                                                                                    embeddirs_fn=embeddirs_fn,
                                                                                    embedwaves_fn=embedwaves_fn,
                                                                                    netchunk=args.netchunk,
                                                                                    wave_both=args.wave_both)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'intermediate' : args.intermediate
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'real' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, intermediate=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 5]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        intermediate: bool. If True, use intermediate representation to compute Stokes vector
        
    Returns:
        stoke_map: [num_rays, 4]. Estimated stoke values of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    # Choose whether we use intermediate representation
    if intermediate:
        stokes_params = torch.cat([torch.sigmoid(raw[...,0:2]), raw[...,2:4]], dim=-1) # s0, dop ~ [0,1], chi, psi
        stoke = params_to_stokes(stokes_params)
        
    else:
        stoke = torch.cat([torch.sigmoid(raw[...,0:1]), 2 * (torch.sigmoid(raw[...,1:4]) - 0.5)], dim=-1) # s0 ~ [0,1], s1~s3 ~ [-1,1]

    noise = 0.
    
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,4].shape) * raw_noise_std
    
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,4].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    alpha = raw2alpha(raw[...,4] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    stoke_map = torch.sum(weights[...,None] * stoke, -2)  # [N_rays, 4] S0~S3
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        stoke_map_tmp = torch.zeros_like(stoke_map)
        stoke_map_tmp[...,0:1] = stoke_map[...,0:1] + (1.-acc_map[...,None]) # For intensity ([0,1]), make empty space to 1 (white)
        stoke_map_tmp[...,1:] = stoke_map[...,1:] * acc_map[...,None] # For stoke vector ([-1,1]), make empty space to 0 (nothing)
        stoke_map = stoke_map_tmp
        
    return stoke_map, disp_map, acc_map, weights, depth_map


def render_rays(wavelength,
                ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                intermediate=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting stokes vector and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
      intermediate: bool. If True, use intermediate representation to compute Stokes vector
      
    Returns:
      stoke_map: [num_rays, 4]. Estimated stokes value of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      stoke0: See stoke_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]    
        
    raw = network_query_fn(pts, viewdirs, wavelength, network_fn)
    stoke_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, intermediate=intermediate)
    
    if N_importance > 0:
        stoke_map_0, disp_map_0, acc_map_0, weights_0, z_vals_0, depth_map_0 = stoke_map, disp_map, acc_map, weights, z_vals, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, wavelength, run_fn)
        
        stoke_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, intermediate=intermediate)

    ret = {'stoke_map' : stoke_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'weights' : weights, 'z_vals' : z_vals, 'depth_map' : depth_map}
    
    if retraw:
        ret['raw'] = raw
        
    if N_importance > 0:
        ret['stoke0'] = stoke_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['weights_0'] = weights_0
        ret['z_vals_0'] = z_vals_0
        ret['depth_map_0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_polarization_cues(stoke, gamma):
    R, C, _ = stoke.shape
    _, dop, top, aolp, cop, _ = polar_inf(stoke)
    
    pol_inf = [dop, top, aolp, cop]
    for index, p in enumerate(pol_inf):
        pol_inf[index] = p.reshape(R, C, 3)
        
    dop, top, aolp, cop = pol_inf
    dop = (dop**(1/gamma)).real
    top = (top**(1/gamma)).real
    alop = (aolp**(1/gamma)).real
    cop = cop**(1/gamma)
    
    return dop, top, aolp, cop


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--debug", type=bool, default=False, 
                        help='whether to debug...')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=512*16, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=512*32, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--loss_mode", type=str, default='vanilla', 
                        help='use [vanilla / polar / static / dynamic] for MSE loss')
    parser.add_argument('--w_scale', type=list, nargs='+')
    parser.add_argument('--w_scale_multiplier', type=float, default=1.)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument("--use_w_reg", action='store_true', 
                        help='whether to use weight variance regularizer')
    parser.add_argument('--weight_reg', type=float, default=1e-3)

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--wave_both", action='store_true', 
                        help='concatenate wavelength input in only input_views or both input_pts and input_views')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_waves", type=int, default=1, 
                        help='log2 of max freq for positional encoding (1D wavelength)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--target_wavelength", type=int, default=550, 
                        help='wavelength for visualization in rendering, set from 450 to 650 integer.')
    parser.add_argument("--scale_rads", type=float, default=0.1, 
                        help='radius of circle near the median pose during rendering.')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, 
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='real', 
                        help='options: real / synthetic ')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--rotate_stoke", action='store_true', 
                        help='Whether to rotate stoke vector for real scene data (measurement -> global coordinates)')
    parser.add_argument("--scale_factor", type=float, default=1., 
                        help='scaling signal of data for real scene..')
    parser.add_argument("--intermediate", action='store_true', 
                        help='set to train the network ouput with(out) interemediate representation..')
    parser.add_argument("--model_type", action='store_true',
                        help='original/spec/pol/spec_pol')
    
    
    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    
    # logging/saving options
    parser.add_argument("--N_iters",   type=int, default=10000, 
                        help='Total iterations for training NeSpoF')
    
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_val",     type=int, default=1000,
                        help='frequency of validation loss logging')
    parser.add_argument("--i_img",     type=int, default=10000,
                        help='frequency of image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video",   type=int, default=10000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_testset", type=int, default=10000, 
                        help='frequency of testset saving')
    
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    
    
    if args.debug:
        print("="*23)
        print("| Start DEBUG Mode!!! |")
        print("="*23)

    # Load openexr data
    K = None
    print(f"start loading exr data... takes long time...")
    print(f"Rotate_Stoke : {args.rotate_stoke}")

    assert args.loss_mode in ['vanilla', 'polar', 'static', 'dynamic'], "Choose loss_mode among vanilla / polar / static / dynamic"
    print(f"USE LOSS MODE : {args.loss_mode}")
    
    if args.dataset_type == 'synthetic':
        waves, images, poses, render_poses, hwf, i_split, w_scale = load_synthetic_data(args.datadir, args.half_res, args.testskip, rotate_stoke=args.rotate_stoke, render_only=args.render_only, target_wavelength=args.target_wavelength, debug=args.debug, scale_rads=args.scale_rads)
        print(f'Loaded openexr', images.shape, render_poses.shape, hwf, args.datadir)
        
        i_train, i_val, i_test = i_split
        
        near = 2.
        far = 6.
        
        images[..., 0] = np.clip(images[..., 0], 0, 1)        # Clip the s0 value in range [0, 1]
        images[..., 1:4] = np.clip(images[..., 1:4], -1, 1)     # Clip the s1~s3 value in range [-1, 1]
        
        if args.white_bkgd:
            images[..., -1] = np.clip(images[..., -1], 0, 1)        # Clip the alpha value in range [0, 1]
            images[...,:1] = images[...,:1]*images[...,-1:] + (1.-images[...,-1:]) # if alpha == 0 (background) :-> make pixel white. (5-channel -> 4-channel), only for s0 (intensity)
            
        images = images[...,:4]
        

    elif args.dataset_type == 'real':
        waves, images, poses, bds, render_poses, i_test, w_scale = load_real_data(args.datadir, recenter=True, bd_factor=.75, spherify=args.spherify, rotate_stoke=args.rotate_stoke, scale_factor = args.scale_factor, render_only=args.render_only, target_wavelength=args.target_wavelength, debug=args.debug, scale_rads=args.scale_rads)
        
        images[..., 0] = np.clip(images[..., 0], 0, 1)        # Clip the s0 value in range [0, 1]
        images[..., 1:] = np.clip(images[..., 1:], -1, 1)     # Clip the s1~s3 value in range [-1, 1]
        
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        
        print('Loaded real scene', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto real scene holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far) 
        
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    
    if args.w_scale is not None and args.loss_mode == 'static':
        print("USE PRE-DEFINED W_SCALE for MSE LOSS")
        w_scale = [int("".join(i)) for i in args.w_scale]
        w_scale = np.array(w_scale)
        w_scale[...,1:] = w_scale[...,1:] * args.w_scale_multiplier
        
    elif args.loss_mode == 'static':
        w_scale[...,1:] = w_scale[...,1:] * args.w_scale_multiplier
        print("USE W_SCALE FROM DATASETS for MSE LOSS")
        
    elif args.loss_mode == 'dynamic':
        print("USE DYNAMIC W_SCALE for MSE LOSS (following RawNeRF)")
        print(f"eps : {args.eps}")
        
    w_scale = np.maximum(w_scale, 1.0)
    w_scale = torch.tensor(w_scale).to(device) # shape [4]
    print(f"|[w_scale]: {w_scale}|")

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create nespof model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nespof(args)
    
    global_step = start
    
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        # print('done, concats')
        
        # 'images' shape = [N, H, W, 4] (4 - Stokes vector s0~s3)
        # 'waves' shape = [N, H, W, 1]
        print('shuffle rays')
        rays_ = np.transpose(rays, [0,2,3,1,4])
        rays_ = np.stack([rays_[i] for i in i_train], 0) # train only
        rays_ = np.reshape(rays_, [-1,2,3])
        rays_ = rays_.astype(np.float32)
        np.random.seed(0)
        np.random.shuffle(rays_)
        
        print('shuffle targets')        
        images_ = np.stack([images[i] for i in i_train], 0) # [N_train, H, W, 4]
        images_ = np.reshape(images_, [-1,4]) # [N_train*H*W, 4]
        np.random.seed(0)
        np.random.shuffle(images_)
        
        print('shuffle waves')
        waves_ = np.stack([waves[i] for i in i_train], 0) # [N_train, H, W, 1]
        waves_ = np.reshape(waves_, [-1,1]) # [N_train*H*W, 1]
        np.random.seed(0)
        np.random.shuffle(waves_)
        
        print('done')
        
        i_batch = 0

    # Move training data to GPU
    # poses = torch.Tensor(poses).to(device)
    
    # N_iters = 200000 + 1
    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    
    try:
        for i in trange(start, N_iters):
            time0 = time.time()
            
            # Sample random ray batch
            if use_batching:
                # Random over all images
                batch_rays = torch.transpose(torch.Tensor(rays_[i_batch:i_batch+N_rand]), 0, 1)[:2].to(device) # [B, 2, 3] -> [2, B, 3]
                target_s = torch.Tensor(images_[i_batch:i_batch+N_rand]).to(device) # [B, 4]
                wavelength = torch.Tensor(waves_[i_batch:i_batch+N_rand]).to(device) # [B, 1]
                
                i_batch += N_rand
                if i_batch >= rays_.shape[0]:
                    print("Shuffle data after an epoch!")
                    
                    rand_idx = np.random.choice(rays_.shape[0], size=rays_.shape[0], replace=False)
                    rays_ = rays_[rand_idx]
                    images_ = images_[rand_idx]
                    waves_ = waves_[rand_idx]
                    i_batch = 0
                
            else:      
                # Random from one image
                img_i = np.random.choice(i_train)
                wavelength = torch.Tensor(waves[img_i]).to(device)   # [H,W,1]
                target = images[img_i]      # [H,W,4]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 4)
                    wavelength = wavelength[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
            
            #####  Core optimization loop  #####
            stoke, disp, acc, extras = render(wavelength, H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)
            
            optimizer.zero_grad()
            if args.loss_mode == 'vanilla':
                img_loss = img2mse(stoke, target_s)
            elif args.loss_mode == 'polar':
                img_loss = img2mse_polar(stoke, target_s)
            elif args.loss_mode == 'static':
                img_loss = img2mse_static_scaled(stoke, target_s, w_scale)
            else:
                img_loss = img2mse_dynamic_scaled(stoke, target_s, args.eps)
                
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)
            REAL_psnr = mse2psnr(img2mse(stoke.detach(), target_s.detach()))
            
            # Weight variance regularizer
            if args.use_w_reg == True:
                w_reg = w_var_reg(extras['weights'], extras['z_vals'], extras['depth_map'])
                loss = loss + args.weight_reg * w_reg
            
            if 'stoke0' in extras:
                if args.loss_mode == 'vanilla':
                    img_loss0 = img2mse(extras['stoke0'], target_s)
                elif args.loss_mode == 'polar':
                    img_loss0 = img2mse_polar(extras['stoke0'], target_s)
                elif args.loss_mode == 'static':
                    img_loss0 = img2mse_static_scaled(extras['stoke0'], target_s, w_scale)
                else:
                    img_loss0 = img2mse_dynamic_scaled(extras['stoke0'], target_s, args.eps)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
                
                if args.use_w_reg == True:
                    w_reg0 = w_var_reg(extras['weights_0'], extras['z_vals_0'], extras['depth_map_0'])
                    loss = loss + args.weight_reg * w_reg0

            loss.backward()
            optimizer.step()
            
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####
                
            # Rest is logging
            if i%args.i_weights==0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            
            
            if i%args.i_testset==0 and i > 0 and args.render_only==False:
                target_waves = [450, 550, 650]
                img_i = np.random.choice(i_val)
                img_i = i_val[0]                
                
                for w in target_waves:
                    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i), str(w))
                    os.makedirs(os.path.join(testsavedir, 'exr'), exist_ok=True)
                    os.makedirs(os.path.join(testsavedir, 'disp'), exist_ok=True)
                    
                    print('test poses shape', poses[img_i].shape)
                    
                    with torch.no_grad():
                        # render_path(w, torch.Tensor(render_poses.to(device)), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                        render_path(w, torch.Tensor(poses[img_i].unsqueeze(0)).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                        
                    print('Saved test set')
                       

            if i%args.i_video==0 and i > 0 and args.render_only:
                target_wavelength = args.target_wavelength
                print('RENDER ONLY')
                testsavedir = os.path.join(basedir, expname)
                os.makedirs(testsavedir, exist_ok=True)
                
                with torch.no_grad():
                    if args.render_test:
                        # render_test switches to test poses
                        images = images[i_test]
                    else:
                        # Default is smoother render_poses path
                        images = None
                        
                    # 1) CASE 1.
                    # For fixed viewpoint, S0, AoLP, DoP, CoP, ToP videos according to wavelength
                    print("Render fixed viewpoint & moving wavelength")
                    stretch = 2
                    start, stop, interval = 450, 650, 10
                    wave_render = range(start, stop+1, interval)
                    pose_median = torch.tensor(poses_median(poses)[np.newaxis,:,:]).to(device)
                    
                    H_, W_, f_ = hwf
                    
                    if args.render_factor == 0:
                        assert args.render_factor > 0, "render_factor should be larger than 0"
                        H_, W_ = H_ // args.render_factor, W_ // args.render_factor
                    
                    save_path_fixed = os.path.join(testsavedir, 'fixed')
                    os.makedirs(os.path.join(save_path_fixed, 's0'), exist_ok = True)
                    os.makedirs(os.path.join(save_path_fixed, 's1'), exist_ok = True)
                    os.makedirs(os.path.join(save_path_fixed, 's2'), exist_ok = True)
                    os.makedirs(os.path.join(save_path_fixed, 's3'), exist_ok = True)
                    
                    list_0 = []
                    list_1 = []
                    list_2 = []
                    list_3 = []
                    
                    for w in wave_render:
                        stokes_render, _ = render_path(w, pose_median, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
                                                
                        dop, top, aolp, cop = get_polarization_cues(stokes_render[0], gamma=2.2) # [H,W,4] -> [H,W,3]
                        dop, top, aolp, cop = to8b(dop), to8b(top), to8b(aolp), to8b(cop)
                        
                        s0 = to8b(stokes_render[0,...,0]) # [H,W]
                        s1 = to8b(stokes_render[0,...,1]) # [H,W]
                        s2 = to8b(stokes_render[0,...,2]) # [H,W]
                        s3 = to8b(stokes_render[0,...,3]) # [H,W]
                        
                        imageio.imwrite(os.path.join(save_path_fixed, 's0', str(w)+'nm.png'), s0)
                        imageio.imwrite(os.path.join(save_path_fixed, 's1', str(w)+'nm.png'), s1)
                        imageio.imwrite(os.path.join(save_path_fixed, 's2', str(w)+'nm.png'), s2)
                        imageio.imwrite(os.path.join(save_path_fixed, 's3', str(w)+'nm.png'), s3)
                        
                        cv2.putText(s0, str(w)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        cv2.putText(s1, str(w)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        cv2.putText(s2, str(w)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        cv2.putText(s3, str(w)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        
                        list_0.append(s0)
                        list_1.append(s1)
                        list_2.append(s2)
                        list_3.append(s3)
                                                
                    list_0 = np.stack(list_0, axis=0)
                    list_1 = np.stack(list_1, axis=0)
                    list_2 = np.stack(list_2, axis=0)
                    list_3 = np.stack(list_3, axis=0)
                                        
                    imageio.mimwrite(os.path.join(testsavedir, 's0_fixed.mp4'), list_0, fps=7, quality=8)
                    imageio.mimwrite(os.path.join(testsavedir, 's1_fixed.mp4'), list_1, fps=7, quality=8)
                    imageio.mimwrite(os.path.join(testsavedir, 's2_fixed.mp4'), list_2, fps=7, quality=8)
                    imageio.mimwrite(os.path.join(testsavedir, 's3_fixed.mp4'), list_3, fps=7, quality=8)
                    
                    
                    # 2) CASE 2.
                    # For fixed wavelength, S0, AoLP, DoP, CoP, ToP videos according to viewpoint (spiral_path)
                    print("Render moving viewpoints & fixed wavelength")
                    print('test poses shape', render_poses.shape)

                    save_path_moving = os.path.join(testsavedir, 'moving')
                    os.makedirs(os.path.join(save_path_moving, 's0'), exist_ok = True)
                    os.makedirs(os.path.join(save_path_moving, 's1'), exist_ok = True)
                    os.makedirs(os.path.join(save_path_moving, 's2'), exist_ok = True)
                    os.makedirs(os.path.join(save_path_moving, 's3'), exist_ok = True)
                    
                    stokes, _ = render_path(target_wavelength, render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor) # [B,H,W,4]
                    list_0 = []
                    list_1 = []
                    list_2 = []
                    list_3 = []
                    
                    for ii, s in enumerate(stokes): # [H,W,4]
                        s0 = to8b(s[...,0]) # [H,W]
                        s1 = to8b(s[...,1]) # [H,W]
                        s2 = to8b(s[...,2]) # [H,W]
                        s3 = to8b(s[...,3]) # [H,W]
                        
                        imageio.imwrite(os.path.join(save_path_moving, 's0', str(target_wavelength)+f'nm_idx{ii:03d}.png'), s0)
                        imageio.imwrite(os.path.join(save_path_moving, 's1', str(target_wavelength)+f'nm_idx{ii:03d}.png'), s1)
                        imageio.imwrite(os.path.join(save_path_moving, 's2', str(target_wavelength)+f'nm_idx{ii:03d}.png'), s2)
                        imageio.imwrite(os.path.join(save_path_moving, 's3', str(target_wavelength)+f'nm_idx{ii:03d}.png'), s3)
                        
                        cv2.putText(s0, str(target_wavelength)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        cv2.putText(s1, str(target_wavelength)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        cv2.putText(s2, str(target_wavelength)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        cv2.putText(s3, str(target_wavelength)+'nm', (int(W_*0.76), int(H_*0.08)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_8)     # fontscale = 2, thickness = 3
                        
                        list_0.append(s0)
                        list_1.append(s1)
                        list_2.append(s2)
                        list_3.append(s3)
                        
                        # dop, top, aolp, cop = get_polarization_cues(s, gamma=2.2)
                                                                    
                    imageio.mimwrite(os.path.join(testsavedir, 's0_moving.mp4'), list_0, fps=30, quality=8)
                    imageio.mimwrite(os.path.join(testsavedir, 's1_moving.mp4'), list_1, fps=30, quality=8)
                    imageio.mimwrite(os.path.join(testsavedir, 's2_moving.mp4'), list_2, fps=30, quality=8)
                    imageio.mimwrite(os.path.join(testsavedir, 's3_moving.mp4'), list_3, fps=30, quality=8)
                                        
                    print('Done rendering')
                    

            if i%args.i_print==0:
                stats_dict = {}
                stats_dict["train/loss"] = loss.item()
                stats_dict["train/PSNR"] = psnr.item()
                if args.use_w_reg:
                    stats_dict["train/W_REG"] = w_reg.item()
                    tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  REAL_PSNR: {REAL_psnr.item()}  W_REG: {w_reg.item()}")
                else:
                    tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  REAL_PSNR: {REAL_psnr.item()}")
            
            
            if i%args.i_val==0:
                img_i=np.random.choice(i_val)
                target = images[img_i]
                target = torch.Tensor(target).to(device)
                
                wavelength = torch.Tensor(waves[img_i:img_i+1]).to(device) # [1,H,W,1]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    stoke, disp, acc, extras = render(wavelength, H, W, K, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)
                val_loss = img2mse(stoke, target)
                psnr = mse2psnr(val_loss)
                stats_dict = {}
                stats_dict["val/loss"] = loss.item()
                stats_dict["val/PSNR"] = psnr.item()

            global_step += 1
        
    except KeyboardInterrupt:
        print("abort!")
        


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()