from dataclasses import field
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from matplotlib import animation 

import numpy as np
import pandas as pd
import os
import json
import argparse
import pdb

res = 512
focal_length = 50
aspect_ratio = 1


# Change the unit of focal length from mm to pixel
def focal_pixel(focal_length, aspect, res_width):
    fov = 2 * np.arctan(np.sqrt(36**2 + 24**2) / (2*focal_length)) * 180/np.pi
    sensor_diagonal = 2 * focal_length * np.tan(fov/2 * np.pi/180)
    sensor_width = sensor_diagonal / np.sqrt(1 + 1/aspect**2)
    
    camera_angle_x = 2 * np.arctan(sensor_width / (2*focal_length)) * 180/np.pi
    focal = .5 * res_width / np.tan(.5 * camera_angle_x)
    
    return focal


# Decide the size and depth of camera frustum
def points_world(focal_length, aspect, res_width): 
    size = res_width * 0.0005
    depth = focal_pixel(focal_length, aspect, res_width) * 0.005
        
    x_near, y_near, z_near = 0, 0, 0
    x_far, y_far = size * aspect, size
    z_far = depth
    
    return np.array(
            [
                [x_near, y_near, -z_near, 1],
                [-x_near, y_near, -z_near, 1],
                [x_near, -y_near, -z_near, 1],
                [-x_near, -y_near, -z_near, 1],
                [x_far, y_far, -z_far, 1],
                [-x_far, y_far, -z_far, 1],
                [x_far, -y_far, -z_far, 1],
                [-x_far, -y_far, -z_far, 1],    ],  dtype=float)
    

# 3D Plot setting and draw line of camera frustum
def render_frustum(points, camera_pos, ax):
    line_indices = [ [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7], ]
    
    for idx_pair in line_indices:
        line = np.transpose([points[idx_pair[0]], points[idx_pair[1]]])
        if idx_pair[0] < 4 and idx_pair[1] == 4:
            ax.plot(line[0], line[1], line[2], "b", linewidth=0.2)
        elif idx_pair[0] < 4 and idx_pair[1] == 5:
            ax.plot(line[0], line[1], line[2], "g", linewidth=0.2)
        elif idx_pair[0] < 4 and idx_pair[1] == 6:
            ax.plot(line[0], line[1], line[2], "m", linewidth=0.2)
        elif idx_pair[0] < 4 and idx_pair[1] == 7:
            ax.plot(line[0], line[1], line[2], "y", linewidth=0.2)
        else:
            ax.plot(line[0], line[1], line[2], "r", linewidth=0.2)
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, 4])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.axis('off')
    ax.set_box_aspect((8, 8, 4))
    
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([0, 1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("z")
    # ax.set_zlabel("y")
    # ax.axis('off')
    # ax.set_box_aspect((2, 2, 1))
    
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], marker="o", color="k", s=0.1) # X,Y,Z

def look_at(camera, target, up_):
    # camera : OpenGL convention. Camera position.
    # target : OpenGL convention. Object center.
    
    dir_ = camera - target # z-dir
    dir_ = dir_/np.linalg.norm(dir_)
    
    left = np.cross(up_, dir_) # x-dir, right?
    left = left/np.linalg.norm(left)
      
    new_up = np.cross(dir_, left) # y-dir
    
    result = np.array(
            [
                [left[0], left[1], left[2], -np.dot(left, camera)],
                [new_up[0], new_up[1], new_up[2], -np.dot(new_up, camera)],
                [dir_[0], dir_[1], dir_[2], -np.dot(dir_, camera)],
                [0, 0, 0, 1]        
            ])
    return result

    
# Plot obj in matplotlib 3D
def plot_obj(filename, color, alpha = 0.3):
    import trimesh
    mesh = trimesh.load(filename)
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)

    x = vertices[:,0]
    y = vertices[:,1]
    z = vertices[:,2]

    ax = plt.gca(projection="3d")
    ax.plot_trisurf(x, z, faces, y, alpha=alpha, color=color, shade=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot cameras in 3D-world.')
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--res', type=int, default=512, required=False)
    parser.add_argument('--focal_length', type=int, default=50, required=False)
    parser.add_argument('--aspect_ratio', type=int, default=1, required=False)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--use_lookat', type=bool, default=False, required=False)
    
    args = parser.parse_args()
      
    
    fig = plt.figure()
    # ax_world = fig.gca(projection="3d")
    ax_world = plt.axes(projection="3d")
    ax_world.set_title("Camera viewpoint")
    
    if args.mode == "ajar":
        target = np.array([0, 3.2, 1.5]) # Default convention
    elif args.mode == "cornell":
        target = np.array([0, 0, 0])
    else:
        target = np.array([0, 0, 0])
    
    points_list = points_world(args.focal_length, args.aspect_ratio, args.res)    
    
    with open(args.json_path, 'r') as fp:
        meta = json.load(fp)
    xyz_list = []
    camera_list = []
    for frame in meta['frames']:
        xyz_list.append(np.array(frame['transform_matrix'])[:3, 3])
        # pdb.set_trace()
        camera_list.append( -np.matmul(np.array(frame['transform_matrix'])[:3,:3].T, np.array(frame['transform_matrix'])[:3, 3]))
    # camera_list = [np.array([xyz_list[i][0], xyz_list[i][2], -xyz_list[i][1]]) for i in range(len(xyz_list))] # Mitsuba2 convention
    camera_list = [np.array([xyz_list[i][0], xyz_list[i][1], xyz_list[i][2]]) for i in range(len(xyz_list))] # Default convention
    if args.use_lookat:
        camera_mat_list = [look_at(camera_position, target, np.array([0, 1, 0])) for camera_position in camera_list]
    else:
        camera_mat_list = []
        for frame in meta['frames']:
            camera_mat_list.append(np.array(frame['transform_matrix']))

    # Plot target point (object center)
    ax_world.scatter(target[0], target[1], target[2], marker="o", color="c", s=0.2)
    # pdb.set_trace()
    # for camera_position in camera_list:
    #     M = look_at(camera_position, target, np.array([0, 1, 0]))
    #     M_inv = np.linalg.inv(M)
        
    #     points_world = []
    #     for i in range(8):
    #         points_world.append(np.matmul(M_inv, points_list[i]))
    #         points_world[i] = points_world[i] / points_world[i][3]
        
    #     render_frustum(points=points_world, camera_pos=camera_position, ax=ax_world)
        
    for camera_position,M in zip(camera_list,camera_mat_list):
        M_inv = np.linalg.inv(M)
        
        points_world = []
        for i in range(8):
            points_world.append(np.matmul(M_inv, points_list[i]))
            points_world[i] = points_world[i] / points_world[i][3]
        
        render_frustum(points=points_world, camera_pos=camera_position, ax=ax_world)

    OBJECT = "hotdog"
    
    file_dir = os.path.join("C:/Users/owner/Desktop/HpNeRF/rendering_mitsuba/mesh", OBJECT)
    
    '''
    plot_obj(os.path.join(file_dir, "hotdog_mayonesa.obj"), "yellow", alpha = 1)
    plot_obj(os.path.join(file_dir, "hotdog_mostaza.obj"), "ghostwhite", alpha = 1)
    plot_obj(os.path.join(file_dir, "hotdog_pan1.obj"), "orange")
    plot_obj(os.path.join(file_dir, "hotdog_pan2.obj"), "orange")
    plot_obj(os.path.join(file_dir, "hotdog_salchicha.obj"), "brown")
    plot_obj(os.path.join(file_dir, "hotdog_plato.obj"), "white", alpha = 0.2)
    plot_obj(os.path.join(file_dir, "hotdog_salsa1.obj"), "lightgrey")
    plot_obj(os.path.join(file_dir, "hotdog_salsa2.obj"), "darkred")
    '''
    '''
    plot_obj(os.path.join(file_dir, "cbox_back.obj"), "yellow")
    plot_obj(os.path.join(file_dir, "cbox_ceiling.obj"), "orange")
    plot_obj(os.path.join(file_dir, "cbox_floor.obj"), "white")
    plot_obj(os.path.join(file_dir, "cbox_greenwall.obj"), "purple")
    plot_obj(os.path.join(file_dir, "cbox_redwall.obj"), "pink")
    
    
    plot_obj(os.path.join(file_dir, "background.obj"), "white")
    plot_obj(os.path.join(file_dir, "floor.obj"), "white")
    '''
    
    os.makedirs(os.path.join("./figures", args.mode), exist_ok=True)
    
    viewing_angles = [(0,0),(0,90),(0,180),(0,270),(90,0)] # 4 horizontal views, 1 top-view
    for i, angle in enumerate(viewing_angles):
        ax_world.view_init(*angle)
        plt.show()
        plt.savefig(os.path.join("./figures", args.mode, f"cam_{args.mode}_Lookat({args.use_lookat})_{angle}.png"))