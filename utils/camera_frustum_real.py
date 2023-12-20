import numpy as np
import open3d

from poses.colmap_read_model import read_model, qvec2rotmat
   
   
def draw_camera(K, R, t, w, h,
                scale=1, color=[0.8, 0.2, 0.8]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    #axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    #axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    #return [axis, plane, line_set]
    return [plane, line_set]


def add_points(vis, path, min_track_len=3, remove_statistical_outlier=True):
    cameras, images, points3D = read_model(path)
    pcd = open3d.geometry.PointCloud()

    xyz = []
    rgb = []
    for point3D in points3D.values():
        track_len = len(point3D.point2D_idxs)
        if track_len < min_track_len:
            continue
        xyz.append(point3D.xyz)
        rgb.append(point3D.rgb / 255)
        
    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(rgb)

    # remove obvious outliers
    if remove_statistical_outlier:
        [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    
    # open3d.visualization.draw_geometries([pcd])
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
     
    
def add_cameras_poses(vis, path_poses, scale=1, color=[0.8, 0.2, 0.8]):
    frames = []
    poses = np.load(path_poses)
    
    H, W, focal = poses[0,:3,-1]
    extrinsic_mat = poses[:,:3,:4]  # (N, 3, 4)
    
    rot = extrinsic_mat[..., :3]   # (N, 3, 3)
    trans = extrinsic_mat[..., -1]     # (N, 3)
    
    for i in range(trans.shape[0]):
        R = rot[i, ...]
        t = trans[i, ...]
        
        #t = -R.T @ t
        #R = R.T
        
        fx = fy = focal
        cx = W/2
        cy = H/2
        
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        cam_model = draw_camera(K, R, t, W, H, scale, color)
        frames.extend(cam_model)

    for i in frames:
        vis.add_geometry(i)
    

def add_cameras(vis, path_bin, scale=1):
    cameras, images, points3D = read_model(path_bin)
    
    frames = []
    for img in images.values():
        # rotation
        R = qvec2rotmat(img.qvec)

        # translation
        t = img.tvec

        # invert
        t = -R.T @ t
        R = R.T

        # intrinsics
        cam = cameras[img.camera_id]

        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
        elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        # create axis, plane and pyramid geometries that will be drawn
        cam_model = draw_camera(K, R, t, cam.width, cam.height, scale)
        frames.extend(cam_model)

    # add geometries to visualizer
    for i in frames:
        vis.add_geometry(i)
        
        
if __name__ == "__main__":
    OBJECT = "billiard"
    path = f"C:/Users/owner/Desktop/frustum/images_{}/sparse/0"    
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    add_points(vis, path)
    add_cameras(vis, path)
    
    # add_cameras_poses(vis, np_path_0, color=[0.5, 0, 0])
    # add_cameras_poses(vis, np_path_1, color=[0.8, 0.2, 0.8])
    
    mesh = open3d.geometry.TriangleMesh.create_coordinate_frame()
    #vis.add_geometry(mesh)
    
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
    