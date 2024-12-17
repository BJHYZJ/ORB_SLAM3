import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm

fx, fy = 385.747223, 385.376587
cx, cy = 327.507355, 243.043869

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

rgb_dir = '/home/yanzj/workspace/code1/ORB_SLAM3/datasets/home/rgb'
depth_dir = '/home/yanzj/workspace/code1/ORB_SLAM3/datasets/home/depth'
pose_file = '/home/yanzj/workspace/code1/ORB_SLAM3/datasets/home/trajectory/KeyFrameTrajectory.txt'

def load_poses(pose_file):
    timestamps = []
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            timestamp = int(float(data[0][:16]))
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
            pose = np.zeros((4, 4))
            pose[:3, :3] = quat_to_rot_matrix(qx, qy, qz, qw)
            pose[:3, 3] = [tx, ty, tz]
            pose[3, 3] = 1
            poses.append(pose)
            timestamps.append(timestamp)
    return timestamps, poses

def quat_to_rot_matrix(qx, qy, qz, qw):
    R = np.array([[1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                  [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                  [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]])
    return R


def get_xyz(depth, intrinsic):
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points_xyz = np.stack((points_x, points_y, points_z), axis=-1).astype(np.float32)
    return points_xyz

def generate_point_cloud(rgb_path, depth_path, K):
    bgr_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32) / 1000.0
    mask = np.logical_and(depth_image > 0.3, depth_image < 3)

    xyz = get_xyz(depth_image, K)
    rgb = rgb_image / 255
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
    
    return pcd

def main():
    timestamps, poses = load_poses(pose_file)
    pcds = []
    rgb_files = sorted(os.listdir(rgb_dir))
    depth_files = sorted(os.listdir(depth_dir))

    rgb_timestamps = np.array([int(float(f.split(".png")[0]) * 1e6)for f in sorted(os.listdir(rgb_dir))])
    depth_timestamps = np.array([int(float(f.split(".png")[0]) * 1e6) for f in sorted(os.listdir(depth_dir))])
    
    for index, ts in enumerate(timestamps):
        print(index, index / len(timestamps))
        min_index = np.argmin(np.abs(rgb_timestamps - ts))
        name = rgb_files[min_index]
        rgb_path = os.path.join(rgb_dir, name)
        depth_path = os.path.join(depth_dir, name)
        pcd = generate_point_cloud(rgb_path, depth_path, K)
        pose = poses[index]
        pcd.transform(pose)
        
        pcds.append(pcd)
    
    all_pcd = pcds[0]
    for pcd in pcds[1:]:
        all_pcd += pcd

    o3d.visualization.draw_geometries([all_pcd])

if __name__ == '__main__':
    main()
