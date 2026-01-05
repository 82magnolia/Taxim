import os
import argparse
from tqdm import tqdm
from OpticalSimulation.simOptical import mesh_simulator
import open3d as o3d
import numpy as np
import cv2
from matplotlib import colormaps
import pyrender
import Basics.sensorParams as psp

os.environ['PYOPENGL_PLATFORM'] = 'egl'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectfoloder_root", help="Root directory of ObjectFolder meshes", default="./data/ObjectFolder/")
    parser.add_argument("--save_root", help="Root directory for saving tactile generation", default="./results/ObjectFolder_touch/")
    parser.add_argument("--point_root", help="Root directory containing keypoints and random points for each object", default="./data/ObjectFolder_sample_points_sparse/")
    parser.add_argument("--scale_method", help="Scaling method to apply", default="max_len")
    parser.add_argument('--override_hw', default=None, type=int, help='Size of image to generate which will be overridden from default', nargs=2)
    parser.add_argument('--depth', default=10.0, type=float, help='Indetation depth into the gelpad.')
    parser.add_argument('--contact_theta', default=0.0, type=float, help='Contact point rotation angle')
    parser.add_argument('--max_len_value', default=50.0, type=float, help="Maximum length value to use for scaling")
    parser.add_argument('--only_render', action='store_true', help="Optionally choose to only perform rendering")
    parser.add_argument('--obj_range', default=None, help="Object index range (start_idx, end_idx) inclusive, to generate data", nargs=2, type=int)
    parser.add_argument('--max_point', default=None, help="Maximum number of points to sample from an object mesh", type=int)
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)

    # List all meshes to process
    mesh_path_list = [os.path.join(args.objectfoloder_root, dir_path, 'model.obj') for dir_path in sorted(os.listdir(args.objectfoloder_root))]

    if args.obj_range is not None:
        filtered_mesh_path_list = []
        for mesh_path in mesh_path_list:
            obj_idx = eval(mesh_path.split("/")[-2])
            if args.obj_range[0] <= obj_idx <= args.obj_range[1]:
                filtered_mesh_path_list.append(mesh_path)
        mesh_path_list = sorted(filtered_mesh_path_list)

    keypoint_root = os.path.join(args.point_root, "keypoints")
    randpoint_root = os.path.join(args.point_root, "randpoints")

    # Optionally override height and width
    if args.override_hw is not None:
        psp_h, psp_w = args.override_hw
        height_psp_mm = psp.pixmm  # NOTE: We use fixed height scaling
    else:
        psp_h, psp_w = psp.h, psp.w
        height_psp_mm = psp.pixmm

    # Set renderer before setting simulator to enable off-screen rendering
    renderer = pyrender.OffscreenRenderer(
        viewport_width=psp_w,
        viewport_height=psp_h
    )
    normal_renderer = pyrender.OffscreenRenderer(
        viewport_width=psp_w,
        viewport_height=psp_h
    )


    for mesh_idx, mesh_path in enumerate(mesh_path_list):
        obj_idx = eval(mesh_path.split("/")[-2])
        save_path = os.path.join(args.save_root, str(obj_idx))

        print(f"Mesh {obj_idx} ({mesh_idx + 1} / {len(mesh_path_list)})")

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        keypoint_path = os.path.join(keypoint_root, f"kpts_{obj_idx}.ply")
        randpoint_path = os.path.join(randpoint_root, f"rpts_{obj_idx}.ply")

        keypts = np.asarray(o3d.io.read_point_cloud(keypoint_path).points)
        randpts = np.asarray(o3d.io.read_point_cloud(randpoint_path).points)
        contactpts = np.concatenate([keypts, randpts], axis=0)

        if args.max_point is not None and contactpts.shape[0] > args.max_point:  # Ensure maximum number of points to be sampled
            contact_pcd = o3d.geometry.PointCloud()
            contact_pcd.points = o3d.utility.Vector3dVector(contactpts)
            contact_pcd = contact_pcd.farthest_point_down_sample(args.max_point)
            contactpts = np.asarray(contact_pcd.points)

        # Set up mesh simulator
        data_folder = os.path.join(os.path.join( ".", "calibs"))
        file_path = os.path.dirname(mesh_path)
        gelpad_model_path = os.path.join( '.', 'calibs', 'gelmap5.npy')

        if args.scale_method == "max_len":
            # Compute scale: normalize so that longest axis is length 1
            long_axis = (contactpts.max(axis=0) - contactpts.min(axis=0)).argmax()
            axis_len = contactpts[:, long_axis].max() - contactpts[:, long_axis].min()
            obj_scale_factor = args.max_len_value / axis_len
        else:
            raise NotImplementedError("Other scaling methods not supported")

        sim = mesh_simulator(data_folder, file_path, 'model.obj', obj_scale_factor, args.override_hw, renderer, normal_renderer)
        press_depth = args.depth

        for contact_idx, contact_point in tqdm(enumerate(contactpts), total=contactpts.shape[0], desc="Touch generation"):
            if contact_idx < keypts.shape[0]:
                pts_type = "kpts"
            else:
                pts_type = "rpts"

            # Save paths
            sim_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_sim.jpg")
            shadow_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_shadow.jpg")
            height_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_height.jpg")
            color_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_color.jpg")
            normal_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_normal.jpg")

            height_map_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_height.npz")
            normal_map_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_normal.npz")
            if args.only_render:
                # generate height map
                height_map, gel_map, contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, 0, 0, contact_point=contact_point, contact_theta=args.contact_theta)

                # Save images
                raw_color_img = np.astype(raw_color_map * 255, np.uint8)
                raw_normal_img = np.astype(vis_raw_normal_map * 255, np.uint8)

                norm_raw_height_map = colormaps.get_cmap("viridis")((raw_height_map - raw_height_map.min()) / (raw_height_map.max() - raw_height_map.min() + 1e-6))
                norm_raw_height_map = np.astype(norm_raw_height_map * 255, np.uint8)
                cv2.imwrite(height_save_path, cv2.cvtColor(norm_raw_height_map, cv2.COLOR_RGB2BGR))

                cv2.imwrite(color_save_path, cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(normal_save_path, cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))

                # Save raw height and normals
                height_map_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_height.npz")
                normal_map_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_normal.npz")
                np.savez_compressed(height_map_save_path, raw_height_map)
                np.savez_compressed(normal_map_save_path, raw_normal_map)
            else:
                # generate height map
                height_map, gel_map, contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, 0, 0, contact_point=contact_point, contact_theta=args.contact_theta)
                # approximate the soft deformation
                heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
                # simulate tactile images
                sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)

                # Save images
                raw_color_img = np.astype(raw_color_map * 255, np.uint8)
                raw_normal_img = np.astype(vis_raw_normal_map * 255, np.uint8)

                cv2.imwrite(sim_save_path, sim_img)
                cv2.imwrite(shadow_save_path, shadow_sim_img)

                norm_raw_height_map = colormaps.get_cmap("viridis")((raw_height_map - raw_height_map.min()) / (raw_height_map.max() - raw_height_map.min() + 1e-6))
                norm_raw_height_map = np.astype(norm_raw_height_map * 255, np.uint8)
                cv2.imwrite(height_save_path, cv2.cvtColor(norm_raw_height_map, cv2.COLOR_RGB2BGR))

                cv2.imwrite(color_save_path, cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(normal_save_path, cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))

                # Save raw height and normals
                height_map_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_height.npz")
                normal_map_save_path = os.path.join(save_path, f"{contact_idx}_scale_{int(args.max_len_value)}_normal.npz")
                np.savez_compressed(height_map_save_path, raw_height_map)
                np.savez_compressed(normal_map_save_path, raw_normal_map)
