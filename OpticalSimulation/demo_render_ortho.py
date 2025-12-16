import numpy as np
import trimesh
import pyrender
from PIL import Image
import argparse
import os


def render_orthographic(
    obj_path,
    R_wc,
    t_wc,
    out_dir,
    img_size=(1024, 1024),
    ortho_scale=1.0
):
    """
    R_wc: (3,3) rotation matrix, camera-to-world
    t_wc: (3,) translation vector, camera-to-world
    """

    # ----------------------------
    # Load mesh
    # ----------------------------
    tr_mesh = trimesh.load(obj_path, force='mesh', process=False)

    if not isinstance(tr_mesh, trimesh.Trimesh):
        raise ValueError("OBJ did not load as a single mesh")

    # Place object onto z=0 plane (fix mininum z value to 0)
    tr_mesh.vertices[:, 2] = np.array(tr_mesh.vertices)[:, 2] - np.array(tr_mesh.vertices)[:, 2].min()

    mesh = pyrender.Mesh.from_trimesh(tr_mesh, smooth=False)

    # ----------------------------
    # Scene
    # ----------------------------
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                            ambient_light=[0.3, 0.3, 0.3])

    scene.add(mesh)

    # ----------------------------
    # Orthographic camera
    # ----------------------------
    znear = 0.01
    zfar = 1000.0
    cam = pyrender.camera.OrthographicCamera(
        xmag=ortho_scale,
        ymag=ortho_scale,
        znear=znear,
        zfar=zfar
    )

    # Camera pose in world frame
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc

    scene.add(cam, pose=T_wc)

    # ----------------------------
    # Lighting (headlight)
    # ----------------------------
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=T_wc)

    # ----------------------------
    # Render
    # ----------------------------
    r = pyrender.OffscreenRenderer(
        viewport_width=img_size[0],
        viewport_height=img_size[1]
    )

    color, depth = r.render(scene)
    r.delete()

    # ==========================================================
    # COLOR
    # ==========================================================
    Image.fromarray(color).save(os.path.join(out_dir, "rgb.png"))

    # ==========================================================
    # DEPTH MAP
    # ==========================================================
    depth_raw = depth.copy()
    depth_raw[depth_raw == 0] = np.nan

    # Normalize for visualization
    d_min = np.nanmin(depth_raw)
    d_max = np.nanmax(depth_raw)

    depth_norm = (depth_raw - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.nan_to_num(depth_norm)

    depth_img = (depth_norm * 255).astype(np.uint8)

    depth_raw = depth.copy()
    noninf = depth_raw > 0
    d_min = np.nanmin(depth_raw)
    d_max = np.nanmax(depth_raw)

    if (d_max - d_min) < (zfar - znear) / 10:  # CHOOSE A ROBUST CHECK FROM THRESHOLDING
        # A hack to fix depth maps for OrthographicCamera.
        # See: https://github.com/mmatl/pyrender/issues/72
        depth_raw[noninf] = zfar + znear - zfar * znear / depth_raw[noninf]

    depth_norm = (depth_raw - d_min) / (d_max - d_min + 1e-8)
    depth_norm = np.nan_to_num(depth_norm)

    depth_img = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(os.path.join(out_dir, "depth.png"))

    # ==========================================================
    # HEIGHT MAP (Assume Object is Placed at z=0)
    # ==========================================================
    height_raw = t_wc[-1] - depth_raw
    height_raw[~noninf] = 0.

    # ----------------------------
    # Encode normals to RGB
    # ----------------------------
    # normals are in [-1, 1]

    # Make a copy mesh storing normals as colors
    copy_tr_mesh = trimesh.Trimesh(vertices=tr_mesh.vertices, faces=tr_mesh.faces)
    copy_tr_mesh.visual.vertex_colors = np.astype(255 * (copy_tr_mesh.vertex_normals + 1.0) / 2., np.uint8)
    copy_tr_mesh.visual.face_colors = np.astype(255 * (copy_tr_mesh.face_normals + 1.0) / 2., np.uint8)
    copy_mesh = pyrender.Mesh.from_trimesh(copy_tr_mesh, smooth=False)
    normal_r = pyrender.OffscreenRenderer(
        viewport_width=img_size[0],
        viewport_height=img_size[1]
    )  # Normal renderer
    normal_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                            ambient_light=[0.3, 0.3, 0.3])
    normal_scene.add(copy_mesh)
    normal_scene.add(cam, pose=T_wc)
    normal_scene.add(light, pose=T_wc)

    normal_rgb, _ = normal_r.render(normal_scene, flags=pyrender.RenderFlags.FLAT)
    normal = 2 * np.astype(normal_rgb, float) / 255 - 1.
    invalid_normal_loc = np.all(normal_rgb == 255, axis=-1)
    normal[invalid_normal_loc] = 0.
    normal[~invalid_normal_loc] = normal[~invalid_normal_loc] / np.linalg.norm(normal[~invalid_normal_loc], axis=-1, keepdims=True)

    Image.fromarray(normal_rgb).save(
        os.path.join(out_dir, "normal_view.png")
    )

    print(f"Saved orthographic render to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ortho_scale", type=float, default=1.0)
    parser.add_argument("--img_size", type=int, nargs=2, default=[1024, 1024])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ----------------------------
    # Example camera pose
    # ----------------------------
    # Camera looking toward -Z, placed at (0, 0, 2)
    R_wc = np.eye(3)
    t_wc = np.array([0.0, 0.0, 3.0])

    render_orthographic(
        obj_path=args.obj,
        R_wc=R_wc,
        t_wc=t_wc,
        out_dir=args.out,
        img_size=tuple(args.img_size),
        ortho_scale=args.ortho_scale
    )
