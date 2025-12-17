from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
import cv2
import argparse

import sys
sys.path.append("..")
from Basics.RawData import RawData
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp
from tqdm import tqdm
import os
import open3d as o3d
import warnings
from matplotlib import colormaps
import trimesh
import pyrender

parser = argparse.ArgumentParser()
parser.add_argument("-obj", nargs='?', default='square',
                    help="Name of Object to be tested, supported_objects_list = [square, cylinder6]")
parser.add_argument('-mode', default="single_press", type=str, help="Type of simulation to apply")
parser.add_argument('-obj_path', default = None, type=str, help='Directory containing object point cloud')
parser.add_argument('-depth', default = 1.0, type=float, help='Indetation depth into the gelpad.')
parser.add_argument('-obj_scale_factor', default = 1.0, type=float, help='Scale factor to multiply to object before simulation.')
parser.add_argument('-depth_range_info', default = [0.1, 1.5, 100.], type=float, help='Indetation depth range information (min_depth, max_depth, num_range) into the gelpad.', nargs=3)
parser.add_argument('-slide_range_info', default = [-100., 100., -100., 100., 100., 1.0], type=float, help='Sliding range information (min_x, max_x, min_y, max_y, num_range, press_depth) into the gelpad.', nargs=6)
parser.add_argument('-rot_range_info', default = [0.3, 0.3, 0.3, 100., 1.0], type=float, help='Rotating range information (yaw_amplitude, pitch_amplitude, roll_amplitude, num_range, press_depth) into the gelpad.', nargs=5)
parser.add_argument('-contact_point', default = None, type=float, help='Contact point location', nargs=3)
parser.add_argument('-contact_theta', default = None, type=float, help='Contact point rotation angle')
parser.add_argument('-sim_type', default = 'pcd', help='type of simulator to use')
args = parser.parse_args()


def rot_from_ypr(ypr_array):
    def _ypr2mtx(ypr):
        # ypr is assumed to have a shape of [3, ]
        yaw, pitch, roll = ypr
        yaw = yaw.reshape(1)
        pitch = pitch.reshape(1)
        roll = roll.reshape(1)

        tensor_0 = np.zeros(1, device=yaw.device)
        tensor_1 = np.ones(1, device=yaw.device)

        RX = np.stack([
                        np.stack([tensor_1, tensor_0, tensor_0]),
                        np.stack([tensor_0, np.cos(roll), -np.sin(roll)]),
                        np.stack([tensor_0, np.sin(roll), np.cos(roll)])]).reshape(3, 3)

        RY = np.stack([
                        np.stack([np.cos(pitch), tensor_0, np.sin(pitch)]),
                        np.stack([tensor_0, tensor_1, tensor_0]),
                        np.stack([-np.sin(pitch), tensor_0, np.cos(pitch)])]).reshape(3, 3)

        RZ = np.stack([
                        np.stack([np.cos(yaw), -np.sin(yaw), tensor_0]),
                        np.stack([np.sin(yaw), np.cos(yaw), tensor_0]),
                        np.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = RZ @ RY
        R = R @ RX

        return R

    if len(ypr_array.shape) == 1:
        return _ypr2mtx(ypr_array)
    else:
        tot_mtx = []
        for ypr in ypr_array:
            tot_mtx.append(_ypr2mtx(ypr))
        return np.stack(tot_mtx)


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotation_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def align_normals(n1, n2):
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    v = np.cross(n1, n2)
    c = np.dot(n1, n2)

    if np.linalg.norm(v) < 1e-8:
        # n1 and n2 are parallel or antiparallel
        if c > 0:
            return np.eye(3)
        else:
            # 180-degree rotation around any axis orthogonal to n1
            a = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, n1)) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            axis = np.cross(n1, a)
            axis /= np.linalg.norm(axis)
            return rotation_from_axis_angle(axis, np.pi)

    axis = v / np.linalg.norm(v)
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    return rotation_from_axis_angle(axis, angle)


def rotation_family(n1, n2, theta):
    """
    Returns a rotation R(theta) such that R(theta) @ n1 = n2
    """
    n1 = n1 / np.linalg.norm(n1)

    R0 = align_normals(n1, n2)
    R_spin = rotation_from_axis_angle(n1, theta)

    return R0 @ R_spin


class simulator(object):
    def __init__(self, data_folder, filePath, obj, obj_scale_factor=1.):
        """
        Initialize the simulator.
        1) load the object,
        2) load the calibration files,
        3) generate shadow table from shadow masks
        """
        # read in object's ply file
        # object facing positive direction of z axis
        objPath = osp.join(filePath,obj)
        self.obj_name = obj.split('.')[0]
        print("load object: " + self.obj_name)

        pcd = o3d.io.read_point_cloud(objPath)
        self.vertices = np.asarray(pcd.points) * obj_scale_factor

        # Paint with uniform color if no colors exist
        if len(pcd.colors) == 0:
            pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.colors = np.asarray(pcd.colors)

        if len(pcd.normals) == 0:  # Esimate normals if none exist
            warnings.warn("No normals exist, resorting to Open3D normal estimation which is noisy")
            pcd.estimate_normals()
        self.vert_normals = np.asarray(pcd.normals)

        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

        return f0

    def simulating(self, heightMap, contact_mask, contact_height, shadow=False):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact
        contact_mask: indicate the contact area
        contact_height: the height of each pix
        shadow: whether add the shadow

        return:
        sim_img: simulated tactile image w/o shadow
        shadow_sim_img: simluated tactile image w/ shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h,psp.w,3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r,axis = 1)
        est_g = np.sum(A * params_g,axis = 1)
        est_b = np.sum(A * params_b,axis = 1)

        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # attach background to simulated image
        sim_img = sim_img_r + self.bg_proc

        if not shadow:
            return sim_img, sim_img

        # add shadow
        cx = psp.w//2
        cy = psp.h//2

        # find shadow attachment area
        kernel = np.ones((5, 5), np.uint8)
        dialate_mask = cv2.dilate(np.float32(contact_mask),kernel,iterations = 2)
        enlarged_mask = dialate_mask == 1
        boundary_contact_mask = 1*enlarged_mask - 1*contact_mask
        contact_mask = boundary_contact_mask == 1

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = normMap // pr.discritize_precision
        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = (contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision
        total_height_idx = self.shadowTable.shape[2]

        shadowSim = np.zeros((psp.h,psp.w,3))

        # all 3 channels
        for c in range(3):
            frame = sim_img_r[:,:,c].copy()
            frame_back = sim_img_r[:,:,c].copy()
            for i in range(len(x_coord)):
                # get the coordinates (x,y) of a certain pixel
                cy_origin = y_coord[i]
                cx_origin = x_coord[i]
                # get the normal of the pixel
                n = int(norm_idx[i])
                # get height of the pixel
                h = int(height_idx[i]) + 6
                if h < 0 or h >= total_height_idx:
                    continue
                # get the shadow list for the pixel
                v = self.shadowTable[c,n,h]

                # number of steps
                num_step = len(v)

                # get the shadow direction
                theta = self.direction[n]
                d_theta = theta
                ct = np.cos(d_theta)
                st = np.sin(d_theta)
                # use a fan of angles around the direction
                theta_list = np.arange(d_theta-pr.fan_angle, d_theta+pr.fan_angle, pr.fan_precision)
                ct_list = np.cos(theta_list)
                st_list = np.sin(theta_list)
                for theta_idx in range(len(theta_list)):
                    ct = ct_list[theta_idx]
                    st = st_list[theta_idx]

                    for s in range(1,num_step):
                        cur_x = int(cx_origin + pr.shadow_step * s * ct)
                        cur_y = int(cy_origin + pr.shadow_step * s * st)
                        # check boundary of the image and height's difference
                        if cur_x >= 0 and cur_x < psp.w and cur_y >= 0 and cur_y < psp.h and heightMap[cy_origin,cx_origin] > heightMap[cur_y,cur_x]:
                            frame[cur_y,cur_x] = np.minimum(frame[cur_y,cur_x],v[s])

            shadowSim[:,:,c] = frame
            shadowSim[:,:,c] = ndimage.gaussian_filter(shadowSim[:,:,c], sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim+ self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        return sim_img, shadow_sim_img

    def generateHeightMap(self, gelpad_model_path, pressing_height_mm, dx, dy, contact_jitter_rot_mtx=None, contact_point=None, contact_theta=0.):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        dx, dy: shift of the object
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # NOTE 1: Tactile sensor is placed at the x,y location of object center with z location at maximum object height, and object points with height over a threshold (0.2) are all considered
        # NOTE 2: Tactile sensor is placed oppositely facing the object placed on top of a virtual plane with z=0, although the object can be "floating"
        # NOTE 3: Currently normals are stored in "raw" xyz coordinates. To transform between two normals, we represent rotation as a 1-parameter family of transformations, controlled by contact_theta.
        # NOTE 4: This contact_theta controls the variation in contact rotations, or "rolling" on the contact point's tangent plane
        # NOTE 5: contact_jitter_rot_mtx additionally applies rotation to a contact point location to enable rotation other than tangent plane rolling

        assert(self.vertices.shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        heightMap = np.zeros((psp.h,psp.w))

        # Raw color and normal maps
        rawcolorMap = np.zeros((psp.h,psp.w,3))
        rawnormalMap = np.zeros((psp.h,psp.w,3))

        # Identify original contact points
        orig_cx = np.mean(self.vertices[:,0])
        orig_cy = np.mean(self.vertices[:,1])
        xy_dist = np.linalg.norm(self.vertices[:, [0, 1]] - np.array([orig_cx, orig_cy]), axis=-1)
        kth = min(100, xy_dist.shape[0] // 2)  # NOTE: This is an arbitrarily chosen number
        topk_idx = np.argpartition(xy_dist, kth=kth)[:kth]
        orig_cz = self.vertices[topk_idx, 2].max()

        # Original concact point and normal direction
        orig_contact = np.array([orig_cx, orig_cy, orig_cz])
        orig_normal = self.vert_normals[np.linalg.norm(self.vertices - orig_contact, axis=-1).argmin()]

        # Copy original vertex set and normals
        sim_vertices = np.copy(self.vertices)
        sim_vert_normals = np.copy(self.vert_normals)

        # Set contact points given as array of shape (3, )
        if contact_point is not None:
            cx = contact_point[0]
            cy = contact_point[1]
            cz = contact_point[2]

            # New contact point and normal direction
            new_contact = np.array([cx, cy, cz])
            new_normal = sim_vert_normals[np.linalg.norm(self.vertices - new_contact, axis=-1).argmin()]

            # Estimate rotation matrix that aligns new normal to the positive z direction
            contact_rot_mtx = rotation_family(new_normal, np.array([0, 0, 1]), contact_theta)

            # Fix contact point and rotate points
            sim_vertices = (sim_vertices - new_contact) @ contact_rot_mtx.T
            sim_vert_normals = sim_vert_normals @ contact_rot_mtx.T
        else:
            cx = orig_cx
            cy = orig_cy
            cz = orig_cz
            contact = np.array([cx, cy, cz])
            sim_vertices = (sim_vertices - contact) @ contact_rot_mtx.T

        if contact_jitter_rot_mtx is not None:
            sim_vertices = sim_vertices @ contact_jitter_rot_mtx.T
            sim_vert_normals = sim_vert_normals @ contact_jitter_rot_mtx.T
        else:
            sim_vert_normals = np.copy(sim_vert_normals)

        # Ensure minimum height is 0. during rendering
        sim_vertices[:, 2] -= sim_vertices[:, 2].min()
        cz = 0.

        # add the shifting and change to the pix coordinate
        uu = ((sim_vertices[:,0])/psp.pixmm + psp.w//2+dx).astype(int)
        vv = ((sim_vertices[:,1])/psp.pixmm + psp.h//2+dy).astype(int)
        vv = psp.h - vv  # NOTE: This is needed to ensure consistency with pyrender's orthographic rendering

        # check boundary of the image
        mask_u = np.logical_and(uu > 0, uu < psp.w)
        mask_v = np.logical_and(vv > 0, vv < psp.h)
        # check the depth
        mask_map = mask_u & mask_v
        heightMap[vv[mask_map],uu[mask_map]] = sim_vertices[mask_map][:,2]/psp.pixmm  # NOTE: We don't re-normalize with minimum value as point projections have holes, causing inaccurate minimum values

        # Fill in raw color and normals
        rawcolorMap[vv[mask_map],uu[mask_map]] = self.colors[mask_map]
        rawnormalMap[vv[mask_map],uu[mask_map]] = sim_vert_normals[mask_map]

        # Normal map for visualization
        vis_rawnormalMap = np.copy(rawnormalMap)
        vis_rawnormalMap[vv[mask_map],uu[mask_map]] = (rawnormalMap[vv[mask_map],uu[mask_map]] + 1.0) * 0.5
        vis_rawnormalMap = np.clip(vis_rawnormalMap, 0, 1)

        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm

        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)  # RHS is gel height map assuming object placed at z = 0

        # get the contact area
        contact_mask = heightMap > gel_map

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]

        return zq, gel_map, contact_mask, rawcolorMap, rawnormalMap, vis_rawnormalMap, heightMap

    def deformApprox(self, pressing_height_mm, height_map, gel_map, contact_mask):
        zq = height_map.copy()
        zq_back = zq.copy()
        pressing_height_pix = pressing_height_mm/psp.pixmm
        # contact mask which is a little smaller than the real contact mask
        mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
        mask = mask & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        for i in range(len(pr.pyramid_kernel_size)):
            zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
            zq[mask] = zq_back[mask]
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        contact_height = zq - gel_map

        return zq, mask, contact_height

    def interpolate(self,img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear', fill_value = 0) # cubic # nearest # linear
        return GD1

    def generate_normals(self,height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h,w] = height_map.shape
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self,img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')


class mesh_simulator(simulator):
    def __init__(self, data_folder, filePath, obj, obj_scale_factor=1.):
        """
        Initialize the simulator.
        1) load the object,
        2) load the calibration files,
        3) generate shadow table from shadow masks
        """
        # read in object's ply file
        # object facing positive direction of z axis
        objPath = osp.join(filePath,obj)
        self.obj_name = obj.split('.')[0]
        print("load object: " + self.obj_name)

        # Load assets for mesh-based rendering
        self.tr_mesh = trimesh.load(objPath, force='mesh', process=False)
        self.tr_mesh.vertices = np.asarray(self.tr_mesh.vertices) * obj_scale_factor
        self.proximitry_query = trimesh.proximity.ProximityQuery(self.tr_mesh)  # Used for nearest neighbor queries
        if not isinstance(self.tr_mesh, trimesh.Trimesh):
            raise ValueError("OBJ did not load as a single mesh")
        self.vertices = self.tr_mesh.vertices
        self.vert_normals = self.tr_mesh.vertex_normals

        # Set orthographic camera (NOTE: we assume camera to be fixed and the object to be moving)
        self.znear = 0.01
        self.zfar = 1000.0

        self.cam = pyrender.camera.OrthographicCamera(
            xmag=psp.pixmm * psp.h / 2.,  # NOTE: This will be automatically re-scaled respecting designated height and width for rendering
            ymag=psp.pixmm * psp.h / 2.,  # NOTE: psp.h / 2. is multiplied to ensure identical orthographic scales as in Taxim
            znear=self.znear,
            zfar=self.zfar
        )

        # Camera pose in world frame
        self.cam_height = 1000.0  # Hard-coded camera height
        self.T_wc = np.eye(4)
        self.T_wc[:3, :3] = np.eye(3)
        self.T_wc[:3, 3] = np.array([0.0, 0.0, self.cam_height])

        # Lighting for rendering
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

        # Set renderer for color and normals
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=psp.w,
            viewport_height=psp.h
        )
        self.normal_renderer = pyrender.OffscreenRenderer(
            viewport_width=psp.w,
            viewport_height=psp.h
        )

        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

    def generateHeightMap(self, gelpad_model_path, pressing_height_mm, dx, dy, contact_jitter_rot_mtx=None, contact_point=None, contact_theta=0.):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        dx, dy: shift of the object
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # NOTE 1: Tactile sensor is placed at the x,y location of object center with z location at maximum object height, and object points with height over a threshold (0.2) are all considered
        # NOTE 2: Tactile sensor is placed oppositely facing the object placed on top of a virtual plane with z=0, although the object can be "floating"
        # NOTE 3: Currently normals are stored in "raw" xyz coordinates. To transform between two normals, we represent rotation as a 1-parameter family of transformations, controlled by contact_theta.
        # NOTE 4: This contact_theta controls the variation in contact rotations, or "rolling" on the contact point's tangent plane
        # NOTE 5: contact_jitter_rot_mtx additionally applies rotation to a contact point location to enable rotation other than tangent plane rolling

        assert(self.vertices.shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        # Copy original vertex set and normals
        sim_vertices = np.copy(self.vertices)

        # Set contact points given as array of shape (3, )
        if contact_point is not None:
            cx = contact_point[0]
            cy = contact_point[1]
            cz = contact_point[2]

            # Contact point array
            contact_arr = np.array([cx, cy, cz])

            # Find normal at contact point
            nn_point, _, nn_fid = self.proximitry_query.on_surface(contact_arr.reshape(1, 3))
            nn_bary = trimesh.triangles.points_to_barycentric(self.tr_mesh.triangles[nn_fid], points=contact_arr.reshape(1, 3))
            new_normal = trimesh.unitize((self.tr_mesh.vertex_normals[self.tr_mesh.faces[nn_fid]] * trimesh.unitize(nn_bary).reshape(-1, 3, 1)).sum(axis=1))
            new_normal = new_normal.reshape(-1)

            # Estimate rotation matrix that aligns new normal to the positive z direction
            contact_rot_mtx = rotation_family(new_normal, np.array([0, 0, 1]), contact_theta)

            # Fix contact point to origin and rotate points
            sim_vertices = (sim_vertices - contact_arr) @ contact_rot_mtx.T

        else:
            # Identify original contact points
            cx = np.mean(sim_vertices[:,0])
            cy = np.mean(sim_vertices[:,1])
            xy_dist = np.linalg.norm(sim_vertices[:, [0, 1]] - np.array([cx, cy]), axis=-1)
            kth = min(100, xy_dist.shape[0] // 2)  # NOTE: This is an arbitrarily chosen number
            topk_idx = np.argpartition(xy_dist, kth=kth)[:kth]
            cz = sim_vertices[topk_idx, 2].max()
            contact_arr = np.array([cx, cy, cz])

            sim_vertices = sim_vertices - contact_arr

        if contact_jitter_rot_mtx is not None:
            sim_vertices = sim_vertices @ contact_jitter_rot_mtx.T

        # Add sensor-plane shifts
        sim_vertices[:, 0] = sim_vertices[:, 0] + dx * psp.pixmm
        sim_vertices[:, 1] = sim_vertices[:, 1] + dy * psp.pixmm

        # Ensure minimum height is 0. during rendering
        sim_vertices[:, 2] -= sim_vertices[:, 2].min()

        # Render heightmap, color, and normals from mesh
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                                ambient_light=[0.3, 0.3, 0.3])
        normal_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                                ambient_light=[0.3, 0.3, 0.3])  # Scene for normal map rendering

        # Set up mesh for rendering
        rgb_tr_mesh = self.tr_mesh.copy()
        rgb_tr_mesh.vertices = sim_vertices
        rgb_mesh = pyrender.Mesh.from_trimesh(rgb_tr_mesh, smooth=False)

        normal_tr_mesh = trimesh.Trimesh(vertices=sim_vertices, faces=self.tr_mesh.faces)
        normal_tr_mesh.visual.vertex_colors = np.astype(255 * (normal_tr_mesh.vertex_normals + 1.0) / 2., np.uint8)
        normal_tr_mesh.visual.face_colors = np.astype(255 * (normal_tr_mesh.face_normals + 1.0) / 2., np.uint8)
        normal_mesh = pyrender.Mesh.from_trimesh(normal_tr_mesh, smooth=False)

        # Set up scene
        scene.add(rgb_mesh)
        normal_scene.add(normal_mesh)
        scene.add(self.cam, pose=self.T_wc)
        normal_scene.add(self.cam, pose=self.T_wc)
        scene.add(self.light, pose=self.T_wc)
        normal_scene.add(self.light, pose=self.T_wc)

        # Render images
        rawcolorMap, depth = self.renderer.render(scene)
        rawcolorMap = rawcolorMap / 255.

        # Obtain depth
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

        if (d_max - d_min) < (self.zfar - self.znear) / 10:  # CHOOSE A ROBUST CHECK FROM THRESHOLDING
            # A hack to fix depth maps for OrthographicCamera.
            # See: https://github.com/mmatl/pyrender/issues/72
            depth_raw[noninf] = self.zfar + self.znear - self.zfar * self.znear / depth_raw[noninf]

        # Obtain height map
        height_raw = self.T_wc[2, -1] - depth_raw
        height_raw[~noninf] = 0.

        # Ensure minimum height is 0. only considering regions within the view
        height_raw[noninf] = height_raw[noninf] - height_raw[noninf].min()
        heightMap = height_raw / psp.pixmm

        # Obtain normal map
        normal_rgb, _ = self.normal_renderer.render(normal_scene, flags=pyrender.RenderFlags.FLAT)
        normal = 2 * np.astype(normal_rgb, float) / 255 - 1.
        invalid_normal_loc = np.all(normal_rgb == 255, axis=-1)
        normal[invalid_normal_loc] = 0.
        normal[~invalid_normal_loc] = normal[~invalid_normal_loc] / np.linalg.norm(normal[~invalid_normal_loc], axis=-1, keepdims=True)

        rawnormalMap = np.copy(normal)
        vis_rawnormalMap = np.copy(normal_rgb)
        vis_rawnormalMap[invalid_normal_loc] = 0
        vis_rawnormalMap = vis_rawnormalMap / 255.

        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm

        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)  # RHS is gel height map assuming object placed at z = 0

        # get the contact area
        contact_mask = heightMap > gel_map

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]

        return zq, gel_map, contact_mask, rawcolorMap, rawnormalMap, vis_rawnormalMap, heightMap


if __name__ == "__main__":
    data_folder = osp.join(osp.join( "..", "calibs"))
    filePath = osp.join('..', 'data', 'objects') if args.obj_path is None else args.obj_path
    gelpad_model_path = osp.join( '..', 'calibs', 'gelmap5.npy')

    if args.contact_point is None:
        contact_point = None
    else:
        contact_point = np.array([args.contact_point[0], args.contact_point[1], args.contact_point[2]])

    if args.sim_type == 'pcd':
        obj = args.obj + '.ply'
        tac_sim = simulator
    elif args.sim_type == 'mesh':
        obj = args.obj + '.obj'
        tac_sim = mesh_simulator
    else:
        raise NotImplementedError("Other simulators not supported")

    if args.mode == "single_press":
        sim = tac_sim(data_folder, filePath, obj, args.obj_scale_factor)
        press_depth = args.depth
        dx = 0
        dy = 0

        # generate height map
        height_map, gel_map, contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy, contact_point=contact_point, contact_theta=args.contact_theta)
        # approximate the soft deformation
        heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
        # simulate tactile images
        sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)
        img_savePath = osp.join('..', 'results', obj[:-4]+'_sim.jpg')
        shadow_savePath = osp.join('..', 'results', obj[:-4]+'_shadow.jpg')
        height_savePath = osp.join('..', 'results', obj[:-4]+'_raw_height.jpg')

        raw_color_savePath = osp.join('..', 'results', obj[:-4]+'_raw_color.jpg')
        raw_normal_savePath = osp.join('..', 'results', obj[:-4]+'_raw_normal.jpg')
        raw_color_img = np.astype(raw_color_map * 255, np.uint8)
        raw_normal_img = np.astype(vis_raw_normal_map * 255, np.uint8)

        cv2.imwrite(img_savePath, sim_img)
        cv2.imwrite(shadow_savePath, shadow_sim_img)

        norm_raw_height_map = colormaps.get_cmap("viridis")((raw_height_map - raw_height_map.min()) / (raw_height_map.max() - raw_height_map.min() + 1e-6))
        norm_raw_height_map = np.astype(norm_raw_height_map * 255, np.uint8)
        cv2.imwrite(height_savePath, cv2.cvtColor(norm_raw_height_map, cv2.COLOR_RGB2BGR))

        cv2.imwrite(raw_color_savePath, cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(raw_normal_savePath, cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))
    elif args.mode == "continuous_press":
        sim = tac_sim(data_folder, filePath, obj, args.obj_scale_factor)
        press_min, press_max, num_step = args.depth_range_info
        num_step = int(num_step)

        for press_idx, press_depth in tqdm(enumerate(np.linspace(press_min, press_max, num_step)), total=num_step):
            dx = 0
            dy = 0

            # generate height map
            height_map, gel_map, contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy, contact_point=contact_point, contact_theta=args.contact_theta)
            # approximate the soft deformation
            heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
            # simulate tactile images
            sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)

            if press_idx == 0:
                sim_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_sim_{press_min}_{press_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (sim_img.shape[1], sim_img.shape[0]))
                shadow_sim_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_shadow_{press_min}_{press_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (shadow_sim_img.shape[1], shadow_sim_img.shape[0]))
                raw_color_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_raw_color_{press_min}_{press_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (raw_color_map.shape[1], raw_color_map.shape[0]))
                raw_normal_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_raw_normal_{press_min}_{press_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (vis_raw_normal_map.shape[1], vis_raw_normal_map.shape[0]))

            sim_video.write(cv2.cvtColor(np.astype(sim_img, np.uint8), cv2.COLOR_RGB2BGR))
            shadow_sim_video.write(cv2.cvtColor(np.astype(shadow_sim_img, np.uint8), cv2.COLOR_RGB2BGR))

            raw_color_img = np.astype(raw_color_map * 255, np.uint8)
            raw_normal_img = np.astype(vis_raw_normal_map * 255, np.uint8)
            raw_color_video.write(cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
            raw_normal_video.write(cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))

            if press_idx == num_step - 1:
                sim_video.release()
                shadow_sim_video.release()
    elif args.mode == "rotating_press":
        sim = tac_sim(data_folder, filePath, obj, args.obj_scale_factor)
        yaw_amplitude, pitch_amplitude, roll_amplitude, num_step, press_depth = args.rot_range_info
        num_step = int(num_step)
        yaw_arr = np.linspace(-yaw_amplitude, yaw_amplitude, num_step)
        pitch_arr = np.linspace(-pitch_amplitude, pitch_amplitude, num_step)
        roll_arr = np.linspace(-roll_amplitude, roll_amplitude, num_step)
        ypr_arr = np.stack([yaw_arr, pitch_arr, roll_arr], axis=-1)

        rot_arr = rot_from_ypr(ypr_arr)

        for press_idx, rot_mtx in tqdm(enumerate(rot_arr), total=num_step):
            dx = 0
            dy = 0

            # generate height map
            height_map, gel_map, contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy, rot_mtx, contact_point=contact_point, contact_theta=args.contact_theta)
            # approximate the soft deformation
            heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
            # simulate tactile images
            sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)

            if press_idx == 0:
                sim_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_sim_rot_{yaw_amplitude}_{pitch_amplitude}_{roll_amplitude}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (sim_img.shape[1], sim_img.shape[0]))
                shadow_sim_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_shadow_rot_{yaw_amplitude}_{pitch_amplitude}_{roll_amplitude}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (shadow_sim_img.shape[1], shadow_sim_img.shape[0]))
                raw_color_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_raw_color_rot_{yaw_amplitude}_{pitch_amplitude}_{roll_amplitude}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (raw_color_map.shape[1], raw_color_map.shape[0]))
                raw_normal_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_raw_normal_rot_{yaw_amplitude}_{pitch_amplitude}_{roll_amplitude}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (vis_raw_normal_map.shape[1], vis_raw_normal_map.shape[0]))

            sim_video.write(cv2.cvtColor(np.astype(sim_img, np.uint8), cv2.COLOR_RGB2BGR))
            shadow_sim_video.write(cv2.cvtColor(np.astype(shadow_sim_img, np.uint8), cv2.COLOR_RGB2BGR))

            raw_color_img = np.astype(raw_color_map * 255, np.uint8)
            raw_normal_img = np.astype(vis_raw_normal_map * 255, np.uint8)
            raw_color_video.write(cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
            raw_normal_video.write(cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))

            if press_idx == num_step - 1:
                sim_video.release()
                shadow_sim_video.release()
    elif args.mode == "sliding_press":
        sim = tac_sim(data_folder, filePath, obj, args.obj_scale_factor)
        dx_min, dx_max, dy_min, dy_max, num_step, press_depth = args.slide_range_info
        num_step = int(num_step)
        slide_x = np.linspace(dx_min, dx_max, num_step)
        slide_y = np.linspace(dy_min, dy_max, num_step)

        for press_idx, (dx, dy) in tqdm(enumerate(zip(slide_x, slide_y)), total=num_step):

            # generate height map
            height_map, gel_map, contact_mask, raw_color_map, raw_normal_map, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy, contact_point=contact_point, contact_theta=args.contact_theta)
            # approximate the soft deformation
            heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
            # simulate tactile images
            sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)

            if press_idx == 0:
                sim_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_sim_slide_{dx_min}_{dx_max}_{dy_min}_{dy_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (sim_img.shape[1], sim_img.shape[0]))
                shadow_sim_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_shadow_slide_{dx_min}_{dx_max}_{dy_min}_{dy_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (shadow_sim_img.shape[1], shadow_sim_img.shape[0]))
                raw_color_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_raw_color_slide_{dx_min}_{dx_max}_{dy_min}_{dy_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (raw_color_map.shape[1], raw_color_map.shape[0]))
                raw_normal_video = cv2.VideoWriter(
                    osp.join('..', 'results', obj[:-4] + f'_raw_normal_slide_{dx_min}_{dx_max}_{dy_min}_{dy_max}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    5.,
                    (vis_raw_normal_map.shape[1], vis_raw_normal_map.shape[0]))

            sim_video.write(cv2.cvtColor(np.astype(sim_img, np.uint8), cv2.COLOR_RGB2BGR))
            shadow_sim_video.write(cv2.cvtColor(np.astype(shadow_sim_img, np.uint8), cv2.COLOR_RGB2BGR))

            raw_color_img = np.astype(raw_color_map * 255, np.uint8)
            raw_normal_img = np.astype(vis_raw_normal_map * 255, np.uint8)
            raw_color_video.write(cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
            raw_normal_video.write(cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))

            if press_idx == num_step - 1:
                sim_video.release()
                shadow_sim_video.release()

    if args.sim_type == 'mesh':
        tac_sim.renderer.delete()
        tac_sim.normal_renderer.delete()
