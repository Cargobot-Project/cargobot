from copy import deepcopy

from pydrake.all import RigidTransform, LeafSystem
from typing import List
import numpy as np

from scene.SceneBuilder import CARGOBOT_CAMERA_POSES

class CameraSystem:
    def __init__(self, idx, meshcat, diagram, context):
        self.idx = idx

        # Read images
        depth_im_read = diagram.GetOutputPort("camera{}_depth_image".format(idx)).Eval(context).data.squeeze()
        self.depth_im = deepcopy(depth_im_read)
        self.depth_im[self.depth_im == np.inf] = 10.0
        self.rgb_im = diagram.GetOutputPort('camera{}_rgb_image'.format(idx)).Eval(context).data

        # Visualize
        point_cloud = diagram.GetOutputPort("camera{}_point_cloud".format(idx)).Eval(context)
        meshcat.SetObject(f"Camera {idx} point cloud", point_cloud)

        # Get other info about the camera
        cam = diagram.GetSubsystemByName('camera' +str(idx))
        cam_context = cam.GetMyMutableContextFromRoot(context)
        self.X_WC = cam.body_pose_in_world_output_port().Eval(cam_context)
        self.cam_info = cam.depth_camera_info()

    def project_depth_to_pC(self, depth_pixel):
        """
        project depth pixels to points in camera frame
        using pinhole camera model
        Input:
            depth_pixels: numpy array of (nx3) or (3,)
        Output:
            pC: 3D point in camera frame, numpy array of (nx3)
        """
        # switch u,v due to python convention
        v = depth_pixel[:,0]
        u = depth_pixel[:,1]
        Z = depth_pixel[:,2]
        cx = self.cam_info.center_x()
        cy = self.cam_info.center_y()
        fx = self.cam_info.focal_x()
        fy = self.cam_info.focal_y()
        X = (u-cx) * Z/fx
        Y = (v-cy) * Z/fy
        pC = np.c_[X,Y,Z]
        return pC

cargobot_num_cameras = 2

def generate_cameras(environment_diagram, environment_context, meshcat, camera_poses: List[RigidTransform]=CARGOBOT_CAMERA_POSES) -> List[CameraSystem]:
    cameras = []

    for i in range(len(camera_poses)):
        cameras.append(CameraSystem(i, meshcat, environment_diagram, environment_context))

    return cameras