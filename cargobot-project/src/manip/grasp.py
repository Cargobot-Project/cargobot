import os
import time
from copy import deepcopy
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as Tf
from IPython.display import clear_output, display
from manipulation import running_as_notebook
#from manipulation.clutter import GenerateAntipodalGraspCandidate
from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import AddPackagePaths, FindResource, LoadDataResource
from pydrake.all import (BaseField, Concatenate, Fields, MeshcatVisualizer,
                         MeshcatVisualizerParams, PointCloud, Quaternion, Rgba,
                         RigidTransform, RotationMatrix, StartMeshcat, CameraInfo)
from pydrake.multibody.parsing import (LoadModelDirectives, Parser,
                                       ProcessModelDirectives)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pydrake.all import RollPitchYaw, AbstractValue, LeafSystem, Image, PixelType

from segmentation.util import *
from scene import WarehouseSceneSystem 
from segmentation.plot import *
def GraspCandidateCost(
    diagram,
    context,
    cloud,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
    adjust_X_G=False,
):
    """
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost

    If adjust_X_G is True, then it also updates the gripper pose in the plant
    context.
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    X_G = plant.GetFreeBodyPose(plant_context, wsg)

    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW @ cloud.xyzs()

    # Crop to a region inside of the finger box.
    """crop_min = [0, -1, 0.07]
    crop_max = [1.8, 1, 0.17]"""
    crop_min = [-0.05, 0.1, -0.00625]
    crop_max = [0.05, 0.1125, 0.00625]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )

    if adjust_X_G and np.sum(indices) > 0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Check collisions between the gripper and the sink
    if query_object.HasCollisions():
        cost = np.inf
        return cost

    # Check collisions between the gripper and the point cloud. `margin`` must
    # be smaller than the margin used in the point cloud preprocessing.
    margin = 0.0
    for i in range(cloud.size()):
        distances = query_object.ComputeSignedDistanceToPoint(
            cloud.xyz(i), threshold=margin
        )
        if distances:
            cost = np.inf
            return cost

    n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0 * X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0, :] ** 2)
    return cost

def GenerateAntipodalGraspCandidate(
    diagram,
    context,
    cloud,
    rng,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
):
    
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    if cloud.size() < 1:
        return np.inf, None

    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame.
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS  # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y, Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return np.inf, None

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    #p_GS_G = [0.054 - 0.01, 0.10625, 0]
    p_GS_G = [0.054,  0.10625, 0]

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # Rotate the object in the hand by a random rotation (around the
        # normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = -R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        
        if np.isfinite(cost):
            return cost, X_G

    return np.inf, None


def find_antipodal_grasp(environment_diagram, environment_context, cameras, predictions, object_idx: int, meshcat=None):
    rng = np.random.default_rng()

    # Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(LoadModelDirectives(FindResource("models/schunk_wsg_50_welded_fingers.dmd.yaml")), plant, parser)
    plant.Finalize()
    
    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    rgb_ims = [c.rgb_im for c in cameras]
    #print("=============rgb im type", type(rgb_ims[0]))
    depth_ims = [c.depth_im for c in cameras]
    #print("=============depth im type", type(depth_ims[0]))
    project_depth_to_pC_funcs = [c.project_depth_to_pC for c in cameras]
    X_WCs = [c.X_WC for c in cameras]
    
    cloud = get_merged_masked_pcd(
        predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs, object_idx, meshcat=meshcat)

    plant_context = plant.GetMyContextFromRoot(context)
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)

    min_cost = np.inf
    best_X_G = None
    #print("cloud size", cloud.size())
    #print("cloud", cloud.xyzs())
    for i in range(100):
        cost, X_G = GenerateAntipodalGraspCandidate(diagram, context, cloud, rng)
        if np.isfinite(cost) and cost < min_cost:
            min_cost = cost
            best_X_G = X_G
    
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), best_X_G)
    diagram.ForcedPublish(context)

def function():
    return

class GraspSelector(LeafSystem):
    def __init__(self, plant, bin_instance, camera_num, model, meshcat=None):
        LeafSystem.__init__(self)
        for i in range(camera_num):
            self.DeclareAbstractInputPort(f"rgb_im_{i}", AbstractValue.Make(Image(1,1)))
            self.DeclareAbstractInputPort(f"depth_im_{i}", AbstractValue.Make(Image[PixelType.kDepth32F](1,1)))
            self.DeclareAbstractInputPort(f"X_WC_{i}", AbstractValue.Make(RigidTransform()))
            self.DeclareAbstractInputPort(f"cam_info_{i}", AbstractValue.Make(CameraInfo(10,10,np.pi/4)))
        self.cam_contexts = []
        self.meshcat = meshcat
        self.DeclareAbstractInputPort("color", AbstractValue.Make(int))
        #self.DeclareAbstractInputPort("diagram", AbstractValue.Make(Diagram))
        #self.DeclareAbstractInputPort("cloud", AbstractValue.Make(PointCloud()))
        #self.DeclareAbstractInputPort("predictions", AbstractValue.Make([]))
        #self.DeclareAbstractInputPort("color", AbstractValue.Make(int))
        #self.DeclareAbstractInputPort("rng", AbstractValue.Make(np.random.Generator))
        
        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make((np.inf, RigidTransform())),
            self.SelectGrasp,
        )
        port.disable_caching_by_default()
        
        # Compute crop box.
        self.context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("table_top_link", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(self.context, bin_body)
        margin = 0.001  # only because simulation is perfect!
    

        self._crop_lower =  [-1, 0, 0.025]
        self._crop_upper =  [1, 1, 0.17]
        
        
        self._internal_model, self._internal_plant, self._internal_scene_graph = WarehouseSceneSystem.make_internal_model()
        self._internal_model_context = (
            self._internal_model.CreateDefaultContext()
        )
        self._rng = np.random.default_rng()
        self.camera_num = camera_num
        self.model = model
       

    def SelectGrasp(self, context, output: AbstractValue):
        rgb_ims = []
        depth_ims = []
        X_WCs = []
        cam_infos = []
        
        # Run predictions
        for i in range(self.camera_num):
            rgb_ims.append(self.GetInputPort(f"rgb_im_{i}").Eval(context).data)
        
        predictions = get_predictions(self.model, rgb_ims)

        # Merging the point clouds
        for i in range(self.camera_num):
            depth_im_read = self.GetInputPort(f"depth_im_{i}").Eval(context).data.squeeze()
            depth_im = deepcopy(depth_im_read)
            depth_im[depth_im == np.inf] = 10.0
            depth_ims.append(depth_im)
            
        for i in range(self.camera_num):
            X_WCs.append(self.GetInputPort(f"X_WC_{i}").Eval(context))
        
        for i in range(self.camera_num):
            cam_infos.append(self.GetInputPort(f"cam_info_{i}").Eval(context))

        object_idx = self.GetInputPort("color").Eval(context)
        
        diagram = self._internal_model
        plant = self._internal_plant
        scene_graph = self._internal_scene_graph
        self._internal_context = diagram.CreateDefaultContext()
        

        cloud = get_merged_masked_pcd(
            predictions, rgb_ims, depth_ims, self.project_depth_to_pC, X_WCs, cam_infos, object_idx, meshcat=self.meshcat)
        
        #print(predictions)
        min_cost = np.inf
        best_X_G = None
        
        if cloud is None:
            output.set_value((min_cost, best_X_G))
            return
        
        for i in range(200):
            cost, X_G = GenerateAntipodalGraspCandidate(diagram, self._internal_context, cloud, self._rng)
            if np.isfinite(cost) and cost < min_cost:
                min_cost = cost
                best_X_G = X_G

        output.set_value((min_cost, best_X_G))
        
    def set_cam_contexts(self, cam_contexts):
        self.cam_contexts = cam_contexts

    def set_context(self, context):
        self.own_context = context

    def project_depth_to_pC(self, cam_info, depth_pixel):
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
            cx = cam_info.center_x()
            cy = cam_info.center_y()
            fx = cam_info.focal_x()
            fy = cam_info.focal_y()
            X = (u-cx) * Z/fx
            Y = (v-cy) * Z/fy
            pC = np.c_[X,Y,Z]
            return pC
    
    def get_pC(self, outer_context):
        rgb_ims = []
        depth_ims = []
        X_WCs = []
        cam_infos = []

        
        context = self.GetMyContextFromRoot(outer_context)
        # Run predictions
        for i in range(self.camera_num):
            rgb_ims.append(self.GetInputPort(f"rgb_im_{i}").Eval(context).data)
            depth_im_read = self.GetInputPort(f"depth_im_{i}").Eval(context).data.squeeze()
            depth_im = deepcopy(depth_im_read)
            depth_im[depth_im == np.inf] = 10.0
            depth_ims.append(depth_im)
            X_WCs.append(self.GetInputPort(f"X_WC_{i}").Eval(context))
            cam_infos.append(self.GetInputPort(f"cam_info_{i}").Eval(context))
        
        predictions = get_predictions(self.model, rgb_ims)

        diagram = self._internal_model
        plant = self._internal_plant
        scene_graph = self._internal_scene_graph

        color = self.GetInputPort("color").Eval(context)
    
        cloud = get_merged_masked_pcd(predictions, rgb_ims, depth_ims, self.project_depth_to_pC, X_WCs, cam_infos, color, meshcat=self.meshcat)
        return cloud

    def get_grasp(self, outer_context):
        cloud = self.get_pC(outer_context)
        min_cost = np.inf
        best_X_G = None
        diagram = self._internal_model
        plant = self._internal_plant
        scene_graph = self._internal_scene_graph
        context = diagram.CreateDefaultContext()
        for i in range(100):
            cost, X_G = GenerateAntipodalGraspCandidate(diagram, context, cloud, self._rng)
            if np.isfinite(cost) and cost < min_cost:
                min_cost = cost
                best_X_G = X_G

        
        return (min_cost, best_X_G)