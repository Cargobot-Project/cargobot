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
from manipulation.clutter import GenerateAntipodalGraspCandidate
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

def find_antipodal_grasp(environment_diagram, environment_context, cameras, meshcat, predictions, object_idx: int):
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
    def __init__(self, plant, bin_instance, camera_num, model, meshcat):
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
        a = X_B.multiply(
            [-0.22 + 0.025 + margin, -0.29 + 0.025 + margin, 0.015 + margin]
        )
        b = X_B.multiply([0.22 - 0.1 - margin, 0.29 - 0.025 - margin, 2.0])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)
        
        
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
            predictions, rgb_ims, depth_ims, self.project_depth_to_pC, X_WCs, cam_infos, object_idx, self.meshcat)
        
        min_cost = np.inf
        best_X_G = None
        
        for i in range(100):
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

        object_idx = self.GetInputPort("color").Eval(context)

        cloud = get_merged_masked_pcd(predictions, rgb_ims, depth_ims, self.project_depth_to_pC, X_WCs, cam_infos, object_idx, meshcat=self.meshcat)

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