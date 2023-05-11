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
                         RigidTransform, RotationMatrix, StartMeshcat)
from pydrake.multibody.parsing import (LoadModelDirectives, Parser,
                                       ProcessModelDirectives)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pydrake.all import RollPitchYaw, AbstractValue, LeafSystem

from segmentation.util import get_merged_masked_pcd
from scene.WarehouseSceneSystem import make_internal_model

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
        predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs, object_idx)

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


class GraspSelector(LeafSystem):
    def __init__(self, plant, bin_instance, camera_body_indices):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("rgb_ims", AbstractValue.Make([np.ndarray()]))
        self.DeclareAbstractInputPort("depth_ims", AbstractValue.Make([np.ndarray()]))
        self.DeclareAbstractInputPort("projection_funcs", AbstractValue.Make([function]))
        self.DeclareAbstractInputPort("X_WCs", AbstractValue.Make([RigidTransform()]))
        #self.DeclareAbstractInputPort("diagram", AbstractValue.Make(Diagram))
        #self.DeclareAbstractInputPort("cloud", AbstractValue.Make(PointCloud()))

        self.DeclareAbstractInputPort("predictions", AbstractValue.Make([]))

        self.DeclareAbstractInputPort("object_type_idx", AbstractValue.Make(int))

        #self.DeclareAbstractInputPort("rng", AbstractValue.Make(np.random.Generator))

        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make((np.inf, RigidTransform())),
            self.SelectGrasp,
        )
        port.disable_caching_by_default()
        """
        # Compute crop box.
        context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply(
            [-0.22 + 0.025 + margin, -0.29 + 0.025 + margin, 0.015 + margin]
        )
        b = X_B.multiply([0.22 - 0.1 - margin, 0.29 - 0.025 - margin, 2.0])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)
        """
        
        self._internal_model, self._internal_plant, self._internal_scene_graph = make_internal_model()
        self._internal_model_context = (
            self._internal_model.CreateDefaultContext()
        )
        self._rng = np.random.default_rng()
        self._camera_body_indices = camera_body_indices

    def SelectGrasp(self, context, output: AbstractValue):
        rgb_ims = self.get_input_port(0).Eval(context)
        depth_ims = self.get_input_port(1).Eval(context)
        project_depth_to_pC_funcs = self.get_input_port(2).Eval(context)
        X_WCs = self.get_input_port(3).Eval(context)
        predictions = self.get_input_port(4).Eval(context)

        diagram = self._internal_model
        plant = self._internal_plant
        scene_graph = self._internal_scene_graph

        object_idx = self.get_input_port(5).Eval(context)

        cloud = get_merged_masked_pcd(
            predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs, object_idx)

        plant_context = plant.GetMyContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)

        min_cost = np.inf
        best_X_G = None
        #print("cloud size", cloud.size())
        #print("cloud", cloud.xyzs())
        for i in range(100):
            cost, X_G = GenerateAntipodalGraspCandidate(diagram, context, cloud, self._rng)
            if np.isfinite(cost) and cost < min_cost:
                min_cost = cost
                best_X_G = X_G
        
        output.set_value((min_cost, best_X_G))
        