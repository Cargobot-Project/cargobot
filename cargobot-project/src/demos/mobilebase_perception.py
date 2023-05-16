import sys
sys.path.append("/usr/cargobot/cargobot-project/src/")

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

from scene.WarehouseSceneSystem import WarehouseSceneSystem
from scene.CameraSystem import CameraSystem, generate_cameras
from segmentation.util import get_instance_segmentation_model, get_predictions, get_merged_masked_pcd
from manip.grasp import find_antipodal_grasp
from segmentation.plot import plot_camera_view, plot_predictions
import pydot
from pydrake.all import Simulator, RandomGenerator

# Fix RNGs
rng = np.random.default_rng(135)  # this is for python
generator = RandomGenerator(rng.integers(0, 1000))  # this is for c++

#print("rng type", type(rng))

# Start the visualizer.
meshcat = StartMeshcat()

# Get instance segmentation model
print("Loading segmentation model...")
model_file = '/usr/cargobot/cargobot-project/res/seg/box_maskrcnn_model_v2.pt'
model = get_instance_segmentation_model(model_path=model_file)
print("Loaded segmentation model.\n")

# Set up the environment and cameras
print("Setting up the environment...")

wh = WarehouseSceneSystem(model, meshcat, scene_path="/usr/cargobot/cargobot-project/res/demo_envs/mobilebase_perception_demo.dmd.yaml")
environment_diagram, environment_context, visualizer, plan = wh.diagram, wh.context, wh.visualizer, wh.planner 
#cameras = generate_cameras(environment_diagram, environment_context, meshcat)
print("Finished setting up the environment.\n")

rgb_ims = wh.get_rgb_ims()

# Make prediction from all cameras
print("Run inference on camera 0...")
object_idx = 1
predictions = get_predictions(model, rgb_ims)
#print("predictions type", type(predictions), type(predictions[0]))

for i in range(len(rgb_ims)):
    #print("Camera", i)
    plot_camera_view(rgb_ims, i, f"./out/camera{i}.png")

plot_predictions(predictions, object_idx, f"./out/")

"""rgb_ims = [c.rgb_im for c in cameras]
depth_ims = [c.depth_im for c in cameras]
project_depth_to_pC_funcs = [c.project_depth_to_pC for c in cameras]
X_WCs = [c.X_WC for c in cameras]"""

pcd = wh.get_pC()
meshcat.SetObject("masked_cloud", pcd, point_size=0.003)
print("Finished running inference on camera 0.\n")

# Find grasp poseprint("Finding optimal grasp pose...")
grasp_cost, grasp_pose = wh.get_grasp()
print("Found optimal grasp pose.\n")

print( "Grasp pose: ", grasp_pose)
print( "Grasp cost: ", grasp_cost)

simulator = Simulator(environment_diagram, environment_context)
context = simulator.get_context()

simulator.Initialize()

graph = pydot.graph_from_dot_data(environment_diagram.GetGraphvizString())[0]
graph.write_jpg("system_output.jpg")
visualizer.StartRecording(False)
simulator.AdvanceTo(10)
visualizer.PublishRecording()

while True:
    time.sleep(1)