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


# Start the visualizer.
meshcat = StartMeshcat()

# Get instance segmentation model
print("Loading segmentation model...")
model_file = '/usr/cargobot/cargobot-project/res/seg/box_maskrcnn_model_v2.pt'
model = get_instance_segmentation_model(model_path=model_file)
print("Loaded segmentation model.\n")

# Set up the environment and cameras
print("Setting up the environment...")
environment_diagram, environment_context = WarehouseSceneSystem(meshcat, scene_path="/usr/cargobot/cargobot-project/res/demo_envs/mobilebase_perception_demo.dmd.yaml")
cameras = generate_cameras(environment_diagram, environment_context, meshcat)
print("Finished setting up the environment.\n")

# Make prediction from all cameras
print("Run inference on camera 0...")
object_idx = 2
predictions = get_predictions(model, cameras)

for i, camera in enumerate(cameras):
    print("Camera", i)
    plot_camera_view(camera, i, f"./out/camera{i}.png")

plot_predictions(predictions, object_idx, f"./out/")

rgb_ims = [c.rgb_im for c in cameras]
depth_ims = [c.depth_im for c in cameras]
project_depth_to_pC_funcs = [c.project_depth_to_pC for c in cameras]
X_WCs = [c.X_WC for c in cameras]

pcd = get_merged_masked_pcd(predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs, object_idx)
meshcat.SetObject("masked_cloud", pcd, point_size=0.003)
print("Finished running inference on camera 0.\n")

# Find grasp pose
print("Finding optimal grasp pose...")
find_antipodal_grasp(environment_diagram, environment_context, cameras, meshcat, predictions, object_idx)
print("Found optimal grasp pose.\n")

while True:
    time.sleep(1)