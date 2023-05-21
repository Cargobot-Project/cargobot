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
#AddIiwaCollision(wh.plant)
#cameras = generate_cameras(environment_diagram, environment_context, meshcat)
print("Finished setting up the environment.\n")


#print("predictions type", type(predictions), type(predictions[0]))


"""pcd = wh.get_pC()
meshcat.SetObject("masked_cloud", pcd, point_size=0.003)
print("Finished running inference on camera 0.\n")

# Find grasp poseprint("Finding optimal grasp pose...")
grasp_cost, grasp_pose = wh.get_grasp()
print("Found optimal grasp pose.\n")

print( "Grasp pose: ", grasp_pose)
print( "Grasp cost: ", grasp_cost)
"""
simulator = Simulator(environment_diagram, environment_context)
context = simulator.get_context()

dimension = 6
num_of_boxes = 5
grid = [f"{x},{y}" for x in range(dimension) for y in range(dimension)]
box_positions = np.random.choice(grid, replace=False, size=num_of_boxes)
plant_context = wh.plant.GetMyMutableContextFromRoot(context)
z=0.1
i = 0
for body_index in wh.plant.GetFloatingBaseBodies():
    tf = RigidTransform(
        RotationMatrix(),
        [0.2*(int(box_positions[i].split(",")[0])-dimension/2)+0.7, 0.2*(int(box_positions[i].split(",")[1])-dimension/2)-0.1, z]
    )
    wh.plant.SetFreeBodyPose(plant_context, wh.plant.get_body(body_index), tf)
    i += 1
rgb_ims = wh.get_rgb_ims()

# Make prediction from all cameras
"""print("Run inference on camera 0...")
object_idx = 1
predictions = get_predictions(model, rgb_ims)
for i in range(len(rgb_ims)):
    #print("Camera", i)
    plot_camera_view(rgb_ims, i, f"./out/camera{i}.png")
plot_predictions(predictions, object_idx, f"./out/")"""

simulator.Initialize()
"""pcd = wh.get_pC()
meshcat.SetObject("masked_cloud", pcd, point_size=0.003)
print("Finished running inference on camera 0.\n")"""
graph = pydot.graph_from_dot_data(environment_diagram.GetGraphvizString())[0]
graph.write_jpg("system_output.jpg")


simulator.AdvanceTo(0.1)
meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
visualizer.StartRecording(True)
meshcat.AddButton("Stop Simulation", "Escape")
while simulator.get_context().get_time() < 4000 and meshcat.GetButtonClicks("Stop Simulation") < 1:
    simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
visualizer.PublishRecording()

while 1:
    time.sleep(1)
