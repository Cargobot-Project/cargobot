#!/usr/bin/env python
# coding: utf-8

# # Antipodal grasp with deep segmentation
# In this problem, you will use same the antipodal grasp strategy we used with geometric perception, but you'll use deep perception to restrict your grasps to a single object (the mustard bottle).
# 
# We'll be using the Mask RCNN model that's trained in [this script](https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/segmentation_train.ipynb) from the textbook. As an input, the model takes an image, and it outputs a series of masks showing where the objects we've trained it on are in the image. (In this case, those objects are images in the YCB dataset.) Once we know which pixels contain the object we wish to grasp, then we can project them back out to point clouds using the depth image and select an antipodal grasp using just those data points.
# 
# Your job in this notebook will be to use the masks output by our neural network to filter the point cloud to only include points on the mustart bottle. Once we have a filtered point cloud, we'll be able to use them to generate antipodal grasps just on our object of interest.

# In[1]:


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


# In[2]:


# Start the visualizer.
meshcat = StartMeshcat()


# ## Load model
# To avoid making you wait and train the model yourself, we'll use the pre-trained model from the textbook. First, we need to load it.

# In[3]:


if running_as_notebook:
    model_file = 'clutter_maskrcnn_model.pt'
    if not os.path.exists(model_file):
        urlretrieve(
            "https://groups.csail.mit.edu/locomotion/clutter_maskrcnn_model.pt", model_file)

model_file = 'clutter_maskrcnn_model.pt'
# In[4]:


mustard_ycb_idx = 3
if running_as_notebook:
    def get_instance_segmentation_model(num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, progress=False)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

        return model

    num_classes = 7
    model = get_instance_segmentation_model(num_classes)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(
        torch.load('clutter_maskrcnn_model.pt', map_location=device))
    model.eval()

    model.to(device)


# ## Set up camera system
# Now that we've loaded our network, we need to set up the Drake model for our system. It has several objects from the YCB data set and two cameras.

# In[5]:


def ClutteredSceneSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(
        LoadModelDirectives(FindResource("models/segmentation_and_grasp_scene.dmd.yaml")),
        plant, parser)
    
    plant.Finalize()

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    context = diagram.CreateDefaultContext()
    return diagram, context


# In[6]:


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

environment_diagram, environment_context = ClutteredSceneSystem()
cameras = []
cameras.append(
    CameraSystem(0, meshcat, environment_diagram, environment_context))
cameras.append(
    CameraSystem(1, meshcat, environment_diagram, environment_context))


# ### Examining camera views
# If you take a look at meshcat, under "scene > drake," you'll find two check boxes: one labeled "Camera 0 point cloud" and one labeled "Camera 1 point cloud." Toggle these to see the different views the two cameras get.
# 
# We can also look directly at the images captured by each camera:

# In[7]:


plt.imshow(cameras[0].rgb_im)
plt.title("View from camera 0")
plt.show()

plt.imshow(cameras[1].rgb_im)
plt.title("View from camera 1")
plt.show()


# ## Generate masks for each image
# Now that we have a network and camera inputs, we can start processing our inputs. First, we will evaluate the mask (which is the output from our network) for each image.

# In[8]:


if running_as_notebook:
    with torch.no_grad():
        predictions = []
        predictions.append(
            model([Tf.to_tensor(cameras[0].rgb_im[:, :, :3]).to(device)]))
        predictions.append(
            model([Tf.to_tensor(cameras[1].rgb_im[:, :, :3]).to(device)]))
    for i in range(2):
        for k in predictions[i][0].keys():
            if k == "masks":
                predictions[i][0][k] = predictions[i][0][k].mul(
                    255).byte().cpu().numpy()
            else:
                predictions[i][0][k] = predictions[i][0][k].cpu().numpy()
else:
    predictions = []
    for i in range(2):
        prediction = []
        prediction.append(
            np.load(LoadDataResource("prediction_{}.npz".format(i))))
        predictions.append(prediction)


# `predictions[0]` was run on the image from Camera 0, while `predictions[1]` was run on the image from Camera 1. 
# 
# Lets take a minute to understand this output. Breaking it down by each key in the output dictionary:
# * The "boxes" correspond to bounding boxes on the regions containing the object
# * The "labels" tell us which class the model has associated with the (as in, whether it's the mustard bottle, the Cheez-it box, the spam container, etc. Each model is identified by a number.)
# * The "scores" are a measure of confidence in the model predictions
# * The "masks" are arrays which indicate which pixels belong to the corresponding class

# In[9]:


predictions[0]


# The two most important elements for our task are `labels`, which tells us which class the mask corresponds to, and `mask`, which gives a higher score for pixels that more likely correspond to points on the mustard bottle.
# 
# Note that we defined `mustard_ycb_idx = 3` at the top of the notebook; that's the value of the label for the class we care about.
# 
# The following cells visualize the masks we get:

# In[10]:


for i, prediction in enumerate(predictions):
    mask_idx = np.argmax(predictions[i][0]['labels'] == mustard_ycb_idx)
    mask = predictions[i][0]['masks'][mask_idx,0]

    plt.imshow(mask)
    plt.title("Mask from Camera " + str(i))
    plt.colorbar()
    plt.show()


# ## Generate point cloud
# ### 6.2a
# Using the masks we've found, generate a filtered point cloud that includes images from both cameras but only includes points within the mustard. (You will fill in code to return both points and colors; technically, we only need the points for finding an antipodal grasp, but the colors are helpful for visualization.)
# 
# You will write code that does the following for each camera and image:
# 1. Extract the pixels from the mask that we consider to be within the mustard bottle (specifically: take values that are above `mask_threshold`)
# 2. Select points in the depth image corresponding to those pixels
# 3. Using the depth values, project those selected pixels back out to be points in the camera frame. You'll use the camera's `project_depth_to_pC` image to do this; refer back to problem 5.2 for how this is used.
# 4. Convert the points to the world frame
# 5. Select color values from the RGB image that correspond to your mask pixels. 

# In[11]:


def get_merged_masked_pcd(predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs, mask_threshold=150):
    """
    predictions: The output of the trained network (one for each camera)
    rgb_ims: RGBA images from each camera
    depth_ims: Depth images from each camera
    project_depth_to_pC_funcs: Functions that perform the pinhole camera operations to convert pixels
        into points. See the analogous function in problem 5.2 to see how to use it.
    X_WCs: Poses of the cameras in the world frame
    """

    pcd = []
    for prediction, rgb_im, depth_im, project_depth_to_pC_func, X_WC in \
            zip(predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs):
        # These arrays aren't the same size as the correct outputs, but we're
        # just initializing them to something valid for now.
        spatial_points = np.zeros((3, 1))
        rgb_points = np.zeros((3, 1))

        ######################################
        # Your code here (populate spatial_points and rgb_points)
        ######################################

        mask_idx = np.argmax(prediction[0]['labels'] == mustard_ycb_idx)
        mask = prediction[0]['masks'][mask_idx, 0]
        mask_uvs = mask >= mask_threshold
        #print(np.sum(mask_uvs))

        img_h, img_w = depth_im.shape
        v_range = np.arange(img_h)
        u_range = np.arange(img_w)
        depth_u, depth_v = np.meshgrid(u_range, v_range)
        #mask_u, mask_v = np.array(prediction["mask"])
        depth_pnts = np.dstack([depth_v, depth_u, depth_im])
        depth_pnts = depth_pnts[mask_uvs].reshape([-1, 3])
        
        # point poses in camera frame
        spatial_points = project_depth_to_pC_func(depth_pnts)
        #spatial_points = X_WC @ spatial_points


        #print(rgb_im.shape)
        rgb_im = rgb_im[:, :, :3]

        #print(rgb_im.shape)
        img_h, img_w, img_c = rgb_im.shape
        v_range = np.arange(img_h)
        u_range = np.arange(img_w)
        rgb_u, rgb_v = np.meshgrid(u_range, v_range)
        #print(rgb_u.shape, rgb_v.shape)
        #rgb_points = np.dstack([rgb_v, rgb_u, rgb_im])
        rgb_points = np.dstack([rgb_im])
        rgb_points = rgb_points[mask_uvs]
        rgb_points = rgb_points.reshape([-1, 3])
        
        spatial_points = spatial_points.T
        rgb_points = rgb_points.T

        spatial_points = X_WC.rotation() @ spatial_points
        #print("final spatial", spatial_points.shape)
        spatial_points = spatial_points + np.concatenate([[X_WC.translation()]] * spatial_points.shape[1]).T

        #print("spatial shape", spatial_points.shape)
        #print("rgb shape", rgb_points.shape)
        # You get an unhelpful RunTime error if your arrays are the wrong
        # shape, so we'll check beforehand that they're the correct shapes.
        assert len(spatial_points.shape
                  ) == 2, "Spatial points is the wrong size -- should be 3 x N"
        assert spatial_points.shape[
            0] == 3, "Spatial points is the wrong size -- should be 3 x N"
        assert len(rgb_points.shape
                  ) == 2, "RGB points is the wrong size -- should be 3 x N"
        assert rgb_points.shape[
            0] == 3, "RGB points is the wrong size -- should be 3 x N"
        assert rgb_points.shape[1] == spatial_points.shape[1]
        print("spatial points shape", spatial_points.shape)
        N = spatial_points.shape[1]
        pcd.append(PointCloud(N, Fields(BaseField.kXYZs | BaseField.kRGBs)))
        pcd[-1].mutable_xyzs()[:] = spatial_points
        pcd[-1].mutable_rgbs()[:] = rgb_points
        # Estimate normals
        pcd[-1].EstimateNormals(radius=0.1, num_closest=30)
        # Flip normals toward camera
        pcd[-1].FlipNormalsTowardPoint(X_WC.translation())
    
    # Merge point clouds.
    merged_pcd = Concatenate(pcd)
    print("merged pcd size", merged_pcd.size())
    print("merged pcd", merged_pcd.xyzs())
    # Voxelize down-sample.  (Note that the normals still look reasonable)
    #return merged_pcd
    return merged_pcd.VoxelizedDownSample(voxel_size=0.005)


# Now let's use this function to visualize the output of `get_merged_masked_pcd`.

# In[12]:


rgb_ims = [c.rgb_im for c in cameras]
depth_ims = [c.depth_im for c in cameras]
project_depth_to_pC_funcs = [c.project_depth_to_pC for c in cameras]
X_WCs = [c.X_WC for c in cameras]

pcd = get_merged_masked_pcd(predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs)
meshcat.SetObject("masked_cloud", pcd, point_size=0.003)


# ## Select a grasp
# The following code uses your point cloud function to find an antipodal grasp, similar to the previous problem set (PSet 6).

# In[13]:


def find_antipodal_grasp(environment_diagram, environment_context, cameras):
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
    depth_ims = [c.depth_im for c in cameras]
    project_depth_to_pC_funcs = [c.project_depth_to_pC for c in cameras]
    X_WCs = [c.X_WC for c in cameras]

    cloud = get_merged_masked_pcd(
        predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs)

    plant_context = plant.GetMyContextFromRoot(context)
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)

    min_cost = np.inf
    best_X_G = None
    print("cloud size", cloud.size())
    print("cloud", cloud.xyzs())
    for i in range(100):
        cost, X_G = GenerateAntipodalGraspCandidate(diagram, context, cloud, rng)
        if np.isfinite(cost) and cost < min_cost:
            min_cost = cost
            best_X_G = X_G
    
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), best_X_G)
    diagram.ForcedPublish(context)


if running_as_notebook:
    find_antipodal_grasp(environment_diagram, environment_context, cameras)


# ## Summary
# If you take a look at meshcat, we've reached our end goal of generating an antipodal grasp! We've now shown that we can leverage the RGB data and our trained network to filter our point clouds and get an antipodal grasp for a specific object. Now, let's think about the implications of some of the design choices we made along the way.

# ## Written questions
# Answer the following  questions in your written submission for this problem set.
# 
# ### 6.2b
# Let's think back to the "Examining camera views" section. Toggling between the views of the two cameras, each of the cameras contributes different information about the scene. Why do we need the information from both of them to find antipodal grasps?

# ### 6.2c
# Our goal in this task is to grasp the mustard bottle. The first step of get_merged_masked_pcd() was to extract the pixels that correspond to the mustard bottle. If we skipped this step and instead considered the entire point cloud, what type of "undesirable" grasps might we select?

# ### 6.2d
# In this notebook, after we mask the point clouds based on the segmentation results, we don't use the other point clouds again (including when we evaluate our grasp candidates). Think about how we have checked whether or not a grasp is feasible. How might discarding the point cloud data for the other objects inadvertently lead us to select an invalid grasp?

# ## How will this notebook be Graded?
# 
# If you are enrolled in the class, this notebook will be graded using [Gradescope](www.gradescope.com). You should have gotten the enrollement code on our announcement in Piazza. 
# 
# For submission of this assignment, you must do two things. 
# - Download and submit the notebook `segmentation_and_grasp.ipynb` to Gradescope's notebook submission section, along with your notebook for the other problems.
# - Answer parts (b) through (d) in the written section of Gradescope as a part of your `pdf` submission. 
# 
# We will evaluate the local functions in the notebook to see if the function behaves as we have expected. For this exercise, the rubric is as follows:
# - [4 pts] `get_merged_masked_pcd` must be implemented correctly. 
# - [2 pts] Correct answer for 6.2b
# - [2 pts] Correct answer for 6.2c
# - [2 pts] Correct answer for 6.2d

# In[14]:


from manipulation.exercises.segmentation.test_segmentation_and_grasp import (
  TestSegmentationAndGrasp
)
from manipulation.exercises.grader import Grader

Grader.grade_output([TestSegmentationAndGrasp], [locals()], 'results.json')
Grader.print_test_results('results.json')


# In[15]:

while True:
    time.sleep(1)



# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=462f51c7-5021-4be0-bbcc-d2020b02a3ec' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
