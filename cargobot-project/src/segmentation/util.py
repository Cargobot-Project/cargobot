import os
import time
from copy import deepcopy
from urllib.request import urlretrieve
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as Tf
from IPython.display import clear_output, display
from manipulation import running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate
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
from manipulation.meshcat_utils import AddMeshcatTriad
from scene.CameraSystem import cargobot_num_cameras, CameraSystem
from segmentation.plot import add_meshcat_triad
from manip.enums import *
cargobot_num_classes = 6 # TBD

def get_instance_segmentation_model(model_path: str, num_classes: int=cargobot_num_classes):
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

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(
        torch.load(model_path, map_location=device))
    model.eval()

    model.to(device)

    return model

def get_predictions(model, rgb_ims):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_cameras = len(rgb_ims)
    with torch.no_grad():
        predictions = []
        for  rgb_im in rgb_ims:
            predictions.append(
                model([Tf.to_tensor(rgb_im[:, :, :3]).to(device)]))
            
    for i in range(num_cameras):
        for k in predictions[i][0].keys():
            if k == "masks":
                predictions[i][0][k] = predictions[i][0][k].mul(
                    255).byte().cpu().numpy()
            else:
                predictions[i][0][k] = predictions[i][0][k].cpu().numpy()
    
    return predictions

def vis_normals(normals, meshcat):
    normals = normals.T
    #print("-----> normal shape:", normals.shape)
    for i in range(int(len(normals))):
        name = f'normal_vec_{i}'
        add_meshcat_triad(meshcat, name, length=0.1,
                        radius=0.01, X_PT=RigidTransform(normals[i]))
    
    """
    while True:
        print("sleeep")
        time.sleep(1)
    """
        
DEFAULT_MASK_THRESHOLD = 150

def get_merged_masked_pcd(predictions, rgb_ims, depth_ims, project_depth_to_pC_func, X_WCs, cam_infos, color: BoxColorEnum, meshcat=None, 
                            mask_threshold=DEFAULT_MASK_THRESHOLD, score_threshold=0.6):
    """
    predictions: The output of the trained network (one for each camera)
    rgb_ims: RGBA images from each camera
    depth_ims: Depth images from each camera
    project_depth_to_pC_funcs: Functions that perform the pinhole camera operations to convert pixels
        into points. See the analogous function in problem 5.2 to see how to use it.
    X_WCs: Poses of the cameras in the world frame
    """

    pcd = []
    crop_min = RigidTransform().multiply(np.array([0.2, -1.5, 0.05]))
    crop_max = RigidTransform().multiply(np.array([2, 1.5, 0.55]))
    avg = 0
    i = 0
    for prediction, rgb_im, depth_im, X_WC, cam_info in \
            zip(predictions, rgb_ims, depth_ims, X_WCs, cam_infos):
        
        # These arrays aren't the same size as the correct outputs, but we're
        # just initializing them to something valid for now.
        spatial_points = np.zeros((3, 1))
        rgb_points = np.zeros((3, 1))

        ######################################
        # Your code here (populate spatial_points and rgb_points)
        ######################################
        i += 1
        #print(prediction[0])
        print("Color given: ", color, "-", color.value)
        mask_idx = 0
        while mask_idx < len(prediction[0]["labels"]):
            #print(prediction[0]["labels"][mask_idx])
            if prediction[0]["labels"][mask_idx] == color.value:
                break
            mask_idx += 1
        if mask_idx >= len(prediction[0]["labels"]):
            continue
        #mask_idx = np.argmax(prediction[0]['labels'] == color)
        avg += prediction[0]["scores"][mask_idx] 
        mask = prediction[0]['masks'][mask_idx, 0]
        mask_uvs = mask >= mask_threshold
        #print(np.sum(mask_uvs))
        #print(depth_im.shape)
        img_h, img_w = depth_im.shape
        v_range = np.arange(img_h)
        u_range = np.arange(img_w)
        depth_u, depth_v = np.meshgrid(u_range, v_range)
        #mask_u, mask_v = np.array(prediction["mask"])
        depth_pnts = np.dstack([depth_v, depth_u, depth_im])
        depth_pnts = depth_pnts[mask_uvs].reshape([-1, 3])
        
        # point poses in camera frame
        spatial_points = project_depth_to_pC_func(cam_info, depth_pnts)
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
        print(rgb_points)
        
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
        #print("spatial points shape", spatial_points.shape)
        
        N = spatial_points.shape[1]
        pcd.append(PointCloud(N, Fields(BaseField.kXYZs | BaseField.kRGBs)))
        pcd[-1].mutable_xyzs()[:] = spatial_points
        pcd[-1].mutable_rgbs()[:] = rgb_points
        pcd[-1].EstimateNormals(radius=0.1, num_closest=30)
        pcd[-1].FlipNormalsTowardPoint(X_WC.translation())
        normals = pcd[-1].normals()
        #if meshcat is not None:
            #vis_normals(normals, meshcat)
    # Merge point clouds.
    avg = avg/i
    print(avg)
    if avg <= score_threshold:
        return None
    
    merged_pcd = Concatenate(pcd)
    #print("merged pcd size", merged_pcd.size())
    #print("merged pcd", merged_pcd.xyzs())
    # Voxelize down-sample.  (Note that the normals still look reasonable)
    merged_cropped_pcd = merged_pcd.Crop(lower_xyz=crop_min, upper_xyz=crop_max)
    meshcat.SetObject("masked_cloud", merged_cropped_pcd, point_size=0.003)
    return merged_cropped_pcd
    #return merged_pcd.VoxelizedDownSample(voxel_size=0.005)