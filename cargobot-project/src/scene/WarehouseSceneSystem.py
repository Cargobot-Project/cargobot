import numpy as np
import time
from pydrake.all import (AbstractValue, AngleAxis, Concatenate, DiagramBuilder,
                         LeafSystem, MeshcatVisualizer, MeshcatPointCloudVisualizer, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform, RotationMatrix,
                         RollPitchYaw, Simulator, StartMeshcat, LoadModelDirectivesFromString, ProcessModelDirectives, 
                         ModelInstanceIndex, PassThrough, Demultiplexer, MultibodyPlant, InverseDynamicsController, 
                         FindResourceOrThrow, RevoluteJoint, Adder, StateInterpolatorWithDiscreteDerivative, 
                         SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem, DifferentialInverseKinematicsParameters, DifferentialInverseKinematicsIntegrator)

from manipulation import running_as_notebook
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardPointCloud
from manipulation.pick import (MakeGripperCommandTrajectory, MakeGripperFrames,
                               MakeGripperPoseTrajectory)
from manipulation.scenarios import (AddIiwaDifferentialIK,
                                    AddIiwa, AddWsg)

from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser

from manipulation.utils import FindResource, AddPackagePaths
from manipulation.scenarios import AddRgbdSensors
from IPython.display import HTML, SVG, display
import pydot

from pydrake.all import DepthImageToPointCloud, BaseField, MeshcatVisualizerParams, Role, MeshcatVisualizer
from manip.motion import *

from scene.CameraSystem import generate_cameras
from scene.SceneBuilder import add_rgbd_sensors, CARGOBOT_CAMERA_POSES
from demos.overwrite import *

def WarehouseSceneSystem(
        meshcat,
        scene_path: str="/usr/cargobot/cargobot-project/res/box_with_cameras.dmd.yaml",
        name="warehouse_scene_system",
        add_cameras: bool=True
        ):
    builder = DiagramBuilder()

    station = builder.AddSystem(
        MakeManipulationStation( time_step=0.002, filename=scene_path))
    
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    # Create the physics engine + scene graph.
    #plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_query_output_port(),
                        meshcat.get_geometry_query_input_port())
        
    # Add arm
    robot = station.GetSubsystemByName("iiwa_controller").get_multibody_plant_for_control()
    
    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    plan = builder.AddSystem(PickAndPlaceTrajectory(plant))
    
    builder.Connect(diff_ik.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(plan.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_state_estimated"),
                    diff_ik.GetInputPort("robot_state"))

    builder.Connect(plan.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))

    builder.Connect(station.GetOutputPort("body_poses"),
                    plan.GetInputPort("body_poses"))
    
    builder.Connect(icp.GetOutputPort("X_WO"), plan.GetInputPort("X_WO"))

    # Adds predefined cameras
    if add_cameras:
        print("--> Adding cameras...")
        for i, X_WC in enumerate(CARGOBOT_CAMERA_POSES):
            camera = AddRgbdSensor(builder, scene_graph, X_WC, output_port=station.GetOutputPort("query_object"))
            camera.set_name(f"camera{i}")
            builder.ExportOutput(camera.color_image_output_port(), f"camera{i}_rgb_image")
            builder.ExportOutput(camera.label_image_output_port(), f"camera{i}_label_image")
            builder.ExportOutput(camera.depth_image_32F_output_port(), f"camera{i}_depth_image")

            to_point_cloud = builder.AddSystem(
                DepthImageToPointCloud(
                    camera_info=camera.depth_camera_info(),
                    fields=BaseField.kXYZs | BaseField.kRGBs,
                )
            )
            builder.Connect(
                camera.depth_image_32F_output_port(),
                to_point_cloud.depth_image_input_port(),
            )
            builder.Connect(
                camera.color_image_output_port(),
                to_point_cloud.color_image_input_port(),
            )
            
            builder.Connect(
                camera.body_pose_in_world_output_port(),
                to_point_cloud.camera_pose_input_port()
            )

            builder.ExportOutput(to_point_cloud.point_cloud_output_port(), f"camera{i}_point_cloud")

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    diagram.set_name(name)
    context = diagram.CreateDefaultContext()

    return diagram, context, visualizer, plan #, cameras