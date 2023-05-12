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

from pydrake.all import (DepthImageToPointCloud, BaseField, MeshcatVisualizerParams, Role,
                         MeshcatVisualizer)

from scene.CameraSystem import generate_cameras
from scene.SceneBuilder import add_rgbd_sensors, CARGOBOT_CAMERA_POSES
from scene.utils import ConfigureParser
from manip.motion import *
from manip.grasp import *
from demos.overwrite import *
from scene.CameraSystem import *


def WarehouseSceneSystem(
        meshcat,
        scene_path: str="/usr/cargobot/cargobot-project/res/box_with_cameras.dmd.yaml",
        name="warehouse_scene_system",
        add_cameras: bool=True
        ):
    builder = DiagramBuilder()
    box_cnt = 5
    station = builder.AddSystem(
        MakeManipulationStation( time_step=0.002, filename=scene_path, box_cnt=box_cnt))
    
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

        
    """# Add arm
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
    
    builder.Connect(icp.GetOutputPort("X_WO"), plan.GetInputPort("X_WO"))"""

    

    to_point_clouds = list()
    rgb_ports = list()
    depth_ports = list()

    # Adds predefined cameras
    if add_cameras:
        print("--> Adding cameras...")
        for i, X_WC in enumerate(CARGOBOT_CAMERA_POSES):
            camera = AddRgbdSensor(builder, scene_graph, X_WC, output_port=station.GetOutputPort("query_object"))
            camera.set_name(f"camera{i}")
            rgb_ports.append(camera.color_image_output_port())
            depth_ports.append(camera.depth_image_32F_output_port())
            #builder.ExportOutput(camera.label_image_output_port(), f"camera{i}_label_image")

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

            to_point_clouds.append(to_point_cloud.point_cloud_output_port())

    planner = wire_ports(builder, plant, station, to_point_clouds, rgb_ports, depth_ports)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    diagram.set_name(name)
    context = diagram.CreateDefaultContext()

    return diagram, context, visualizer, planner #, cameras



def make_internal_model():
    """
    Makes an internal diagram for only the objects we know about, like truck, floor, gripper etc.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.AddModels("/usr/cargobot/cargobot-project/res/demo_envs/mobilebase_perception_demo.dmd.yaml")
    plant.Finalize()
    return builder.Build(), plant, scene_graph


def wire_ports( builder, plant, station, to_point_clouds, rgb_ports, depth_ports):
    grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("table_top")#,
            #camera_body_indices=[
                #plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[
                 #   0
                #],
               # plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[
                #    0
                #],
            #],
        )
    )
    
    builder.Connect(
        rgb_ports[0],
        grasp_selector.get_input_port(0),
    )

    builder.Connect(
        depth_ports[0],
        grasp_selector.get_input_port(1),
    )
    
    builder.Connect(

        grasp_selector.get_input_port(1)
    )

    builder.Connect(
        station.GetOutputPort("body_poses"),
        grasp_selector.GetInputPort("body_poses"),
    )

    planner = builder.AddSystem(Planner(plant))
    builder.Connect(
        station.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(
        grasp_selector.get_output_port(),
        planner.GetInputPort("grasp"),
    )
    
    builder.Connect(
        station.GetOutputPort("wsg_state_measured"),
        planner.GetInputPort("wsg_state"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa_position_measured"),
        planner.GetInputPort("iiwa_position"),
    )

    robot = station.GetSubsystemByName(
        "iiwa_controller"
    ).get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    builder.Connect(
        station.GetOutputPort("iiwa_state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        station.GetInputPort("wsg_position"),
    )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(plant.num_positions()))
    builder.Connect(
        diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik")
    )
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        switch.DeclareInputPort("position"),
    )
    builder.Connect(
        switch.get_output_port(), station.GetInputPort("iiwa_position")
    )
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    return planner