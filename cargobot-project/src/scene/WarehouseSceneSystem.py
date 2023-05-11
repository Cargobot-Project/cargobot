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
                                    MakeManipulationStation, AddIiwa, AddWsg)

from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser

from manipulation.utils import FindResource, AddPackagePaths
from manipulation.scenarios import AddRgbdSensor, AddRgbdSensors
from IPython.display import HTML, SVG, display
import pydot
from scene.CameraSystem import generate_cameras
from scene.SceneBuilder import add_rgbd_sensors, CARGOBOT_CAMERA_POSES

def WarehouseSceneSystem(
        meshcat,
        scene_path: str="/usr/cargobot/cargobot-project/res/box_with_cameras.dmd.yaml",
        name="warehouse_scene_system",
        add_cameras: bool=True
        ):
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.AddModels(scene_path)
    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_query_output_port(),
                        meshcat.get_geometry_query_input_port())

    #AddRgbdSensors(builder, plant, scene_graph)

    # Adds predefined cameras
    if add_cameras:
        print("--> Adding cameras...")
        add_rgbd_sensors(builder, plant, scene_graph, poses=CARGOBOT_CAMERA_POSES)
        """
        for i, X_WC in enumerate(CARGOBOT_CAMERA_POSES):
            camera = AddRgbdSensor(builder, scene_graph, X_WC)
            builder.ExportOutput(camera.color_image_output_port(), f"camera{i}_rgb_image")
            builder.ExportOutput(camera.label_image_output_port(), f"camera{i}_label_image")
            builder.ExportOutput(camera.depth_image_32F_output_port(), f"camera{i}_depth_image")
        """

    diagram = builder.Build()
    diagram.set_name(name)
    context = diagram.CreateDefaultContext()

    return diagram, context #, cameras