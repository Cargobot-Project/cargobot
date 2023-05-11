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

from pydrake.all import DepthImageToPointCloud, BaseField

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

    # Adds predefined cameras
    if add_cameras:
        print("--> Adding cameras...")
        for i, X_WC in enumerate(CARGOBOT_CAMERA_POSES):
            camera = AddRgbdSensor(builder, scene_graph, X_WC)
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
            """
            for index in range(plant.num_model_instances()):
                model_instance_index = ModelInstanceIndex(index)
                model_name = plant.GetModelInstanceName(model_instance_index)
                print("-------> model name", model_name)
                if model_name.startswith(f"camera{i}"):
                    print("-------> found index", index)
                    body_index = plant.GetBodyIndices(model_instance_index)[0]

                    builder.Connect(
                        scene_graph.get_query_output_port(),
                        camera.query_object_input_port(),
                    )

                    # Export the camera outputs
                    builder.ExportOutput(
                        camera.color_image_output_port(), f"{model_name}_rgb_image"
                    )
                    builder.ExportOutput(
                        camera.depth_image_32F_output_port(), f"{model_name}_depth_image"
                    )
                    builder.ExportOutput(
                        camera.label_image_output_port(), f"{model_name}_label_image"
                    )

                    # Add a system to convert the camera output into a point cloud
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

                    class ExtractBodyPose(LeafSystem):
                        def __init__(self, body_index):
                            LeafSystem.__init__(self)
                            self.body_index = body_index
                            self.DeclareAbstractInputPort(
                                "poses",
                                plant.get_body_poses_output_port().Allocate(),
                            )

                            self.DeclareAbstractOutputPort(
                                "pose",
                                #lambda: AbstractValue.Make(RigidTransform()),
                                lambda: AbstractValue.Make(RigidTransform()),
                                self.CalcOutput,
                            )

                        def CalcOutput(self, context, output):
                            poses = self.EvalAbstractInput(context, 0).get_value()
                            pose = poses[int(self.body_index)]
                            output.get_mutable_value().set(
                                pose.rotation(), pose.translation()
                            )

                    camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                    builder.Connect(
                        plant.get_body_poses_output_port(),
                        camera_pose.get_input_port(),
                    )
                    builder.Connect(
                        camera_pose.get_output_port(),
                        to_point_cloud.GetInputPort("camera_pose"),
                    )

                    # Export the point cloud output.
                    builder.ExportOutput(
                        to_point_cloud.point_cloud_output_port(),
                        f"{model_name}_point_cloud",
                    )
                """


        #add_rgbd_sensors(builder, plant, scene_graph, poses=CARGOBOT_CAMERA_POSES)
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