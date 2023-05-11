import numpy as np
import sys
import os

from typing import List

from pydrake.all import RigidTransform, RollPitchYaw
from pydrake.all import MakeRenderEngineVtk, RenderEngineVtkParams, DepthRenderCamera, RenderCameraCore, CameraInfo, ClippingRange, DepthRange
from pydrake.all import ModelInstanceIndex, RgbdSensor, DepthImageToPointCloud, BaseField, LeafSystem, AbstractValue

CARGOBOT_CAMERA_POSES = [
    RigidTransform(RollPitchYaw(np.pi, -np.pi/4,  np.pi / 2.0), [0, -1.5, 1.5]),
    RigidTransform(RollPitchYaw(np.pi,  np.pi/4,  np.pi / 2.0), [0,  1.5, 1.5])    
]

# Custom version of manipulation library's AddRgbdSensors()
def add_rgbd_sensors(builder, plant, scene_graph, poses: List[RigidTransform]=CARGOBOT_CAMERA_POSES):
    """
    Custom version of manipulation.scenarios.AddRgbdSensors. It uses our camera poses instead of default RigidTransform().
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(
            renderer, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

    depth_camera = DepthRenderCamera(
        RenderCameraCore(
            renderer,
            CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
            ClippingRange(near=0.1, far=10.0),
            RigidTransform(),
        ),
        DepthRange(0.1, 10.0),
    )

    model_instance_prefix = "camera"
    cam_index = 0
    print("poses", poses)
    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)
        print("-------> model name", model_name)
        if model_name.startswith(model_instance_prefix):
            print("-------> found index", index)
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(
                    parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                    X_PB=poses[cam_index],
                    depth_camera=depth_camera,
                    show_window=False,
                )
            )
            rgbd.set_name(model_name)

            builder.Connect(
                scene_graph.get_query_output_port(),
                rgbd.query_object_input_port(),
            )

            # Export the camera outputs
            builder.ExportOutput(
                rgbd.color_image_output_port(), f"{model_name}_rgb_image"
            )
            builder.ExportOutput(
                rgbd.depth_image_32F_output_port(), f"{model_name}_depth_image"
            )
            builder.ExportOutput(
                rgbd.label_image_output_port(), f"{model_name}_label_image"
            )

            # Add a system to convert the camera output into a point cloud
            to_point_cloud = builder.AddSystem(
                DepthImageToPointCloud(
                    camera_info=rgbd.depth_camera_info(),
                    fields=BaseField.kXYZs | BaseField.kRGBs,
                )
            )
            builder.Connect(
                rgbd.depth_image_32F_output_port(),
                to_point_cloud.depth_image_input_port(),
            )
            builder.Connect(
                rgbd.color_image_output_port(),
                to_point_cloud.color_image_input_port(),
            )

            class ExtractBodyPose(LeafSystem):
                def __init__(self, body_index, pose):
                    LeafSystem.__init__(self)
                    self.body_index = body_index
                    self.DeclareAbstractInputPort(
                        "poses",
                        plant.get_body_poses_output_port().Allocate(),
                    )

                    self.DeclareAbstractOutputPort(
                        "pose",
                        #lambda: AbstractValue.Make(RigidTransform()),
                        lambda: AbstractValue.Make(pose),
                        self.CalcOutput,
                    )

                def CalcOutput(self, context, output):
                    poses = self.EvalAbstractInput(context, 0).get_value()
                    pose = poses[int(self.body_index)]
                    output.get_mutable_value().set(
                        pose.rotation(), pose.translation()
                    )

            camera_pose = builder.AddSystem(ExtractBodyPose(body_index, poses[cam_index]))
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

            cam_index += 1


def add_point_cloud_port(builder, plant, rgbd: RgbdSensor):
    """
    DO NOT USE
    """
    to_point_cloud = builder.AddSystem(
        DepthImageToPointCloud(
            camera_info=rgbd.depth_camera_info(),
            fields=BaseField.kXYZs | BaseField.kRGBs,
        )
    )
    builder.Connect(
        rgbd.depth_image_32F_output_port(),
        to_point_cloud.depth_image_input_port(),
    )
    builder.Connect(
        rgbd.color_image_output_port(),
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
                lambda: AbstractValue.Make(RigidTransform()),
                self.CalcOutput,
            )

        def CalcOutput(self, context, output):
            poses = self.EvalAbstractInput(context, 0).get_value()
            pose = poses[int(self.body_index)]
            output.get_mutable_value().set(
                pose.rotation(), pose.translation()
            )

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        model_instance_prefix = "camera"
        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            
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