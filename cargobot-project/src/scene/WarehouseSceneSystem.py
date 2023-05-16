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
from manip.enums import *

class WarehouseSceneSystem:
    def __init__(self,
            segmentation_model,
            meshcat,
            scene_path: str="/usr/cargobot/cargobot-project/res/box_with_cameras.dmd.yaml",
            name="warehouse_scene_system",
            add_cameras: bool=True
            ):
        self.meshcat = meshcat
        
        self.builder = DiagramBuilder()
        self.box_cnt = 5
        self.station = self.builder.AddSystem(MakeManipulationStation( time_step=0.002, filename=scene_path, box_cnt=self.box_cnt))
        self.plant = self.station.GetSubsystemByName("plant")
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.station.CreateDefaultContext())
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        self.segmentation_model = segmentation_model
        
        use_meshcat = False
        if use_meshcat:
            meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
            builder.Connect(scene_graph.get_query_output_port(),
                            meshcat.get_geometry_query_input_port())


        self.to_point_clouds = []
        self.cameras = []
        # Adds predefined cameras
        if add_cameras:
            print("--> Adding cameras...")
            for i, X_WC in enumerate(CARGOBOT_CAMERA_POSES):
                camera = AddRgbdSensor(self.builder, self.scene_graph, X_WC, output_port=self.station.GetOutputPort("query_object"))
                camera.set_name(f"camera{i}")
                AddMeshcatTriad(meshcat, f"initial{i}", X_PT=X_WC)
                self.cameras.append(camera)
                #builder.ExportOutput(camera.label_image_output_port(), f"camera{i}_label_image")

                to_point_cloud = self.builder.AddSystem(
                    DepthImageToPointCloud(
                        camera_info=camera.depth_camera_info(),
                        fields=BaseField.kXYZs | BaseField.kRGBs,
                    )
                )

                self.builder.Connect(
                    camera.depth_image_32F_output_port(),
                    to_point_cloud.depth_image_input_port(),
                )

                self.builder.Connect(
                    camera.color_image_output_port(),
                    to_point_cloud.color_image_input_port()
                )
                
                self.builder.Connect(
                    camera.body_pose_in_world_output_port(),
                    to_point_cloud.camera_pose_input_port()
                )

                self.to_point_clouds.append(to_point_cloud)

        self.grasp_selector = self.builder.AddSystem(
            GraspSelector(
                self.plant,
                self.plant.GetModelInstanceByName("table_top"),
                len(self.cameras),
                self.segmentation_model,
                meshcat=self.meshcat
            )
        )

        self.planner = self.wire_ports()

        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.station.GetOutputPort("query_object"), meshcat)

        self.diagram = self.builder.Build()
        self.diagram.set_name(name)
        self.context = self.diagram.CreateDefaultContext()
        gs_context = self.grasp_selector.GetMyMutableContextFromRoot(self.context)
        
        for i, camera in enumerate(self.cameras):
            self.grasp_selector.GetInputPort(f"cam_info_{i}").FixValue(gs_context, camera.depth_camera_info())
            
        self.grasp_selector.GetInputPort("color").FixValue(gs_context, 1)
        #self.grasp_selector.set_cam_contexts([self.cameras[i].GetMyMutableContextFromRoot(self.context) for i in range(len(self.cameras))])
        self.grasp_selector.set_context( gs_context)


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


    def wire_ports(self):
        # Camera bindings
        for i, camera in enumerate(self.cameras):
            self.builder.Connect(
                camera.color_image_output_port(),
                self.grasp_selector.GetInputPort(f"rgb_im_{i}")
            )
            
            self.builder.Connect(
                camera.depth_image_32F_output_port(),
                self.grasp_selector.GetInputPort(f"depth_im_{i}")
            )

            self.builder.Connect(
                camera.body_pose_in_world_output_port(),
                self.grasp_selector.GetInputPort(f"X_WC_{i}")
            )
        
        # Planner and Grasp Selector Bindings
        box_list = [{"id": 1, "dimensions": (0.1, 0.1, 0.2), "labels": (LabelEnum.HIGH_PRIORTY, LabelEnum.HEAVY), "color": BoxColorEnum.BLUE}]
        planner = self.builder.AddSystem(Planner(self.plant, box_list=box_list))
        
        self.builder.Connect(
            self.grasp_selector.get_output_port(),
            planner.GetInputPort("grasp"),
        )
        # we only use wsg's pose :3
        self.builder.Connect(
            self.station.GetOutputPort("body_poses"),
            planner.GetInputPort("body_poses")
        )
        self.builder.Connect(
            self.station.GetOutputPort("wsg_state_measured"),
            planner.GetInputPort("wsg_state"),
        )
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_position_measured"),
            planner.GetInputPort("iiwa_position"),
        )

        robot = self.station.GetSubsystemByName(
            "iiwa_controller"
        ).get_multibody_plant_for_control()

        # Set up differential inverse kinematics.
        diff_ik = AddIiwaDifferentialIK(self.builder, robot)
        self.builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
        self.builder.Connect(
            self.station.GetOutputPort("iiwa_state_estimated"),
            diff_ik.GetInputPort("robot_state"),
        )
        self.builder.Connect(
            planner.GetOutputPort("reset_diff_ik"),
            diff_ik.GetInputPort("use_robot_state"),
        )

        self.builder.Connect(
            planner.GetOutputPort("wsg_position"),
            self.station.GetInputPort("wsg_position"),
        )

        # The DiffIK and the direct position-control modes go through a PortSwitch
        switch = self.builder.AddSystem(PortSwitch(10))
        self.builder.Connect(
            diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik")
        )
        self.builder.Connect(
            planner.GetOutputPort("iiwa_position_command"),
            switch.DeclareInputPort("position"),
        )
        self.builder.Connect(
            switch.get_output_port(), self.station.GetInputPort("iiwa_position")
        )
        self.builder.Connect(
            planner.GetOutputPort("control_mode"),
            switch.get_port_selector_input_port(),
        )
        
        return planner
    
    def get_rgb_ims(self):
        return [self.cameras[i].color_image_output_port().Eval(self.cameras[i].GetMyContextFromRoot(self.context)).data for i in range(len(self.cameras))]

    def get_pC(self):
        return self.grasp_selector.get_pC(self.context)

    def get_grasp(self):
        return self.grasp_selector.get_grasp(self.context)

def make_internal_model():
        """
        Makes an internal diagram for only the objects we know about, like truck, floor, gripper etc.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        parser.AddModels("/usr/cargobot/cargobot-project/res/demo_envs/mobilebase_perception_demo_without_robot.dmd.yaml")
        parser.AddModels("/usr/cargobot/cargobot-project/res/demo_envs/wsg_fixed.sdf")
        plant.Finalize()
        return builder.Build(), plant, scene_graph