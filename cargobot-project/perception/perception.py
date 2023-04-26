import numpy as np
import time
from pydrake.all import (AbstractValue, AngleAxis, Concatenate, DiagramBuilder,
                         LeafSystem, MeshcatVisualizer, MeshcatPointCloudVisualizer, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform, RotationMatrix,
                         RollPitchYaw, Simulator, StartMeshcat)

from manipulation import running_as_notebook
from manipulation.icp import IterativeClosestPoint
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.mustard_depth_camera_example import MustardPointCloud
from manipulation.pick import (MakeGripperCommandTrajectory, MakeGripperFrames,
                               MakeGripperPoseTrajectory)
from manipulation.scenarios import (AddIiwaDifferentialIK,
                                    MakeManipulationStation)

from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser

from manipulation.utils import FindResource, AddPackagePaths
from manipulation.scenarios import AddRgbdSensors

# Start the visualizer.
meshcat = StartMeshcat()

def BoxSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.AddModels("models/box_with_cameras.dmd.yaml")
    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_query_output_port(),
                        meshcat.get_geometry_query_input_port())

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram
BoxSystem()

def BoxPointCloud(normals=False, down_sample=True):
    system = BoxSystem()
    context = system.CreateDefaultContext()
    plant = system.GetSubsystemByName("plant")
    plant_context = plant.GetMyMutableContextFromRoot(context)

    pcd = []
    for i in range(3):
        cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)

        # Crop to region of interest.
        pcd.append(cloud.Crop(lower_xyz=[-.3, -.3, -.3], upper_xyz=[.3, .3, .3]))

        if normals:
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            camera = plant.GetModelInstanceByName(f"camera{i}")
            body = plant.GetBodyByName("base", camera)
            X_C = plant.EvalBodyPoseInWorld(plant_context, body)
            pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds.
    merged_pcd = Concatenate(pcd)
    if not down_sample:
        return merged_pcd
    # Down sample.
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    return down_sampled_pcd

# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.
class BoxIterativeClosestPoint(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2", model_point_cloud)

        self.DeclareAbstractOutputPort(
            "X_WO", lambda: AbstractValue.Make(RigidTransform()),
            self.EstimatePose)

        self.box = BoxPointCloud()
        meshcat.SetObject("icp_scene", self.box)

    def EstimatePose(self, context, output):
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(
                cloud.Crop(lower_xyz=[.4, -.2, 0.001], upper_xyz=[.6, .3, .3]))
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        meshcat.SetObject("icp_observations",
                          down_sampled_pcd,
                          point_size=0.001)

        X_WOhat, chat = IterativeClosestPoint(
            self.box.xyzs(),
            down_sampled_pcd.xyzs(),
            meshcat=meshcat,
            meshcat_scene_path="icp_scene")

        output.set_value(X_WOhat)

#BoxIterativeClosestPoint()


class PickAndPlaceTrajectory(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))
        self.DeclareAbstractInputPort("X_WO",
                                      AbstractValue.Make(RigidTransform()))

        self.DeclareInitializationUnrestrictedUpdateEvent(self.Plan)
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))

        self.DeclareAbstractOutputPort(
            "X_WG", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

    def Plan(self, context, state):
        X_G = {
            "initial":
                self.get_input_port(0).Eval(context)
                [int(self._gripper_body_index)]
        }
        X_O = {
            "initial": self.get_input_port(1).Eval(context),
            "goal": RigidTransform([0, -.6, 0])
        }
        X_GgraspO = RigidTransform(RollPitchYaw(np.pi / 2, np.pi / 2, 0), [0, 0.07, 0])
        X_OGgrasp = X_GgraspO.inverse()
        X_G["pick"] = X_O["initial"] @ X_OGgrasp
        X_G["place"] = X_O["goal"] @ X_OGgrasp
        X_G, times = MakeGripperFrames(X_G) # TODO: this takes a t0 argument, maybe it delays?
        print(f"Planned {times['postplace']} second trajectory.")

        if True:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(
            self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(
            self._traj_wsg_index)).set_value(traj_wsg_command)

    def start_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().start_time()

    def end_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().end_time()

    def CalcGripperPose(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.
        output.set_value(context.get_abstract_state(int(
            self._traj_X_G_index)).get_value().GetPose(context.get_time()))

    def CalcWsgPosition(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.
        output.SetFromVector(
            context.get_abstract_state(int(
                self._traj_wsg_index)).get_value().value(context.get_time()))


x = np.random.rand(2)
r = np.random.randint(-180, 180, 2)
print([.55 + 0.05 * (x[0] - 0.5), 0.1 + 0.05 * (x[1] - 0.5), 0.02515])
print(r)
model_directives = f"""
directives:
- add_directives:
    file: package://manipulation/iiwa_and_wsg.dmd.yaml
- add_directives:
    file: file:///usr/cargobot/cargobot-project/perception/models/warehouse_w_cameras.dmd.yaml
- add_model:
    name: box-vertical
    file: file:///usr/cargobot/cargobot-project/perception/models/box-vertical.urdf
    default_free_body_pose:
        base_link_box-vertical:
            translation: [{.55 + 0.05 * (x[0] - 0.5)}, {0.1 + 0.05 * (x[1] - 0.5)}, 0.02515]
            #rotation: !Rpy {{ deg: [-90, 0, 45] }}
            rotation: !Rpy {{ deg: [0, 0, {r[1]}]}}
"""


def icp_pick_and_place_demo():
    builder = DiagramBuilder()

    station = builder.AddSystem(
        MakeManipulationStation(model_directives, time_step=0.002))
    plant = station.GetSubsystemByName("plant")

    icp = builder.AddSystem(BoxIterativeClosestPoint())
    builder.Connect(station.GetOutputPort("camera3_point_cloud"),
                    icp.get_input_port(0))
    builder.Connect(station.GetOutputPort("camera4_point_cloud"),
                    icp.get_input_port(1))
    builder.Connect(station.GetOutputPort("camera5_point_cloud"),
                    icp.get_input_port(2))
    plan = builder.AddSystem(PickAndPlaceTrajectory(plant))
    builder.Connect(station.GetOutputPort("body_poses"),
                    plan.GetInputPort("body_poses"))
    builder.Connect(icp.GetOutputPort("X_WO"), plan.GetInputPort("X_WO"))

    robot = station.GetSubsystemByName(
        "iiwa_controller").get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(diff_ik.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(plan.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_state_estimated"),
                    diff_ik.GetInputPort("robot_state"))

    builder.Connect(plan.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    #pc_visualizer = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, path="icp_place"))
    #builder.Connect(station.GetOutputPort(f"camera0_point_cloud"), pc_visualizer.cloud_input_port())


    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_context()

    simulator.Initialize()
    if False: # draw the trajectory triads
        X_G_traj = plan.GetMyContextFromRoot(context).get_abstract_state(
            0).get_value()
        for t in np.linspace(X_G_traj.start_time(), X_G_traj.end_time(), 40):
            AddMeshcatTriad(meshcat,
                            f"X_G/({t})",
                            X_PT=X_G_traj.GetPose(t),
                            length=.1,
                            radius=0.004)

    print(plan.end_time(plan.GetMyContextFromRoot(context)))
    visualizer.StartRecording(False)
    simulator.AdvanceTo(plan.end_time(plan.GetMyContextFromRoot(context)))
    visualizer.PublishRecording()

#icp_pick_and_place_demo()

while True:
    time.sleep(1)