import numpy as np
import os

from pydrake.common import FindResourceOrThrow, temp_directory
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from manipulation.utils import AddPackagePaths, FindResource
from manipulation.meshcat_utils import PublishPositionTrajectory
from IPython.display import clear_output
from manipulation.meshcat_utils import MeshcatPoseSliders
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.visualization import ModelVisualizer
from manipulation import running_as_notebook
from manipulation.scenarios import MakeManipulationStation, AddIiwa, AddWsg, AddPlanarIiwa
from pydrake.all import ( AddMultibodyPlantSceneGraph, AngleAxis, BsplineTrajectory, 
                         DiagramBuilder, FindResourceOrThrow, Integrator,InverseKinematics,
                         JacobianWrtVariable, KinematicTrajectoryOptimization, LeafSystem, MeshcatVisualizer,
                         MinimumDistanceConstraint, MultibodyPlant, MultibodyPositionToGeometryPose,
                         Parser, PiecewisePolynomial, PiecewisePose, PositionConstraint, 
                         Quaternion, Rgba, RigidTransform, RotationMatrix,
                         SceneGraph, Simulator, Solve,  StartMeshcat, TrajectorySource)
from IPython.display import HTML, SVG, display
import pydot

meshcat = StartMeshcat()

def register_plant_with_scene_graph(builder, scene_graph, plant):
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port(),
    )

def Optimize( plant, plant_context, X_WStart, X_WGoal, wsg):
    num_q = plant.num_positions() 
    q0 = plant.GetPositions(plant_context)
    gripper_frame = plant.GetFrameByName("body", wsg)
    trajopt = KinematicTrajectoryOptimization(num_q, 10)
    prog = trajopt.get_mutable_prog()


    q_guess = np.tile(q0.reshape((num_q,1)), (1, trajopt.num_control_points()))
    q_guess[0,:] = np.linspace(0, -np.pi/2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)

    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(),
                            plant.GetPositionUpperLimits())
    
    # bypass the querternions - it is not neat
    lower_limits = plant.GetVelocityLowerLimits()[:-6]
    upper_limits = plant.GetVelocityUpperLimits()[:-6]
    ths = -1000*np.ones(7)
    lower_limits = np.append(lower_limits, ths)
    upper_limits = np.append(upper_limits, -1*ths)
    
    trajopt.AddVelocityBounds(lower_limits, upper_limits)
    
    trajopt.AddDurationConstraint(2.5, 5)
    model = plant.GetModelInstanceByName("horizontal_box")
    model = plant.GetModelInstanceByName("iiwa7")
    
    # start constraint
    start_constraint = PositionConstraint(plant, plant.world_frame(),
                                        X_WStart.translation(),
                                        X_WStart.translation(), gripper_frame,
                                        [0, 0.1, 0], plant_context)
    

    trajopt.AddPathPositionConstraint(start_constraint, 0)
    
    prog.AddQuadraticErrorCost(np.eye(num_q), q0, trajopt.control_points()[:, 0])
    

    # goal constraint
    goal_constraint = PositionConstraint(plant, plant.world_frame(),
                                        X_WGoal.translation(),
                                        X_WGoal.translation(), gripper_frame,
                                        [0, 0.1, 0], plant_context)
    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    
    prog.AddQuadraticErrorCost(np.eye(num_q), q0, trajopt.control_points()[:, -1])

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)
    
    # collision constraints
    collision_constraint = MinimumDistanceConstraint(plant, 0.001,
                                                    plant_context, None, 0.01)

    evaluate_at_s = np.linspace(0, 1, 50)
    for s in evaluate_at_s:
        trajopt.AddPathPositionConstraint(collision_constraint, s)
                            
    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed")
        print(result.get_solver_id().name())

    return trajopt, result

def MakeTrajectoryOptimized(plant, plant_context, X_G, X_O, wsg):
    traj = []

    p_GgraspO = [0, 0.13, 0]
    R_GgraspO = RotationMatrix.MakeXRotation(
        np.pi / 2.0) @ RotationMatrix.MakeZRotation(np.pi / 2.0)
    #R_GgraspO = RotationMatrix()
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)

    X_OGgrasp = X_GgraspO.inverse()
    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.13, 0])
    X_G["pick"] = X_O["initial"] @ X_OGgrasp
    X_G["prepick"] = X_G["pick"] @ X_GgraspGpregrasp
    X_G["place"] = X_O["goal"] @ X_OGgrasp
    X_G["preplace"] = X_G["place"] @ X_GgraspGpregrasp

    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    X_G["postpick"] = X_G["prepick"]
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    X_G["postplace"] = X_G["preplace"]

    trajs = []
    times = {"initial":0}
    
    trajopt, result = Optimize(plant, plant_context, X_G["initial"], X_G["prepick"], wsg)
    times ["prepick"] = times["initial"] + result.get_x_val()[-1]
    trajs.append(trajopt.ReconstructTrajectory(result))

    trajopt, result = Optimize(plant, plant_context, X_G["prepick"], X_G["pick"],wsg)
    times["pick_start"] = times["prepick"] + result.get_x_val()[-1]
    trajs.append(trajopt.ReconstructTrajectory(result))
    #trajopt, result =Optimize(plant, plant_context, X_G["pick"], X_G["pick"],wsg)
    #trajs.append(trajopt.ReconstructTrajectory(result))
    times["pick_end"] = times["pick_start"] + result.get_x_val()[-1]
    
    trajopt, result = Optimize(plant, plant_context, X_G["pick"], X_G["prepick"],wsg)
    times ["postpick"] = times ["pick_end"] + result.get_x_val()[-1]
    trajs.append(trajopt.ReconstructTrajectory(result))
    
    trajopt, result = Optimize(plant, plant_context, X_G["prepick"], X_G["preplace"],wsg)
    times ["preplace"] = times ["postpick"] + result.get_x_val()
    trajs.append(trajopt.ReconstructTrajectory(result))
    
    trajopt, result =Optimize(plant, plant_context, X_G["preplace"], X_G["place"],wsg)
    times["place_start"] = times["preplace"] + result.get_x_val()[-1]
    trajs.append(trajopt.ReconstructTrajectory(result))
    #trajopt, result =Optimize(plant, plant_context, X_G["place"], X_G["place"],wsg)
    #trajs.append(trajopt.ReconstructTrajectory(result))
    times["place_end"] = times["place_start"] + result.get_x_val()[-1]
    
    trajopt, result = Optimize(plant, plant_context, X_G["place"], X_G["preplace"],wsg)
    times["postplace"] = times["place_end"] + result.get_x_val()[-1]
    trajs.append(trajopt.ReconstructTrajectory(result))
    return trajs, times

def PublishPositionTrajectory(trajectory,
                            root_context,
                            plant,
                            visualizer,
                            time_step=1.0 / 33.0):
    """
    Args:
        trajectory: A Trajectory instance list.
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)
    dt = 0
    for traj in trajectory:
        for t in np.append(
                np.arange(traj.start_time(), traj.end_time(),
                        time_step), traj.end_time()):
            root_context.SetTime(t+dt)
            plant.SetPositions(plant_context, traj.value(t))
            visualizer.ForcedPublish(visualizer_context)
        dt += t
    
    visualizer.StopRecording()
    visualizer.PublishRecording()

def cargobot_inverse_kinematics(sim_time_step=0.001):
    # Clean up the Meshcat instance.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    
    #all_plant = builder.AddSystem(MultibodyPlant(time_step=sim_time_step))
    #register_plant_with_scene_graph(builder ,scene_graph, all_plant)
    
    parser = Parser(plant)
    #all_parser = Parser(all_plant)

    AddPackagePaths(parser)
    base_models = parser.AddAllModelsFromFile("/usr/cargobot/cargobot-project/trajopt/cargobot-models/all.dmd.yaml")
    iiwa = AddIiwa(plant,collision_model="with_box_collision")
    wsg = AddWsg(plant, iiwa, roll=0.0, welded=True, sphere=False)
    
    #all_models = all_parser.AddAllModelsFromFile("/work/cargobot-models/object_models.dmd.yaml")
    #iiwa_all = AddIiwa(all_plant,collision_model="with_box_collision")
    #wsg_all = AddWsg(all_plant, iiwa_all, roll=0.0, welded=True, sphere=True)
    
    initial = [0.7, 0.0, 0.3]
    final = [-0.5, -0.2, 0.0]
    X_O = {"initial": RigidTransform(RotationMatrix(), [0.6, 0.1, .1]),
            "goal": RigidTransform(RotationMatrix(), final)}
    box = plant.GetBodyByName("box-horizontal")
    plant.SetDefaultFreeBodyPose(box, X_O["initial"])
    

    X_WStart = RigidTransform(RotationMatrix.MakeZRotation(np.pi),initial)
    X_WGoal = X_O["goal"]
    X_G = {"initial": X_WStart, "goal": X_WGoal}

    # Finalize the plant after loading the scene.
    plant.Finalize()
    #all_plant.Finalize()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
    collision_visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(prefix="collision", role=Role.kProximity))
    meshcat.SetProperty("collision", "visible", False)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    
    trajs, times = MakeTrajectoryOptimized(plant, plant_context, X_G, X_O, wsg)
    
    graph = pydot.graph_from_dot_data(diagram.GetGraphvizString())[0]
    graph.write_jpg("system-diagrams/trajopt_output.jpg")

    PublishPositionTrajectory(trajs, context,
                            plant, visualizer)
    collision_visualizer.ForcedPublish(
        collision_visualizer.GetMyContextFromRoot(context))
    
cargobot_inverse_kinematics()