import numpy as np
import time
from pydrake.all import (AbstractValue, AngleAxis, Concatenate, DiagramBuilder,
                         LeafSystem, MeshcatVisualizer, MeshcatPointCloudVisualizer, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform, RotationMatrix,
                         RollPitchYaw, Simulator, StartMeshcat, LoadModelDirectivesFromString, ProcessModelDirectives, 
                         ModelInstanceIndex, PassThrough, Demultiplexer, MultibodyPlant, InverseDynamicsController, 
                         FindResourceOrThrow, RevoluteJoint, Adder, StateInterpolatorWithDiscreteDerivative, 
                         SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem,
                         DifferentialInverseKinematicsParameters, DifferentialInverseKinematicsIntegrator,
                         KinematicTrajectoryOptimization,BsplineTrajectory,PositionConstraint,MinimumDistanceConstraint, Solve)

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
from manipulation.scenarios import AddRgbdSensors
from IPython.display import HTML, SVG, display
import pydot

class TrajectoryOptimizationSystem(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
     
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

    def set_plant_context(self, plant_context):
        self.plant_context = plant_context

    def Plan(self, context, state):
        
        X_O = {
            "initial": self.get_input_port(1).Eval(context),
            "goal": RigidTransform([0, -.6, 0])
        }

        X_G = {
            "initial":
                self.get_input_port(0).Eval(context)
                [int(self._gripper_body_index)],
            "goal":
                RigidTransform([0, -.6, 0])
        }

        """X_GgraspO = RigidTransform(RollPitchYaw(np.pi / 2, np.pi / 2, 0), [0, 0.07, 0])
        X_OGgrasp = X_GgraspO.inverse()
        X_G["pick"] = X_O["initial"] @ X_OGgrasp
        X_G["place"] = X_O["goal"] @ X_OGgrasp"""
        wsg = self.plant.GetModelInstanceByName("wsg")
        
        X_G, trajs, times = self.MakeGripperFramesOptimized(self.plant, self.plant_context, X_G, X_O, wsg) 
        print(f"Planned {times['postplace']} second trajectory.")

        traj_X_G = self.MakeGripperPoseTrajectoryOptimized(trajs, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(
            self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(
            self._traj_wsg_index)).set_value(traj_wsg_command)

    def MakeGripperPoseTrajectoryOptimized(self, trajs, times,time_step=1.0 / 33.0):
        sample_times=[]
        poses=[]
        dt = 0
        for traj in trajs: 
            for t in np.append(
                    np.arange(traj.start_time(), traj.end_time(),
                            time_step), traj.end_time()):
                sample_times.append(t+dt)
                poses.append( traj.value(t))
               
            dt += t
        return sample_times, poses

    def MakeGripperFramesOptimized(self, plant, plant_context, X_G, X_O, wsg):
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
        
        trajopt, result = self.Optimize(plant, plant_context, X_G["initial"], X_G["prepick"], wsg)
        times["prepick"] = times["initial"] + result.get_x_val()[-1]
        trajs.append(trajopt.ReconstructTrajectory(result))

        trajopt, result = self.Optimize(plant, plant_context, X_G["prepick"], X_G["pick"],wsg)
        times["pick_start"] = times["prepick"] + result.get_x_val()[-1]
        trajs.append(trajopt.ReconstructTrajectory(result))
        #trajopt, result =Optimize(plant, plant_context, X_G["pick"], X_G["pick"],wsg)
        #trajs.append(trajopt.ReconstructTrajectory(result))
        times["pick_end"] = times["pick_start"] + result.get_x_val()[-1]
        
        trajopt, result = self.Optimize(plant, plant_context, X_G["pick"], X_G["prepick"],wsg)
        times["postpick"] = times ["pick_end"] + result.get_x_val()[-1]
        trajs.append(trajopt.ReconstructTrajectory(result))
        
        trajopt, result = self.Optimize(plant, plant_context, X_G["prepick"], X_G["preplace"],wsg)
        times["preplace"] = times ["postpick"] + result.get_x_val()[-1]
        trajs.append(trajopt.ReconstructTrajectory(result))
        
        trajopt, result = self.Optimize(plant, plant_context, X_G["preplace"], X_G["place"],wsg)
        times["place_start"] = times["preplace"] + result.get_x_val()[-1]
        trajs.append(trajopt.ReconstructTrajectory(result))
        #trajopt, result =Optimize(plant, plant_context, X_G["place"], X_G["place"],wsg)
        #trajs.append(trajopt.ReconstructTrajectory(result))
        times["place_end"] = times["place_start"] + result.get_x_val()[-1]
        
        trajopt, result = self.Optimize(plant, plant_context, X_G["place"], X_G["preplace"],wsg)
        times["postplace"] = times["place_end"] + result.get_x_val()[-1]
        trajs.append(trajopt.ReconstructTrajectory(result))
        print(times)
        return X_G, trajs, times

    def Optimize(self, plant, plant_context, X_WStart, X_WGoal, wsg):
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
        lower_limits = plant.GetVelocityLowerLimits()[:-9]
        upper_limits = plant.GetVelocityUpperLimits()[:-9]
        ths = -1000*np.ones(10)
        lower_limits = np.append(lower_limits, ths)
        upper_limits = np.append(upper_limits, -1*ths)
        
        trajopt.AddVelocityBounds(lower_limits, upper_limits)
        
        trajopt.AddDurationConstraint(2.5, 5)
        model = plant.GetModelInstanceByName("box-vertical")
        model = plant.GetModelInstanceByName("iiwa")
        
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
        print(result)
        return trajopt, result

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
    
