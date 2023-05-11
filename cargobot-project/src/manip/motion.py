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
from manipulation.scenarios import AddRgbdSensors
from IPython.display import HTML, SVG, display
import pydot
from enum import Enum
from copy import copy

from pydrake.all import InputPortIndex

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    PICKING_BOX = 2
    EMPTY_TRUCK = 3
    GO_HOME = 4


class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        """
        self._x_bin_grasp_index = self.DeclareAbstractInputPort(
            "x_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        self._y_bin_grasp_index = self.DeclareAbstractInputPort(
            "y_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        """
        self._pickup_grasp_index = self.DeclareAbstractInputPort(
            "pickup_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        self._truck_grasp_index = self.DeclareAbstractInputPort(
            "truck_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort(
            "wsg_state", 2
        ).get_index()

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        )
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._attempts_index = self.DeclareDiscreteState(1)

        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        # For GoHome mode.
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions
        ).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", num_positions, self.CalcIiwaPosition
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.rng = np.random.default_rng(135)

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times["initial"] > 1.0:
                self.Plan(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()
            if not traj_q.is_time_in_range(current_time):
                self.Plan(context, state)
            return

        # If we are between pick and place and the gripper is closed, then
        # we've missed or dropped the object.  Time to replan.
        if (
            current_time > times["postpick"]
            and current_time < times["preplace"]
        ):
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(
                context
            )
            if wsg_state[0] < 0.01: # closed gripper
                attempts = state.get_mutable_discrete_state(
                    int(self._attempts_index)
                ).get_mutable_value()
                if attempts[0] > 5:
                    # If I've failed 5 times in a row, then switch bins.
                    print(
                        "Switching to the other bin after 5 consecutive failed attempts"
                    )
                    attempts[0] = 0
                    if mode == PlannerState.PICKING_BOX:
                        state.get_mutable_abstract_state(
                            int(self._mode_index)
                        ).set_value(PlannerState.EMPTY_TRUCK)
                    else:
                        state.get_mutable_abstract_state(
                            int(self._mode_index)
                        ).set_value(PlannerState.PICKING_BOX)
                    self.Plan(context, state)
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                times = {"initial": current_time}
                state.get_mutable_abstract_state(
                    int(self._times_index)
                ).set_value(times)
                X_G = self.get_input_port(0).Eval(context)[
                    int(self._gripper_body_index)
                ]
                state.get_mutable_abstract_state(
                    int(self._traj_X_G_index)
                ).set_value(
                    PiecewisePose.MakeLinear(
                        [current_time, np.inf], [X_G, X_G]
                    )
                )
                return

        traj_X_G = context.get_abstract_state(
            int(self._traj_X_G_index)
        ).get_value()
        if not traj_X_G.is_time_in_range(current_time):
            self.Plan(context, state)
            return

        X_G = self.get_input_port(0).Eval(context)[
            int(self._gripper_body_index)
        ]
        # if current_time > 10 and current_time < 12:
        #    self.GoHome(context, state)
        #    return
        if (
            np.linalg.norm(
                traj_X_G.GetPose(current_time).translation()
                - X_G.translation()
            )
            > 0.2
        ):
            # If my trajectory tracking has gone this wrong, then I'd better
            # stop and replan.  TODO(russt): Go home, in joint coordinates,
            # instead.
            self.GoHome(context, state)
            return

    def GoHome(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.GO_HOME
        )
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T
        )
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(
            q_traj
        )

    def Plan(self, context, state):
        mode = copy(
            state.get_mutable_abstract_state(int(self._mode_index)).get_value()
        )

        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }

        cost = np.inf
        for i in range(5):
            if mode == PlannerState.EMPTY_TRUCK:
                cost, X_G["pick"] = self.get_input_port(
                    self._truck_grasp_index
                ).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_BOX
            else:
                cost, X_G["pick"] = self.get_input_port(
                    self._pickup_grasp_index
                ).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.EMPTY_TRUCK
                else:
                    mode = PlannerState.PICKING_BOX

            if not np.isinf(cost):
                break

        assert not np.isinf(
            cost
        ), "Could not find a valid grasp in either bin after 5 attempts"
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        if mode == PlannerState.PICKING_BOX:
            # Place in truck:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [self.rng.uniform(-0.25, 0.15), self.rng.uniform(-0.6, -0.4), 0.3],
            )
        else:
            # Place in pickup area:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                [self.rng.uniform(0.35, 0.65), self.rng.uniform(-0.12, 0.28), 0.3],
            )

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )

        if False:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            traj_X_G
        )
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def start_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .start_time()
        )

    def end_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .end_time()
        )

    def CalcGripperPose(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(
            int(self._traj_X_G_index)
        ).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(self._traj_X_G_index))
                .get_value()
                .GetPose(context.get_time())
            )
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(
            self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        )

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(
            int(self._traj_wsg_index)
        ).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context),
        )

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_mutable_abstract_state(
            int(self._traj_q_index)
        ).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))


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

        if False:  # Useful for debugging
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