import numpy as np
import time
from pydrake.all import (AbstractValue, AngleAxis, Concatenate, DiagramBuilder,
                         LeafSystem, MeshcatVisualizer, MeshcatPointCloudVisualizer, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform, RotationMatrix,
                         RollPitchYaw, Simulator, StartMeshcat, LoadModelDirectivesFromString, ProcessModelDirectives, 
                         ModelInstanceIndex, PassThrough, Demultiplexer, MultibodyPlant, InverseDynamicsController, 
                         FindResourceOrThrow, RevoluteJoint, Adder, StateInterpolatorWithDiscreteDerivative, 
                         SchunkWsgPositionController, MakeMultibodyStateToWsgStateSystem, DifferentialInverseKinematicsParameters, DifferentialInverseKinematicsIntegrator,
                         PortSwitch)

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
from manip.enums import *

class Planner(LeafSystem):
    def __init__(self, plant, box_list):
        LeafSystem.__init__(self)
        self.box_list = box_list
        self.truck_box_list = []
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._gripper_body_index = plant.GetBodyByName("body").index()
        
        
        self._x_bin_grasp_index = self.DeclareAbstractInputPort(
            "grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        
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
        num_positions = 10
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
        self.DeclareAbstractOutputPort("color", lambda: AbstractValue.Make(BoxColorEnum.RED), self.CalcColor, prerequisites_of_calc=set([self.xc_ticket()]))
        self.properties = (LabelEnum.LOW_PRIORTY, LabelEnum.HEAVY) # default
        self.color = BoxColorEnum.RED # default
        self.output_color = BoxColorEnum.RED # default

    def CalcGraspColor(self):
        color_list = np.array([BoxColorEnum.RED,BoxColorEnum.BLUE,BoxColorEnum.GREEN,BoxColorEnum.MAGENTA,BoxColorEnum.CYAN, BoxColorEnum.YELLOW])
        choice = self.output_color
        while choice != self.output_color:
            choice = np.random.choice(color_list, replace=False, size=1)
        self.color = choice


    def CalcColor(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE or mode == PlannerState.PICKING_BOX:
            if self.box_list:
                for box in self.box_list:
                    if LabelEnum.LOW_PRIORTY in box["labels"] and LabelEnum.HEAVY in box["labels"]:
                        output.set_value(box["color"])
                        self.output_color = box["color"]
                        placed_box = self.box_list.pop(self.box_list.index(box))
                        self.truck_box_list.append(placed_box)
                        self.properties = box["labels"]
                        return

                for box in self.box_list:    
                    if LabelEnum.LOW_PRIORTY in box["labels"] and LabelEnum.LIGHT in box["labels"]:
                        output.set_value(box["color"])
                        self.output_color = box["color"]
                        placed_box = self.box_list.pop(self.box_list.index(box))
                        self.truck_box_list.append(placed_box)
                        self.properties = box["labels"]
                        return

                for box in self.box_list:    
                    if LabelEnum.MID_PRIORTY in box["labels"] and LabelEnum.HEAVY in box["labels"]:
                        output.set_value(box["color"])
                        self.output_color = box["color"]
                        placed_box = self.box_list.pop(self.box_list.index(box))
                        self.truck_box_list.append(placed_box)
                        self.properties = box["labels"]
                        return

                for box in self.box_list:    
                    if LabelEnum.MID_PRIORTY in box["labels"] and LabelEnum.LIGHT in box["labels"]:
                        output.set_value(box["color"])
                        self.output_color = box["color"]
                        placed_box = self.box_list.pop(self.box_list.index(box))
                        self.truck_box_list.append(placed_box)
                        self.properties = box["labels"]
                        return

                for box in self.box_list:    
                    if LabelEnum.HIGH_PRIORTY in box["labels"] and LabelEnum.HEAVY in box["labels"]:
                        output.set_value(box["color"])
                        self.output_color = box["color"]
                        placed_box = self.box_list.pop(self.box_list.index(box))
                        self.truck_box_list.append(placed_box)
                        self.properties = box["labels"]
                        return

                for box in self.box_list:    
                    if LabelEnum.HIGH_PRIORTY in box["labels"] and LabelEnum.LIGHT in box["labels"]:
                        output.set_value(box["color"])
                        self.output_color = box["color"]
                        placed_box = self.box_list.pop(self.box_list.index(box))
                        self.truck_box_list.append(placed_box)
                        self.properties = box["labels"]
                        return
    
        elif mode == PlannerState.SHUFFLE_BOXES:
            color = self.CalcGraspColor()
            output.set_value(color)
            return
        return
        

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
                        ).set_value(PlannerState.SHUFFLE_BOXES)
                        self.Plan(context, state)
                        
                    elif mode == PlannerState.SHUFFLE_BOXES:
                        state.get_mutable_abstract_state(
                            int(self._mode_index)
                        ).set_value(PlannerState.GO_HOME)
                        self.GoHome(context, state)
                    # TODO What if the system is in another state?
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                times = {"initial": current_time}
                state.get_mutable_abstract_state(
                    int(self._times_index)
                ).set_value(times)
                X_G = self.get_input_port(0).Eval(context)[ #TODO
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

        X_G = self.get_input_port(0).Eval(context)[ #TODO
            int(self._gripper_body_index)
        ]
       
        if (
            np.linalg.norm(
                traj_X_G.GetPose(current_time).translation()
                - X_G.translation()
            )
            > 10 # TODO use as hyperparameter
        ):
            print("------->Current time: ", current_time)
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
        #q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q = [1,1,0, 0.0, 0.1, 0, -1.2, 0, 1.6, 0] # TODO change according to the run
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[3] = q[3]  # Safer to not reset the first joint.

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
            "initial": self.get_input_port(0).Eval(context)[ #TODO
                int(self._gripper_body_index)
            ]
        }
        
        cost = np.inf
        for i in range(5):
            if mode == PlannerState.SHUFFLE_BOXES:
                cost, X_G["pick"] = self.GetInputPort("grasp").Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_BOX
            
            else:
                print("---------->X_G", self.GetInputPort("grasp").Eval(context))
                cost, X_G["pick"] = self.GetInputPort("grasp").Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.SHUFFLE_BOXES
                else:
                    mode = PlannerState.PICKING_BOX

            if not np.isinf(cost):
                break

        assert not np.isinf(
            cost
        ), "Could not find a valid grasp after 5 attempts"
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        if mode == PlannerState.PICKING_BOX:
            x = 0
            z = 0
            if LabelEnum.LOW_PRIORTY in self.properties:
                x = -1.5
            elif LabelEnum.MID_PRIORTY in self.properties:
                x = -1
            elif LabelEnum.HIGH_PRIORTY in self.properties:
                x = -0.5
            
            if LabelEnum.HEAVY in self.properties:
                z = 0.3
            elif LabelEnum.LIGHT in self.properties:
                z = 0.6

            # Place in truck:
            X_G["place"] = RigidTransform(RollPitchYaw(-np.pi / 2, 0, 0), [x,0,z])
    
        elif mode == PlannerState.SHUFFLE_BOXES:
            dimension = 6
            num_of_boxes = 5
            grid = [f"{x},{y}" for x in range(dimension) for y in range(dimension)]
            box_positions = np.random.choice(grid, replace=False, size=1)
            tf = RigidTransform(
                        RotationMatrix(),
                        [0.15*(int(box_positions.split(",")[0])-dimension/2)+0.7, 0.15*(int(box_positions.split(",")[1])-dimension/2)-0.1, z])
            
            
            # Place in pickup area:
            X_G["place"] = tf
        
        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )

        if False:  
            AddMeshcatTriad(meshcat, "initial", X_PT=["initial"])
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
            self.get_input_port(0).Eval(context)[int(self._gripper_body_index)] # TODO
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
            #"initial": self.get_input_port(1).Eval(context), 
            "goal": RigidTransform([0, -.6, 0]) #TODO use if else to decide which color goes to which part of the truck grid
        }
        X_GgraspO = RigidTransform(RollPitchYaw(np.pi / 2, np.pi / 2, 0), [0, 0.07, 0])
        X_OGgrasp = X_GgraspO.inverse()
        #X_G["pick"] = X_O["initial"] @ X_OGgrasp
        X_G["pick"] = self.GetInputPort("grasp").Eval(context)
        X_G["place"] = X_O["goal"] @ X_OGgrasp
        X_G, times = MakeGripperFrames(X_G) 
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