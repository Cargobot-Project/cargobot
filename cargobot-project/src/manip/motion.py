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

import time
from collections import namedtuple
from functools import partial

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    Cylinder,
    Rgba,
    Sphere,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.solvers import BoundingBoxConstraint
from pydrake.systems.framework import (
    EventStatus,
    LeafSystem,
)

class Planner(LeafSystem):
    def __init__(self, plant, box_list, box_list2, meshcat):
        LeafSystem.__init__(self)
        self.box_list = box_list
        self.box_list2 = box_list2
        self.truck_box_list = []
        self.truck2_box_list = []
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._gripper_body_index = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).index()
        
        
        self._x_bin_grasp_index = self.DeclareAbstractInputPort(
            "grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()

        self._y_bin_grasp_index = self.DeclareAbstractInputPort(
            "grasp_shuffle", AbstractValue.Make((np.inf, RigidTransform()))
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
        self.DeclareAbstractOutputPort(
            "second_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcSecondMode
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
        self.DeclareAbstractOutputPort("color_shuffle", lambda: AbstractValue.Make(BoxColorEnum.RED), self.CalcShuffleColor, prerequisites_of_calc=set([self.xc_ticket()]))
        self.properties = (LabelEnum.LOW_PRIORTY, LabelEnum.HEAVY) # default
        self.color = BoxColorEnum.RED # default
        self.color_shuffle = BoxColorEnum.RED # default
        self.output_color = BoxColorEnum.RED # default
        self.current_box = self.box_list[0]
        self.second_mode = 0
        """self.max_widths = [[0]]
        self.max_depths = [[0]]
        self.max_heights = [[0]]"""
        self.max_dims = []
        self.current_pillar = [0,0]
        self.start_point = [-2.5,-0.75,0]
        self.limits = [0, 1.5, 0.7]
        self.plant = plant
        self.meshcat = meshcat
        #self.ghost = self.plant.GetBodyByName("ghost")
        


    def set_start_point(start_point):
        self.start_point = start_point
    
    def CalcShuffleColor(self, context, output):
        color_list = np.array([box["color"] for box in self.box_list])
        choice = self.output_color
       
        if color_list.size == 1:
            print("NO OTHER BOX")
            output.set_value(choice)
            return

        while choice == self.output_color or choice in self.truck_box_list:
            choice = np.random.choice(color_list, replace=False, size=1)
        print(choice)
        self.color_shuffle = choice
        print("CALCSHUFFLECOLOR")
        output.set_value(choice[0])

    def CalcColor(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if self.box_list:
            for box in self.box_list:
                if LabelEnum.LOW_PRIORTY in box["labels"] and LabelEnum.HEAVY in box["labels"]:
                    output.set_value(box["color"])
                    print("----CalcColor: ", box["color"])
                    self.output_color = box["color"]
                    self.properties = box["labels"]
                    self.current_box = box
                    return

            for box in self.box_list:    
                if LabelEnum.LOW_PRIORTY in box["labels"] and LabelEnum.LIGHT in box["labels"]:
                    output.set_value(box["color"])
                    print("----CalcColor: ", box["color"])
                    self.output_color = box["color"]
                    self.properties = box["labels"]
                    self.current_box = box
                    return

            for box in self.box_list:    
                if LabelEnum.MID_PRIORTY in box["labels"] and LabelEnum.HEAVY in box["labels"]:
                    output.set_value(box["color"])
                    print("----CalcColor: ", box["color"])
                    self.output_color = box["color"]
                    self.properties = box["labels"]
                    self.current_box = box
                    return

            for box in self.box_list:    
                if LabelEnum.MID_PRIORTY in box["labels"] and LabelEnum.LIGHT in box["labels"]:
                    output.set_value(box["color"])
                    print("----CalcColor: ", box["color"])
                    self.output_color = box["color"]
                    self.properties = box["labels"]
                    self.current_box = box
                    return

            for box in self.box_list:    
                if LabelEnum.HIGH_PRIORTY in box["labels"] and LabelEnum.HEAVY in box["labels"]:
                    output.set_value(box["color"])
                    print("----CalcColor: ", box["color"])
                    self.output_color = box["color"]
                    self.properties = box["labels"]
                    self.current_box = box
                    return

            for box in self.box_list:    
                if LabelEnum.HIGH_PRIORTY in box["labels"] and LabelEnum.LIGHT in box["labels"]:
                    output.set_value(box["color"])
                    print("----CalcColor: ", box["color"])
                    self.output_color = box["color"]
                    self.properties = box["labels"]
                    self.current_box = box
                    return
        return
        

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            print("UPDATE: FIRST WFOTS STATE")
            if context.get_time() - times["initial"] > 1.0:
                if self.second_mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
                    self.GoHome(context, state, None)
                else:
                    self.GoHome(context, state, self.second_mode)
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
            and current_time + 4 < times["preplace"]
        ):
            self.second_mode = 1
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(
                context
            )
            if wsg_state[0] < 0.01: # closed gripper
                print("UPDATE: DROPPED THE BOX - OR COULD NOT GRASP")
                
                if self.current_box not in self.box_list:
                    self.box_list.append(self.current_box)
                attempts = state.get_mutable_discrete_state(
                    int(self._attempts_index)
                ).get_mutable_value()
                
                print("-----Attempt: ", attempts[0])
                if attempts[0] > 2:
                    # If I've failed 2 times in a row, then switch bins.
                    print(
                        "Switching to the other bin after 2 consecutive failed attempts"
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
                        ).set_value(PlannerState.PICKING_BOX)
                        self.Plan(context, state)
                        #self.Plan(context, state)
                    # TODO What if the system is in another state?
                    
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                self.second_mode = mode
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
            self.GoHome(context, state, None)
            #self.Plan(context, state)
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
            
            print("UPDATE: TOO FAR AWAY")
            # If my trajectory tracking has gone this wrong, then I'd better
            # stop and replan.  TODO(russt): Go home, in joint coordinates,
            # instead.
            self.GoHome(context, state, None)
            return

        """print("Mode: ", mode)
        print("Current box: ", self.current_box)
        print("Current Time: ", current_time)
        print("End Time: ", times["preplace"])"""
        if mode == PlannerState.PICKING_BOX and np.linalg.norm(traj_X_G.GetPose(times["preplace"]).translation()- X_G.translation()) < 0.2 and self.current_box in self.box_list:
            print("INSIDE BOX POP")
            tmp_box = self.box_list.pop(self.box_list.index(self.current_box))
            tmp_box["pillar"] = self.current_pillar 
            self.truck_box_list.append(tmp_box)
            self.second_mode +=1
            self.second_mode = self.second_mode%2



    
    def GoHome(self, context, state, from_dropped):
        print("GO HOME.")
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.GO_HOME
        )
        """if from_dropped:
            state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                from_dropped
            )"""
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        #q = [0,0,0, 0.0, 0.1, 0, -1.2, 0, 1.6, 0] # TODO change according to the run
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[3] = q[3]  # Safer to not reset the first joint.
        
        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 10.0], np.vstack((q, q0)).T
        )
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(
            q_traj
        )
        print("Done go home")


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
        print("FOR PLANNING")
        for i in range(5):
            print("FOR IN PLANNING")
            if mode == PlannerState.SHUFFLE_BOXES:
                print("PLAN: SHUFFLEBOX")
                
                cost, X_G["pick"] = self.GetInputPort("grasp_shuffle").Eval(context)
                
                if np.isinf(cost):
                    mode = PlannerState.PICKING_BOX
            
            else:
                print("PLAN: PICKINGBOX")
                cost, X_G["pick"] = self.GetInputPort("grasp").Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.SHUFFLE_BOXES
                else:
                    mode = PlannerState.PICKING_BOX

            if not np.isinf(cost):
                break

        """if np.isinf(cost):
            if mode == PlannerState.SHUFFLE_BOXES:
                mode = PlannerState.PICKING_BOX
            elif mode == PlannerState.PICKING_BOX:
                mode = PlannerState.SHUFFLE_BOXES"""
        
        
        print("SET PLANNING")
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        if mode == PlannerState.PICKING_BOX:
            continue_pillar = False
            pr = LabelEnum.MID_PRIORTY
            
            if LabelEnum.HIGH_PRIORTY in self.properties:
                pr = LabelEnum.HIGH_PRIORTY
            elif LabelEnum.LOW_PRIORTY:
                pr = LabelEnum.LOW_PRIORTY

            for placed_box in self.truck_box_list:
                if pr in placed_box["labels"]:
                    continue_pillar = True
            
            x = 0
            y = 0
            z = 0
            if not continue_pillar:
                current_x = self.current_pillar[0]
                self.current_pillar = [current_x+1, 1]
                max_dim = max( float(self.current_box["dimensions"][0]), float(self.current_box["dimensions"][1]), float(self.current_box["dimensions"][2]))
                self.max_dims.append([max_dim])
                
                x = self.start_point[0]+0.03
                
                for i in range(current_x):
                    x += max(self.max_dims[i])
                    x += 0.03
            
                y = self.start_point[1]+ 0.03
                z = float(self.current_box["dimensions"][2])/2+0.03

            else: # we are not adding a new column (y-oriented vector)
                total_z = 0
              
                for placed_box in self.truck_box_list:
                    if placed_box["pillar"] == self.current_pillar:
                        total_z += float(placed_box["dimensions"][2])
                
                x = self.start_point[0]+0.03
                if self.current_pillar[0] >1:
                    for i in range(1, self.current_pillar[0]):
                        x += max(self.max_dims[i])
                        x += 0.03

                max_dim = max(float(self.current_box["dimensions"][0]),float(self.current_box["dimensions"][1]),float(self.current_box["dimensions"][2]))
                if total_z + float(self.current_box["dimensions"][2]) + 0.02 < self.limits[2]:
                    if max_dim > self.max_dims[self.current_pillar[0]-1][self.current_pillar[1]-1]:
                        self.max_dims[self.current_pillar[0], self.current_pillar[1]] = max_dim
                    z = self.start_point[2]+ total_z + float(self.current_box["dimensions"][2]) 

                else:
                    self.current_pillar = [self.current_pillar[0], self.current_pillar[1]+1]
                    self.max_dims[self.current_pillar[0]-1].append(max_dim)
                    z = self.start_point[2] + float(self.current_box["dimensions"][2])/2 + 0.02
                    
                y += self.start_point[1]+0.03
                if self.current_pillar[1] > 1:
                    for i in range(1, self.current_pillar[1]):
                            y += self.max_dims[self.current_pillar[0]-1][i]
                            y += 0.03

            print("XYZ: ", x, y, z)
            print("Pillar: ", self.current_pillar)
            print("Max Dims: ", self.max_dims)
            # Place in truck:
            X_G["place"] = RigidTransform(RollPitchYaw(-np.pi / 2, 0, 0), [x,y,z])
    
        elif mode == PlannerState.SHUFFLE_BOXES:
            dimension = 6
            num_of_boxes = 5
            grid = [f"{x},{y}" for x in range(dimension) for y in range(dimension)]
            box_positions = np.random.choice(grid, replace=False, size=1)
            z=0.1
            random_z = np.random.uniform(0, 2*np.pi)
            tf = RigidTransform(
                        RotationMatrix(RollPitchYaw(0,0,random_z)),
                        [1.1/dimension*(int(box_positions[0].split(",")[0]))+0.6, 1.1/dimension*(int(box_positions[0].split(",")[0])-dimension/2), z])
            # Place in pickup area:
            X_G["place"] = tf

        
        #TODO HEREEEE
        #self.plant.SetDefaultFreeBodyPose( self.ghost, X_G["place"])
        AddMeshcatTarget(self.meshcat, f"target", X_PT=X_G["place"])

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print("GRIPPER PLANNING")
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )
        print("TRAJ PLANNING")
        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            traj_X_G
        )
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )
        print("DONE PLANNING")

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

    def CalcSecondMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        
        if self.second_mode == 1:
            output.set_value(InputPortIndex(2))  # second iiwa
        else:
            output.set_value(InputPortIndex(1))  # first iiwa

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        if mode == PlannerState.GO_HOME:
            
            output.set_value(True)
        else:
            output.set_value(False)


    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            [0,0,0, -1.57, 0.1, 0, -1.2, 0, 1.6, 0],
        )

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_mutable_abstract_state(
            int(self._traj_q_index)
        ).get_value()
        output.SetFromVector(traj_q.value(context.get_time()))


def AddMeshcatTarget(
    meshcat, path, length=0.2, radius=0.06, opacity=1.0, X_PT=RigidTransform()
):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi/2), [length / 2.0, 0, 0]
    )
    meshcat.SetTransform(path + "/target", X_TG)
    meshcat.SetObject(
        path + "/target", Cylinder(radius, length), Rgba(0, 1, 1, opacity)
    )

    