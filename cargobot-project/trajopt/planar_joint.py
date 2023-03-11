import numpy as np
import os
import matplotlib.pyplot as plt

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
                         SceneGraph, Simulator, Solve,  StartMeshcat, TrajectorySource, InverseDynamicsController, SpatialInertia, UnitInertia)
from manipulation.utils import colorize_labels

from pydrake.all import (AddMultibodyPlantSceneGraph, Box,
                         ConnectPlanarSceneGraphVisualizer, CoulombFriction,
                         DiagramBuilder, DrakeVisualizer, FindResourceOrThrow,
                         FixedOffsetFrame, JointIndex, Parser, PlanarJoint,
                         RandomGenerator, RigidTransform, RollPitchYaw,
                         RotationMatrix, Simulator,
                         UniformlyRandomRotationMatrix, PlanarJoint, Context,JointActuator, PrismaticJoint, RevoluteJoint, WeldJoint)

from manipulation import running_as_notebook
from manipulation.scenarios import (AddRgbdSensor, AddShape, ycb)

import json
from IPython.display import HTML, SVG, display
import pydot

meshcat = StartMeshcat()

meshcat.Flush()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.05)

parser = Parser(plant)

scene_models = parser.AddModels("cargobot-models/scene_without_robot.dmd.yaml")
#iiwa = plant.GetModelInstanceByName("box")

"""false_body = plant.AddRigidBody(
        "false_body", iiwa,
        SpatialInertia(1, [0, 0, 0], UnitInertia(0.0041666666666666675, 0.0041666666666666675, 0.001666666666666667)))
false_body2 = plant.AddRigidBody(
        "false_body2", iiwa,
        SpatialInertia(1, [0, 0, 0], UnitInertia(0.0041666666666666675, 0.0041666666666666675, 0.001666666666666667)))

mobile_base_y = plant.AddJoint(PrismaticJoint(
        "mobile_base_y", plant.GetFrameByName("table_top_link"), plant.GetFrameByName("false_body"), 
        [0, 1, 0], -.3, .3))
plant.AddJointActuator("mobile_base_y_actuator", mobile_base_y)

mobile_base_x = plant.AddJoint(PrismaticJoint(
        "mobile_base_x", plant.GetFrameByName("false_body"), plant.GetFrameByName("false_body2"), 
        [1, 0, 0], -.3, .3))
plant.AddJointActuator("mobile_base_x_actuator", mobile_base_x)

mobile_base_theta = plant.AddJoint(RevoluteJoint(
        "mobile_base_theta", plant.GetFrameByName("false_body2"), plant.GetFrameByName("box"), 
        [0, 0, 1],  -np.pi, np.pi))
plant.AddJointActuator("mobile_base_theta_actuator", mobile_base_theta)

#weld_joint = plant.AddJoint(WeldJoint("weld_joint", plant.GetFrameByName("box"), plant.GetFrameByName("iiwa_link_0"), RigidTransform()))
plant.WeldFrames(plant.GetFrameByName("box"), plant.GetFrameByName("iiwa_link_0"))

mobile_base_theta.set_acceleration_limits([0], [1])
mobile_base_y.set_acceleration_limits([0], [1])
mobile_base_x.set_acceleration_limits([0], [1])
"""

plant.Finalize()

visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat)

plant_context = plant.CreateDefaultContext()

kp=[1] * plant.num_positions()
ki=[1] * plant.num_positions()
kd=[1] * plant.num_positions()

planar_controller = builder.AddSystem(
    InverseDynamicsController(plant, kp, ki, kd, False))
planar_controller.set_name("planar_controller")
builder.Connect(plant.get_state_output_port(),
                planar_controller.get_input_port_estimated_state())
builder.Connect(planar_controller.get_output_port_control(),
                plant.get_actuation_input_port())

diagram = builder.Build()
context = diagram.CreateDefaultContext()

states = [1, 1, 0, -1.57, 0.1, 0, -1.2, 0, 1.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
planar_controller.GetInputPort('desired_state').FixValue(
    planar_controller.GetMyMutableContextFromRoot(context), states)

simulator = Simulator(diagram, context)
"""simulator.get_mutable_integrator().set_fixed_step_mode(True)"""
visualizer.StartRecording()
"""SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0]).create_svg()"""

print(context)
simulator.AdvanceTo(12.0)
visualizer.StopRecording()
visualizer.PublishRecording()

while True:
    continue