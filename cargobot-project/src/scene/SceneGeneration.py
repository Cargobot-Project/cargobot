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
                         SceneGraph, Simulator, Solve,  StartMeshcat, TrajectorySource)

                        
from pydrake.all import (AddMultibodyPlantSceneGraph, Box,
                         ConnectPlanarSceneGraphVisualizer, CoulombFriction,
                         DiagramBuilder, DrakeVisualizer, FindResourceOrThrow,
                         FixedOffsetFrame, JointIndex, Parser, PlanarJoint,
                         RandomGenerator, RigidTransform, RollPitchYaw,
                         RotationMatrix, Simulator,
                         UniformlyRandomRotationMatrix)

from manipulation import running_as_notebook
from manipulation.scenarios import (AddRgbdSensor, AddShape, ycb)
import json
from old.BoxObjectString import *




def generate_scene(number_of_times):
    meshcat = StartMeshcat()

    rs = np.random.RandomState()  # this is for python
    generator = RandomGenerator(rs.randint(1000))  # this is for c++

    path = "dataset2/"

    for epoch in range(number_of_times):
        if epoch % 100 == 0:
            print(epoch)

        filename_base = os.path.join(path,  f"{epoch+168:05d}")

        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
        inspector = scene_graph.model_inspector()
        parser = Parser(plant)

        parser.AddModels("/usr/cargobot/cargobot-project/perception/models/scene_without_robot.dmd.yaml")
        
        box_count= 5 + rs.randint(7)
        instance_id_to_class_name = dict()
        for i in range(box_count):
            color = rs.randint(5)
            box = BoxObjectString(color, .1, .1, .2, 1, "")
            sdf = box.generate_sdf_string(f"box{i}")
            instance = parser.AddModelsFromString(sdf, "sdf")
            
            frame_id = plant.GetBodyFrameIdOrThrow(
                plant.GetBodyIndices(instance[0])[0])
            
            geometry_ids = inspector.GetGeometries(frame_id, Role.kPerception)
           
            for geom_id in geometry_ids:
                instance_id_to_class_name[int(
                    inspector.GetPerceptionProperties(geom_id).GetProperty(
                        "label", "id"))] = color

        plant.Finalize()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration))
        
        with open( f"{filename_base}.json", "w") as f:
            json.dump(instance_id_to_class_name, f)

        camera1 = AddRgbdSensor(builder, scene_graph, RigidTransform(
            RollPitchYaw(np.pi, -np.pi/4,  np.pi / 2.0), [0, -1.5, 1.5]))
        builder.ExportOutput(camera1.color_image_output_port(), "color_image1")
        builder.ExportOutput(camera1.label_image_output_port(), "label_image1")

        """camera2 = AddRgbdSensor(builder, scene_graph, RigidTransform(
            RollPitchYaw(np.pi, -np.pi/4,  np.pi / 2.0), [0, -1.5, 1.5]))
        builder.ExportOutput(camera2.color_image_output_port(), "color_image2")

        camera3 = AddRgbdSensor(builder, scene_graph, RigidTransform(
            RollPitchYaw(np.pi, 0,  np.pi / 2.0), [0, 0, 1.5]))
        builder.ExportOutput(camera3.color_image_output_port(), "color_image3")"""

        #vis = DrakeVisualizer.AddToBuilder(builder, scene_graph)

        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)
        #visualizer.StartRecording()
        
        z = 0.2
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                    UniformlyRandomRotationMatrix(generator),
                    [rs.uniform(-.15,.15), rs.uniform(-.2, .2), z])
            plant.SetFreeBodyPose(plant_context,
                                plant.get_body(body_index),
                                tf)
            z += 0.1
        
        #meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
        simulator.set_target_realtime_rate(5.0)
        simulator.AdvanceTo(5.0)

        color_image1 = diagram.GetOutputPort("color_image1").Eval(context)
        plt.figure()
        plt.axis('off')
        plt.imshow(color_image1.data)
        plt.savefig(f"{filename_base}.png")
        plt.close()

        label_image1 = diagram.GetOutputPort("label_image1").Eval(context)
        np.save(f"{filename_base}_mask", label_image1.data)

        """color_image2 = diagram.GetOutputPort("color_image2").Eval(context)
        plt.figure()
        plt.imshow(color_image2.data)
        plt.axis('off')
        plt.savefig(f"dataset/scene{epoch}_cam2.png"  )

        color_image3 = diagram.GetOutputPort("color_image3").Eval(context)
        plt.figure()
        plt.imshow(color_image3.data)
        plt.axis('off')
        plt.savefig(f"dataset/scene{epoch}_cam3.png"  )"""

        #visualizer.PublishRecording()


def generate_boxes( plant, parser, num_of_boxes=None, list_of_boxes=None):
    rs = np.random.RandomState()  # this is for python
    generator = RandomGenerator(rs.randint(1000))  # this is for c+
   
    instance_id_to_class_name = dict()

    dimension = 6
    grid = [f"{x},{y}" for x in range(dimension) for y in range(dimension)]
    box_positions = np.random.choice(grid, replace=False, size=num_of_boxes)
    box_positions = ["0,0"]
    for i in range(num_of_boxes):
        color = 1
        box_rotation = [np.pi/2*np.random.randint(4), np.pi/2*np.random.randint(4), np.pi/2*np.random.randint(4)]
        box_position = [0.2*int(box_positions[i][0])-0.15, 0.2*int(box_positions[i][2])-0.15, 0.1]
        box = BoxObjectString(color, .1, .1, .2, 1, "", box_position, box_rotation)
        sdf = box.generate_sdf_string(f"box{i}")
        instance = parser.AddModelsFromString(sdf, "sdf")
        
    