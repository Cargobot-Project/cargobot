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





def generate_scene(number_of_times):
    rs = np.random.RandomState()  # this is for python
    generator = RandomGenerator(rs.randint(1000))  # this is for c++
    for epoch in range(number_of_times):
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)

        parser = Parser(plant)

        parser.AddModels("/work/scene_without_robot.dmd.yaml")
        
        box_count = color = rs.randint(16)
        for i in range(box_count if running_as_notebook else 2):
            color = rs.randint(5)
            box = BoxObjectString(color, .1, .1, .2, 1, "")
            sdf = box.generate_sdf_string(f"box{i}")
            parser.AddModelsFromString(sdf, "sdf")

        plant.Finalize()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration))
        

        camera1 = AddRgbdSensor(builder, scene_graph, RigidTransform(
            RollPitchYaw(np.pi, np.pi/4,  np.pi / 2.0), [0, 1.5, 1.5]))
        builder.ExportOutput(camera1.color_image_output_port(), "color_image1")

        camera2 = AddRgbdSensor(builder, scene_graph, RigidTransform(
            RollPitchYaw(np.pi, -np.pi/4,  np.pi / 2.0), [0, -1.5, 1.5]))
        builder.ExportOutput(camera2.color_image_output_port(), "color_image2")

        camera3 = AddRgbdSensor(builder, scene_graph, RigidTransform(
            RollPitchYaw(np.pi, 0,  np.pi / 2.0), [0, 0, 1.5]))
        builder.ExportOutput(camera3.color_image_output_port(), "color_image3")

        #vis = DrakeVisualizer.AddToBuilder(builder, scene_graph)

        diagram = builder.Build()
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)
        visualizer.StartRecording()
        
        z = 0.2
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                    UniformlyRandomRotationMatrix(generator),
                    [rs.uniform(-.15,.15), rs.uniform(-.2, .2), z])
            plant.SetFreeBodyPose(plant_context,
                                plant.get_body(body_index),
                                tf)
            z += 0.1
        
        meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    
        simulator.AdvanceTo(2.0 if running_as_notebook else 0.1)

        color_image1 = diagram.GetOutputPort("color_image1").Eval(context)
        plt.figure()
        plt.imshow(color_image1.data)
        plt.axis('off')
        plt.savefig(f"/datasets/segmentatiton-database/Segmentation Data/scene{epoch}_cam1.png")

        color_image2 = diagram.GetOutputPort("color_image2").Eval(context)
        plt.figure()
        plt.imshow(color_image2.data)
        plt.axis('off')
        plt.savefig(f"/datasets/segmentatiton-database/Segmentation Data/scene{epoch}_cam2.png"  )

        color_image3 = diagram.GetOutputPort("color_image3").Eval(context)
        plt.figure()
        plt.imshow(color_image3.data)
        plt.axis('off')
        plt.savefig(f"/datasets/segmentatiton-database/Segmentation Data/scene{epoch}_cam3.png"  )

        visualizer.PublishRecording()

generate_scene(2)