from base_manipulation_station import BaseManipulationStation
import numpy as np
import os

from pydrake.common import FindResourceOrThrow, temp_directory
from pydrake.math import RigidTransform
from pydrake.all import RotationMatrix
from pydrake.geometry import (
    StartMeshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
)

class ManipulationStation(BaseManipulationStation):
    def __init__(self,time_step: float,add_iiwa: bool = True,collision_model: str = "with_box_collision"):
        BaseManipulationStation.__init__(self,time_step,add_iiwa,collision_model)
        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.meshcat.DeleteAddedControls()
        self.build_scenario()
        super().plant.Finalize()
        self.visualizer = MeshcatVisualizer.AddToBuilder(super().builder, super().scene_graph, self.meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
        self.collision_visualizer = MeshcatVisualizer.AddToBuilder(
        super().builder, super().scene_graph, self.meshcat,
        MeshcatVisualizerParams(prefix="collision", role=Role.kProximity))
        self.meshcat.SetProperty("collision", "visible", False)

        self.diagram = super().builder.Build()
        self.context = super().diagram.CreateDefaultContext()
        self.plant_context = super().plant.GetMyContextFromRoot(self.context)

    
    def build_scenario(self):
        initial = [0.7, 0.0, 0.3]
        final = [-0.5, -0.2, 0.0]
        X_O = {"initial": RigidTransform(RotationMatrix(), [0.6, 0.1, .1]),
                "goal": RigidTransform(RotationMatrix(), final)}
        box = super().plant.GetBodyByName("box-horizontal")
        super().plant.SetDefaultFreeBodyPose(box, X_O["initial"])
ManipulationStation(1.0)
    






    