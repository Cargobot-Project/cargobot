from base_manipulation_station import BaseManipulationStation
import numpy as np
import os
import time
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
    def __init__(self, time_step: float, add_iiwa: bool = True, collision_model: str = "with_box_collision"):
        super(ManipulationStation, self).__init__(time_step, add_iiwa, collision_model)
        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.meshcat.DeleteAddedControls()
        self.build_scenario()
        super().get_plant().Finalize()
        self.visualizer = MeshcatVisualizer.AddToBuilder(super().get_builder(), super().get_scene_graph(), self.meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
        self.collision_visualizer = MeshcatVisualizer.AddToBuilder(
                                    super().get_builder(), super().get_scene_graph(), self.meshcat,
                                    MeshcatVisualizerParams(prefix="collision", role=Role.kProximity)
                                )
        self.meshcat.SetProperty("collision", "visible", False)

        self.diagram = super().get_builder().Build()
        self.context = super().get_diagram().CreateDefaultContext()
        self.plant_context = super().get_plant().GetMyContextFromRoot(self.context)

    
    def build_scenario(self):
        initial = [0.7, 0.0, 0.3]
        final = [-0.5, -0.2, 0.0]
        X_O = {"initial": RigidTransform(RotationMatrix(), [0.6, 0.1, .1]),
                "goal": RigidTransform(RotationMatrix(), final)}
        box = super().get_plant().GetBodyByName("base_link_box-horizontal")
        super().get_plant().SetDefaultFreeBodyPose(box, X_O["initial"])
        
ManipulationStation(1.0)

while True:
    time.sleep(1)
    






    