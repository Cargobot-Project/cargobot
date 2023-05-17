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



class BoxObjectString():
    def __init__(self, color, w, d, h, mass, texture, position, rotation):
        self.color = color
        self.inertia = self.calculate_box_inertia(mass, w, d, h)
        self.w = w
        self.d = d
        self.h = h
        self.mass = mass
        self.texture = texture
        self.position = position
        self.rotation = rotation

    def calculate_box_inertia(self, m, w, d, h):
        Iw = (m/12.0)*(pow(d,2)+pow(h,2))
        Id = (m / 12.0) * (pow(w, 2) + pow(h, 2))
        Ih = (m / 12.0) * (pow(w, 2) + pow(d, 2))
        return ('<ixx>' + str(Iw) + '</ixx> <ixy>0.0</ixy> <ixz>0.0</ixz> <iyy>' 
                + str(Id) + ' </iyy> <iyz>0.0</iyz> <izz>' + str(Ih) + '</izz>')

    def return_color_and_texture(self):
        color_text = ""
        if self.color == 0:
            color_text = "<ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse><emissive>1 0 0 1</emissive>"
        elif self.color == 1:
            color_text = "<ambient>0 0 1 1</ambient><diffuse>0 0 1 1</diffuse><emissive>0 0 1 1</emissive>"
        elif self.color == 2:
            color_text = "<ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse><emissive>0 1 0 1</emissive>"
        elif self.color == 3:
            color_text = "<ambient>1 1 0 1</ambient><diffuse>1 1 0 1</diffuse><emissive>1 1 0 1</emissive>"
        elif self.color == 4:
            color_text = "<ambient>1 0 1 1</ambient><diffuse>1 0 1 1</diffuse><emissive>1 0 1 1</emissive>"
        
        return color_text

    def generate_sdf_string(self, name):
        color_text = self.return_color_and_texture()
        sdf = """
              <?xml version="1.0"?>
              <sdf version="1.7">
                <model name=""" + '"'+  name + '"'+ """>
                  <link name=""" + '"'+  name + '"'+ """>
                    <inertial>
                      <mass>""" + str(self.mass) + """</mass>
                      <inertia>
                          """ + self.inertia+ """ 
                      </inertia>
                    </inertial>
                    <visual name="visual">
                      <pose>""" + f"{self.position[0]} {self.position[1]} {self.position[2]} " + f"{self.rotation[0]} {self.rotation[1]} {self.rotation[2]}" +"""</pose>
                      <geometry>
                        <box>
                          <size>"""+ str(self.w) + """ """ + str(self.d) + """ """ + str(self.h) + """</size>
                        </box>
                      </geometry>
                      <material>
                      """+color_text+"""
                    </material>
                    </visual>
                    <collision name="collision">
                      <pose>""" + f"{self.position[0]} {self.position[1]} {self.position[2]} " + f"{self.rotation[0]} {self.rotation[1]} {self.rotation[2]}" +"""</pose>
                      <geometry>
                        <box> 
                          <size>"""+ str(self.w) + """ """ + str(self.d) + """ """ + str(self.h) + """</size>
                        </box>
                      </geometry>
                    </collision>
                    
                  </link>
                </model>
              </sdf>"""
        
        return sdf

