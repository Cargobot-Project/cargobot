from manipulation.utils import AddPackagePaths
from manipulation.scenarios import AddIiwa, AddWsg
from pydrake.all import ( AddMultibodyPlantSceneGraph, 
                         DiagramBuilder,Parser)

class BaseManipulationStation:
    def __init__(self, time_step: float, add_iiwa: bool = True, collision_model: str = "with_box_collision"):
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step
        )
        self.parser = Parser(self.plant)
        AddPackagePaths(self.parser)
        #Relative path, dont change the model yaml folder !
        self.base_models = self.parser.AddAllModelsFromFile("/usr/cargobot/cargobot-project/perception/models/all.dmd.yaml")
        if add_iiwa:
            self.iiwa = AddIiwa(self.plant, collision_model=collision_model)
            self.wsg = AddWsg(self.plant, self.iiwa, roll=0.0, welded=True, sphere=False)
        #if scenario:
            #scene.add_base_station(self.plant)
            #scenario.add_scenario_objects(self.plant)

        #self.parser = Parser(self.plant)
        #self.query_output_port = self.scene_graph.GetOutputPort("query")

    # get functions
    def get_plant(self):
        return self.plant
    
    def get_builder(self):
        return self.builder
    
    def get_diagram(self):
        return self.diagram
    
    def get_scene_graph(self):
        return self.scene_graph
    
    def get_parser(self):
        return self.parser
    
    def get_iiwa(self):
        return self.iiwa
    
    def get_wsg(self):
        return self.wsg

    

