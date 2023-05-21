import sys
sys.path.append("/usr/cargobot/cargobot-project/src/")
import webbrowser

from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect
from django.template import loader

from demos.mobilebase_perception import run_demo
from manip.enums import LabelEnum, BoxColorEnum
from enum import Enum


from copy import deepcopy
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as Tf
from IPython.display import clear_output, display
from manipulation import running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate
from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import AddPackagePaths, FindResource, LoadDataResource
from pydrake.all import (BaseField, Concatenate, Fields, MeshcatVisualizer,
                         MeshcatVisualizerParams, PointCloud, Quaternion, Rgba,
                         RigidTransform, RotationMatrix, StartMeshcat)

status = "NOT_WORKING" #task durumu 
progress = "0%" # task runlandıktan sonraki process
config = "UNDONE" #kutu ve label secimi boolean valuesu
box_container = []
number_of_boxes = 0
label_container = []

class Label:
    def __init__(self,name,priority=None,weight=None):
        self.priority = priority
        self.name = name
        self.weight = weight
    def __str__(self):
        return "Label: {}, Weight Type: {} Priority Type: {}".format(self.name,self.weight,self.priority)
class Box:
    def __init__(self,id,label=None,height=None,width=None,depth=None):
        self.id = id
        self.label = label
        self.height = height
        self.width = width
        self.depth = depth
    def __str__(self):
        return "Box {} height={} width={} depth={} label={}".format(self.id,self.height,self.width,self.depth,self.label)


def match_enums(label):
    weight = None
    priority = None
    color = None
    if label.weight == 'light':
        weight = LabelEnum.LIGHT
    if label.weight == 'heavy':
        weight = LabelEnum.HEAVY
    if label.priority == 'low':
        priority = LabelEnum.LOW_PRIORTY
    if label.priority == 'medium':
        priority = LabelEnum.MID_PRIORTY
    if label.priority == 'high':
        priority = LabelEnum.HIGH_PRIORTY
    if label.name == 'red':
        color = BoxColorEnum.RED
    elif label.name == 'green':
        color = BoxColorEnum.GREEN
    elif label.name == 'blue':
        color = BoxColorEnum.BLUE
    elif label.name == 'cyan':
        color = BoxColorEnum.CYAN
    elif label.name == 'magenta':
        color = BoxColorEnum.MAGENTA
    elif label.name == 'yellow':
        color = BoxColorEnum.YELLOW
    
    return (weight,priority,color)



def construct_labels(labels=[Label('red'),Label('green'),Label('blue'),Label('cyan'),Label('magenta'),Label('yellow')]):
    global label_container
    label_container = labels

def box_tuple_adaptor():
    global box_container
    global label_container
    adapted_boxes = []
    weight = None
    priority = None
    color = None
    for box in box_container:
        matched_enums = match_enums(box.label)
        weight = matched_enums[0]
        priority = matched_enums[1]
        color = matched_enums[2]
        adapted_boxes.append({
            "id":box.id,
            "dimensions":(box.height,box.width,box.depth),
            "labels":(weight,priority),
            "color":(color)
        })
    return adapted_boxes


def modify_status(code):
    if code == 0:
        status = "NOT_WORKING"
    elif code == 1:
        status = "WORKING"
    elif code == 2:
        status = "PAUSED"
def set_number_of_boxes(number):
    global number_of_boxes
    number_of_boxes = number

def update_progress(time,total_time): # kutu başı progress daha mantıklı olabilirmiş
    progress = "{}%".format((time / total_time) * 100)

def index(request):
    context = {
        "status": status,
        "progress":progress,
        "config":config
    }
    template = loader.get_template("gui/index.html")
    return HttpResponse(template.render(context, request))


def set_num_boxes(request):
    if request.method == 'POST':
        set_number_of_boxes(request.POST.get('num_boxes', 0)) #updates global var number_of_boxes
        return redirect('set_box_size')
    else:
        template = loader.get_template("gui/set_boxes.html")
        context = {}
        return HttpResponse(template.render(context, request))
    
def set_box_size(request):
    if request.method == 'GET':
        template = loader.get_template("gui/set_box_size.html")
        global box_container
        for i in range(int(number_of_boxes)):
            box_container.append(Box(i))
        context = {
            "box_container":box_container
        }
        return HttpResponse(template.render(context, request))

def process_box_size(request):
    if request.method == 'POST':
        counter = 0
        indexer = 0
        for key in request.POST.keys():
            if key == 'csrfmiddlewaretoken':
                continue
            else:
                if key.startswith('box_height_'):
                    box_container[counter].height = request.POST[key]
                elif key.startswith('box_width_'):
                    box_container[counter].width = request.POST[key]
                elif key.startswith('box_depth_'):
                    box_container[counter].depth = request.POST[key]
                indexer += 1
                if indexer % 3 == 0:
                    counter += 1
        context = {
            "box_container":box_container
        }
        return redirect('set_label_priority')

            

def set_label_priority_view(request):
    if request.method == 'GET':
        template = loader.get_template("gui/set_label_priority.html")
        context = {}
        return HttpResponse(template.render(context, request))
    if request.method == 'POST':
        construct_labels()
        global label_container
        red_1 = request.POST.get('red_1') #weight
        red_2 = request.POST.get('red_2') #priority
        green_1 = request.POST.get('green_1')
        green_2 = request.POST.get('green_2')
        blue_1 = request.POST.get('blue_1')
        blue_2 = request.POST.get('blue_2')
        cyan_1 = request.POST.get('cyan_1')
        cyan_2 = request.POST.get('cyan_2')
        magenta_1 = request.POST.get('magenta_1')
        magenta_2 = request.POST.get('magenta_2')
        yellow_1 = request.POST.get('yellow_1')
        yellow_2 = request.POST.get('yellow_2')
        for label in label_container:
            if label.name == 'red':
                label.weight = red_1
                label.priority = red_2
            elif label.name == 'green':
                label.weight = green_1
                label.priority = green_2
            elif label.name == 'blue':
                label.weight = blue_1
                label.priority = blue_2
            elif label.name == 'cyan':
                label.weight = cyan_1
                label.priority = cyan_2
            elif label.name == 'magenta':
                label.weight = magenta_1
                label.priority = magenta_2
            elif label.name == 'yellow':
                label.weight = yellow_1
                label.priority = yellow_2
            
        return redirect('assign_label_to_boxes')

def assign_label_to_boxes(request):
    global box_container
    if request.method == 'GET':
        template = loader.get_template("gui/assign_label_to_boxes.html")
        context = {"box_container":box_container,"label_container":label_container}
        return HttpResponse(template.render(context, request))
    if request.method == 'POST':
        for box in box_container:
            label_name = request.POST.get('box{}_label'.format(box.id))
            for label in label_container:
                if label.name == label_name:
                    box.label = label
    box_list = box_tuple_adaptor()
    webbrowser.open('localhost:7000/')
    simulator, meshcat, visualizer = run_demo(box_list)
    meshcat.AddButton("Stop Simulation", "Escape")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    visualizer.PublishRecording()
    
    return redirect('index')

def meshcat_opener():
    redirect_url = 'localhost:7000/'
    return HttpResponseRedirect(redirect_url)

def start_process(request):
    pass
def pause_process(request):
    pass
def continue_process(request):
    pass
def stop_process(request):
    pass
    
