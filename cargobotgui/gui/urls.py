from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("setboxsize/",views.set_box_size,name="set_box_size"),
    path("processboxsize/",views.process_box_size,name="process_box_size"),
    path("setlabelpriority/",views.set_label_priority_view,name="set_label_priority"),
    path("assign_label_to_boxes/",views.assign_label_to_boxes,name="assign_label_to_boxes"),
    path("setboxes/",views.set_num_boxes,name="set_box_amount"),
    path("pauseprocess/",views.pause_process,name="pause_process"),
    path("stopprocess/",views.stop_process,name="stop_process"),
    path("continueprocess/",views.continue_process,name="continue_process"),
    path("start_process/",views.start_process,name="start_process"),
]