import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import RigidTransform, RotationMatrix, Cylinder, Rgba

from scene.CameraSystem import CameraSystem

def plot_camera_view(rgb_ims, camera_idx: int, output_path: str=None):
    plt.clf()
    plt.imshow(rgb_ims[camera_idx])
    plt.title(f"View from camera {camera_idx}")
    #plt.show()
    if output_path is not None:
        plt.savefig(output_path)

def plot_predictions(predictions, object_idx: int, output_dir: str=None):
    
    for i, prediction in enumerate(predictions):
        #print(prediction)
        plt.clf()
        mask_idx = np.argmax(prediction[0]['labels'] == object_idx)
        mask = prediction[0]['masks'][mask_idx,0]

        plt.imshow(mask)
        plt.title("Mask from Camera " + str(i))
        plt.colorbar()
        #plt.show()

        if output_dir is not None:
            plt.savefig(output_dir + f"prediction{i}.png")


def add_meshcat_triad(
    meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()
):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0]
    )
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(
        path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0]
    )
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(
        path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(
        path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )