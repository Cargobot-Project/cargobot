import numpy as np
import matplotlib as plt

from scene.CameraSystem import CameraSystem

def plot_camera_view(camera: CameraSystem, camera_idx: int):
    plt.imshow(camera.rgb_im)
    plt.title(f"View from camera {camera_idx}")
    plt.show()

def plot_predictions(predictions, object_idx: int):
    for i, prediction in enumerate(predictions):
        mask_idx = np.argmax(predictions[i][0]['labels'] == object_idx)
        mask = predictions[i][0]['masks'][mask_idx,0]

        plt.imshow(mask)
        plt.title("Mask from Camera " + str(i))
        plt.colorbar()
        plt.show()