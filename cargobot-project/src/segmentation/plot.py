import numpy as np
import matplotlib.pyplot as plt

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