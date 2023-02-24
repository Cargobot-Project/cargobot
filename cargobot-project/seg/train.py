# Standard library
import sys

# Third-party libraries and modules
import albumentations as album
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Local modules
from unet import UNet
from utils import *

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1000
NUM_WORKERS = 0
IMAGE_HEIGHT = 512  # original size 512
IMAGE_WIDTH = 512  # original size 512
PIN_MEMORY = True
LOAD_MODEL = False

# Todo
# Change the namings according to the setup
DATA_COMB = "name_of_the_data_combination"
DATASET = "task_name_such_as_single_task"
TRAIN_IMG_DIR = f"image/directory/for/training"
TRAIN_MASK_DIR = f"mask/directory/for/training"
VAL_IMG_DIR = f"image/directory/for/validation"
VAL_MASK_DIR = f"mask/directory/for/validation"
TEST_IMG_DIR = f"image/directory/for/testing"
TEST_MASK_DIR = f"mask/directory/for/testing"
# Save predictions as images to a folder
PARENT_DIR = f"/target/output/folder/{DATASET}"


def model_eval(model, loader, loss_fn):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for index, (data, target, name) in enumerate(loader):
            data = data.to(device=DEVICE)
            target = target.float().unsqueeze(1).to(device=DEVICE)

            with torch.cuda.amp.autocast():
                prediction = model(data)
                loss = loss_fn(prediction, target)
                total_loss += loss
    average_loss = total_loss / len(loader)
    model.train()

    return average_loss


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, file=sys.stdout)
    total_loss = 0

    for index, (data, target, name) in enumerate(loop):
        data = data.to(device=DEVICE)
        target = target.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, target)
            total_loss += loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

    average_loss = total_loss / len(loader)
    return average_loss


def main():
    train_transform = [
        normalize,
        album.Compose([ToTensorV2()])
    ]
    val_transform = [
        normalize,
        album.Compose([ToTensorV2()])
    ]
    test_transform = [
        normalize,
        album.Compose([ToTensorV2()])
    ]

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scaler = torch.cuda.amp.GradScaler()

    # Early stopping
    min_val_loss = np.Inf
    min_epoch = 0
    min_delta = 0
    patience = 100
    val_losses = []

    # Create output directory
    config_stamp = create_config_stamp(type(optimizer).__name__, LEARNING_RATE, patience)
    timestamp = create_timestamp()
    paths = make_directory(PARENT_DIR, DATASET, DATA_COMB, config_stamp, timestamp)
    # Create log file to keep info
    log_file_path = paths["output_path"] + "/execution.log"
    log_file = open(log_file_path, "w")

    if LOAD_MODEL:
        checkpoint_file_path = paths["output_path"] + "/checkpoint.pth.tar"
        load_checkpoint(torch.load(checkpoint_file_path), model)
    # compute_scores(val_loader, model, device=DEVICE)

    # training
    for epoch in range(NUM_EPOCHS):
        print_and_log(f"\nTraining EPOCH {epoch + 1}", log_file)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        print_and_log(f"\nAverage Training Loss for EPOCH {epoch + 1}: {train_loss}", log_file)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        checkpoint_file_path = paths["output_path"] + "/checkpoint.pth.tar"
        save_checkpoint(checkpoint, filename=checkpoint_file_path)

        val_loss = model_eval(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        print_and_log(f"\nAverage Validation Loss for EPOCH {epoch + 1}: {val_loss}", log_file)

        if min_val_loss > val_loss.item() + min_delta:
            min_val_loss = val_loss.item()
            min_epoch = epoch + 1
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print_and_log(f"\nMinimum Validation Loss: {min_val_loss} at EPOCH {min_epoch}", log_file)
            print_and_log(f"\nTotal Epoch Count: {epoch + 1}", log_file)
            break

    log_file.close()
    # create txt file to keep scores
    txt_file_path = paths["output_path"] + "/scores.txt"
    txt_file = open(txt_file_path, "w")

    # compute scores
    print_and_log(f"\nTest Scores", txt_file)
    compute_scores(test_loader, model, device=DEVICE, file=txt_file)
    txt_file.close()

    save_predictions_as_images(test_loader, model, paths=paths, device=DEVICE)


if __name__ == "__main__":

    repetition_counter = 4  # To train 4 different models (can be changed)
    for repetition in range(repetition_counter):
        main()
