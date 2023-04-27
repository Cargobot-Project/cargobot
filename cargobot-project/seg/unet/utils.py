# Standard library
import os
from datetime import datetime

# Third-party libraries and modules
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

# Local modules
from dataset import CustomDataset


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint saved")


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded")


def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        test_dir,
        test_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        test_transform,
        num_workers=0,
        pin_memory=True,
):
    train_ds = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = CustomDataset(
        image_dir=test_dir,
        mask_dir=test_mask_dir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def compute_recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def compute_specificity(true_negative, false_positive):
    return true_negative / (true_negative + false_positive)


def compute_accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)


def compute_precision(true_positive, false_positive):
    if (true_positive + false_positive) != 0.0:
        return true_positive / (true_positive + false_positive)
    else:
        return 0.0


def compute_dice(true_positive, false_positive, false_negative):
    return (2 * true_positive) / (2 * true_positive + false_positive + false_negative)


def compute_f1score(true_positive, false_positive, false_negative):
    precision = compute_precision(true_positive, false_positive)
    recall = compute_recall(true_positive, false_negative)
    if (precision + recall) != 0.0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0.0


def compute_essentials(prediction, target):
    true_positive = torch.sum((target == 1) * (prediction == 1)).float()
    false_positive = torch.sum((prediction == 1) * (target != prediction)).float()
    true_negative = torch.sum((target == 0) * (prediction == 0)).float()
    false_negative = torch.sum((prediction == 0) * (target != prediction)).float()
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative
    }


def compute_scores(loader, model, device='cuda', file=None):
    f1scores = []
    precisions = []
    recalls = []
    dice_scores = []
    accuracies = []
    precision_reporter = []
    f1_reporter = []

    model.eval()
    for index, (data, target, name) in enumerate(loader):
        data = data.to(device=device)
        target = target.to(device).unsqueeze(1)
        prediction = torch.sigmoid(model(data))
        prediction = (prediction > 0.5).float()

        essentials = compute_essentials(prediction, target)
        true_positive = essentials["true_positive"]
        true_negative = essentials["true_negative"]
        false_positive = essentials["false_positive"]
        false_negative = essentials["false_negative"]

        accuracy = compute_accuracy(true_positive, true_negative, false_positive, false_negative)
        dice = compute_dice(true_positive, false_positive, false_negative)
        recall = compute_recall(true_positive, false_negative)
        precision = compute_precision(true_positive, false_positive)
        f1score = compute_f1score(true_positive, false_positive, false_negative)

        if precision == 0.0:
            precision_reporter.append(index)
        if f1score == 0.0:
            f1_reporter.append(index)

        accuracies.append(accuracy)
        dice_scores.append(dice)
        recalls.append(recall)
        precisions.append(precision)
        f1scores.append(f1score)

    scores = f"\nAccuracy: {sum(accuracies) / len(accuracies)}"
    scores += f"\nF1: {sum(f1scores) / len(f1scores)}"
    scores += f"\nPrecision: {sum(precisions) / len(precisions)}"
    scores += f"\nRecall: {sum(recalls) / len(recalls)}"
    scores += f"\nDice: {sum(dice_scores) / len(dice_scores)}"

    if len(precision_reporter) != 0:
        scores += f"\nPrecision Reporter: {str(precision_reporter)}"
    if len(f1_reporter) != 0:
        scores += f"\nF1 Reporter: {str(f1_reporter)}"

    print_and_log(scores, file)


def create_timestamp():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def create_config_stamp(optim_name, learning_rate, patience):
    return f"{optim_name}_{learning_rate}_{patience}"


def make_directory(parent_dir, dataset, data_comb, config_stamp, timestamp):
    directory = f"{dataset}_{data_comb}_{config_stamp}_{timestamp}"
    output_path = os.path.join(parent_dir, directory)
    predictions_path = os.path.join(output_path, "predictions")
    golds_path = os.path.join(output_path, "golds")
    arrays_path = os.path.join(output_path, "arrays")
    processed_path = os.path.join(output_path, "processed")
    try:
        os.mkdir(output_path)
        print(f"\n{output_path} successfully created")
        os.mkdir(predictions_path)
        print(f"\n{predictions_path} successfully created")
        os.mkdir(golds_path)
        print(f"\n{golds_path} successfully created")
        os.mkdir(arrays_path)
        print(f"\n{arrays_path} successfully created")
        os.mkdir(processed_path)
        print(f"\n{processed_path} successfully created")
    except OSError as error:
        print(error)
    return {
        "output_path": output_path,
        "predictions_path": predictions_path,
        "golds_path": golds_path,
        "arrays_path": arrays_path,
        "processed_path": processed_path
    }


def save_prediction_tensor_as_txt(tensor, file_path):
    file = open(file_path, "w")
    for row in tensor:
        np.savetxt(file, row)
    file.close()


def save_predictions_as_images(loader, model, paths, device="cuda"):
    model.eval()
    for index, (data, target, name) in enumerate(loader):
        data = data.to(device=device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(data))
            prediction = (probabilities > 0.5).float()
        torchvision.utils.save_image(prediction, f"{paths['predictions_path']}/{name}_pred.png")
        torchvision.utils.save_image(target.unsqueeze(1), f"{paths['golds_path']}/{name}_gold.png")
        array_file_path = paths["arrays_path"] + f"/{name}_array.txt"
        if index == 2:
            print(probabilities)
            print(probabilities.shape)
        probabilities = probabilities.cpu().numpy()
        probabilities = probabilities[0][0]
        save_prediction_tensor_as_txt(probabilities, array_file_path)


def print_and_log(data, file=None):
    print(data)
    if file is not None:
        file.write(data)


def normalize(matrix):
    return (matrix - matrix.mean()) / (matrix.std())


