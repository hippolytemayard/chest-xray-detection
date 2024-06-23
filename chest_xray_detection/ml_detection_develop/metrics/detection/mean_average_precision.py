import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.models.detection.faster_rcnn import GeneralizedRCNN
from torchvision.utils import make_grid
from tqdm import tqdm


def get_ap_from_boxes(
    model: GeneralizedRCNN,
    loader: DataLoader,
    device: torch.device,
    iou_thresh: float = 0.5,
    box_thresh: float = 0.5,
    tensorboard_writer: SummaryWriter = None,
    epoch=None,
) -> dict[str, dict[str, list[float]]]:
    """This function performs an AP-like evaluation of the model, by class.
    The function returns a detailled evaluation (AP-score at threshold, precision, recall).

    Parameters
    ----------
    model : GeneralizedRCNN
        the model to evaluate
    loader : DataLoader
        the dataloader related to the evaluation dataset
    device : torch.device
        either "cpu" or "cuda"
    iou_thresh : float, optional
        the IoU evaluation threshold, by default 0.5

    Returns
    -------
    dict[dict[str, float]]
        evaluation dict is composed of recall, precision, AP and number of ground truth instances
        for each class
    """
    classes = loader.dataset.classes
    dict_metrics = {
        i + 1: {"TP": [], "FP": [], "FN": [], "scores": [], "total_true_polygons": []}
        for i, c in enumerate(classes)
    }

    model.eval()
    i_batch = 0
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets
        ]

        with torch.inference_mode():
            predictions = model(images)

        for prediction in predictions:
            prediction["boxes"] = (prediction["boxes"] > box_thresh).squeeze(1).to(torch.uint8)

        for prediction, target in zip(predictions, targets):
            if len(target["labels"]) == 0:
                continue

            if len(prediction["labels"]) == 0:
                continue

            dict_metrics = get_image_metrics(
                prediction=prediction,
                target=target,
                dict_metrics=dict_metrics,
                classes=classes,
                device=device,
                iou_thresh=iou_thresh,
            )

        if tensorboard_writer:
            im1, im2 = torch.clone(images[0]), torch.clone(images[0])
            im1[2] = (predictions[0]["boxes"] > 0.5).int().sum(dim=0)
            im2[2] = targets[0]["boxes"].sum(dim=0)

            tensorboard_writer.add_images(
                tag="pred(left) vs target(right)",
                img_tensor=torch.stack([im1, im2]),
                global_step=epoch + i_batch,
            )
        i_batch += 1
    ap_metrics = compute_metrics(dict_metrics=dict_metrics, classes=classes)

    return ap_metrics


def get_image_metrics(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    dict_metrics: dict[int, dict[str, list[float]]],
    classes: list,
    device: torch.device,
    iou_thresh: float = 0.5,
) -> dict[int, dict[str, list[float]]]:
    """using a prediction and the ground truth of an image, this function computes the TP, FP as well as
    the total number of ground truth instances. The computed values feed the metrics dictionary.

    Parameters
    ----------
    prediction : dict[str, torch.Tensor]
        the model prediction
    target : dict[str, torch.Tensor]
        ground truth
    dict_metrics : dict[dict[str, list]]
        dictionnary metrics
    classes : list
        dataset classes
    device : torch.device
        either "cpu" or "cuda"
    iou_thresh : float, optional
        the IoU evaluation threshold, , by default 0.5

    Returns
    -------
    dict[dict[str, list]]
        evaluation output dict of TP, FP lists and total ground truth instances by class
    """

    scores = prediction["scores"]
    _, indexes_sorted_by_score = torch.sort(scores, descending=True)

    predicted_boxes = prediction["boxes"][indexes_sorted_by_score]
    predicted_labels = prediction["labels"][indexes_sorted_by_score]
    target_box = target["boxes"]
    target_labels = target["labels"]

    iou_matrix = get_iou_matrix(
        predicted_boxes=predicted_boxes,
        target_boxes=target_box,
        predicted_labels=predicted_labels,
        target_labels=target_labels,
        device=device,
    )

    detected_labels_counter = torch.bincount(predicted_labels, minlength=len(classes) + 1)
    target_labels_counter = torch.bincount(target_labels, minlength=len(classes) + 1)

    # print(f"target_labels_counter : {target_labels_counter}")

    maximum_iou_preds = iou_matrix.amax(dim=0)
    maximum_iou_targets = iou_matrix.amax(dim=1)

    for detected_labels, occurences in enumerate(detected_labels_counter):
        total_true_polygons = target_labels_counter[detected_labels]

        if occurences.item():
            label_indexes = predicted_labels == detected_labels
            label_targets_indexes = target_labels == detected_labels

            best_ious = maximum_iou_preds[label_indexes]
            best_ious_targets_list = torch.clone(
                maximum_iou_targets[label_targets_indexes]
            ).tolist()

            if not total_true_polygons:
                nb_occurences = occurences.item()
                TP = torch.zeros(size=(nb_occurences,)).int()
                FN = torch.zeros(size=(nb_occurences,)).int()
                FP = torch.ones(size=(nb_occurences,)).int()

            else:
                match_detected_target = []
                for iou_pred in best_ious.tolist():
                    is_in_targets = iou_pred in best_ious_targets_list
                    match_detected_target.append(is_in_targets)
                    if is_in_targets:
                        matching_index = best_ious_targets_list.index(iou_pred)
                        best_ious_targets_list.pop(matching_index)

                match_detected_target = torch.tensor(match_detected_target, dtype=torch.bool).to(
                    device
                )

                TP = ((best_ious >= iou_thresh) * match_detected_target).int()
                FN = ((best_ious < iou_thresh) * match_detected_target).int()
                FP = torch.ones(size=TP.shape, device=device).int() - TP - FN

            dict_metrics[detected_labels]["TP"] += TP.tolist()
            dict_metrics[detected_labels]["FP"] += FP.tolist()
            dict_metrics[detected_labels]["FN"] += FN.tolist()
            dict_metrics[detected_labels]["scores"] += scores[label_indexes].tolist()
            dict_metrics[detected_labels]["total_true_polygons"] += [total_true_polygons.item()]

        elif total_true_polygons.item():
            dict_metrics[detected_labels]["total_true_polygons"] += [total_true_polygons.item()]

        else:
            continue

    return dict_metrics


def get_iou_matrix(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    target_labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """this function computes the IoU matrix between the predicted boxes and the ground truth.

    Parameters
    ----------
    predicted_boxes : torch.Tensor
        predicted boxes tensor, dimensions [N_instances x height x width]
    target_boxes : torch.Tensor
        target boxes tensor, dimensions [N_ground_truth x height x width]
    predicted_labels : torch.Tensor
        predicted labels tensor, dimensions [N_instances]
    target_labels : torch.Tensor
        ground truth labels, , dimensions [N_ground_truth]
    device : torch.device
        either "cpu" or cuda

    Returns
    -------
    torch.Tensor
        the IoU matrix (dimension [N_ground_truth x N_instances])
    """
    # print(target_boxes.shape)
    target_box_area = target_boxes.sum((0, 1)).float()
    pred_box_area = predicted_boxes.sum((0, 1)).float()

    number_of_boxes_target = len(target_boxes)
    number_of_boxes_pred = len(predicted_boxes)

    flatten_boxes_target = target_boxes.reshape(number_of_boxes_target, -1).float()
    flatten_boxes_pred = predicted_boxes.reshape(number_of_boxes_pred, -1).float()

    expanded_box_area_target = target_box_area.expand(number_of_boxes_pred, number_of_boxes_target)
    expanded_box_area_pred = pred_box_area.expand(number_of_boxes_target, number_of_boxes_pred)

    intersection_matrix = torch.mm(flatten_boxes_target, flatten_boxes_pred.transpose(1, 0))
    union_matrix = (
        expanded_box_area_pred + expanded_box_area_target.transpose(1, 0) - intersection_matrix
    )
    iou_matrix = intersection_matrix / union_matrix

    expanded_labels_pred = predicted_labels.expand(number_of_boxes_target, number_of_boxes_pred)
    expanded_labels_target = target_labels.expand(number_of_boxes_pred, number_of_boxes_target)
    label_matrix = (expanded_labels_pred == expanded_labels_target.transpose(1, 0)).to(device)
    iou_matrix = iou_matrix * label_matrix

    return iou_matrix


def get_precision_recall_from_dict(
    TP: list[int], FP: list[int], total_ground_truth: int, epsilon: float = 1.0e-6
) -> tuple[float, float, float]:
    """this function computes the precision, recall and ap scores from a list of
    TP, FP and total ground truth instances

    Parameters
    ----------
    TP : list[int]
        list of True Positives (list of 0 and 1)
    FP : list[int]
        list of False Positives (list of 0 and 1)
    total_ground_truth : int
        total number of ground truth instances
    epsilon : float, optional
        constant to avoid division by zero, by default 1.0e-6

    Returns
    -------
    tuple[float, float, float]
        tuples of metrics at threshold : tuple[ap, precision, recall]
    """
    TP_cumsum = np.cumsum(TP, axis=0)
    FP_cumsum = np.cumsum(FP, axis=0)

    recalls = TP_cumsum / (total_ground_truth + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

    recall_ = np.sum(TP) / (total_ground_truth + epsilon)
    precision_ = np.sum(TP) / (np.sum(TP) + np.sum(FP) + epsilon)

    precisions = np.concatenate((np.array([1]), precisions))
    recalls = np.concatenate((np.array([0]), recalls))

    area = np.trapz(precisions, recalls)

    return float(np.round(area, 4)), float(np.round(precision_, 4)), float(np.round(recall_, 4))


def compute_metrics(
    dict_metrics: dict[int, dict[str, list[float]]], classes: list[str], epsilon: float = 1.0e-6
) -> dict[str, dict[str, list[float]]]:
    """computes the metrics from the evaluation output dict of TP, FP and total
    ground truth instances by class

    Parameters
    ----------
    dict_metrics : dict[dict[str, list]]
        evaluation output dict of TP, FP and total ground truth instances by class
    classes : list[str]
        list of classes
    epsilon : float, optional
        constant to avoid division by zero, by default 1.0e-6

    Returns
    -------
    dict[dict[str, int]]
        dict of ap, precision, recall scores at threshol, by class
    """
    ap_dict = {classes[c]: {} for c in range(len(classes))}

    for label, values in dict_metrics.items():
        scores = np.array(dict_metrics[label]["scores"])

        sort_indexes = np.argsort(-scores)

        TP = np.array(dict_metrics[label]["TP"])[sort_indexes]
        FP = np.array(dict_metrics[label]["FP"])[sort_indexes]

        total_true_polygons = np.array(dict_metrics[label]["total_true_polygons"])
        total_ground_truth = total_true_polygons.sum()

        area, precision, recall = get_precision_recall_from_dict(
            TP=TP, FP=FP, total_ground_truth=total_ground_truth, epsilon=epsilon
        )

        ap_dict[classes[label - 1]]["recall"] = recall
        ap_dict[classes[label - 1]]["precision"] = precision
        ap_dict[classes[label - 1]]["AP"] = area
        ap_dict[classes[label - 1]]["total_ground_truth"] = total_ground_truth

    return ap_dict
