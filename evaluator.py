#!/usr/bin/env python

import argparse
import json
import csv
import torch
from torchvision import ops


class Utils:
    @staticmethod
    def xywh_to_xyxy(box):
        x, y, w, h = box
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def xyxy_to_xywh(box):
        xmin, ymin, xmax, ymax = box
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        return [x, y, w, h]


def parse_custom(path):
    result = {}
    with open(path, "r") as file:
        data = json.load(file)
    frames = data["frames"]
    for frame in frames:
        boxes = []
        for object in frame["objects"]:
            boxes.append(
                Utils.xywh_to_xyxy([object["x"], object["y"], object["w"], object["h"]])
            )
        result[frame["number"]] = boxes
    average_time = data["average_time"]
    return result, average_time


def parse_mot(path):
    with open(path, "r") as file:
        reader = csv.reader(file)
        matrix = []
        for row in reader:
            matrix.append([int(i) for i in row])
    frames = set([row[0] for row in matrix])
    result = {frame: [] for frame in frames}
    for row in matrix:
        result[row[0]].append(Utils.xywh_to_xyxy([row[2], row[3], row[4], row[5]]))
    return result


def parse_got10k(path):
    with open(path, "r") as file:
        reader = csv.reader(file)
        matrix = []
        for row in reader:
            matrix.append([int(float(i)) for i in row])
    result = {}
    for i, row in enumerate(matrix):
        result[i] = [Utils.xywh_to_xyxy(row)]
    return result


def calculate_iou(predicted, groundtruth):
    predicted_bbox = torch.tensor([predicted], dtype=torch.float)
    groundtruth_bbox = torch.tensor([groundtruth], dtype=torch.float)
    iou = ops.box_iou(groundtruth_bbox, predicted_bbox)
    return iou.numpy()[0][0]


def get_pe(predicted, groundtruth):
    all_pe = []
    for frame, predicted_boxes in predicted.items():
        pe = len(predicted_boxes) / len(groundtruth[frame])
        all_pe.append(pe)
    min_r, max_r, avg_r = min(all_pe), max(all_pe), sum(all_pe) / len(all_pe)
    return min_r, max_r, avg_r


def get_iou(predicted, groundtruth):
    all_iou = []
    for frame, groundtruth_boxes in groundtruth.items():
        for groundtruth_box in groundtruth_boxes:
            predicted_boxes = predicted[frame]
            iou = max(
                [
                    calculate_iou(predicted_box, groundtruth_box)
                    for predicted_box in predicted_boxes
                ]
            )
            all_iou.append(iou)
    min_r, max_r, avg_r = min(all_iou), max(all_iou), sum(all_iou) / len(all_iou)
    return min_r, max_r, avg_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Video Annotator Evaluator",
        description="Evaluates tracked objects annotation",
    )
    parser.add_argument(
        "--predicted",
        dest="predicted",
        required=True,
        help="predicted annotation file path",
    )
    parser.add_argument(
        "--groundtruth",
        dest="groundtruth",
        required=True,
        help="groundtruth annotation file path",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=True,
        choices=["mot", "got10k"],
        help="dataset associated with annotation format",
    )

    args = parser.parse_args()

    predicted, average_time = parse_custom(args.predicted)
    print(f"Average frame annotation time: {round(average_time)} ms")

    if args.dataset == "mot":
        groundtruth = parse_mot(args.groundtruth)
    elif args.dataset == "got10k":
        groundtruth = parse_got10k(args.groundtruth)

    min_iou, max_iou, avg_iou = get_iou(predicted, groundtruth)
    print(f"IOU: min = {min_iou},  avg = {avg_iou}, max = {max_iou}")

    if args.dataset == "mot":
        min_pe, max_pe, avg_pe = get_pe(predicted, groundtruth)
        print(f"PE: min = {min_pe}, avg = {avg_pe}, max = {max_pe}")
