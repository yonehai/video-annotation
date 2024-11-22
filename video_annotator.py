#!/usr/bin/env python

import argparse
import json
import time
import cv2
from ultralytics import YOLO
from evaluator import Utils, calculate_iou

opencv_trackers = ["mil", "nano", "vit", "csrt", "kcf"]
yolo_trackers = ["botsort", "bytetrack"]
yolo_detectors = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


class TrackedObject:
    def __init__(self, box, frame, tracker_type):
        self.__box = box
        self.__tracker_type = tracker_type
        self.__tracker = self.__create_opencv_tracker()
        self.__tracker.init(frame, box)

    def get_box(self):
        return self.__box

    def get_tracker(self):
        return self.__tracker

    def set_box(self, box):
        self.__box = box

    def set_tracker(self, tracker):
        self.__tracker = tracker

    def __create_opencv_tracker(self):
        if self.__tracker_type == "mil":
            tracker = cv2.TrackerMIL.create()
        if self.__tracker_type == "nano":
            tracker = cv2.TrackerNano.create()
        if self.__tracker_type == "vit":
            tracker = cv2.TrackerVit.create()
        if self.__tracker_type == "csrt":
            tracker = cv2.TrackerCSRT.create()
        if self.__tracker_type == "kcf":
            tracker = cv2.TrackerKCF.create()
        return tracker


class VideoAnnotator:
    def __init__(self, tracker_type, detector, mot):
        self.__tracker_type = tracker_type
        self.__detector = YOLO(f"{detector}.pt")
        self.__mot = mot
        self.__frames_data = []
        self.__current_frame_number = 0
        self.__time = []
        self.__tracked_objects_list = []

    def annotate(self, video_path, show_output):
        if self.__tracker_type in yolo_trackers:
            return self.__annotate_yolo(video_path, show_output)
        else:
            return self.__annotate_opencv(video_path, show_output)

    def __annotate_yolo(self, video_path, show_output):
        capture = cv2.VideoCapture(video_path)
        while capture.isOpened():
            success, frame = capture.read()
            if success:
                start_time = time.time_ns()
                results = self.__detector.track(
                    frame,
                    persist=True,
                    tracker=f"{self.__tracker_type}.yaml",
                    show=show_output,
                    verbose=False,
                )
                end_time = time.time_ns()
                self.__time.append((end_time - start_time) / 1000000)  # miliseconds
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
                objects = [self.__box_to_json(Utils.xyxy_to_xywh(box)) for box in boxes]
                self.__frames_data.append(
                    {"number": self.__current_frame_number, "objects": objects}
                )
                self.__current_frame_number += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        capture.release()
        result = {
            "average_time": sum(self.__time) / len(self.__time),
            "frames": self.__frames_data,
        }
        self.__reset()
        return result

    def __annotate_opencv(self, video_path, show_output):
        if self.__mot:
            return self.__opencv_mot(video_path, show_output)
        else:
            return self.__opencv_sot(video_path, show_output)

    def __opencv_sot(self, video_path, show_output):
        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()
        height, width, _ = frame.shape
        if success:
            if show_output:
                video = cv2.VideoWriter("output.mp4", -1, 1, (width, height))
                video.write(frame)
            results = self.__detector.predict(frame, verbose=False)
            box = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()[0]
            box = Utils.xyxy_to_xywh(box)
            self.__tracked_objects_list.append(
                TrackedObject(box, frame, self.__tracker_type)
            )
            if show_output:
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[0] + box[2], box[1] + box[3]),
                    (255, 0, 0),
                    2,
                    1,
                )
            self.__frames_data.append(
                {
                    "number": self.__current_frame_number,
                    "objects": [self.__box_to_json(box)],
                }
            )
            self.__current_frame_number += 1

            object = self.__tracked_objects_list[0]
            while capture.isOpened():
                success, frame = capture.read()
                if success:
                    start_time = time.time_ns()
                    success, result = object.get_tracker().update(frame)
                    if success:
                        object.set_box(result)
                    end_time = time.time_ns()
                    self.__time.append((end_time - start_time) / 1000000)  # miliseconds
                    box = object.get_box()
                    self.__frames_data.append(
                        {
                            "number": self.__current_frame_number,
                            "objects": [self.__box_to_json(box)],
                        }
                    )
                    self.__current_frame_number += 1
                    if show_output:
                        cv2.rectangle(
                            frame,
                            (box[0], box[1]),
                            (box[0] + box[2], box[1] + box[3]),
                            (255, 0, 0),
                            2,
                            1,
                        )
                        cv2.imshow("Tracking", frame)
                        video.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
            capture.release()
            result = {
                "average_time": sum(self.__time) / len(self.__time),
                "frames": self.__frames_data,
            }
            self.__reset()
            return result

    def __opencv_mot(self, video_path, show_output):
        iou_min_threshold = 0.1
        iou_max_threshold = 0.8
        out_of_frame_threshold = 0.5
        iou_displacement_threshold = 0.5
        size_change_threshold = 0.5

        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()
        height, width, _ = frame.shape
        if success:
            if show_output:
                video = cv2.VideoWriter("output.mp4", -1, 1, (width, height))
                video.write(frame)
            results = self.__detector.predict(frame, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
            boxes = [Utils.xyxy_to_xywh(box) for box in boxes]
            for box in boxes:
                self.__tracked_objects_list.append(
                    TrackedObject(box, frame, self.__tracker_type)
                )

            objects_json = []
            for object in self.__tracked_objects_list:
                box = object.get_box()
                objects_json.append(self.__box_to_json(box))
                if show_output:
                    cv2.rectangle(
                        frame,
                        (box[0], box[1]),
                        (box[0] + box[2], box[1] + box[3]),
                        (255, 0, 0),
                        2,
                        1,
                    )
            self.__frames_data.append(
                {"number": self.__current_frame_number, "objects": objects_json}
            )
            self.__current_frame_number += 1

            while capture.isOpened():
                success, frame = capture.read()
                if success:
                    start_time = time.time_ns()

                    object_indexes_to_remove = set()

                    for i, object in enumerate(self.__tracked_objects_list):
                        success, result = object.get_tracker().update(frame)
                        if success:
                            # check if box size or position has changed significantly
                            is_position_changed = self.__is_position_changed(
                                object.get_box(), result, iou_displacement_threshold
                            )
                            is_size_changed = self.__is_size_changed(
                                object.get_box(), result, size_change_threshold
                            )
                            if is_size_changed or is_position_changed:
                                object_indexes_to_remove.add(i)
                            else:
                                object.set_box(result)
                        else:
                            object_indexes_to_remove.add(i)

                    new_results = self.__detector.predict(frame, verbose=False)
                    new_boxes = (
                        new_results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
                    )
                    for new_box in new_boxes:
                        object_indexes_to_iou = {}
                        for i, object in enumerate(self.__tracked_objects_list):
                            iou = calculate_iou(
                                Utils.xywh_to_xyxy(object.get_box()), new_box
                            )
                            object_indexes_to_iou[i] = iou

                        max_iou = max(object_indexes_to_iou.values())

                        # add new non-overlapping object
                        if max_iou < iou_min_threshold:
                            new_box = Utils.xyxy_to_xywh(new_box)
                            self.__tracked_objects_list.append(
                                TrackedObject(new_box, frame, self.__tracker_type)
                            )
                        # remove overlapping objects but not the original one
                        elif max_iou > iou_max_threshold:
                            sorted_data = dict(
                                sorted(
                                    object_indexes_to_iou.items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            iterator = iter(sorted_data.items())
                            next(iterator)
                            for i, iou in iterator:
                                if iou <= iou_max_threshold:
                                    break
                                object_indexes_to_remove.add(i)

                    # remove objects that are out of frame
                    for i, object in enumerate(self.__tracked_objects_list):
                        is_out_of_frame = self.__is_out_of_frame(
                            object.get_box(), width, height, out_of_frame_threshold
                        )
                        if is_out_of_frame:
                            object_indexes_to_remove.add(i)

                    for index in sorted(object_indexes_to_remove, reverse=True):
                        del self.__tracked_objects_list[index]

                    end_time = time.time_ns()
                    self.__time.append((end_time - start_time) / 1000000)  # miliseconds

                    objects_json = []
                    for object in self.__tracked_objects_list:
                        box = object.get_box()
                        objects_json.append(self.__box_to_json(box))
                        if show_output:
                            cv2.rectangle(
                                frame,
                                (box[0], box[1]),
                                (box[0] + box[2], box[1] + box[3]),
                                (255, 0, 0),
                                2,
                                1,
                            )
                    self.__frames_data.append(
                        {"number": self.__current_frame_number, "objects": objects_json}
                    )
                    self.__current_frame_number += 1
                    if show_output:
                        cv2.imshow("Tracking", frame)
                        video.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
            capture.release()
            result = {
                "average_time": sum(self.__time) / len(self.__time),
                "frames": self.__frames_data,
            }
            self.__reset()
            return result

    def __box_to_json(self, box):
        box_json = {
            "x": box[0],
            "y": box[1],
            "w": box[2],
            "h": box[3],
        }
        return box_json

    def __is_out_of_frame(self, box, frame_width, frame_height, threshold):
        xmin, ymin, w, h = box
        xmax, ymax = xmin + w, ymin + h
        return (
            (xmin < 0 and abs(xmin) > w * threshold)
            or (xmax > frame_width and xmax - frame_width > w * threshold)
            or (ymin < 0 and abs(ymin) > h * threshold)
            or (ymax > frame_height and ymax - frame_height > h * threshold)
        )

    def __is_position_changed(self, box, new_box, threshold):
        iou = calculate_iou(
            Utils.xywh_to_xyxy(box),
            Utils.xywh_to_xyxy(new_box),
        )
        return iou < threshold

    def __is_size_changed(self, box, new_box, threshold):
        s_old = box[2] * box[3]
        s_new = new_box[2] * new_box[3]
        return abs(s_old - s_new) / max(s_old, s_new) > threshold

    def __reset(self):
        self.__frames_data = []
        self.__current_frame_number = 0
        self.__time = []
        self.__tracked_objects_list = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Video Annotator",
        description="Generates a json file with tracked objects annotation",
    )
    parser.add_argument("--input_path", required=True, help="input video file path")
    parser.add_argument(
        "--tracker",
        required=False,
        choices=opencv_trackers + yolo_trackers,
        default="botsort",
        help="tracker to use",
    )
    parser.add_argument(
        "--detector",
        required=False,
        choices=yolo_detectors,
        default="yolov8n",
        help="detector to use",
    )
    parser.add_argument(
        "--mot",
        action=argparse.BooleanOptionalAction,
        help="specify if multiple objects should be tracked",
    )
    parser.add_argument(
        "--video_output",
        action=argparse.BooleanOptionalAction,
        help="specify if an output video should be played while annotating",
    )
    args = parser.parse_args()

    output_data = VideoAnnotator(args.tracker, args.detector, args.mot).annotate(
        args.input_path, args.video_output
    )
    output_file_name = f"{args.input_path}_annotated.json"
    with open(output_file_name, "w") as file:
        json.dump(output_data, file)
