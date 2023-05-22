import cv2
import numpy as np
from typing import Tuple
import tensorflow as tf
from pathlib import Path
from collections import namedtuple

from tflite_runtime.interpreter import Interpreter, load_delegate

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

DetectedObject = namedtuple('DetectedObject', ['bbox', 'class_id', 'score'])

class ModelProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.edge = 'edgetpu' in model_path
        self.interpreter = self._initialize_interpreter()
        self.input_size = self._get_input_size()

    def _initialize_interpreter(self) -> Interpreter:
        if self.edge:
            return Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
        else:
            return Interpreter(model_path=str(self.model_path))

    def _get_input_size(self) -> Tuple[int, int]:
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_size = input_shape[1:3]  # Assuming the input shape is in the format [batch, height, width, channels]
        return input_size

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, self.input_size)
        normalized_image = resized_image.astype(np.uint8)
        input_data = np.expand_dims(normalized_image, axis=0)
        return input_data

    # ... The rest of the methods from your original code

    def extract_detected_objects(self, interpreter, min_score_thresh=0.5):
        output_details = interpreter.get_output_details()
        scores = interpreter.get_tensor(output_details[0]['index'])
        boxes = interpreter.get_tensor(output_details[1]['index'])
        num_detections = interpreter.get_tensor(output_details[2]['index'])
        classes = interpreter.get_tensor(output_details[3]['index'])
        detected_objects_list = []
        for i in range(int(num_detections[0])):
            if scores[0][i] >= min_score_thresh:
                bbox = boxes[0][i].tolist()
                normalized_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                detected_obj = DetectedObject(
                    bbox=normalized_bbox,
                    class_id=int(classes[0][i]),
                    score=scores[0][i]
                )
                detected_objects_list.append(detected_obj)

        return detected_objects_list

    def run_prediction_tflite(self, input_data, min_score_thresh=0.5):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        detected_objects = self.extract_detected_objects(self.interpreter, min_score_thresh=min_score_thresh)
        return detected_objects

    def run_prediction_edgetpu(self, frame, min_score_thresh=0.5):
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inference_size = self.input_size
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(self.interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(self.interpreter, min_score_thresh)

        # Normalize bounding boxes
        normalized_objs = []
        for obj in objs:
            normalized_bbox = [obj.bbox.xmin / inference_size[0], obj.bbox.ymin / inference_size[1],
                            obj.bbox.xmax / inference_size[0], obj.bbox.ymax / inference_size[1]]
            normalized_obj = DetectedObject(bbox=normalized_bbox, class_id=obj.id, score=obj.score)
            normalized_objs.append(normalized_obj)

        return normalized_objs
    
    def run_prediction(self, frame, min_score_thresh=0.5):
        preprocessed_input = self.preprocess_input(frame)
        if self.edge:
            detected_objects = self.run_prediction_edgetpu(frame, min_score_thresh=min_score_thresh)
        else:
            detected_objects = self.run_prediction_tflite(preprocessed_input, min_score_thresh=min_score_thresh)
        return detected_objects









'''
        
    def run_prediction(self, image: np.ndarray, detection_threshold: float):
        input_data = self.preprocess_input(image)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
        self.interpreter.invoke()
        detected_objects = self.extract_detected_objects(self.interpreter, min_score_thresh=detection_threshold)
        return detected_objects

def run_prediction(self, frame):
    preprocessed_input = self.preprocess_input(frame)
    if self.edge:
        detected_objects = self.run_prediction_edgetpu(frame)
    else:
        detected_objects = self.run_prediction_tflite(preprocessed_input)
    return detected_objects


            if edge:
                input_data = cv2.resize(frame.copy(), inference_size)
                cv2_im_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
                run_inference(interpreter, cv2_im_rgb.tobytes())
                objs = get_objects(interpreter, DETECTION_THRESHOLD)
                scale_x, scale_y = (width / inference_size[0], height / inference_size[1])
            else:
                input_data = preprocess_input(frame.copy(), input_size=inference_size)    
                run_prediction(interpreter, input_data)
                objs = extract_detected_objects(interpreter, min_score_thresh=DETECTION_THRESHOLD)
                scale_x, scale_y = width, height
                
                
                
# Normalize and batchify the image
def pre_process_image(image_np, type='normalized', input_size=(256,256)):
    if type=='normalized': 
        image = cv2.resize(image_np, input_size)
        image = np.expand_dims(image/255.0, axis=0).astype(np.float32)
    else:    
        image = np.expand_dims(image_np, axis=0).astype(np.uint8)
    return image
    '''