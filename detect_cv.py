# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.
TEST_DATA=../all_models
Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
from collections import namedtuple
import cv2
import os
import time
import numpy as np 

from tools.filter_detected_objects import filter_horizontal_distance, filter_overlap_threshold, filter_roi_ranges
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from tools.pedestrian_counter import LineCounter, LineCounterAnnotator

#trackers
from tools.sort.sort7 import Sort
from tools.byte.byte_tracker import BYTETracker
from tools.model_processor import ModelProcessor

# deepSort
from tools.deep_sort_pytorch.utils.parser import get_config
from tools.deep_sort_pytorch.deep_sort import DeepSort
from tools.deep_sort_pytorch.deep_sort.deep.feature_extractor import Extractor
from tools.deep_sort_pytorch.deep_sort.sort.detection import Detection

version = 5
edge = True
video = "/home/cappittall/Videos/seperatorlu.mp4"

history = 15
### SELECT MODEL AND ARGUMENTS !!
default_model_dir = f'../ekmeksay/models/{version}/'
default_model = f'modelv{version}_edgetpu.tflite' if edge else f'modelv{version}.tflite'
default_labels = 'ekmek-labels.txt'
model = os.path.join(default_model_dir, default_model)

frame_count = 0
start_time = 0
yon = "horizontal"
roi = 0.65
offset = 1
DETECTION_THRESHOLD = 0.4

# Sort tracker 
tracker = Sort(max_age=history + 1  , min_hits=10, iou_threshold=.30)

# Initialize bread types and bread counts
bread_types = [
    "Francala",
    "Tambugdayli",
    "Kepekli",
    "Cavdarli",
    "Ruseymli",
    "Yulafli",
    "Cekirdekli",
    "Tamtahilli",
]

bread_counts = {bread_type: 0 for bread_type in bread_types}
current_bread_type = bread_types[0]
separator_seen = False
counted_ids = set()

deepsort = None
encoder = None

def init_tracker():
    global deepsort, encoder
    cfg_deep = get_config()
    cfg_deep.merge_from_file("tools/deep_sort_pytorch/configs/deep_sort.yaml")

    encoder = Extractor(cfg_deep.DEEPSORT.REID_CKPT, use_cuda=True)

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
    
def main():
    global frame_count, start_time
   
    print(f'Loading Model: {model} \nVideo: {video}')
    cap = cv2.VideoCapture(video) 
    model_processor = ModelProcessor(model)
    model_processor.interpreter.allocate_tensors()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h,w,_ = frame.shape
        ratio = h/w
        cv2_im = cv2.resize(frame, (960, int(960 * ratio)))
        height, width, _ = cv2_im.shape

        #create scale instace throug infrence size and actual size
        if frame_count % history == 0 :
            frame_count = 0           
            print(f'İnf. time {((time.monotonic() - start_time) * 1000)/history:.0f}')             
            detected_objects = model_processor.run_prediction(cv2_im.copy(), min_score_thresh = DETECTION_THRESHOLD)
            
            # apply filters : 
            detected_objects = filter_roi_ranges(detected_objects, roi_range_start= roi - 0.05, roi_range_end=roi + 0.10)
            #detected_objects = filter_horizontal_distance(detected_objects, distance_threshold_factor = 0.005)        
            #detected_objects = filter_overlap_threshold(detected_objects, overlap_threshold=0.6)
            
            start_time = time.monotonic()                        
            # detections = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], obj.score, obj.class_id] for obj in objs for bbox in [obj.bbox]]) if len(objs) > 0 else np.empty((0, 7))
            
            detections = np.array([det_obj.bbox for det_obj in detected_objects])
            scores = np.array([det_obj.score for det_obj in detected_objects])
            label_ids = np.array([det_obj.class_id for det_obj in detected_objects])
            
            abs_detections = np.array([[x1 * width, y1 * height, x2 * width, y2 * height] for x1, y1, x2, y2 in detections])

            
            tracked_objects = tracker.update(abs_detections, scores, label_ids) 



        """
        detection_images = [cv2_im[int(y0):int(y1), int(x0):int(x1), :] for x0, y0, x1, y1 in abs_detections]
        features = encoder(detection_images)
        detections = [Detection(bbox, score, label_id, feature) for bbox, score, label_id, feature in zip(abs_detections, scores, label_ids, features)]
        confidences = np.array([d.confidence for d in detections])
        oids = np.array([d.oid for d in detections])
        ori_img = cv2_im.copy()
        bbox_tlwh = np.array([det.tlwh for det in detections])
        deepsort.update(bbox_tlwh, confidences, oids, ori_img)
        tracked_objects = [[*det.to_tlbr(), det.track_id, det.confidence, det.label] for det in deepsort.tracker.tracks if det.is_confirmed() and det.time_since_update < 1]
        ## """   
                        
        cv2_im = append_objs_to_img(cv2_im.copy(), tracked_objects, int(height * roi))
        cv2.imshow(f'frame Edge ? {model_processor.edge}', cv2_im)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def append_objs_to_img(frame, trdata, roiline):
    global current_bread_type, separator_seen
    heigh, width, _ = frame.shape
    
    for tr in trdata:
        print(tr)
        x0_, y0_, x1_, y1_, trackID, score, labelid = tr        
        x0, y0, x1, y1 = int(x0_ * 1), int( y0_ * 1), int(x1_ * 1), int( y1_ * 1)
        cx, cy =int((x0 + x1)/2 ), int((y0 + y1)/2 )
           
        percent = int(100 * score)
        labell = f'{percent}% \n{trackID:.0f}\n{labelid}' #.format(percent, labels.get(obj.id, obj.id)[:1])
        color = (0, 255, 0) if labelid == 0.0 else (255, 0, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
        cv2.rectangle(frame, (cx - offset, cy - offset), (cx + offset, cy + offset), color, 1)
        
        yy = y0+10
        for label in labell.split('\n'):
            cv2.putText(frame, label, (x0, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            yy += 15
        
        if cy >= roiline and trackID not in counted_ids:
            if labelid == 1.0:  # Separator
                if not separator_seen:
                    # Update the bread type to the next one
                    separator_seen = True
                    current_bread_type = bread_types[
                        (bread_types.index(current_bread_type) + 1) % len(bread_types)
                    ]
            else:
                # Bread
                separator_seen = False
                # Increment the count for the current bread type
                # count the bread and update the counts dictionary
                bread_counts[current_bread_type] += 1
                counted_ids.add(trackID)
                    
   
    cv2.line(frame, (0, roiline), (frame.shape[1], roiline), (0,0,255),1) 
    

    # Display bread counts on the frame
    y = 0
    for bread_type, count in bread_counts.items():
        y += 20
        text = f"{bread_type} : {count} -{len(trdata)}"
        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2 if bread_type==current_bread_type else 1)
    return frame

if __name__ == '__main__':
    main()
    
