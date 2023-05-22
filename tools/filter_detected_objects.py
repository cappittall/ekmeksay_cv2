
def filter_roi_ranges(detected_objects, roi_range_start=0.1, roi_range_end=0.9 ): 
        filtered_objs= [obj for obj in detected_objects if 
                                roi_range_start <= ((obj.bbox[1] + obj.bbox[3])/2) <= roi_range_end]
        return filtered_objs
        
def filter_horizontal_distance(detected_objects, distance_threshold_factor = 0.05 ):
   # Sort detected_objects by x1 coordinate
    detected_objects = sorted(detected_objects, key=lambda x: x.bbox[0])

    # Calculate average horizontal distance between detected objects
    distances = []
    for i in range(len(detected_objects) - 1):
        x1_1, y1_1, x2_1, y2_1 = detected_objects[i].bbox
        x1_2, y1_2, x2_2, y2_2 = detected_objects[i + 1].bbox
        distance = x1_2 - x2_1
        distances.append(distance)

    average_distance = sum(distances) / len(distances)

    # Filter out detected objects with significantly smaller horizontal distance to their neighbors
    filtered_detected_objects = [detected_objects[0]]
    
    for i in range(1, len(detected_objects) - 1):
        x1_1, y1_1, x2_1, y2_1 = detected_objects[i - 1].bbox
        x1, y1, x2, y2 = detected_objects[i].bbox
        x1_2, y1_2, x2_2, y2_2 = detected_objects[i + 1].bbox

        distance_to_prev = x1 - x2_1
        distance_to_next = x1_2 - x2

        if (distance_to_prev >= average_distance * distance_threshold_factor
                and distance_to_next >= average_distance * distance_threshold_factor):
            filtered_detected_objects.append(detected_objects[i])

    filtered_detected_objects.append(detected_objects[-1])  # Always keep the last object
    
    return filtered_detected_objects

def filter_overlap_threshold(detected_objects, overlap_threshold=0.5):
    # Sort detected_objects by x1 coordinate
    detected_objects = sorted(detected_objects, key=lambda x: x.bbox[0])

    # Calculate the overlapping area between consecutive bounding boxes
    overlaps = []
    for i in range(len(detected_objects) - 1):
        x1_1, y1_1, x2_1, y2_1 = detected_objects[i].bbox
        x1_2, y1_2, x2_2, y2_2 = detected_objects[i + 1].bbox
        overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        overlaps.append(overlap)

    # Filter out detected objects with a high overlapping area
    filtered_detected_objects = [detected_objects[0]]

    for i in range(1, len(detected_objects) - 1):
        x1, y1, x2, y2 = detected_objects[i].bbox
        area = (x2 - x1) * (y2 - y1)

        if overlaps[i - 1] / area < overlap_threshold and overlaps[i] / area < overlap_threshold:
            filtered_detected_objects.append(detected_objects[i])

    filtered_detected_objects.append(detected_objects[-1])  # Always keep the last object

    return filtered_detected_objects