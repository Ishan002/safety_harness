#safety_harness
from pathlib import Path
import torch
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from customsort import *
# import DataBase_SFH
# dbdata = DataBase_SFH.dbconnect("localhost", "root","toor","safety_harness", "harness_monitoring")

# This list will store the information to be displayed on the web page
# detection_info = []
sort_tracker = None
def tracker_initialize():
    global sort_tracker
    sort_max_age = 7
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    return sort_tracker

def draw_boxes(img, bbox, masks, identities=None, categories=None, names=None, offset=(0, 0), rs=None):
    for i, (box, mask_coords) in enumerate(zip(bbox, masks.xy)):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        label = names[cat]
        colorcode=[(80,255, 30),(86, 86, 255), (80,200, 160), (40,255,249),(80,200, 160), (40,255,249)]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), colorcode[cat], 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 10, y1), colorcode[cat], -1)
        cv2.putText(img, f'{id}:{label}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 2)
        
        # Overlay mask on the frame using alpha blending
        mask = np.array(mask_coords).astype(np.int32)
        mask_color =colorcode[cat]
        alpha = 0.5  # Transparency level
        overlay = img.copy()
        cv2.fillPoly(overlay, [mask], mask_color)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img




def detect(src, Camname):
    sort_tracker = tracker_initialize()
    device = torch.device('cuda')
    model = YOLO('sfh_09_04_24.pt').to(device)
    classnames = ["harness_with_hook", "harness_without_hook", "noharness"]

    cap = cv2.VideoCapture(src)
    # framenum = 0

# Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the video file
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_video1.mp4', fourcc, fps, (frame_width, frame_height))


    id_start_time = {}
    id_end_time = {}    
    identity_category_map = {}
    id_tracked = set()  # Set to keep track of currently tracked IDs
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            results = model(source=img, stream=True)
            for result in results:
                detections = np.empty((0, 6))
                bboxes = result.boxes
                for box in bboxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confd = box.conf[0].cpu().numpy()
                    class_ = box.cls[0].cpu().numpy()
                    class_ = int(class_)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    current_detections = np.array([x1, y1, x2, y2, confd, class_])
                    detections = np.vstack((detections, current_detections))

                tracked_dets = sort_tracker.update(detections)
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    masks=result.masks
                    # print(masks)
                    for identity, category in zip(identities, categories):
                        identity_category_map[identity] = category
                        id_tracked.add(identity)  
                    img = draw_boxes(img, bbox_xyxy,masks, identities, categories, classnames)
                    
                    # Update start times for each ID
                    current_time = time.time()
                    for identity in identities:
                        if identity not in id_start_time:
                            id_start_time[identity] = current_time
                  

                    # Update end time for IDs that are not currently detected
                    for identity in id_tracked.copy():
                        if identity not in identities:
                            id_end_time[identity] = current_time
                            id_tracked.remove(identity)  # Remove from tracked set
                            # Calculate duration for the ID and add to database
                            duration = current_time - id_start_time[identity]
                            category = int(identity_category_map.get(identity))
                            class_name = classnames[category]
                            # dbdata.add_dbdata([Camname, identity, class_name, duration])
                            #print(f"ID: {identity}, Duration: {duration} seconds,Class name: {class_name}")
                            # detection_info.append({"Camname": Camname, "ID": identity, "Duration": duration, "Class Name": class_name})
            # Write the frame to the video file
            out.write(img)
            cv2.imshow('Camname', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break  # 1 millisecond
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    

  


if __name__ == '__main__':
    source1 = "pv (2).mp4"
    Camname1 = "Default1"

    with torch.no_grad():
        detect(source1, Camname1)
