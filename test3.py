from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # Note: ultralytics.yolo.utils.plotting is deprecated
import time
import torch

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained YOLOv8l model
model = YOLO('yolov8l.pt')
model.to(device)

#Now starting for Task 1 Other parts

import json

# Path to the JSON file
json_file_path = "data/train.jsonl"

# Dictionary to store image information
image_data = {}

# Open and read the JSON file
with open(json_file_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
    
    # Iterate through each image entry in the JSON data
        # print(entry)
        image_id = int(entry['id'])
        image_path = "data/"+entry['img']
        label = int(entry['label'])
        text = entry['text']
        
        # Store image information in dictionary
        image_data[image_id] = {
            'image_id' : image_id,
            'image_path': image_path,
            'label': label,
            'text': text
        }
object_detection_results={}

for key,elem in image_data.items():
    # print(elem)
    # if (elem['image_path']):
    results = model(elem['image_path'])
    class_ids=[]
    confs=[]
    bboxs=[]
    
    for result in results:
        for conf,class_id,bbox in zip(result.boxes.conf,result.boxes.cls,result.boxes.xyxy):
            if conf > 0.5:
                class_ids.append(class_id)
                confs.append(conf)
                bboxs.append(bbox)
    object_detection_results[key]={
        'class_ids': class_ids,
        'confs': confs,
        'bboxs': bboxs,
        'label': elem['label']
    }

import json
annotations_file = 'annotations/instances_train2017.json'  # Adjust the path as needed
with open(annotations_file, 'r') as f:
    coco_annotations = json.load(f)

# Extract class labels from COCO annotations
class_labels = {}
for category in coco_annotations['categories']:
    class_labels[category['id']] = category['name']
  
class_ids_hate_weight={}
for category in coco_annotations['categories']:
    class_ids_hate_weight[category['id']] = {
        'curr_val' : int(0),
        'freq_total' : int(0),
    }
for key,img_result in object_detection_results.items():
    for index,class_id in enumerate(img_result['class_ids']):
        class_ids_hate_weight[class_id]['freq_total']+=1
        if (img_result['label']==1):
            class_ids_hate_weight[class_id]['curr_val']-=1
        else:
            class_ids_hate_weight[class_id]['curr_val']+=1


print(class_ids_hate_weight)
with open("trained.json", 'w') as f:
    json.dump(class_ids_hate_weight, f)
    
with open("object_details.json", 'w') as f:
    json.dump(object_detection_results, f)
    
import matplotlib.pyplot as plt


object_ids = list(class_ids_hate_weight.keys())
obj_names=[]
for elem in object_ids:
    obj_names.append(class_labels[elem])
freq_totals = [entry['freq_total'] for entry in class_ids_hate_weight.values()]

# Plot the graph
# plt.figure(figsize=(10, 6))
plt.bar(obj_names, freq_totals, color='blue')
plt.xlabel('Object ID')
plt.ylabel('Frequency Total')
plt.title('Object ID vs Frequency Total')
plt.grid(True)
plt.show()
