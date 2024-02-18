    import tensorflow as tf
    tf.config.list_physical_devices('GPU')
    import tensorflow as tf
    import tensorflow_hub as hub
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # Load the pre-trained SSD MobileNet V2 model from TensorFlow Hub
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
    model = hub.load(model_url)

    def run_inference(model, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = model(input_tensor)

        return detections

    def visualize_detections(image, detections, l,confidence_threshold=0.5):
        # Visualize the detections on the image
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        scores = detections['detection_scores'][0].numpy()

        class_mapping = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
        59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
        64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
        73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
        78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
        84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
        }
        for i in range(len(boxes)):
            if scores[i] >= confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                xmin, xmax, ymin, ymax = int(xmin * image.shape[1]), int(xmax * image.shape[1]), int(ymin * image.shape[0]), int(ymax * image.shape[0])

                # Draw bounding box and label
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
                # print(class_mapping[classes[i]])
                l[classes[i]]+=1
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # plt.imshow(image)
        # plt.show()

    import json
    # import cv2
    # Your file path
    file_path = "./drive/MyDrive/data1/train.jsonl"
    lst=[]
    c=0
    text_file_path_b = "freqbad.txt"
    text_file_path_g = "freqgood.txt"
    file = open("freqbad.txt", 'w')
    file.close()  # Close the file explicitly
    file = open("freqgood.txt", 'w')
    file.close()  # Close the file explicitly
    # Read and parse each line separately
    with open(file_path, 'r') as file:
        for line in file:
            c+=1
            print(c)
            data_dict = json.loads(line)
            # print(data_dict)
            v=data_dict['id']
            image_path = "./drive/MyDrive/data1/img/"  # Replace with the actual path to your image
            s=str(v)
            if(len(s)==4):
                s='0'+s
            image_path+=s
            image_path+=".png"
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform object detection
            detections = run_inference(model, image_rgb)
            l=[]
            for i in range(91):
                l.append(0)
            # Visualize the results
            visualize_detections(image_rgb, detections,l)
            lst.append(l)

            if(data_dict['label']):
                with open(text_file_path_b, 'a') as text_file:
                    text_file.write(str(l) + '\n')
            else:
                with open(text_file_path_g, 'a') as text_file:
                    text_file.write(str(l) + '\n')

    # print(lst)
