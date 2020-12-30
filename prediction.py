# %matplotlib inline
import numpy as np
import imutils
import cv2
# import matplotlib.pyplot as plt


CONF_THRESH, NMS_THRESH = 0.5, 0.5


weightsPath = 'my_data/weights/yolov3_8000.weights'
configPath = 'my_data/yolov3.cfg'
namesPath = 'my_data/classes.names'
image = 'img7.jpg'

# Load the network using openCV
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLOv3
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Read and convert the image to blob and perform forward pass
img = cv2.imread(image)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(output_layers)


class_ids, confidences, b_boxes = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESH:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            b_boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))


# Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

if len(indices) > 0:

    # Draw the filtered bounding boxes with their class to the image
    with open(namesPath, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        (x,y) = (b_boxes[index][0], b_boxes[index][1])
        (w,h) = (b_boxes[index][2], b_boxes[index][3])
        
        # Blur the ROI of the detected licence plate 
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w] ,(35,35),0)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "{}: {:.4f}".format("LP", confidences[index])
        cv2.putText(img, text, (x, y - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, .75 , (0, 255, 0), 1)

cv2.imshow('final',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
