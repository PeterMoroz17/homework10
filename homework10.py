# Step 0
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1
folder = 'C:/r_d/car_data2'
frames = os.listdir(folder)
idx = frames.index('frame_0300.png')

tracker_type="KCF"
if tracker_type == 'KCF':
    # Step 2
    # KCF works better with a bigger bbox
    x, y = 610, 325
    x1, y1 = 740, 450
    w = x1 - x
    h = y1 - y
    search = 50
    tracker = cv2.TrackerKCF_create()

if tracker_type == "CSRT":
    # Step 5
    x, y = 640, 365
    x1, y1 = 710, 430
    w = x1 - x
    h = y1 - y
    search = 50
    tracker = cv2.TrackerCSRT_create()

img = cv2.imread(os.path.join(folder, frames[idx]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bbox = (x, y, w, h)
ok = tracker.init(img, bbox)

# Step 3
for i in range(idx, idx + 40) :
    img = cv2.imread(os.path.join(folder, frames[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    ok, bbox = tracker.update(img)
    print(ok, bbox)
    
    # Step 4
    x, y = bbox[0], bbox[1]
    w, h = bbox[2], bbox[3]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    plt.imshow(img)
    plt.axis(False)
    plt.show()
    plt.pause(0.1)

# Step 6
# 1. Yes, unlike KCF, CSRT changes bbox width and height
# 2. CSRT works better than KCF, it doesn't lose the car when camera changes position