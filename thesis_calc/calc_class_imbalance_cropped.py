import os
import numpy as np
import tensorflow as tf
from PIL import Image
from cmath import inf

annotated_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\cropped\\out\\mask2\\"

files = os.listdir(annotated_loc)
file_count = len(files)

# We will contain all the class data in a master list.
all_class_data = []

for f in files:
    target = annotated_loc + f
    print(target)
    np_data = np.load(target)
    
    # The classes are 0 (not in the given class) and 1 (is in the given class)
    classes = [0,0]
    
    for x in range(len(np_data)):
        for y in range(len(np_data[0])):
           pixel_class = np_data[x,y]
           classes[pixel_class]= classes[pixel_class] + 1
        
    print(classes)
    all_class_data.append(classes)

# We now have the counts of all classes.  Figure out some data from that.
highest_class_0_ratio = 0
highest_class_1_ratio = 0
lowest_class_0_ratio = inf
lowest_class_1_ratio = inf

total_class_0_ratio = 0
total_class_1_ratio = 0

for c in all_class_data:
    cls0 = c[0]
    cls1 = c[1]
    cls_all = cls0 + cls1
    
    class_0_ratio = float(cls0) / float(cls_all)
    class_1_ratio = float(cls1) / float(cls_all)
    
    total_class_0_ratio = total_class_0_ratio + class_0_ratio
    total_class_1_ratio = total_class_1_ratio + class_1_ratio
    
    if class_0_ratio > highest_class_0_ratio:
        highest_class_0_ratio = class_0_ratio
        
    if class_1_ratio > highest_class_1_ratio:
        highest_class_1_ratio = class_1_ratio

    if class_0_ratio < lowest_class_0_ratio:
        lowest_class_0_ratio = class_0_ratio
        
    if class_1_ratio < lowest_class_1_ratio:
        lowest_class_1_ratio = class_1_ratio

print("lowest ratio NOT in class: " + str(lowest_class_0_ratio))
print("lowest ratio in class: " + str(lowest_class_1_ratio))
print("highest ratio NOT in class: " + str(highest_class_0_ratio))
print("highest ratio in class: " + str(highest_class_1_ratio))
print("Average ratio NOT in class: " + str(total_class_0_ratio / float(len(all_class_data))))
print("Average ratio in class: " + str(total_class_1_ratio / float(len(all_class_data))))
