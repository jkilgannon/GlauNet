import os
import numpy as np
from cmath import inf

# For the trimmed images: Because the trimmed images are one-vs-many, you will need to
# run this program for each class (i.e. need to copy in the contents of maskN and maskvalidateN
# where N is [0..2], and noting down the avg pixels per file for each class. 
#
# The "images" are really saved NumPy arrays.  They're called images for historical reasons (the
# original class masks were bitmaps)

#annotated_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\glaucomaData\\segmented\\output-7\\inseg\\"

# For the full-sized images, copy contents of 
#  "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\glaucomaData\\segmented\\output-7\\inseg\\"
# and
#  "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\glaucomaData\\segmented\\output-7\\insegval\\"
# to this directory so we capture both the training and validation data.
#
# For the cropped images, instead copy contents of:
#   "E:\!data\college\2020 Spring\CSC620 - Deep Learning\cropped_all\3\maskN"
# and
#   "E:\!data\college\2020 Spring\CSC620 - Deep Learning\cropped_all\3\maskvalidateN"
# to this directory.
annotated_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\thesis\\temp_data\\"

files = os.listdir(annotated_loc)
file_count = len(files)

# We will contain all the class data in a master list.
all_class_data = []

# Count of total number of pixels in each class, among all files.
total_class_pixel_count = [0,0,0]

# Image size
img_x = 0
img_y = 0

for f in files:
    target = annotated_loc + f
    print(target)
    np_data = np.load(target)
    
    # The classes are 0, 1, and 2, which lend themselves nicely to using a simple list as the data type
    # The last element will contain the file name.
    classes = [0, 0, 0, f]
    
    img_x = len(np_data[0])
    img_y = len(np_data)
    
    for x in range(len(np_data)):
        for y in range(len(np_data[0])):
           pixel_class = np_data[x,y]
           classes[pixel_class]= classes[pixel_class] + 1
        
    print(classes)
    all_class_data.append(classes)
    
    for idx in range(3):
        total_class_pixel_count[idx] = total_class_pixel_count[idx] + classes[idx]

# We now have the counts of all classes.  Figure out some data from that.
highest_class_ratio = [0,0,0]
lowest_class_ratio = [inf,inf,inf]
total_class_ratio = [0,0,0]
highest_class_count = [0,0,0]
lowest_class_count = [inf,inf,inf]

# Contains names of files with min and max pixels for class 1.
min_file = ["","",""]
max_file = ["","",""]

for c in all_class_data:
    cls0 = c[0]
    cls1 = c[1]
    cls2 = c[2]
    cls_all = cls0 + cls1 + cls2
    
    cls_counts = [cls0, cls1, cls2]
    
    class_0_ratio = float(cls0) / float(cls_all)
    class_1_ratio = float(cls1) / float(cls_all)
    class_2_ratio = float(cls2) / float(cls_all)
    
    class_ratio = [class_0_ratio, class_1_ratio, class_2_ratio]
    
    for idx in range(3):
        total_class_ratio[idx] = total_class_ratio[idx] + class_ratio[idx]
        
    for idx in range(3):
        if class_ratio[idx] > highest_class_ratio[idx]:
            highest_class_ratio[idx] = class_ratio[idx]
        if class_ratio[idx] < lowest_class_ratio[idx]:
            lowest_class_ratio[idx] = class_ratio[idx]
        if cls_counts[idx] > highest_class_count[idx]:
            highest_class_count[idx] = cls_counts[idx]
            max_file[idx] = c[3]
        if cls_counts[idx] < lowest_class_count[idx]:
            lowest_class_count[idx] = cls_counts[idx]
            min_file[idx] = c[3]

for idx in range(3):
    print("Lowest ratio, class " + str(idx) + ": " + str(lowest_class_ratio[idx]))
    print("Highest ratio, class " + str(idx) + ": " + str(highest_class_ratio[idx]))
    print("Average ratio, class " + str(idx) + ": " + str(total_class_ratio[idx] / float(len(all_class_data))))
    print("Lowest pixel count, class " + str(idx) + ": " + str(lowest_class_count[idx]))
    print("Lowest pixel ratio, class " + str(idx) + ": " + str(lowest_class_count[idx] / float(img_x*img_y)))
    print("Lowest pixel count, class " + str(idx) +  " is in file: " + min_file[idx])
    print("Highest pixel count, class " + str(idx) + ": " + str(highest_class_count[idx]))
    print("Highest pixel ratio, class " + str(idx) + ": " + str(highest_class_count[idx] / float(img_x*img_y)))
    print("Highest pixel count, class " + str(idx) +  " is in file: " + max_file[idx])
    
    
print("------")
print(total_class_pixel_count)
for idx in range(3):
    print("Average pixels per file for class" + str(idx) + ": " + str(float(total_class_pixel_count[idx]) / float(file_count)))
