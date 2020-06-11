import os
import numpy as np
#import tensorflow as tf
from PIL import Image
#from cmath import inf

# This code will read in the predicted images made by the modely on Cloudlab.  An image was 
# made after each epoch for each learning rate.  This program will output a file that can 
# be read by an R program to make a graph.


def epoch_num_from_file_name(x):
    # Strips out the epoch number from the name of the predicted file.
    temp_str = x[10:]
    return(int(temp_str[:-4]))


print("Started.")

# This is the learning rate being worked on.
learning_rate = 4

# This is the class that the model was used to predict, on Cloudlab.
class_processed = 1

# The base location for everything being done today.
base_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\LR_comparisons\\"

# The directory containing the images predicted by the model as run on Cloudlab
predicted_loc = base_loc + "predicted-lr" + str(learning_rate) + "\\"
print(predicted_loc)

# The location of the ground truth .npy file (the real mask for the given class)
ground_truth_file = base_loc + "image333prime.tif.npy"

# Read in the .npy file. We will check all the predicted files against it.
ground_truth = np.load(ground_truth_file)

# The length and width of ground_truth; cuts down on the number of calls
# to the len() function.
x_len = len(ground_truth)
y_len = len(ground_truth[0])

# Get a list of the predicted masks
files = os.listdir(predicted_loc)
file_count = len(files)

# Sort the data so that it is in order by epoch (we read the files in, in whatever order
# the OS decided to give it to us in).
files.sort(key = epoch_num_from_file_name)

print("----------------------------")
print(files)
print("----------------------------")

# We will contain all the "correctness" data in a master list.  Put the
# count of pixels into element 0 of the list.  (The epoch numbering starts with 1,
# so this makes life somewhat simpler in the end.)
correctness_data = []
correctness_data.append(len(ground_truth) * len(ground_truth[0]))
#correctness_data = [0] * (len(ground_truth) * len(ground_truth[0]))
#correctness_data[0] = len(ground_truth) * len(ground_truth[0])

for f in files:
    target = predicted_loc + f
    
    # Read in the predicted image
    predicted_image = Image.open(target)
    predicted_array = np.asarray(predicted_image)
    
    # Count the correct/incorrect pixels. This could be done more quickly, but
    # it would take longer to figure out and test a clever algorithm than to 
    # come up with this simpler one.
    total_correct = 0
    total_incorrect = 0
    for x in range(x_len):
        for y in range(y_len):
            #print(ground_truth[x,y])
            #print(predicted_array[x,y,0])
            #print(predicted_array[x,y,1])
            #print(predicted_array[x,y,2])
            if (ground_truth[x,y] == class_processed) and (predicted_array[x,y,0] != 255) and (predicted_array[x,y,1] != 255) and (predicted_array[x,y,2] != 255):
               total_correct += 1
            if (ground_truth[x,y] != class_processed) and (predicted_array[x,y,0] == 255) and (predicted_array[x,y,1] == 255) and (predicted_array[x,y,2] == 255):
               total_correct += 1
            if (ground_truth[x,y] != class_processed) and (predicted_array[x,y,0] != 255) and (predicted_array[x,y,1] != 255) and (predicted_array[x,y,2] != 255):
               total_incorrect += 1
            if (ground_truth[x,y] == class_processed) and (predicted_array[x,y,0] == 255) and (predicted_array[x,y,1] == 255) and (predicted_array[x,y,2] == 255):
               total_incorrect += 1
                                   
    print(target + ", total correct: " + str(total_correct) + ", total incorrect: " + str(total_incorrect))
    correctness_data.append(total_correct)
    correctness_data.append(total_incorrect)


# We now have a list of the number of correct pixels in each image, with element 0 
# containing the count of all pixels in an image.  Element n contains the correctness
# data for epoch 0 for the given learning rate. Write out to a file.
outfile = open(base_loc + "correct_pixels-LR" + str(learning_rate) + ".txt", "w")

for lr_data in correctness_data:
    outfile.write(str(lr_data) + '\n')

outfile.close()
    
print("Done.")
