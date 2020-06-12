import os
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# Jon Kilgannon
# This code will read in the predicted images made by the models on Cloudlab.  An image was 
# made after each epoch for each learning rate.  This program will output a file that can 
# be read by an R program to make a graph.


def epoch_num_from_file_name(x):
    # Strips out the epoch number from the name of the predicted file.
    temp_str = x[10:]
    return(int(temp_str[:-4]))


print("Started.")

# How many epochs will we graph?
max_epoch = 70

# This is the class that the model was used to predict, on Cloudlab.
class_processed = 1

# The base location for everything being done today.
base_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\LR_comparisons\\"

# The location of the ground truth .npy file (the real mask for the given class)
ground_truth_file = base_loc + "image333prime.tif.npy"

# Read in the .npy file. We will check all the predicted files against it.
ground_truth = np.load(ground_truth_file)

# The length and width of ground_truth; cuts down on the number of calls
# to the len() function.
x_len = len(ground_truth)
y_len = len(ground_truth[0])

# Master list of correctnesses
correctness_for_LRs = []
incorrectness_for_LRs = []
aggregate_correctness_for_LRs = []

# This is the learning rate being worked on.
all_learning_rates = ['2', '3', '4', '5', '5x6' , '6', '7']
for learning_rate in all_learning_rates:
    # The directory containing the images predicted by the model as run on Cloudlab
    predicted_loc = base_loc + "predicted-lr" + str(learning_rate) + "\\"
    print("=============================")
    print(predicted_loc)
    
    # Get a list of the predicted masks
    files = os.listdir(predicted_loc)
    file_count = len(files)
    
    # Sort the data so that it is in order by epoch (we had read the files in, in whatever order
    # the OS decided to give it to us).
    files.sort(key = epoch_num_from_file_name)
    
    print(files)
    print("----------------------------")
    
    # We will contain all the "correctness" data in master lists.
    correctness_data = []       # Correct pixels per epoch
    incorrectness_data = []     # Incorrect pixels per epoch
    aggregate_correctness_data = []     # # correct minus # incorrect
    
    #current_epoch = 0
    
    for f in files:
        if epoch_num_from_file_name(f) > max_epoch:
            # Don't bother with the ones that are beyond the "edge of the graph" that we want
            continue
        
        # Read in the predicted image
        target = predicted_loc + f
        predicted_image = Image.open(target)
        predicted_array = np.asarray(predicted_image)
        
        # Count the correct/incorrect pixels. This could be done more quickly, but
        # it would take longer to figure out and test a clever algorithm than to 
        # come up with this simpler one.  Basically, a white pixel (255,255,255)
        # means "not a pixel in this class" and a different color means "is a
        # pixel in this class."  We use that to figure out the pixels' correctness.
        total_correct = 0
        total_incorrect = 0
        for x in range(x_len):
            for y in range(y_len):
                if (ground_truth[x,y] == class_processed) and (predicted_array[x,y,0] != 255) and (predicted_array[x,y,1] != 255) and (predicted_array[x,y,2] != 255):
                   total_correct += 1
                if (ground_truth[x,y] != class_processed) and (predicted_array[x,y,0] == 255) and (predicted_array[x,y,1] == 255) and (predicted_array[x,y,2] == 255):
                   total_correct += 1
                if (ground_truth[x,y] != class_processed) and (predicted_array[x,y,0] != 255) and (predicted_array[x,y,1] != 255) and (predicted_array[x,y,2] != 255):
                   total_incorrect += 1
                if (ground_truth[x,y] == class_processed) and (predicted_array[x,y,0] == 255) and (predicted_array[x,y,1] == 255) and (predicted_array[x,y,2] == 255):
                   total_incorrect += 1
                                       
        print(target + ", learning rate: " + str(learning_rate) + ", total correct: " + str(total_correct) + ", total incorrect: " + str(total_incorrect))
        correctness_data.append(total_correct)
        incorrectness_data.append(total_incorrect)
        aggregate_correctness_data.append(total_correct - total_incorrect)

    correctness_for_LRs.append(correctness_data)
    incorrectness_for_LRs.append(incorrectness_data)
    aggregate_correctness_for_LRs.append(aggregate_correctness_data)
    
# We now have a list of the number of correct/incorrect pixels in each image 
# Element n contains the correctness data for epoch n+1 for the given learning rate.

# Put the data we have into arrays that matplotlib's 3D tools can use.
# See https://pundit.pratt.duke.edu/wiki/Python:Plotting_Surfaces for some
# guidance (but not much guidance, sadly).

# Y-axis: the learning rates.  
# One of the learning rates is fractional (5 * 10e-6); give it a 
# representation on a log-10 scale.  log_10(5*10e-6) = 5.301
LRs = np.array([2, 3, 4, 5, 5.3, 6, 7])

# X-axis: the epochs
epochs = np.array(list(range(1,max_epoch+1)))

(x, y) = np.meshgrid(epochs, LRs)

# ******************* GRAPH BOTH THE CORRECTNESS AND  INCORRECTNESS DATA! *********************************************
#z = np.array(correctness_for_LRs)
#z = np.array(incorrectness_for_LRs)
z = np.array(aggregate_correctness_for_LRs)

# Make a 3D display
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')   

# Plot the data on it.
ax.plot_surface(x, y, z)

#ax.set(xlabel='Epoch', ylabel='-log_10(Learning Rate)', zlabel='Correct Pixels', title='Correctness by Learning Rate and Epoch')
#ax.set(xlabel='Epoch', ylabel='-log_10(Learning Rate)', zlabel='Incorrect Pixels', title='Incorrectness by Learning Rate and Epoch')
ax.set(xlabel='Epoch', ylabel='-log_10(Learning Rate)', zlabel='Aggregate Correct Pixels', title='Aggregate Correctness by Learning Rate and Epoch')

fig.tight_layout()
#fig.savefig('PatchExOrig_py.png')

# Show the graph!
plt.show()

    
print("Done.")
