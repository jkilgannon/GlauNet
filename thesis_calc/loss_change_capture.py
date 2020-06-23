import os
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pandas as pd
import copy

# This program lifts the loss rates from the end of each epoch out of 
# the output.txt files that are the output of glaunet.py and reload_model.py 
# (the Python programs that create the model on Cloudlab). The loss rates are 
# saved in a CSV with the columns: annotator, epoch_num, training loss, and
# validation loss.

# General form of the loss file:
#  TensorFlow setup (discard; unimportant to this program)
#  Three lines of "++++++++++++++" inserted by glaunet.py as a separator.
#  Header of an epoch: "Epoch X/Y\n\n", with X and Y being integers, X <= Y.
#  Several dozen lines of training losses, one per batch.
#  Validation loss.
#  Header of next epoch (then more lines of training losses)
# 
# Validation loss.  We are looking for a line of the form:
# Epoch nnnnn: saving model to /outgoing/best_model_incremental-cls1-runNNNNNNN.h5
# nnnnn is a zero-padded integer, the epoch number.
# The capital N's stand in for one or more digits [0..9].  This is the run number, which
# is immaterial.
#
# Training loss.  We are looking for several dozen lines of the form:
# X/Y [============================>.] - ETA: 3s - loss: 45.5706 - soft_loss: -44.5706
# X and Y are integers.  X less than or equal to Y.
# We care about the loss.  Capture all of them, average them to get the training loss.
#
# When we get to the end of the file, if we haven't found a Validation Loss section after 
# a set of Training Losses, discard the Training Losses as this is an epoch that
# was interrupted because the server was about to end its reservation time.

print("Started.")

# Base dir where the directories of annotator data can be found
base_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\!!results\\training_logs\\"

# File where we output the CSV
output_file = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\!!results\\training_logs\\loss_data.csv"
outf = open(output_file, "w")

# A list of the directory names that hold the annotators' data.
annotators = ['1', '2', '3', '4', '5', '6', '4_xfer', '5_xfer']
#annotators = ['1', '2', '3', '4', '5', '6']
#annotators = ['4_xfer', '5_xfer']

# Will build lists of the data here, to make into graphs
loss_data = []
val_loss_data = []

for annotator in annotators:
    # Directory name
    ann_dir = base_loc + annotator + "\\"
    
    # Get list of subdirectories in the directory.
    subdirs = os.listdir(ann_dir)
    subdirs.sort()
    
    epoch_count = 1
    
    loss_by_ann = []
    val_loss_by_ann = []
    
    # Run through the directories.
    for subdir in subdirs:
        data_file = ann_dir + subdir + "\\output.txt"
        
        # Open the file containing loss info
        f = open(data_file, "r")
        
        # Get a handler to pull lines out of the file.
        ln = f.readlines()
        
        prevline = ""
        first_epoch = True        
        
        for l in ln:
            if l[:5] == "Epoch" and l[-3:-1] != "h5":
                if first_epoch:
                    first_epoch = False
                else:
                    loss_field_loc = prevline.find(' loss: ')
                    soft_loss_field_loc = prevline.find(' - soft_loss: ')
                    
                    loss = float(prevline[loss_field_loc+6 : soft_loss_field_loc])
                    
                    # Original code is: return (1 - jaccard) * 100, so divide by 100 to get Jaccard Distance (1 - Jacc index)
                    loss = loss / 100.0

                    val_loss_field_loc = prevline.find(' val_loss: ')
                    val_soft_loss_field_loc = prevline.find(' - val_soft_loss: ')
                    
                    validation_loss = float(prevline[val_loss_field_loc+11 : val_soft_loss_field_loc])

                    # Original code is: return (1 - jaccard) * 100, so divide by 100 to get Jaccard Distance (1 - Jacc index)
                    validation_loss = validation_loss / 100.0
                    
                    print(annotator + ", " + str(epoch_count) + ", " + str(loss) + ", " + str(validation_loss))
                    outf.write(annotator + ", " + str(epoch_count) + ", " + str(loss) + ", " + str(validation_loss) + '\n')
                    
                    # Save the losses into lists for this annotator's data
                    loss_by_ann.append(loss)
                    val_loss_by_ann.append(validation_loss)
            
                    epoch_count += 1
                    
            prevline = l
        
        # Close the loss info file.
        f.close()

    # We have a list of all the losses by this annotator; put them into 2-d lists
    loss_data.append(copy.deepcopy(loss_by_ann))
    val_loss_data.append(copy.deepcopy(val_loss_by_ann))
    
outf.close()

print('CSV file has been created: ' + output_file)

##################################
# Make graphs.
for annotator_index in range(len(annotators)):
    plt.plot(loss_data[annotator_index], color='k', linewidth=0.7)
    plt.plot(val_loss_data[annotator_index], color='b',  linewidth=0.3)
    
    current_annotator = annotators[annotator_index]
    if current_annotator == '4_xfer' or current_annotator == '5_xfer':
        plt.title('Training and Validation Loss per Epoch, Annotator ' + str(current_annotator)[0] + ", with Transfer Learning")
    else:
        plt.title('Training and Validation Loss per Epoch, Annotator ' + str(current_annotator))

    plt.xlabel('Epoch')
    plt.ylabel('Loss (Jaccard Distance)')
    plt.show()

print("Done.")
