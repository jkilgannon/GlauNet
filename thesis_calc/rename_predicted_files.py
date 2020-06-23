import os

# Bulk-rename the predicted-mask files.


def epoch_num_from_file_name(x):
    # Strips out the epoch number from the name of the predicted file, x.
    temp_str = x[10:]
    return(int(temp_str[:-4]))

def new_file_name(x):
    # Returns a new name for the filename in argument x.
    new_epoch = epoch_num_from_file_name(x) + starting_epoch
    return("predicted_" + str(new_epoch) + ".png")
    

print("Started.")

###################################################################################
## This is the learning rate being worked on. (used when finding the learning rate)
###################################################################################
#learning_rate = '5x6'

###########################################################################
### This is the annotator being worked on (used when training by annotator)
###########################################################################
# IN PROGRESS - FILES BEING RENAMED
# done.
# COMPLETED - ALL FILES RENAMED
#annotator = 1
#annotator = 2
#annotator = 3           # This annotator is done, and was in fact done with the learning_rate version of the program
#annotator = 4
#annotator = 5
#annotator = '4_xfer'   
#annotator = '5_xfer'
#annotator = 6
# NOT STARTED - START WHEN TRAINING FINISHES
# done.

# I could write something to get the largest file number, but I probably won't need
# to do this more than twice, so just giving the program the number is much faster
# in the long run.
#  THIS IS THE EPOCH OF THE LAST FILE IN THE FIRST BATCH OF FILES!
starting_epoch = 1330

# The base location for everything being done today.
base_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\!!results\\mask_images\\"
#base_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\LR_comparisons\\"

# The directory containing the images predicted by the model as run on Cloudlab
predicted_loc = base_loc + str(annotator) + "\\"
#predicted_loc = base_loc + "predicted-lr" + str(learning_rate) + "\\"
print(predicted_loc)

predicted_part2_loc = predicted_loc + "part2\\"
predicted_part2_update_loc = predicted_loc 

# Get a list of the files in part 2.
files = os.listdir(predicted_part2_loc)

print("----------------------------")
print(files)
print("----------------------------")

for f in files:
    os.rename(predicted_part2_loc + f, predicted_part2_update_loc + new_file_name(f))
    print(predicted_part2_loc + f + " to " + predicted_part2_update_loc + new_file_name(f))
    
print("Done.")
