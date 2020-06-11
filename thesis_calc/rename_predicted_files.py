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

# This is the learning rate being worked on.
learning_rate = 5

# I could write something to get the largest file number, but I probably won't need
# to do this more than twice, so just giving the program the number is much faster
# in the long run.
#  THIS IS THE EPOCH OF THE LAST FILE IN THE FIRST BATCH OF FILES!
starting_epoch = 293

# The base location for everything being done today.
base_loc = "E:\\!data\\college\\2020 Spring\\CSC620 - Deep Learning\\LR_comparisons\\"

# The directory containing the images predicted by the model as run on Cloudlab
predicted_loc = base_loc + "predicted-lr" + str(learning_rate) + "\\"
print(predicted_loc)

predicted_part2_loc = predicted_loc + "part2\\"

# Get a list of the files in part 2.
files = os.listdir(predicted_part2_loc)

print("----------------------------")
print(files)
print("----------------------------")


for f in files:
    os.rename(predicted_part2_loc + f, predicted_part2_loc + new_file_name(f))
    print(predicted_part2_loc + f + " to " + predicted_part2_loc + new_file_name(f))
   
    
print("Done.")
