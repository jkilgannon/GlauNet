import os
import numpy as np
from PIL import Image
import sys

annotator = int(sys.argv[1])

#path_fundus = '/local/repository/training_data/other_annotators/' + str(annotator) +  '/fundus/'
#path_annotated = '/local/repository/training_data/other_annotators/' + str(annotator) +  '/mask1/'

path_fundus = '/local/repository/training_data/other_annotators/' + str(annotator) +  '/fundusvalidate/'
path_annotated = '/local/repository/training_data/other_annotators/' + str(annotator) +  '/maskvalidate1/'

out_path = '/outgoing/'
input_size = (160, 160, 3)
preferred_size = (160, 160)

#####################################

def main():
    # Place the fundus TIF in the <path_fundus> directory.
    files = os.listdir(path_fundus)
    files.sort()

    for f in files:
        if f.lower().endswith('.tif'):
            img = path_fundus + f

            empty_color= [255,255,255]   # white; used for the "not the right class" color.
            class_color = [0, 0, 0]      # black; color to make the pixels marked as being of the given class

            ground_truth_file = path_annotated + f + '.npy'
            ground_truth = np.load(ground_truth_file)
            ground_truth = ground_truth.reshape(preferred_size)

            # Print out the ground truth image.
            img = np.zeros((preferred_size[0], preferred_size[1], 3), dtype=np.uint8)
            correct_color = [40, 255, 0]      # green; color to make the pixels marked as being right
            for x in range(preferred_size[0]):
                for y in range(preferred_size[1]):
                    if ground_truth[x,y] == 1.0:
                        img[x,y] = correct_color
                    else:
                        img[x,y] = empty_color
                        
            img_file = Image.fromarray(img)            
            img_file.save(out_path + f + "_ground-truth_annotator-" + str(annotator) + ".png")
            
    print("Done")


# Launch!
main()
