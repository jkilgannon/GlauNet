# Counts the number of classes in the csv output
import numpy as np

fl_path = '/users/jk880380/'
input_file = fl_path + 'predicted.csv'

f = open(input_file, "r")
contents = f.read()
elements = contents.split(",")	
valid_list = []
count = 0
for num in elements:
    count = count + 1
    h = hash(str(num))
    if not h in valid_list:
        valid_list.append(h)

print(valid_list)
print(count)

print("Done")
