# Counts the number of classes in the csv output
import numpy as np

fl_path = '/users/jk880380/'
input_file = fl_path + 'predicted.csv'

f = open(input_file, "r")
contents = f.read()
elements = contents.split(",")	

valid_list = []
elements_list = []
count_list = []

totalcount = 0
for num in elements:
    h = hash(str(num))
    if not h in valid_list:
        valid_list.append(h)
        elements_list.append(num)
        count_list.append(0)
    
    idx = valid_list.index(h)
    count_list[idx] = count_list[idx] + 1
    totalcount = totalcount + 1

print(valid_list)
print(elements_list)
print(count_list)
print(totalcount)

print("Done")
