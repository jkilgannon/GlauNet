import numpy as np

input_size = (160, 160, 3)

fl = np.fromfile('predicted.csv', sep=",")
fl = fl.reshape(input_size)

print(type(fl))
print(fl.shape)

#resized = np.zeros((320,480,3), dtype=np.single)
#resized_max = np.zeros((320,480,3), dtype=np.single)
resized = np.zeros(input_size, dtype=np.uint8)
resized_max = np.zeros(input_size, dtype=np.uint8)

count = [0,0,0]

for x in range(input_size[0]):
    for y in range(input_size[1]):
        if fl[x,y,0] > fl[x,y,1] and fl[x,y,0] > fl[x,y,2]:
            resized[x,y,0] = 11
            count[0] = count[0] + 1
        if fl[x,y,1] > fl[x,y,0] and fl[x,y,1] > fl[x,y,2]:
            resized[x,y,1] = 22
            count[1] = count[1] + 1
        if fl[x,y,2] > fl[x,y,1] and fl[x,y,2] > fl[x,y,0]:
            resized[x,y,2] = 33
            count[2] = count[2] + 1

print(count)
            
count_solo = [0,0,0]

for x in range(input_size[0]):
    for y in range(input_size[1]):
        if fl[x,y,0] > 0.00001:
            resized_max[x,y,0] = 11
            count_solo[0] = count_solo[0] + 1
        if fl[x,y,1] > 0.00001:
            resized_max[x,y,1] = 22
            count_solo[1] = count_solo[1] + 1
        if fl[x,y,2] > 0.00001:
            resized_max[x,y,2] = 33
            count_solo[2] = count_solo[2] + 1

print(count_solo)

#fl = np.savetxt('predicted_compiled.csv', resized.flatten(), delimiter=",")
#fl2 = np.savetxt('maxed_compiled.csv', resized_max.flatten(), delimiter=",")

resized.tofile('predicted_compiled.csv', sep=',')
resized_max.tofile('maxed_compiled.csv', sep=',')
print('predicted_compiled.csv: greatest class for each element')
print('maxed_compiled.csv: classes gt 0.00001 for each element')

print("Done")

