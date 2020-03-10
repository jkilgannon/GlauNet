import numpy as np

shaper = np.zeros((160,160,3), dtype = np.double)
desired_shape = type(shaper)

#fl = np.fromfile('predicted.csv', dtype = desired_shape)
fl = np.fromfile('predicted.csv', sep=",")

fl = fl.reshape((160,160,3))

print(type(fl))
print(fl.shape)

resized = np.zeros((160,160,3), dtype=np.single)

count = [0,0,0]

for x in range(160):
    for y in range(160):
        if fl[x,y,0] > fl[x,y,1] and fl[x,y,0] > fl[x,y,2]:
            resized[x,y] = 0
            count[0] = count[0] + 1
        if fl[x,y,1] > fl[x,y,0] and fl[x,y,1] > fl[x,y,2]:
            resized[x,y] = 1
            count[1] = count[1] + 1
        if fl[x,y,2] > fl[x,y,1] and fl[x,y,2] > fl[x,y,0]:
            resized[x,y] = 2
            count[2] = count[2] + 1

print(count)

fl = np.savetxt('predicted_compiled.csv', resized.flatten(), delimiter=",")

print("Done")
