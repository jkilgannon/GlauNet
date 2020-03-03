import numpy as np

shaper = np.zeros((320,480,3), dtype = np.double)
desired_shape = type(shaper)

#fl = np.fromfile('predicted.csv', dtype = desired_shape)
fl = np.fromfile('predicted.csv', sep=",")

fl = fl.reshape((320,480,3))

print(type(fl))
print(fl.shape)

resized = np.zeros((320,480,3), dtype=np.single)

for x in range(320):
    for y in range(480):
        if fl[x,y,0] > fl[x,y,1] and fl[x,y,0] > fl[x,y,2]:
            resized[x,y] = 0
        if fl[x,y,1] > fl[x,y,0] and fl[x,y,1] > fl[x,y,2]:
            resized[x,y] = 1
        if fl[x,y,2] > fl[x,y,1] and fl[x,y,2] > fl[x,y,0]:
            resized[x,y] = 2

fl = np.savetxt('predicted_compiled.csv', resized.flatten(), delimiter=",")

print("Done")
