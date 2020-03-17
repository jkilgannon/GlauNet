import numpy as np

fl = np.fromfile('predicted.csv', sep=",")
fl = fl.reshape((320,480,3))

print(type(fl))
print(fl.shape)

resized = np.zeros((320,480,3), dtype=np.single)

count = [0,0,0]

for x in range(320):
    for y in range(480):
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
            
count_solo = [0,0,0]

for x in range(320):
    for y in range(480):
        if fl[x,y,0] > 0.1:
            count_solo[0] = count_solo[0] + 1
        if fl[x,y,1] > 0.1:
            count_solo[1] = count_solo[1] + 1
        if fl[x,y,2] > 0.1:
            count_solo[2] = count_solo[2] + 1

print(count_solo)

fl = np.savetxt('predicted_compiled.csv', resized.flatten(), delimiter=",")

print("Done")
