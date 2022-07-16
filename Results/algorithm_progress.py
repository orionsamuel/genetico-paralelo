import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

error = []

for i in range(10):
    inputFile = open("../Output/"+str(i)+"/error.txt")
    count = 0
    for line in inputFile:
        if(count == 0):
            error.append(float(line))
        count = count + 1

population = []
    
for j in range(10):
    population.append(j+1)

plt.figure(figsize=(10, 4), dpi=300)
plt.title("Genetic Progress")
plt.ylabel("Best-So-Far Misfit Value")
plt.xlabel("Iterations")
#plt.yscale('log')
#plt.xscale('log')

plt.plot(population, error, color='red', label ='Error')
plt.legend(loc = 'upper right')
plt.savefig("Genetic Progress.png")
