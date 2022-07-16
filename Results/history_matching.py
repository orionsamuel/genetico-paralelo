import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

if(not os.path.exists("Matchs")):
    os.system("mkdir Matchs")
else:
    os.system("rm -f Matchs/*")


inputFile_RealWater = open("../Input/agua.txt", "r")
inputFile_RealOil = open("../Input/oleo.txt", "r")
inputFile_RealGas = open("../Input/gas.txt", "r")

real_water = []
real_oil = []
real_gas = []

for line in inputFile_RealWater:
    real_water.append(float(line))

inputFile_RealWater.close()

for line in inputFile_RealOil:
    real_oil.append(float(line))

inputFile_RealOil.close()

for line in inputFile_RealGas:
    real_gas.append(float(line))

inputFile_RealGas.close()

for i in range(10): 
    water = []
    water2 = []
    oil = []
    oil2 = []
    gas = []
    gas2 = []
    real_water2 = []
    real_oil2 = []
    real_gas2 = []

    inputFile_Water = open("Output_Simulation/agua/"+str(i)+".txt", "r")
    inputFile_Oil = open("Output_Simulation/oleo/"+str(i)+".txt", "r")
    inputFile_Gas = open("Output_Simulation/gas/"+str(i)+".txt", "r")

    for line in inputFile_Water:
        water.append(float(line))
        
    inputFile_Water.close()

    for line in inputFile_Oil:
        oil.append(float(line))

    inputFile_Oil.close()

    for line in inputFile_Gas:
        gas.append(float(line))

    inputFile_Gas.close()

    for j in range(120):
        water2.append(water[j])
        oil2.append(oil[j])
        gas2.append(gas[j])
        real_water2.append(real_water[j])
        real_oil2.append(real_oil[j])
        real_gas2.append(real_gas[j])

    print(str(i)+" - MAPE Error Water: ", mean_absolute_percentage_error(real_water2, water2))
    print(str(i)+" - R² Error Water: ", r2_score(real_water2, water2))
    print(str(i)+" - MAPE Error Oil: ", mean_absolute_percentage_error(real_oil2, oil2))
    print(str(i)+" - R² Error Oil: ", r2_score(real_oil2, oil2))
    print(str(i)+" - MAPE Error Gas: ", mean_absolute_percentage_error(real_gas2, gas2))
    print(str(i)+" - R² Error Gas: ", r2_score(real_gas2, gas2))
    print("")

    output = open("Matchs/resultado_"+str(i)+".txt", "w")

    output.write("MAPE Error Water: "+str(mean_absolute_percentage_error(real_water2, water2))+"\n")
    output.write("R² Error Water: "+str(r2_score(real_water2, water2))+"\n")
    output.write("MAPE Error Oil: "+str(mean_absolute_percentage_error(real_oil2, oil2))+"\n")
    output.write("R² Error Oil: "+str(r2_score(real_oil2, oil2))+"\n")
    output.write("MAPE Error Gas: "+str(mean_absolute_percentage_error(real_gas2, gas2))+"\n")
    output.write("R² Error Gas: "+str(r2_score(real_gas2, gas2))+"\n")


    output.close()

    values = [[real_water2, water2],[real_oil2, oil2],[real_gas2, gas2]]
    
    count = 0
    for value in values:
        instances = []
        for j in range(len(real_water2)):
            instances.append((j+1)*250)
            j = j + 250

        if(count == 0):
            #plt.title("Water Flow")
            plt.subplots_adjust(left=0.13, right=0.78, bottom=0.1, top=0.95)
            plt.ylabel("Water Flow")
            plt.xlabel("Time (in hours)")
            #plt.yscale('log')
            plt.plot(instances, values[0][0], color='red', label ='Real')
            plt.plot(instances, values[0][1], color='purple', label ='Predicted', linestyle = '--')
            plt.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
            plt.savefig("Matchs/Matching Water_"+str(i)+" - Linhas.png")
        elif(count == 1):
            #plt.title("Oil Flow")
            plt.subplots_adjust(left=0.13, right=0.78, bottom=0.1, top=0.95)
            plt.ylabel("Oil Flow")
            plt.xlabel("Time (in hours)")
            #plt.yscale('log')
            plt.plot(instances, values[1][0], color='red', label ='Real')
            plt.plot(instances, values[1][1], color='purple', label ='Predicted', linestyle = '--')
            plt.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
            plt.savefig("Matchs/Matching Oil_"+str(i)+" - Linhas.png")
        else:
            #plt.title("Oil Flow")
            plt.subplots_adjust(left=0.13, right=0.78, bottom=0.1, top=0.95)
            plt.ylabel("Gas Flow")
            plt.xlabel("Time (in hours)")
            #plt.yscale('log')
            plt.plot(instances, values[2][0], color='red', label ='Real')
            plt.plot(instances, values[2][1], color='purple', label ='Predicted', linestyle = '--')
            plt.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
            plt.savefig("Matchs/Matching Gas_"+str(i)+" - Linhas.png")
        
        plt.clf()

        count = count + 1
