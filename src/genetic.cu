#include "../include/genetic.hpp"
#include "cuda_runtime.h"

__global__ void FillPopulation(individual *d_population, int size_population, int n_permeability){
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    if((i < size_population) && (j < n_permeability)){
        d_population[i].permeability_x[j] = 0;
        d_population[i].permeability_y[j] = 0;
        d_population[i].permeability_z[j] = 0;
        d_population[i].porosity = 0;
        d_population[i].error_rank = 0;
    }
}

genetic_algorithm::genetic_algorithm(){
    srand (time(0));
}

genetic_algorithm::~genetic_algorithm(){

}

void genetic_algorithm::FirstPopulation(){
    srand((unsigned)time(0));
    
    CreateResultDir(0);

    individual *d_population;

    this->population.resize(SIZE_POPULATION);

    dim3 blocks(SIZE_POPULATION,N_PERMEABILITY);

    cudaMalloc((void **)&d_population, SIZE_POPULATION * sizeof(individual));

    cudaMemcpy(d_population, this->population.data(), SIZE_POPULATION * sizeof(individual), cudaMemcpyHostToDevice);

    FillPopulation<<<1,blocks>>>(d_population, SIZE_POPULATION, N_PERMEABILITY);
    cudaDeviceSynchronize();

    cudaMemcpy(this->population.data(), &d_population, SIZE_POPULATION * sizeof(individual), cudaMemcpyDeviceToHost);

    cudaFree(d_population);

    for(int i = 0; i < SIZE_POPULATION; i++){
        this->population[i].porosity = Rand_double(MIN_POROSITY, MAX_POROSITY);
        for(int j = 0; j < N_PERMEABILITY; j++){
            this->population[i].permeability_x[j] = Rand_double(MIN_PERMEABILITY, MAX_PERMEABILITY);
            this->population[i].permeability_y[j] = Rand_double(MIN_PERMEABILITY, MAX_PERMEABILITY);
            this->population[i].permeability_z[j] = Rand_double(MIN_PERMEABILITY, MAX_PERMEABILITY);
        }
    }
 
    for(int i = 0; i < SIZE_POPULATION; i++){
        WriteSimulationFile(0, i, simulationFile, fileName, population);
    }
    
    Simulation(0, fileName);
    Fitness(0);
    sort(begin(this->population), end(this->population), Compare);
  
    WriteErrorFile(0, population);

    for(int i = 0; i < SIZE_POPULATION; i++){
        WriteSimulationFile(0, i, simulationFile, fileName, population);
    }
      
}

void genetic_algorithm::Fitness(int idIteration){
    if(idIteration == 0){
        for(int i = 0; i < SIZE_POPULATION; i++){
            string oilOutputResult = "../Output/"+to_string(idIteration)+"/oleo/"+to_string(i)+".txt";
            string waterOutputResult = "../Output/"+to_string(idIteration)+"/agua/"+to_string(i)+".txt";
            string gasOutputResult = "../Output/"+to_string(idIteration)+"/gas/"+to_string(i)+".txt";
            this->population[i].error_rank = activationFunction(waterOutputResult, oilOutputResult, gasOutputResult, realResults, idIteration, i);
        }
    }else{
        for(int i = SIZE_POPULATION; i < (SIZE_POPULATION + this->crossover_rate); i++){
            string oilOutputResult = "../Output/"+to_string(idIteration)+"/oleo/"+to_string(i)+".txt";
            string waterOutputResult = "../Output/"+to_string(idIteration)+"/agua/"+to_string(i)+".txt";
            string gasOutputResult = "../Output/"+to_string(idIteration)+"/gas/"+to_string(i)+".txt";
            this->population[i].error_rank = activationFunction(waterOutputResult, oilOutputResult, gasOutputResult, realResults, idIteration, i);
        }
    }
}

void genetic_algorithm::Init(){
    CreateOutputDir();
    
    string oilInputResult = ReadFileInput(inputOil);
    string waterInputResult = ReadFileInput(inputWater);
    string gasInputResult = ReadFileInput(inputGas);
    
    realResults = ConvertStringInputToDoubleResult(waterInputResult, oilInputResult, gasInputResult);    

    FirstPopulation();
    // int count = 1;
    // while(count < N_GENERATIONS){
    //     OtherPopulations(count);
    //     count++;
    // }

}