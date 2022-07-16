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

__global__ void FillChildren(individual* d_children, int crossover_rate, int n_permeability){
    int i = threadIdx.x;
    int j = threadIdx.y;

    if((i < crossover_rate) && (j < n_permeability)){
        d_children[i].porosity = 0;
        d_children[i].error_rank = 0;
        d_children[i].permeability_x[j] = 0;
        d_children[i].permeability_y[j] = 0;
        d_children[i].permeability_z[j] = 0;
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
        #pragma omp parallel for
        for(int j = 0; j < N_PERMEABILITY; j++){
            this->population[i].permeability_x[j] = Rand_double(MIN_PERMEABILITY, MAX_PERMEABILITY);
            this->population[i].permeability_y[j] = Rand_double(MIN_PERMEABILITY, MAX_PERMEABILITY);
            this->population[i].permeability_z[j] = Rand_double(MIN_PERMEABILITY, MAX_PERMEABILITY);
        }
    }
 
    #pragma omp parallel for
    for(int i = 0; i < SIZE_POPULATION; i++){
        WriteSimulationFile(0, i, simulationFile, fileName, population);
    }
    
    Simulation(0, fileName);
    Fitness(0);
    sort(begin(this->population), end(this->population), Compare);
  
    WriteErrorFile(0, population);

    #pragma omp parallel for
    for(int i = 0; i < SIZE_POPULATION; i++){
        WriteSimulationFile(0, i, simulationFile, fileName, population);
    }
      
}

void genetic_algorithm::OtherPopulations(int idIteration){
    Crossover();

    CreateResultDir(idIteration);

    #pragma omp parallel for
    for(int i = 0; i < SIZE_POPULATION; i++){
        system(Command("cp ../Output/"+to_string(idIteration-1)+"/"+to_string(i)+"-"+fileName+".DATA ../Output/"+to_string(idIteration)+"/"+to_string(i)+"-"+fileName+".DATA"));
    }

    #pragma omp parallel for
    for(int i = SIZE_POPULATION; i < (SIZE_POPULATION + this->crossover_rate); i++){
        WriteSimulationFile(idIteration, i, simulationFile, fileName, population);
    }

    Simulation(idIteration, fileName);
    Fitness(idIteration);
    sort(begin(this->population), end(this->population), Compare);

    WriteErrorFile(idIteration, population);

    system(Command("rm -f ../Output/"+to_string(idIteration)+"/*.DATA"));

    #pragma omp parallel for
    for(int i = SIZE_POPULATION; i < (SIZE_POPULATION + this->crossover_rate); i++){
        this->population.pop_back();
    }

    #pragma omp parallel for
    for(int i = 0; i < SIZE_POPULATION; i++){
        WriteSimulationFile(idIteration, i, simulationFile, fileName, population);
    }

}

void genetic_algorithm::Fitness(int idIteration){
    if(idIteration == 0){
        #pragma omp parallel for
        for(int i = 0; i < SIZE_POPULATION; i++){
            string oilOutputResult = "../Output/"+to_string(idIteration)+"/oleo/"+to_string(i)+".txt";
            string waterOutputResult = "../Output/"+to_string(idIteration)+"/agua/"+to_string(i)+".txt";
            string gasOutputResult = "../Output/"+to_string(idIteration)+"/gas/"+to_string(i)+".txt";
            this->population[i].error_rank = activationFunction(waterOutputResult, oilOutputResult, gasOutputResult, realResults, idIteration, i);
        }
    }else{
        #pragma omp parallel for
        for(int i = SIZE_POPULATION; i < (SIZE_POPULATION + this->crossover_rate); i++){
            string oilOutputResult = "../Output/"+to_string(idIteration)+"/oleo/"+to_string(i)+".txt";
            string waterOutputResult = "../Output/"+to_string(idIteration)+"/agua/"+to_string(i)+".txt";
            string gasOutputResult = "../Output/"+to_string(idIteration)+"/gas/"+to_string(i)+".txt";
            this->population[i].error_rank = activationFunction(waterOutputResult, oilOutputResult, gasOutputResult, realResults, idIteration, i);
        }
    }
}

void genetic_algorithm::Crossover(){
    individual *d_children;

    dim3 blocks(crossover_rate,N_PERMEABILITY);

    cudaMalloc((void **)&d_children, crossover_rate * sizeof(individual));

    cudaMemcpy(d_children, &this->children, crossover_rate * sizeof(individual), cudaMemcpyHostToDevice);

    FillChildren<<<1,blocks>>>(d_children, crossover_rate, N_PERMEABILITY);
    cudaDeviceSynchronize();

    cudaMemcpy(&this->children, &d_children, crossover_rate * sizeof(individual), cudaMemcpyDeviceToHost);

    cudaFree(d_children);

    for(int i = 0; i < crossover_rate; i++){
        this->children[i].porosity = 0;
        this->children[i].error_rank = 0;
        #pragma omp parallel for
        for(int j = 0; j < N_PERMEABILITY; j++){
            this->children[i].permeability_x[j] = 0;
            this->children[i].permeability_y[j] = 0;
            this->children[i].permeability_z[j] = 0;
        }
    }

    int percent = rand() % 2;
    if(percent == 0){
        int count = 0;
        while(count < crossover_rate){
            this->children[count].porosity = this->population[count].porosity;
            this->children[count + 1].porosity = this->population[count + 1].porosity;
            #pragma omp parallel for
            for(int i = 0; i < N_PERMEABILITY; i++){
                this->children[count].permeability_x[i] = this->population[count + 1].permeability_x[i];
                this->children[count].permeability_y[i] = this->population[count + 1].permeability_y[i];
                this->children[count].permeability_z[i] = this->population[count + 1].permeability_z[i];

                this->children[count + 1].permeability_x[i] = this->population[count].permeability_x[i];
                this->children[count + 1].permeability_y[i] = this->population[count].permeability_y[i];
                this->children[count + 1].permeability_z[i] = this->population[count].permeability_z[i];
            }
            count = count + 2;
        }
    }else {
        int count = 0;
        while(count < crossover_rate){
            this->children[count].porosity = this->population[count].porosity;
            this->children[count + 1].porosity = this->population[count + 1].porosity;
            #pragma omp parallel for
            for(int i = 0; i < N_PERMEABILITY; i++){
                this->children[count].permeability_x[i] = this->population[count].permeability_x[i];
                this->children[count].permeability_y[i] = this->population[count + 1].permeability_y[i];
                this->children[count].permeability_z[i] = this->population[count + 1].permeability_z[i];

                this->children[count + 1].permeability_x[i] = this->population[count + 1].permeability_x[i];
                this->children[count + 1].permeability_y[i] = this->population[count].permeability_y[i];
                this->children[count + 1].permeability_z[i] = this->population[count].permeability_z[i];
            }
            count = count + 2;
        }
    }

    //Mutation();

    for(int i = 0; i < crossover_rate; i++){
        this->children[i].porosity = floor(this->children[i].porosity * 100) / 100;
        #pragma omp parallel for
        for(int j = 0; j < N_PERMEABILITY; j++){
            this->children[i].permeability_x[j] = floor(this->children[i].permeability_x[j] * 100) / 100;
            this->children[i].permeability_y[j] = floor(this->children[i].permeability_y[j] * 100) / 100;
            this->children[i].permeability_z[j] = floor(this->children[i].permeability_z[j] * 100) / 100;
        }
    }
    
    for(int i = 0; i < this->crossover_rate; i++){
        this->population.push_back(children[i]);
    }
}

void genetic_algorithm::Mutation(){
    for(int i = 0; i < this->mutation_rate; i++){
        int gene = rand() % 3;
        int tunning = rand() % 2;

        mutationValue newValue;

        if(tunning == 0){
            newValue = RandMutationValue(this->children[i], gene, true);
        }else{
            newValue = RandMutationValue(this->children[i], gene, false);
        }

        this->children[i].porosity = newValue.porosity;
        this->children[i].permeability_x[gene] = newValue.permeability_x;
        this->children[i].permeability_y[gene] = newValue.permeability_y;
        this->children[i].permeability_z[gene] = newValue.permeability_z;
        
    }
}

void genetic_algorithm::Init(){
    CreateOutputDir();
    
    string oilInputResult = ReadFileInput(inputOil);
    string waterInputResult = ReadFileInput(inputWater);
    string gasInputResult = ReadFileInput(inputGas);
    
    realResults = ConvertStringInputToDoubleResult(waterInputResult, oilInputResult, gasInputResult);    

    FirstPopulation();
    int count = 1;
    while(count < N_GENERATIONS){
        OtherPopulations(count);
        count++;
    }

}