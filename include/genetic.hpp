#include <iostream>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include <random>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include "functions.hpp"

using namespace std;

class genetic_algorithm: public functions{
    private:
    vector<individual> population;
    vector<result> realResults;
    individual children[((SIZE_POPULATION * CROSSOVER_RATE) / 100)];
    int crossover_rate = ((SIZE_POPULATION * CROSSOVER_RATE) / 100);
    int mutation_rate = ((((SIZE_POPULATION * CROSSOVER_RATE) / 100) * MUTATION_RATE) / 100);
    string decimal[2];

    public:
    genetic_algorithm();
    ~genetic_algorithm();

    void Init();
    void FirstPopulation();
    void OtherPopulations(int idIteration);
    void Fitness(int idIteration);
    void Crossover();
    void Mutation(); 
};
