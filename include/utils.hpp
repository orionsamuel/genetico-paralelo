#include <string>

using namespace std;

const string inputOil = "../Input/oleo.txt";
const string inputWater = "../Input/agua.txt";
const string inputGas = "../Input/gas.txt";
const string simulationFile = "../Input/SPE1CASE1.DATA";
const string fileName = "SPE1CASE1";


#define SIZE_POPULATION 10
#define N_GENERATIONS 10

#define CROSSOVER_RATE 80
#define MUTATION_RATE 50

#define MIN_POROSITY 0.1
#define MAX_POROSITY 0.3

#define MIN_PERMEABILITY 50.0
#define MAX_PERMEABILITY 500.0

#define WATER_WEIGHT 0.2
#define OIL_WEIGHT 0.5
#define GAS_WEIGHT 0.3

#define N_PERMEABILITY 3
#define TOTAL_CELLS 300

#define N_METRICS 3

struct individual{
    double porosity;
    double permeability_x[N_PERMEABILITY];
    double permeability_y[N_PERMEABILITY];
    double permeability_z[N_PERMEABILITY];
    double error_rank;
};

struct result{
    double water;
    double oil;
    double gas;
};

struct mutationValue{
    double porosity;
    double permeability_x;
    double permeability_y;
    double permeability_z;
};


