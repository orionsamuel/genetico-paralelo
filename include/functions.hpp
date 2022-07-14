#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <random>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>
#include "utils.hpp"

using namespace std;

class functions{
    public:

    void Simulation(int idIteration, string file);
    double Rand_double(double min, double max);
    double Max(double num1, double num2);
    double Min(double num1, double num2);
    const vector<string> split(const string& s, const char& c);
    void CreateOutputDir();
    string ReadFileInput(string file);
    vector<result> ConvertStringInputToDoubleResult(string water, string oil, string gas);
    const char* Command(string inputCommand);
    void CreateResultDir(int idIteration);
    void WriteSimulationFile(int idIteration, int interator, string inputFile, string file, vector<individual> population);
    void WriteErrorFile(int idIteration, vector<individual> population);
    mutationValue RandMutationValue(individual children, int gene, bool soma);
    double activationFunction(string waterResult, string oilResult, string gasResult, vector<result> results, int idIteration, int iterator);

    static bool Compare(const individual n, const individual m){
        return n.error_rank < m.error_rank;
    }

};