#include <random>
#include <iostream>
#include <cmath>
using namespace std;
#include <stdio.h>
#include "methods.h"
#include "noise.h"
#include "privBayes_model.h"


string c_get_model(const int* data, const string& config, double eps, double theta, int seed, int m, int n) {

    table tbl(data, config, true, m, n);

    engine eng(seed);                       //deterministic engine with a random seed

    //Vanilla privbayes implementation from the paper
    bayesian bayesian1(eng, tbl, eps, theta, 1);
    string m1 = bayesian1.print_model();
    return m1;
}

