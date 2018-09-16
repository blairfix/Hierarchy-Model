//File: k_func.h
#ifndef K_FUNC_H
#define K_FUNC_H


#include <algorithm>
#include <math.h>
#include <vector>

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>



/*
k_func calculates capitalist income for individuals in the pay vector (a vector of total income).
Capitalist income fraction is modelled as a logarithmic function of power (number of subordinates +1).
Resulting capitalist income is equal to pay * capitalist income fraction.

k_parameters are determined from regressions on Execucomp CEO data.
*/


arma::vec k_function(arma::vec pay, arma::vec power, arma::vec k_parameters)
{

    // k_parameter[0] = parameter of capitalist income fraction vs power regression

    // get capitalist income
    int n_people = pay.size();
    arma::vec k_income(n_people);

    for(int i = 0; i < n_people; ++i){
        k_income[i] = pay[i] * k_parameters[0]*std::log(power[i]);

        // fudge
        //k_income[i] = pay[i] * k_parameters[0]*pow(std::log(power[i]), 2);
    }


    return k_income;

}



#endif
