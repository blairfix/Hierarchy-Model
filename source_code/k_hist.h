//File: k_hist.h
#ifndef K_HIST_H
#define K_HIST_H


#include <algorithm>
#include <math.h>
#include <vector>

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>

/*
k_hist generates a relative histogram of modelled capitalist income.
Since the goal is to compare this to empirical (US) data, the first step is
to normalize the mean capitalist income of the model to the means of US data
(contained in k_parameter vector). We then get the relative histogram with for
incomes less than $100K, in with bin sizes of #2.5K.
*/




arma::rowvec k_hist(arma::vec &k_income, arma::vec k_parameters)
{

    // k_parameter[0] = paramter of capitalist income fraction vs power regression
    // k_parameter[1] = minimum mean of US dividend income
    // k_parameter[2] = maximum mean of US dividence income

    // get capitalist income
    int n_people = k_income.size();


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(k_parameters[1], k_parameters[2]);

    double mean_k_goal = dis(gen); // stochastic goal for mean income of capitalist income
    double mean_k_exist = arma::mean(k_income); // mean of unnormalized model data

    for(int i = 0; i < n_people; ++i){
        k_income[i] = k_income[i]/mean_k_exist*mean_k_goal;
    }




    // histogram function (only on income under $200k)
    arma::vec histogram = arma::zeros<arma::vec>(80);
    double bin_size = 2.5;


    for(int i = 0; i < n_people; ++i){
        if(k_income[i] < 200){
            int bin = (int)std::floor(k_income[i]/bin_size);
            histogram[bin] += 1;
        }
    }

    // get relative histogram
    arma::rowvec density(80);
    for(int i = 0; i < 80; ++i){
        density[i] = histogram[i]/(double)n_people;
    }


    return density;

}



#endif
