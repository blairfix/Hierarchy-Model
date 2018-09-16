//File: power_mean.h
#ifndef POWER_MEAN_H
#define POWER_MEAN_H


#include <vector>
#include <algorithm>


#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>


/*
*/

arma::rowvec  power_mean(const arma::vec &pay, const arma::vec &power)
{

    arma::vec power_group = arma::round(arma::log(power));
    arma::vec power_group_unique = arma::unique(power_group);

    int n_bins = power_group_unique.size();
    arma::rowvec power_mean(40);
    power_mean.fill(9999);

    double base_mean = arma::mean( pay.elem( arma::find(power_group == 0) ) );


    // loop over power bins
    for(int i = 0; i < n_bins; ++i){

        double power_find = power_group_unique[i];

        arma::uvec ids = arma::find(power_group == power_find);    // Find indices
        arma::vec pay_select = pay.elem(ids);
        power_mean[i] = arma::mean(pay_select)/base_mean;        // mean income of power bin
    }

    return power_mean;

}


#endif
