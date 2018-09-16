//File: fit_power_law.h
#ifndef FIT_POWER_LAW_H
#define FIT_POWER_LAW_H

#include "sample.h"
#include <math.h>

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>


/*
This function uses the maximum likelihood method to find
the best fit power law exponent for the tail of vector 'pay'
where the tail is defined by the cutoff percentiles in 'percentile_vec'.
The function picks a random sample cutoff from 'percecntile_vec'
*/

double fit_power_law(  arma::vec pay, arma::vec percentile_vec)
{

    arma::vec sample = sample_no_replace(percentile_vec, 1);
    double percentile = sample[0];


    int n_total = pay.size();
    int n_top = percentile*pay.size();
    int top_begin = n_total - n_top;

    std::nth_element(pay.begin(), pay.end() - n_top, pay.end());

    double xmin = arma::min( pay.subvec(top_begin, n_total-1) );
    double sum_log = 0;

    for(int i = top_begin; i < n_total ; ++i){
        sum_log = sum_log + std::log( pay[i] / xmin );
    }


    double alpha = 1 + n_top/sum_log;

    return alpha;

}



#endif
