//File: h_mean.h
#ifndef H_MEAN_H
#define H_MEAN_H


#include <vector>
#include <algorithm>
#include "gini_fast.h"

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>


/*
This function finds mean pay by hierarchical level.
Inputs are pay vector 'pay' and hierarchy vector 'h'.
*/

arma::rowvec  h_mean(const arma::vec &pay, const arma::vec &h)
{

    int h_max =  h.max();       // get maximum hierarchical level
    arma::rowvec h_mean(20);
    h_mean.fill(9999);

    double base_mean = arma::mean( pay.elem( arma::find(h == 1) ) );


    // loop over hierarchical levels
    for(int i = 1; i < h_max + 1; ++i){
        arma::uvec ids = arma::find(h == i);                    // Find indices
        arma::vec pay_select = pay.elem(ids);
        h_mean[i-1] = arma::mean(pay_select)/base_mean;        // mean income of hierarchical level
    }

    return h_mean;

}


#endif
