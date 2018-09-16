//File: pdf.h
#ifndef PDF_H
#define PDF_H


#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>

/*
Gets the probability density function (relative histogram) of x
*/




arma::rowvec pdf(   const arma::vec &x,
                    double left_bound,
                    double right_bound,
                    int n_bins
                        )
{

    int n = x.size();
    double bin_width = (right_bound - left_bound) / (double) n_bins;

    arma::vec edges = arma::linspace<arma::vec>(left_bound, right_bound, n_bins);
    arma::uvec counts = arma::histc(x, edges );


    arma::rowvec density(n_bins);

    for(int i = 0; i < n_bins; ++i){
        density[i] = (double)  counts[i] / (double) n / bin_width ;
    }


    return density;

}



#endif
