//File: log_hist.h
#ifndef LOG_HIST_H
#define LOG_HIST_H

#include <math.h>

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>

/*
Takes the histogram of the logarithm (base 10) of vector x
*/




arma::urowvec log_hist(     const arma::vec &x,
                            double left_bound,
                            double right_bound,
                            int n_bins
                        )
{

    double m = arma::mean(x);

    int n = x.size();
    arma::vec log_norm_x(n);    // log of normalized x

    for(int i = 0; i < n; ++i){
        log_norm_x[i] = std::log10( x[i]/m );
    }


    arma::vec edges = arma::linspace<arma::vec>(left_bound, right_bound, n_bins);
    arma::uvec h = arma::histc(log_norm_x, edges );

    arma::urowvec h_row = arma::conv_to<arma::urowvec>::from(h);

    return h_row;

}



#endif
