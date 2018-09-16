//File: gini_fast.h
#ifndef GINI_FAST_H
#define GINI_FAST_H


#include <algorithm>
#include <vector>

#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>


/*
gini calculates the gini index of the vector x. If corr = true,
the gini is corrected for small sample bias. Code is based off
of the gini function contained in the R 'ineq' package

**** Warning ****
For speed purposes, this function is passed a reference and will alter the original
input vector.

*/


double gini_fast(arma::vec &x, bool corr = false)
{


    int n = x.size();

    std::sort(x.begin(), x.end());
    double G = 0;

    for(int i = 0; i < n; i++){
        G = G + x[i]*(i+1);
    }

    double sum = arma::sum(x);

    G = 2*G/sum - (n + 1);

        if(corr){
          G = G/(n-1);
        } else {
          G = G/n;
        }

    if(x.size() == 1){G = 0;}

    return G;
}



#endif
