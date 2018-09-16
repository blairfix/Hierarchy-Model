//File: base_pay_sim.h
#ifndef BASE_PAY_SIM_H
#define BASE_PAY_SIM_H


#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <math.h>
#include <vector>
#include <random>
#include <algorithm>


#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>



/*
This function is virtually identical to the one found in base_pay_sim.h
The only difference is that I introduce a fudge factor to increase/decrease
the fitted gamma k parameter. This is used to increase/decrease inter-firm
income dispersion.
*/


arma::vec base_pay_sim( const arma::vec &base_pay_empirical,
                        int n_sim,
                        double fudge_factor
                        )
{

    int n = base_pay_empirical.size(); // number of firms in emprical sample

    // Get initial guess for k
    double sum = 0;
    double sum_log = 0;


    for(int i = 0; i < n; ++i){
        sum = sum + base_pay_empirical[i];
        sum_log = sum_log + log(base_pay_empirical[i]);
    }

    double s = log(sum/n) - sum_log/n;

    double k = (3 - s + pow( pow(s-3, 2) + 24*s, 0.5  ) )/(12*s)  ;
    double k_new;
    double error = 1;

    // Newton method to estimate k
    while(error > 0.00001){

        k_new = k -   (log(k) - boost::math::digamma(k) -s) / (1/k  - boost::math::polygamma(1, k) )  ;
        error = abs(k_new - k);
        k = k_new;

    }

    double theta = 1/(k*n)*sum; // get other gamma parameter,theta


    // fudge factor ********************************************************
    k = k*fudge_factor;


    // generate random gamma distribution for simulated firm base pay distribution
    arma::vec x(n_sim);

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::gamma_distribution<double> distribution(k, theta);

    for(int i = 0; i < n_sim; ++i){
        x[i] = distribution(gen);
    }


    return x;

}


#endif

