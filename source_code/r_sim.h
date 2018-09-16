//File: r_sim.h
#ifndef R_SIM_H
#define R_SIM_H

#include <math.h>


#include <math.h>
#include <random>
#include <vector>



#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>

/*
r_sim is a function that generates a simulated distribution of
pay scaling parameters. The function is designed to inputs
of parameters fitted to Compustat firms.

    employment = Compustat firm employment
    r = fitted pay scaling value for each Compustat firm

Fitting the hierarchical model to Compustat firms generally results
in r values that decrease in dispersion as employment increases. This dispersion
is modelled as lognormal, and the decline is assumed to be linear in with
the log of employment. This regression is then used to inform the lognormal scale
parameter sigma for each firm.

The third input, sim_employment, is a vector of simulated firm sizes
that is inputed into the model to generate a random pay scaling parameter
for each firm.

*/



arma::vec r_sim(   const arma::uvec &employment,
                   const arma::vec &r,
                   const arma::uvec &sim_employment)
{

    arma::vec emp = arma::conv_to<arma::vec>::from(employment); // convert employment to arma::vec

    int n = r.size();

    // subtract 1 from r to model with lognormal dist
    arma::vec r0(n);

    for(int i = 0; i < n; ++i){
        r0[i] = r[i] - 1;
    }

    double mu_r =    arma::mean(arma::log(r0));


    // group firms into rounded log of employment
    double bin_factor = 3;  // greater = more employment bins

    arma::vec group =  arma::round( bin_factor*arma::log(emp) )/bin_factor ;
    arma::vec group_unique = arma::unique(group);
    int n_group = group_unique.size();
    arma::vec sigma(n_group);

    // get standard deviation of log(r0) for each group
    // this is equivalent to sigma parameter for a lognormal distribution

    for(int i = 0; i < n_group; ++i){

        arma::uvec ids = find(group == group_unique[i]); // Find indices
        arma::vec r0_sub = r0.elem(ids);     // subset r.0
        sigma[i] = arma::stddev(arma::log(r0_sub));

    }

    // fill response matrix
    arma::mat X(n_group, 2);
    X.ones();
    X.col(1) = group_unique;


    // regression on sigma vs group (log of employment)
    arma::vec coef = arma::solve(X, sigma);


    // generate simulated r distribution
    // each firm gets sigma based on employment
    // every firm gets same mu

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    n = sim_employment.size();
    arma::vec r_output(n);

    for(int i = 0; i < n; ++i){

        double sigma_firm = coef[0] + coef[1]*std::log(sim_employment[i]);
        r_output[i] = std::exp(  sigma_firm*d(gen) + mu_r) + 1;

    }


    return r_output;
}



#endif
