//File: ceo_ratio.h
#ifndef CEO_RATIO_H
#define CEO_RATIO_H

#include "core/exponents.h"
#include "core/hierarchy.h"
#include "utils/top_k.h"

#include <algorithm>
#include <math.h>
#include <vector>

 #define ARMA_DONT_USE_WRAPPER
 //#define ARMA_NO_DEBUG
 #define ARMA_DONT_USE_HDF5
 #include <armadillo>


/*
This function calculates the mean ceo pay ratio for the top  firms ranked by total payroll.
Input parameters:

    a & b = span of control parameters
    base_emp_vec = vector of employment in firm base hierarchical levels
    emp_vec = vector of firm employment
    base_pay_vec = vector of firm base level mean pay
    r_vec = vector of firm hierarchical pay scaling parameters
    n_top_firms = number of top firms to use
    mean_pay = mean pay of model

*/


double   mean_ceo_pay_ratio  (  double a,
                                double b,
                                const arma::vec &base_emp_vec,
                                const arma::uvec &emp_vec,
                                const arma::vec &base_pay_vec,
                                const arma::vec &r_vec,
                                int n_top_firms,
                                double mean_pay
                                )

{

    int   n_firm = base_emp_vec.size(); // number of firms
    arma::vec  firm_payroll(n_firm);
    arma::vec  ceo_pay(n_firm);


    // get span product and pay exponent
    arma::vec sprod = s_func(a, b);
    arma::uvec e_pay = p_func(a, b);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // loop over firms

    for(int firm_index = 0; firm_index < n_firm; firm_index++){

        double base = base_emp_vec[firm_index];
        double emp = emp_vec[firm_index];
        double r = r_vec[firm_index];
        double base_pay = base_pay_vec[firm_index];


        // employment model
        arma::uvec h = hierarchy_func(emp, base, sprod);
        int maximum = h[19];


        // calculate firm mean pay (weighted mean of p by h vectors)
        double total = 0;
        double total_w = 0;
        double p = 0;

        for(int i = 0; i < maximum; ++i){
            p = base_pay*std::pow( r, e_pay[i]) ;
            total += p * h[i];
            total_w += h[i];
        }


        firm_payroll(firm_index) = total;       // firm total payroll
        ceo_pay(firm_index) = p;                // ceo pay


    }

    arma::rowvec top_ceo = top_k(firm_payroll, ceo_pay, n_top_firms);       // get ceo pay for top payroll firms (proxy for sales)
    double mean_ceo_ratio = arma::mean(top_ceo/mean_pay);       // mean ceo pay ratio


    return mean_ceo_ratio;
}


#endif

