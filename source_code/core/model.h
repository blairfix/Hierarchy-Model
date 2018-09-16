//File: model.h
#ifndef MODEL_H
#define MODEL_H


#include <algorithm>
#include "exponents.h"
#include "hierarchy.h"

#include <math.h>
#include <random>
#include <vector>



#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>


/*

The model function creates a simulated income distribution.
Inputs are span of control parameters a and b, as well as the
lognormal parameter sigma, which determines the income inequality within
hierarchical levels of all firms (a, b, and sigma are the same
for all firms)

The following 4 vectors containing data for individual firms:

    base_emp_vec =  a vector of firm base level employment
    emp_vec =  a vector of firm total employment
    base_pay_vec = a vector of firm base level mean pay
    r_vec =  vector containing pay scaling parameters for each form

Three boolean variables --- firm_size, hierarchy, and power --- determine
optional output. If firm_size = true, the output matrix contains an
additional column indicating the size of the firm that each individual
is employed by. If hierarchy = true, the output matrix contains an
additional column with the hierarchical level of each individual.
Lastly, if power = true, the output matrix contains a column
with the power of each individual (number of subordinates + 1).

*/



arma::mat model(    double a,
                    double b,
                    const arma::vec &base_emp_vec,
                    const arma::uvec &emp_vec,
                    const arma::vec &base_pay_vec,
                    const arma::vec &r_vec,
                    double sigma,
                    bool firm_size = false,
                    bool hierarchy = false,
                    bool power = false
                    )

{

    //////////////////////////////////////////////////////////////////////////////////////////////
    // mandatory output
    int n_cols = 1; // keep track of number of columns in output matrix
    int pay_col = 0;       int p_index = 0;

    // optional outputs
    int firm_size_col;  int firm_size_index = 0;
    int hier_col;       int hier_index = 0;
    int power_col;      int power_index = 0;

    int lnorm_index = 0;


        // firm size export
        if( firm_size == true ){
            firm_size_col = n_cols;
            //names[n_cols] = "firm_size";
            n_cols++;
        }

        // hierarchical levels export
        if( hierarchy == true ){
            hier_col = n_cols;
            //names[n_cols] = "h_level";
            n_cols++;
        }


        // power export
        if( power == true ){
            power_col = n_cols;
            //names[n_cols] = "power";
            n_cols++;
        }


    // output matrix
    double n_people = arma::sum(emp_vec);
    arma::mat output( n_people, n_cols);



    ////////////////////////////////////////////////////////////////////////////////////////////////
    // begin model

    int   n_firm = base_emp_vec.size();
    double mu = log(1) - 0.5*pow(sigma, 2);


    // get span product and pay exponent
    arma::vec sprod = s_func(a, b);
    arma::uvec e_pay = p_func(a, b);


    // generate intra-hierarchical level lognormal distribution
    // to speed up the model, we sample from a pregenerated distribution,
    // with size equal to the largest base hierarchical level. This shortcut is faster
    // than generating a new random number for each individual
    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<> d(mu, sigma);


    int n_samp = base_emp_vec.max();
    arma::vec lnorm(n_samp);

    for(int i = 0; i < n_samp; ++i) {
        lnorm[i] =  d(gen);
    }



    ////////////////////////////////////////////////////////////////////////////////
    // loop over firms
    for(int firm_index = 0; firm_index < n_firm; firm_index++){

        double base = base_emp_vec[firm_index];
        double emp = emp_vec[firm_index];
        double r = r_vec[firm_index];
        double base_pay = base_pay_vec[firm_index];


        // employment model
        arma::uvec h = hierarchy_func(emp, base, sprod);

        int maximum = h[19];


        // pay model
        double p;  // pay


        //loop over hierarchical levels
        for (int i = 0; i < maximum; ++i){
            p = base_pay*std::pow( r, e_pay[i]) ;

            // loop over individuals
            for (int j = 0; j < h[i]; ++j){

                if(lnorm_index == n_samp){lnorm_index = 0;}     // check if at end of lnorm sample
                output(p_index, pay_col) = p*lnorm[lnorm_index];

                lnorm_index++;
                p_index++;

            }
        } // end loop over levels



        //////////////////////////////////////////////////////////////////////////////
        // optional calculations

        // firm size
        if( firm_size == true){
            for(int i = 0; i < emp; ++i){
                output(firm_size_index, firm_size_col) = emp;
                firm_size_index++;
            }
        }



        // hierarchical levels
        if(hierarchy == true){
            //loop over levels
            for (int i = 0; i < maximum; ++i){

                // loop over individuals
                for (int j = 0; j < h[i]; ++j){
                    output(hier_index, hier_col) = i + 1;
                    hier_index++;
                }
            } // end loop over levels
        } // end if


        // power
        if(power == true){

            double n_subord = 0;

            //loop over levels
            for (int i = 0; i < maximum; ++i){

                double power_mean = n_subord/h[i] + 1;

                // loop over individuals
                for (int j = 0; j < h[i]; ++j){
                    output(power_index, power_col) = power_mean;
                    power_index++;
                }

                n_subord = n_subord + h[i];

            } // end loop over levels
        } // end if




    } // end loop over firms


  return output;
}



#endif
