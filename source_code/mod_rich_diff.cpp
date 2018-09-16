#include "base_fit.h"
#include "base_pay_sim.h"
#include "boot_sigma.h"
#include "boot_span.h"
#include "change_dir.h"
#include "fit_model.h"
#include "log_hist.h"
#include "model.h"
#include "pdf.h"
#include "rpld.h"
#include "r_sim.h"
#include "sample.h"



#include <algorithm>
#include <boost/progress.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>


#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_HDF5
#include <armadillo>


/*
This code implements the hierarchy model and calculates mean
hierarchical power by income percentile.
*/


int main()
{
    std::cout  << "Rich Are Powerful Model" << std::endl;


    // model parameters
    int n_iterations = 1000;    // number of model iterations

    int n_firms = 1000000;      // number of firms in simulation
    double tol = 0.01;          // error tolerance for ceo ratio fit

    int n_bins = 1000;    // number of percentiles

    // load  data
    /////////////////////////////////////////////////////////////
    change_directory("executables", "data");

    arma::mat s_empirical;
    s_empirical.load("s_empirical.txt");

    arma::vec g_empirical;
    g_empirical.load("g_empirical.txt");

    arma::mat compustat;
    compustat.load("compustat.txt");

    arma::uvec  comp_employment = arma::conv_to<arma::uvec>::from(compustat.col(0));
    arma::vec   comp_ceo_ratio = compustat.col(1);
    arma::vec   comp_mean_pay = compustat.col(2);



    // output matrices
    //////////////////////////////////////////////////////////////////////////////////

    arma::mat power_mean(n_iterations, n_bins);


    //********************************************************************************************
    //********************************************************************************************
    // MODEL


    auto start = std::chrono::system_clock::now();
    boost::progress_display show_progress(n_iterations);
    #pragma omp parallel for firstprivate(s_empirical, g_empirical, comp_employment, comp_ceo_ratio, comp_mean_pay)

    for(int iteration = 0; iteration < n_iterations; iteration++){

        // tune model with compustat data
        ////////////////////////////////////////////////////////////////
        arma::vec coef = boot_span(s_empirical.col(0), s_empirical.col(1));


        double a = coef[0];
        double b = coef[1];
        double sigma = boot_sigma(g_empirical);


        arma::vec comp_base = base_fit(a, b, comp_employment);
        arma::mat comp_r = fit_model(a, b, comp_base, comp_employment, comp_ceo_ratio, comp_mean_pay, tol);

        // keep only firms with ceo ratio fit within error tolerance
        arma::uvec  ids = find(comp_r.col(1) < tol);
        arma::mat   comp_r_good = comp_r.rows(ids);
        arma::uvec  comp_employment_good = comp_employment(ids);



        // simulation
        //////////////////////////////////////////////////////////////////
        arma::uvec  sim_employment = rpld(n_firms, 1, 2.01, 2300000, 2300000, true); // power law firm size distribution
        arma::vec   sim_base = base_fit(a, b, sim_employment); // fit base level
        arma::vec   sim_base_pay = base_pay_sim(comp_r_good.col(2), n_firms); // modelled base pay distribution
        arma::vec   sim_r = r_sim(comp_employment_good, comp_r_good.col(0), sim_employment); // modelled pay scaling


        //****************************************************************
        // hf model     (hierarchy and intrafirm dispersion)
        arma::mat   mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r, sigma, false, false, true);


            // mean power by percentile
            arma::uvec sort_id = sort_index(mod.col(0), "descend");

            arma::uvec column(1);
            column[0] = 1;
            arma::vec power_sort = mod(sort_id, column);


            arma::vec percentile = arma::logspace<arma::vec>(-6, 0 , n_bins + 1  );
            arma::vec breaks_dec = percentile % arma::linspace<arma::vec>(0, power_sort.size() -1, n_bins + 1  );
            arma::uvec breaks = arma::conv_to< arma::uvec >::from(breaks_dec);



            for(int i = 0; i < n_bins; ++i){

                power_mean(iteration, i) = arma::mean( power_sort.subvec( breaks[i], breaks[i+1]  ));

            }




        ++show_progress;
   }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout  << "elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    std::cout << std::endl;



    // output results
    ////////////////////////////////////////////////////////////////
    change_directory("data", "results");


    power_mean.save("power_mean_percentile.csv", arma::csv_ascii);



	return 0;
}



