#include "core/base_fit.h"
#include "core/base_pay_sim.h"
#include "core/boot_sigma.h"
#include "core/boot_span.h"
#include "core/fit_model.h"
#include "core/model.h"
#include "core/rpld.h"
#include "core/r_sim.h"



#include "utils/change_dir.h"
#include "utils/ceo_pay_ratio.h"
#include "utils/fit_power_law.h"
#include "utils/gini_fast.h"
#include "utils/k_frac_percentile.h"
#include "utils/k_func.h"
#include "utils/k_hist.h"
#include "utils/lorenz.h"
#include "utils/log_hist.h"
#include "utils/pdf.h"
#include "utils/power_mean.h"
#include "utils/sample.h"
#include "utils/sample_index.h"
#include "utils/top.h"
#include "utils/top_k.h"


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
This code implements the hierarchy model with a firm size that varies
stochasticaly between iterations (via the power law exponent).
The output is the gini index of hierarchical power concentration.
This forms the basis for the energy-hierarchy model.
*/


int main()
{
    std::cout  << "Power Concentration Model" << std::endl;

    // model parameters
    int n_iterations = 10000;   // number of model iterations

    int n_firms = 1000000;      // number of firms in simulation
    double tol = 0.01;          // error tolerance for ceo ratio fit



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

    arma::vec   k_parameters;
    k_parameters.load("k_parameters.txt");


    // output matrix
    //////////////////////////////////////////////////////////////////////////////////
    arma::mat result(n_iterations, 2);



    //********************************************************************************************
    //********************************************************************************************
    // MODEL

    auto start = std::chrono::system_clock::now();
    boost::progress_display show_progress(n_iterations);
    #pragma omp parallel for firstprivate(s_empirical, g_empirical, comp_employment, comp_ceo_ratio, comp_mean_pay, k_parameters)

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

            // stochastic firm size dist exponent
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_alpha(1.8, 4);

        double alpha = dis_alpha(gen);

        arma::uvec  sim_employment = rpld(n_firms, 1, alpha, 2300000, 2300000, true); // power law firm size distribution
        arma::vec   sim_base = base_fit(a, b, sim_employment); // fit base level
        arma::vec   sim_base_pay = base_pay_sim(comp_r_good.col(2), n_firms); // modelled base pay distribution
        arma::vec   sim_r = r_sim(comp_employment_good, comp_r_good.col(0), sim_employment); // modelled pay scaling


        arma::mat   mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r, sigma, false, false, true); // main model


        // get stats on sample of 1 million people
        double  sample_size = 1000000;

        arma::vec power_sample = sample_no_replace(mod.col(1), sample_size);

        double mean_firm_size = (double) arma::sum(sim_employment) / sample_size;


        double power_gini = gini_fast(power_sample);

        result(iteration, 0) =  mean_firm_size;
        result(iteration, 1) =  power_gini;


        ++show_progress;
   }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout  << "elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    std::cout << std::endl;



    // output results
    ////////////////////////////////////////////////////////////////
    change_directory("data", "results");

    result.save("power_concentration.csv", arma::csv_ascii);



	return 0;
}



