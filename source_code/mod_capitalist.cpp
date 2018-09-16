#include "base_fit.h"
#include "base_pay_sim.h"
#include "boot_sigma.h"
#include "boot_span.h"
#include "change_dir.h"
#include "fit_model.h"
#include "fit_power_law.h"
#include "gini_fast.h"
#include "k_frac_percentile.h"
#include "k_func.h"
#include "k_hist.h"
#include "lorenz.h"
#include "model.h"
#include "rpld.h"
#include "r_sim.h"
#include "sample.h"
#include "top.h"
#include "top_k.h"


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
This code implements the "Capitalist Gradient Model".
This model consists of the basic hierarchy model with an additional function
that determines capitalist income from an individual's hierarchical power.
*/


int main()
{
    std::cout  << "Capitalist Gradient Model" << std::endl;

    // model parameters
    int n_iterations = 1000;     // number of model iterations

    int n_firms = 1000000;      // number of firms in simulation
    double tol = 0.001;          // error tolerance for ceo ratio fit


    double percentile_start = 0.9;   // lowest percentile for k fraction calc.
    double n_bins = 1000;            // number of bins for k fraction calc.

    int n_bounds = 200;             // number of bounds in lorenz
    double lorenz_left = 0.00001;    // lower bound of lorenz
    double lorenz_right = 100000;     // upper bond of lorenz


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

    arma::mat   k_exec;
    k_exec.load("k_exec.txt");


    // output matrix
    //////////////////////////////////////////////////////////////////////////////////
    arma::mat k_hist_out(n_iterations, 80);         // capitalist income histogram output
    arma::mat k_percentile(n_iterations, n_bins);   // capitalist fraction percentile output
    arma::vec k_gini(n_iterations);                 // capitalist income gini index output
    arma::vec k_top1(n_iterations);                 // capitalist income top 1% share
    arma::vec k_alpha(n_iterations);                // capitalist power law exponent
    arma::vec k_share(n_iterations);                // capitalist share of income
    arma::mat k_lorenz(2*n_iterations, n_bounds);   // lorenz output matrix


    arma::vec k_gini_epsilon(n_iterations);                 // capitalist income gini index output
    arma::vec k_top1_epsilon(n_iterations);                 // capitalist income top 1% share
    arma::mat k_lorenz_epsilon(2*n_iterations, n_bounds);   // lorenz output matrix



    // model
    //////////////////////////////////////////////////////////////////////////////////

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

        arma::uvec  sim_employment = rpld(n_firms, 1, 2.01, 2300000, 2300000, true); // power law firm size distribution
        arma::vec   sim_base = base_fit(a, b, sim_employment); // fit base level
        arma::vec   sim_base_pay = base_pay_sim(comp_r_good.col(2), n_firms); // modelled base pay distribution
        arma::vec   sim_r = r_sim(comp_employment_good, comp_r_good.col(0), sim_employment); // modelled pay scaling

        arma::mat   mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r, sigma, false, false, true); // main model

        arma::vec   k_income = k_function(mod.col(0), mod.col(1), k_parameters); // get capitalist income
        arma::vec   k_frac = k_income/mod.col(0);


        k_percentile.row(iteration) = k_frac_percentile( mod.col(0), k_frac, percentile_start, n_bins );
        k_share(iteration) = arma::sum(k_income)/arma::sum(mod.col(0));



        // statistics on non zero income
        arma::vec k_sample = sample_no_replace(k_income, 1000000);
        arma::vec k_nonzero = k_sample.elem( arma::find(k_sample > 0 ));    // keep only non-zero capitalist income
        k_hist_out.row(iteration)  = k_hist(k_nonzero, k_parameters);   // histogram of capitalist income
        k_gini[iteration] = gini_fast(k_nonzero, true);           // gini index
        k_top1[iteration] = top_frac(k_nonzero, 0.01);         // top 1% share

        arma::vec p;
        p << 0.01;
        k_alpha[iteration] = fit_power_law(k_nonzero, p);       // power law exponent of top 1%

        k_lorenz.rows(2*iteration, 2*iteration+1) = lorenz(k_nonzero, lorenz_left, lorenz_right, n_bounds); // lorenz curve




        // statistics when zero income changed to very small amount
        int n_zeros = k_sample.size() - k_nonzero.size();
        double k_mean = arma::mean(k_sample);
        arma::vec k_small(n_zeros);
        k_small.fill(k_mean/3000);
        arma::vec k_epsilon = join_cols(k_nonzero, k_small);   // change 0 income to small amount


        k_gini_epsilon[iteration] = gini_fast(k_epsilon, true);           // gini index
        k_top1_epsilon(iteration, 0) = top_frac(k_epsilon, 0.01);         // top 1% share
        k_lorenz_epsilon.rows(2*iteration, 2*iteration+1) = lorenz(k_epsilon, lorenz_left, lorenz_right, n_bounds); // lorenz curve


        ++show_progress;
   }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout  << "elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    std::cout << std::endl;


    // output results
    ////////////////////////////////////////////////////////////////
    change_directory("data", "results");

    k_hist_out.save("k_hist.csv", arma::csv_ascii);
    k_percentile.save("k_frac_percentile.csv", arma::csv_ascii);

    k_gini.save("k_gini.csv", arma::csv_ascii);
    k_top1.save("k_top1.csv", arma::csv_ascii);
    k_alpha.save("k_alpha.csv", arma::csv_ascii);
    k_lorenz.save("k_lorenz.csv", arma::csv_ascii);
    k_share.save("k_share.csv", arma::csv_ascii);

    k_gini_epsilon.save("k_gini_epsilon.csv", arma::csv_ascii);
    k_top1_epsilon.save("k_top1_epsilon.csv", arma::csv_ascii);
    k_lorenz_epsilon.save("k_lorenz_epsilon.csv", arma::csv_ascii);



	return 0;
}



