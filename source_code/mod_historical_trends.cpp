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
This code implements the Hierarchical Redistribution Model.
The mean of the hierarchical pay scaling parameter distribution is
allowed to vary stochastically over each iteration.
This produces models with different pay by hierarchical level.
*/


int main()
{
    std::cout  << "Hierarchical Redistribution Model" << std::endl;

    // model parameters
    int n_iterations = 50000;   // number of model iterations

    int n_firms = 1000000;      // number of firms in simulation
    double tol = 0.01;          // error tolerance for ceo ratio fit

    double dividend_fraction = 0.5;    // dividend fraction of capitalist income


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
    arma::mat result(n_iterations, 6);
    arma::mat power_mean_out(n_iterations, 40);
    arma::mat hierarchy_mean_out(n_iterations,20);


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

        arma::uvec  sim_employment = rpld(n_firms, 1, 2.01, 2300000, 2300000, true); // power law firm size distribution
        arma::vec   sim_base = base_fit(a, b, sim_employment); // fit base level
        arma::vec   sim_base_pay = base_pay_sim(comp_r_good.col(2), n_firms); // modelled base pay distribution
        arma::vec   sim_r = r_sim(comp_employment_good, comp_r_good.col(0), sim_employment); // modelled pay scaling


            // stochastically shift mean of pay scaling parameter (r) distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.25, 1.35);
            double r_shift = dis(gen);

            for(int i = 0; i < n_firms; ++i){
                sim_r[i] = (sim_r[i] - 1)*r_shift + 1;
            }


        arma::mat   mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r, sigma, false, true, true); // main model




        // get stats on sample of 1 million people
        double  sample_size = 1000000;
        arma::uvec sample_id = sample_index(mod.col(0), sample_size);

        arma::mat mod_sample = mod.rows(sample_id);
        arma::vec   k_income = k_function(mod_sample.col(0), mod_sample.col(2), k_parameters); // get capitalist income

        // calculations
        double sum_all_pay = arma::sum(mod_sample.col(0));
        double sum_capital = arma::sum(k_income);
        double mean_all_pay = sum_all_pay / mod_sample.n_rows;

        double dividend_share = sum_capital/sum_all_pay*dividend_fraction;
        double top_1_frac = top_frac(mod_sample.col(0), 0.01); // top 1%  share of national income
        double mean_r = arma::mean(sim_r);              // mean pay scaling parameter
        double mean_ceo_ratio = mean_ceo_pay_ratio(a, b, sim_base, sim_employment, sim_base_pay, sim_r, 350, mean_all_pay);


        double top1_total_sum = top_1_frac*sum_all_pay;
        int k_top1 = mod_sample.n_rows*0.01;
        double top1_k_sum = arma::sum( top_k( mod_sample.col(0), k_income, k_top1 ) );
        double top1_k_share = top1_k_sum/top1_total_sum;

        arma::vec percentile;
        percentile << 0.01;
        double alpha = fit_power_law(mod_sample.col(0), percentile);


        result(iteration, 0) = dividend_share;
        result(iteration, 1) = top_1_frac;
        result(iteration, 2) = mean_r;
        result(iteration, 3) = mean_ceo_ratio;
        result(iteration, 4) = top1_k_share;
        result(iteration, 5) = alpha;

        power_mean_out.row(iteration) = power_mean(mod_sample.col(0), mod_sample.col(2));
        hierarchy_mean_out.row(iteration) = h_mean(mod_sample.col(0), mod_sample.col(1));


        ++show_progress;
   }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout  << "elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    std::cout << std::endl;



    // output results
    ////////////////////////////////////////////////////////////////
    change_directory("data", "results");

    result.save("dividend_top_share.csv", arma::csv_ascii);
    power_mean_out.save("power_mean.csv", arma::csv_ascii);
    hierarchy_mean_out.save("hierarchy_mean.csv", arma::csv_ascii);

	return 0;
}



