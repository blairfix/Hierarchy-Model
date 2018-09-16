#include "base_fit.h"
#include "base_pay_sim_fudge.h"
#include "boot_sigma.h"
#include "boot_span.h"
#include "lorenz.h"
#include "change_dir.h"
#include "fit_model.h"
#include "fit_power_law.h"
#include "gini_fast.h"
#include "log_hist.h"
#include "model.h"
#include "pdf.h"
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
This code is implements counterfactual models (see mod_counter_fact.cpp).
The difference here is that I introduce a fudge factor to increase/decrease
inter-firm dispersion as I see fit. This fudge factor is chosens so that
outputs best match US data.
*/


int main()
{
    std::cout  << "Counter-Factual Model" << std::endl;


    // model parameters
    int n_iterations = 1000;    // number of model iterations

    int n_firms = 1000000;      // number of firms in simulation
    double tol = 0.01;          // error tolerance for ceo ratio fit
    int n_top_income = 500;     // number of top incomes to select

    double hist_left = -5;         // left bound of histogram
    double hist_right = 5;         // right bound of histogram
    int n_bin = 100;            // number of bins

    int n_bounds = 200;         // number of bounds in lorenz
    double lorenz_left = 0.01;     // lower bound of lorenz
    double lorenz_right = 1000;    // upper bond of lorenz

    double pdf_left = 0;           // left bound for probability density
    double pdf_right = 5;          // right bound for probability density
    int pdf_bins = 1000;        // number of bins for pdf

    double base_pay_fudge = 0.5;    // fudge factor for base pay sim


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


    arma::vec power_law_percentile;
    power_law_percentile.load("power_law_percentile.txt");


    // output matrices
    //////////////////////////////////////////////////////////////////////////////////
    change_directory("data", "results");


    arma::mat hf(n_iterations, n_top_income);   // hf mod output matrix
    arma::mat nhf(n_iterations, n_top_income);  // nhf mod output matrix
    arma::mat hnf(n_iterations, n_top_income);  // hnf mod output matrix
    arma::mat nhnf(n_iterations, n_top_income); // nhnf mod output matrix

    arma::mat gini_result(n_iterations, 4);     // gini output matrix
    arma::mat top1_result(n_iterations, 4);     // top 1% output matrix
    arma::mat alpha_result(n_iterations, 4);           // power law exponent vector

    arma::umat hf_hist(n_iterations, n_bin);    // hf mod histogram output matrix
    arma::umat nhf_hist(n_iterations, n_bin);   // nhf mod histogram output matrix
    arma::umat hnf_hist(n_iterations, n_bin);   // hnf mod histogram output matrix

    arma::mat hf_lorenz(2*n_iterations, n_bounds);      // hf  lorenz output matrix
    arma::mat nhf_lorenz(2*n_iterations, n_bounds);     // nhf lorenz output matrix
    arma::mat hnf_lorenz(2*n_iterations, n_bounds);     // hnf lorenz output matrix

    arma::mat hf_pdf(n_iterations, pdf_bins);      // pdf output matrix




    //********************************************************************************************
    //********************************************************************************************
    // MODEL


    auto start = std::chrono::system_clock::now();
    boost::progress_display show_progress(n_iterations);
    #pragma omp parallel for firstprivate(s_empirical, g_empirical, comp_employment, comp_ceo_ratio, comp_mean_pay, power_law_percentile) //num_threads(4)

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
        arma::uvec  sim_employment = rpld(n_firms, 1, 2.01, 2300000, 2300000, true);        // power law firm size distribution
        arma::vec   sim_base = base_fit(a, b, sim_employment);                              // fit base level
        arma::vec   sim_base_pay = base_pay_sim(comp_r_good.col(2), n_firms, base_pay_fudge);          // modelled base pay distribution
        arma::vec   sim_r = r_sim(comp_employment_good, comp_r_good.col(0), sim_employment);// modelled pay scaling


            //****************************************************************
            // hf model     (model with all dispersion sources)
            arma::mat   mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r, sigma, true);
            hf.row(iteration) = top_k(mod.col(0), mod.col(1), n_top_income);

            arma::vec pay_sample = sample_no_replace(mod.col(0), 1000000);  // sample

            alpha_result(iteration, 0) = fit_power_law(pay_sample, power_law_percentile );
            gini_result(iteration, 0) = gini_fast(pay_sample, true);        // get gini of sample
            top1_result(iteration, 0) = top_frac(pay_sample, 0.01);         // top 1% share

            hf_hist.row(iteration) = log_hist(pay_sample, hist_left, hist_right, n_bin); // histogram of pay_sample
            hf_lorenz.rows(2*iteration, 2*iteration+1) = lorenz(pay_sample, lorenz_left, lorenz_right, n_bounds); // lorenz of sample
            hf_pdf.row(iteration) = pdf(pay_sample, pdf_left, pdf_right, pdf_bins); // prob. density



            //****************************************************************
            // nhf model     (model with inter-firm dispersion only)
            arma::vec sim_r_flat =  arma::ones<arma::vec>(n_firms);
            mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r_flat, sigma, true);
            nhf.row(iteration) = top_k(mod.col(0), mod.col(1), n_top_income);

            pay_sample = sample_no_replace(mod.col(0), 1000000);        // sample

            alpha_result(iteration, 1)  = fit_power_law(pay_sample, power_law_percentile );
            gini_result(iteration, 1) = gini_fast(pay_sample, true);    // get gini of sample
            top1_result(iteration, 1) = top_frac(pay_sample, 0.01);     // top 1% share

            nhf_hist.row(iteration) = log_hist(pay_sample, hist_left, hist_right, n_bin); // histogram of pay_sample
            nhf_lorenz.rows(2*iteration, 2*iteration+1) = lorenz(pay_sample, lorenz_left, lorenz_right, n_bounds); // lorenz of sample


            //****************************************************************
            // hnf model     (model with inter-hierarchical dispersion only)
            arma::vec sim_base_pay_ones =  arma::ones<arma::vec>(n_firms);
            mod = model(a, b, sim_base, sim_employment, sim_base_pay_ones, sim_r, sigma, true);
            hnf.row(iteration) = top_k(mod.col(0), mod.col(1), n_top_income);

            pay_sample = sample_no_replace(mod.col(0), 1000000);        // sample

            alpha_result(iteration, 2) = fit_power_law(pay_sample, power_law_percentile );
            gini_result(iteration, 2) = gini_fast(pay_sample, true);    // get gini of sample
            top1_result(iteration, 2) = top_frac(pay_sample, 0.01);     // top 1% share

            hnf_hist.row(iteration) = log_hist(pay_sample, hist_left, hist_right, n_bin); // histogram of pay_sample
            hnf_lorenz.rows(2*iteration, 2*iteration+1) = lorenz(pay_sample, lorenz_left, lorenz_right, n_bounds); // lorenz of sample



            //****************************************************************
            // nhnf model     (model with intra-hierarchical dispersion only)
            mod = model(a, b, sim_base, sim_employment, sim_base_pay_ones, sim_r_flat, sigma, true);
            nhnf.row(iteration) = top_k(mod.col(0), mod.col(1), n_top_income);

            pay_sample = sample_no_replace(mod.col(0), 1000000);        // sample
            gini_result(iteration, 3) = gini_fast(pay_sample, true);    // get gini of sample
            top1_result(iteration, 3) = top_frac(pay_sample, 0.01);     // top 1% share
            alpha_result(iteration, 3) = fit_power_law(pay_sample, power_law_percentile );


        ++show_progress;
   }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout  << "elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    std::cout << std::endl;



    // output results
    ////////////////////////////////////////////////////////////////
    hf.save("hf_fudge.csv", arma::csv_ascii);
    nhf.save("nhf_fudge.csv", arma::csv_ascii);
    hnf.save("hnf_fudge.csv", arma::csv_ascii);
    nhnf.save("nhnf_fudge.csv", arma::csv_ascii);

    gini_result.save("gini_counterfact_fudge.csv", arma::csv_ascii);
    top1_result.save("top1_counterfact_fudge.csv", arma::csv_ascii);
    alpha_result.save("alpha_counterfact_fudge.csv", arma::csv_ascii);

    hf_hist.save("hf_hist_fudge.csv", arma::csv_ascii);
    nhf_hist.save("nhf_hist_fudge.csv", arma::csv_ascii);
    hnf_hist.save("hnf_hist_fudge.csv", arma::csv_ascii);

    hf_lorenz.save("hf_lorenz_fudge.csv", arma::csv_ascii);
    nhf_lorenz.save("nhf_lorenz_fudge.csv", arma::csv_ascii);
    hnf_lorenz.save("hnf_lorenz_fudge.csv", arma::csv_ascii);

    hf_pdf.save("hf_pdf_fudge.csv", arma::csv_ascii);


	return 0;
}



