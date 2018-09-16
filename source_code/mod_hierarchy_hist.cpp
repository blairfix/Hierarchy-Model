#include "core/base_fit.h"
#include "core/base_pay_sim.h"
#include "core/boot_sigma.h"
#include "core/boot_span.h"
#include "core/fit_model.h"
#include "core/model.h"
#include "core/rpld.h"
#include "core/r_sim.h"



#include "utils/change_dir.h"
#include "utils/fit_power_law.h"
#include "utils/gini_fast.h"
#include "utils/k_frac_percentile.h"
#include "utils/k_func.h"
#include "utils/k_hist.h"
#include "utils/lorenz.h"
#include "utils/log_hist.h"
#include "utils/pdf.h"
#include "utils/sample.h"
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
This code implements the hierarchy model and generates statistics
(i.e. pay distribution) by hierarchical level.
*/

int main()
{
    std::cout  << "Hierarchy Histogram Model" << std::endl;


    // model parameters
    int n_iterations = 10000;    // number of model iterations

    int n_firms = 1000000;      // number of firms in simulation
    double tol = 0.01;          // error tolerance for ceo ratio fit
    int n_top_income = 500;     // number of top incomes to select


    double hist_left = -5;      // left bound of histogram
    double hist_right = 5;      // right bound of histogram
    int n_bin = 100;            // number of bins


    int h_max = 14;          // number of hierarchical levels in stats

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

    arma::umat h_level_hist(n_iterations*n_bin, h_max); // histogram by hierarchical level output
    arma::mat h_mean_out(n_iterations, h_max);          // hierarchical mean pay output
    arma::mat h_emp_frac_out(n_iterations, h_max);      // hierarchical employment fraction output
    arma::mat h_pay_frac_out(n_iterations, h_max);      // hierarchcial pay fraction output


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
        // hierarchy model
        arma::mat   mod = model(a, b, sim_base, sim_employment, sim_base_pay, sim_r, sigma, false, true);



            // get stats
            arma::umat h_hist(n_bin, h_max);            // histogram .... counts in rows, hierarchy in columns
            arma::rowvec h_mean(n_bin);                 // mean pay in each h level
            arma::rowvec h_pay_frac(n_bin);             // fraction pay in each h level
            arma::rowvec h_emp_frac(n_bin);             // fraction of employment in each h level


            double pay_sum = arma::sum(mod.col(0));     // model total payroll
            int n_people = mod.n_rows;                  // model number of people
            double pay_mean = pay_sum/n_people;         // model mean pay


            arma::vec log_norm_pay = arma::log10( mod.col(0)/pay_mean );     // log of normalized pay
            arma::vec edges = arma::linspace<arma::vec>(hist_left, hist_right, n_bin);     // histogram bins



            // loop over hierarchical levels
            for(int i = 1; i < h_max + 1; ++i){

                arma::uvec ids = arma::find( mod.col(1)  == i);        // Find indices for h

                // log hist
                arma::vec pay_select = log_norm_pay.elem(ids);
                h_hist.col(i-1) = arma::histc(pay_select, edges );

                // h level summary stats
                arma::uvec column(1);
                column[0] = 0;

                double h_number = ids.size();         // number of people in h level
                arma::mat h_pay = mod(ids, column) ;
                double h_pay_sum = arma::sum(h_pay.col(0)); // sum pay in h level


                h_mean[i-1] = h_pay_sum / h_number;
                h_pay_frac[i-1] = h_pay_sum / pay_sum;
                h_emp_frac[i-1] = h_number / n_people;


            }


            h_level_hist.rows(n_bin*iteration, n_bin*iteration + n_bin -1) = h_hist;
            h_mean_out.row(iteration) = h_mean;
            h_pay_frac_out.row(iteration) = h_pay_frac;
            h_emp_frac_out.row(iteration) = h_emp_frac;




        ++show_progress;
   }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout  << "elapsed time: " << elapsed_seconds.count() << " s" << std::endl;
    std::cout << std::endl;



    // output results
    ////////////////////////////////////////////////////////////////
    change_directory("data", "results");


    h_level_hist.save("h_level_hist.csv", arma::csv_ascii);
    h_mean_out.save("h_mean_standard.csv", arma::csv_ascii);
    h_pay_frac_out.save("h_pay_frac.csv", arma::csv_ascii);
    h_emp_frac_out.save("h_emp_frac.csv", arma::csv_ascii);


	return 0;
}



