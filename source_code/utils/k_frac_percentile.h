//File: k_frac_percentile.h
#ifndef K_FRAC_PERCENTILE_H
#define K_FRAC_PERCENTILE_H

#include <algorithm>
#include <math.h>

#include <queue>
#include <vector>


 #define ARMA_DONT_USE_WRAPPER
 #define ARMA_NO_DEBUG
 #include <armadillo>


/*
This function finds the mean capitalist income for top income percentiles.
Input 'pay' is a vector of incomes, and 'k_frac_vec' is a vector of corresponding capitalist income fractions.
'percentile_start' is the starting percentile (i.e. 90th percentile), and 'n_bins' is the number of percenile bins.
Top percentiles (or more properly, fractiles) are logarithmicaly spaced to isolate the very top.
*/



arma::rowvec k_frac_percentile ( const arma::vec &pay,
                                 const arma::vec &k_frac_vec,
                                 double percentile_start,
                                 int n_bins
                                )
{

    // sort k_frac_vec by pay
    int n = pay.size()*(1-percentile_start) + 1;

    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;


    for (int i = 0; i < pay.size(); ++i) {
        if(q.size() < n)
            q.push(std::pair<double, int>(pay[i], i));
        else if( q.top().first < pay[i] ){
            q.pop();
            q.push(std::pair<double, int>(pay[i], i));
        }
    }

    n = q.size();
    arma::uvec res(n);

    for (int i = 0; i < n; ++i) {
        res[n - i - 1] = q.top().second;
        q.pop();
    }


    arma::vec k_sort = k_frac_vec(res);


    // get average capitalist fraction by percentile
    //arma::uvec breaks = arma::linspace<arma::uvec>(0, k_sort.size() -1, n_bins + 1  );


    arma::vec percentile = arma::logspace<arma::vec>(-5, 0 , n_bins + 1  );
    arma::vec breaks_dec = percentile % arma::linspace<arma::vec>(0, k_sort.size() -1, n_bins + 1  );
    arma::uvec breaks = arma::conv_to< arma::uvec >::from(breaks_dec);


    arma::rowvec k_frac_mean(n_bins);

    for(int i = 0; i < n_bins; ++i){

        k_frac_mean[i] = arma::mean( k_sort.subvec( breaks[i], breaks[i+1]  ));

    }

    return k_frac_mean;
}


#endif

