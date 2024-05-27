/* Ckmeans_1d_dp.cpp -- Performs 1-D k-means by a dynamic programming
 * approach that is guaranteed to be optimal.
 *
 * Copyright (C) 2010-2016 Mingzhou Song and Haizhou Wang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 Joe Song
 Computer Science Department
 New Mexico State University
 joemsong@cs.nmsu.edu

 Haizhou Wang
 Computer Science Department
 New Mexico State University
 hwang@cs.nmsu.edu

 Created: May 19, 2007
 Updated: September 3, 2009
 Updated: September 6, 2009.  Handle special cases when K is too big or array
 contains identical elements.  Added number of clusters selection by the
 MCLUST package.
 Updated: Oct 6, 2009 by Haizhou Wang Convert this program from R code into
 C++ code.
 Updated: Oct 20, 2009 by Haizhou Wang, add interface to this function, so that
 it could be called directly in R

 Updated: March 20, 2014. Joe Song.
 1.  When the number of clusters is not uniquely specified, the code now
 automatically select an optimal number of clusters by Bayesian
 information criterion.
 2.  Reduced unnecessary sorting performed on the input string in
 kmeans_1d_dp().

 Updated: February 8, 2015.
 1.  Cleaned up the code by removing commented sections (Haizhou Wang)
 2.  Speed up the code slightly as suggested by a user (Joe Song)
 3.  Throw exceptions for fatal errors (Joe Song)

 Updated: May 3, 2016
 1. Changed from 1-based to 0-based C implementation (MS)
 2. Optimized the code by reducing overhead. See 22% reduction in runtime to
 cluster one million points into two clusters. (MS)
 3. Removed the result class ClusterResult

 Updated: May 7, 2016
 1. Substantial runtime reduction. Added code to check for an upper bound
 for the sum of within cluster square distances. This reduced the runtime
 by half when clustering 100000 points (from standard normal distribution)
 into 10 clusters.
 2. Eliminated the unnecessary calculation of (n-1) elements in the dynamic
 programming matrix that are not needed for the final result. This
 resulted in enormous reduction in run time when the number of cluster
 is 2: assigning one million points into two clusters took half a
 a second on iMac with 2.93 GHz Intel Core i7 processor.

 Updated: May 9, 2016
 1. Added an alternative way to fill in the dynamic programming
 matix by i (columns in the matrix) to achieve speedup

 Updated: May 21, 2016
 1. Moved the weighted solutions to new files. They will be updated separately

 Updated: May 26, 2016
 1. Implemented log-linear algorithm and tested successfully

 Updated: May 29, 2016
 1. Implemented linear algorithm but have bugs

 Updated: May 30, 2016
 1. Debugged the linear algorithm and tested successfully on all examples
 2. Added new test cases in the C++ testing project

 Updated: July 19, 2016.
 1. If the input array is already sorted, not sorting
 is performed.

 Updated: August 20, 2016
 1. The weighted univariate k-means now runs in O(kn), down from O(kn^2).
 This is a result of integrating weighted and unweighted k-means
 clustering into a unified dynamic programming function without sacrificing
 performance.

 */

//#include "EWL2.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>
#include "tasks/regression/ckmeans/Ckmeans.1d.dp.h"

template <class ForwardIterator>
size_t numberOfUnique(ForwardIterator first, ForwardIterator last)
{
  size_t nUnique;

  if (first == last) {
    nUnique = 0;
  } else {
    nUnique = 1;
    for (ForwardIterator itr=first+1; itr!=last; ++itr) {
      if (*itr != *(itr -1)) {
        nUnique ++;
      }
    }
  }
  return nUnique;
}

static const double * px;
bool compi(size_t i, size_t j)
{
  return px[i] < px[j];
}

//void kmeans_1d_dp(const double* x, const size_t N, const double* y,
//                  size_t Kmin, size_t Kmax,
//                  int* cluster, double* centers,
//                  double* withinss, double *size, // int* size,
//                  double* BIC,
//                  const std::string & estimate_k,
//                  const std::string & method,
//                  const enum DISSIMILARITY criterion)
//{
//  // Input:
//  //  x -- an array of double precision numbers, not necessarily sorted
//  //  Kmin -- the minimum number of clusters expected
//  //  Kmax -- the maximum number of clusters expected
//  // NOTE: All vectors in this program is considered starting at position 0.
//
//  std::vector<size_t> order(N);
//
//  //Number generation using lambda function, not supported by all g++:
//  //std::size_t n(0);
//  //std::generate(order.begin(), order.end(), [&]{ return n++; });
//
//  for(size_t i=0; i<order.size(); ++i) {
//    order[i] = i;
//  }
//
//  bool is_sorted(true);
//  for(size_t i=0; i<N-1; ++i) {
//    if(x[i] > x[i+1]) {
//      is_sorted = false;
//      break;
//    }
//  }
//
//  std::vector<double> x_sorted(x, x+N);
//
//  std::vector<double> y_sorted;
//  bool is_equally_weighted = true;
//
//  if(! is_sorted) {
//    // Sort the index of x in increasing order of x
//
//    // Option 1.
//    // Sorting using lambda function, not supported by all g++ versions:
//    // std::sort(order.begin(), order.end(),
//    //           [&](size_t i1, size_t i2) { return x[i1] < x[i2]; } );
//
//    /* Option 2. The following is not supported by C++98:
//     struct CompareIndex {
//     const double * m_x;
//     CompareIndex(const double * x) : m_x(x) {}
//     bool operator() (size_t i, size_t j) { return (m_x[i] < m_x[j]);}
//     } compi(x);
//
//     std::sort(order.begin(), order.end(), compi);
//     */
//
//    // Option 3:
//    px = x;
//    std::sort(order.begin(), order.end(), compi);
//
//    for(size_t i=0ul; i<order.size(); ++i) {
//      x_sorted[i] = x[order[i]];
//    }
//  }
//
//  // check to see if unequal weight is provided
//  if(y != NULL) {
//    is_equally_weighted = true;
//    for(size_t i=1; i<N; ++i) {
//      if(y[i] != y[i-1]) {
//        is_equally_weighted = false;
//        break;
//      }
//    }
//  }
//
//  if(! is_equally_weighted) {
//    y_sorted.resize(N);
//    for(size_t i=0; i<order.size(); ++i) {
//      y_sorted[i] = y[order[i]];
//    }
//  }
//
//  const size_t nUnique = numberOfUnique(x_sorted.begin(), x_sorted.end());
//
//  Kmax = nUnique < Kmax ? nUnique : Kmax;
//
//  if(nUnique > 1) { // The case when not all elements are equal.
//
//    std::vector< std::vector< ldouble > > S( Kmax, std::vector<ldouble>(N) );
//    std::vector< std::vector< size_t > > J( Kmax, std::vector<size_t>(N) );
//
//    size_t Kopt;
//
//    // Fill in dynamic programming matrix
//    if(is_equally_weighted) {
//
//      //if (criterion == L2) {
//      //  EWL2::fill_dp_matrix(x_sorted, y_sorted, S, J, method);
//      //} else {
//        fill_dp_matrix(x_sorted, y_sorted, S, J, method, criterion);
//      //}
//
//      // Choose an optimal number of levels between Kmin and Kmax
//      if(estimate_k=="BIC") {
//        Kopt = select_levels(x_sorted, J, Kmin, Kmax, BIC);
//      } else {
//        Kopt = select_levels_3_4_12(x_sorted, J, Kmin, Kmax, BIC);
//      }
//
//    } else {
//
//      fill_dp_matrix(x_sorted, y_sorted, S, J, method, criterion);
//
//      switch(criterion) {
//      case L2Y:
//        if(estimate_k=="BIC") {
//          Kopt = select_levels(y_sorted, J, Kmin, Kmax, BIC);
//        } else {
//          Kopt = select_levels_3_4_12(y_sorted, J, Kmin, Kmax, BIC);
//        }
//        break;
//
//      default:
//        if(estimate_k=="BIC") {
//          // Choose an optimal number of levels between Kmin and Kmax
//          Kopt = select_levels_weighted(x_sorted, y_sorted, J, Kmin, Kmax, BIC);
//        } else {
//          Kopt = select_levels_weighted_3_4_12(x_sorted, y_sorted, J, Kmin, Kmax, BIC);
//        }
//      }
//    }
//
//    if (Kopt < Kmax) { // Reform the dynamic programming matrix S and J
//      J.erase(J.begin() + Kopt, J.end());
//    }
//
//    std::vector<int> cluster_sorted(N);
//
//    // Backtrack to find the clusters beginning and ending indices
//    if(is_equally_weighted && criterion == L1) {
//        backtrack_L1(x_sorted, J, &cluster_sorted[0], centers, withinss, size);
//    } else if (is_equally_weighted && criterion == L2) {
//        backtrack(x_sorted, J, &cluster_sorted[0], centers, withinss, size);
//    } else if(criterion == L2Y) {
//      backtrack_L2Y(x_sorted, y_sorted, J, &cluster_sorted[0],
//                    centers, withinss, size);
//
//    } else {
//      backtrack_weighted(x_sorted, y_sorted, J, &cluster_sorted[0],
//                         centers, withinss, size);
//    }
//
//#ifdef DEBUG
//    std::cout << "backtrack done." << std::endl;
//#endif
//
//    for(size_t i = 0; i < N; ++i) {
//      // Obtain clustering on data in the original order
//      cluster[order[i]] = cluster_sorted[i];
//    }
//
//  } else {  // A single cluster that contains all elements
//
//    for(size_t i=0; i<N; ++i) {
//      cluster[i] = 0;
//    }
//
//    centers[0] = x[0];
//    withinss[0] = 0.0;
//    size[0] = N * (is_equally_weighted ? 1 : y[0]);
//  }
//}  //end of kmeans_1d_dp()

std::pair<ldouble,int> fill_dp_matrix_dynamic_stop(const std::vector<WeightValuePair>& data, // data
    std::vector< std::vector< ldouble > >& S,
    std::vector< std::vector< size_t > >& J,
    ldouble reg)
    /*
     x: One dimension vector to be clustered, must be sorted (in any order).
     S: K x N matrix. S[q][i] is the sum of squares of the distance from
     each x[i] to its cluster mean when there are exactly x[i] is the
     last point in cluster q
     J: K x N backtrack matrix

     NOTE: All vector indices in this program start at position 0
     */
{
    const int K = (int)S.size();
    const int N = (int)S[0].size();

    std::vector<ldouble> sum_x(N), sum_x_sq(N);
    std::vector<ldouble> sum_w(data.size()), sum_w_sq(data.size());

    std::vector<int> jseq;

    ldouble shift = data[N / 2].value; // median. used to shift the values of x to
    //  improve numerical stability

    const auto& value = data[0].value;
    bool equally_weighted = data[0].weight == 0;
    if (equally_weighted) { // equally weighted
        sum_x[0] = value - shift;
        sum_x_sq[0] = (value - shift) * (value - shift);
    } else { // unequally weighted
        const auto& weight = data[0].weight;
        sum_x[0] = weight * (value - shift);
        sum_x_sq[0] = weight * (value - shift) * (value - shift);
        sum_w[0] = weight;
        sum_w_sq[0] = weight * weight;
    }

    S[0][0] = 0;
    J[0][0] = 0;

    for (int i = 1; i < N; ++i) {
        const auto& value = data[i].value;
        if (equally_weighted) { // equally weighted
            sum_x[i] = sum_x[i - 1] + value - shift;
            sum_x_sq[i] = sum_x_sq[i - 1] + (value - shift) * (value - shift);
        } else { // unequally weighted
            auto& weight = data[i].weight;
            sum_x[i] = sum_x[i - 1] + weight * (value - shift);
            sum_x_sq[i] = sum_x_sq[i - 1] + weight * (value - shift) * (value - shift);
            sum_w[i] = sum_w[i - 1] + weight;
            sum_w_sq[i] = sum_w_sq[i - 1] + weight * weight;
        }

        // Initialize for q = 0
        S[0][i] = ssq(0, i, sum_x, sum_x_sq, sum_w);
        J[0][i] = 0;
    }


    for (int q = 1; q < K; ++q) {
        int imin;
        if (q < K - 1) {
            imin = std::max(1, q);
        } else {
            // No need to compute S[K-1][0] ... S[K-1][N-2]
            imin = N - 1;
        }
        // Faster when doing <300 data pts
        if (N > 300) {
            fill_row_q_SMAWK(imin, N - 1, q, S, J, sum_x, sum_x_sq, sum_w, sum_w_sq, L2);
        } else {
            fill_row_q_log_linear(imin, N - 1, q, q, N - 1, S, J, sum_x, sum_x_sq, sum_w, sum_w_sq, L2);
        }

        if (S[q - 1][N - 1] - S[q][N - 1] < reg) {
            return std::make_pair<ldouble, int>(S[q - 1][N - 1] + (q - 1) * reg, q-1);
        }
    }
    // TODO: fix this ad hoc thingy 
    runtime_assert(K < 50);
    return std::make_pair<ldouble, int>(S[K - 1][N - 1] + (K - 1) * reg, K-1);

}