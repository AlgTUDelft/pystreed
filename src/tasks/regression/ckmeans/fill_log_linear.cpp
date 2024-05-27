/* fill_log_linear.cpp --- a divide-and-conquer algorithm to compute a
 *   row in the dynamic programming matrix in O(n lg n) time.
 *
 * Copyright (C) 2016 Mingzhou Song
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

//
// Joe Song
// Created: June 4, 2016
// Extracted from Ckmeans.1d.dp.cpp

#include <vector>
#include <iostream>
#include "tasks/regression/ckmeans/Ckmeans.1d.dp.h"

void fill_row_q_log_linear(int imin, int imax, int q,
                           int jmin, int jmax,
                           std::vector< std::vector<ldouble> > & S,
                           std::vector< std::vector<size_t> > & J,
                           const std::vector<ldouble> & sum_x,
                           const std::vector<ldouble> & sum_x_sq,
                           const std::vector<ldouble> & sum_w,
                           const std::vector<ldouble> & sum_w_sq,
                           const enum DISSIMILARITY criterion)
{
  if(imin > imax) {
    return;
  }

  const int N = (int) S[0].size();

  int i = (imin + imax) / 2;

#ifdef DEBUG
  // std::cout << "  i=" << i << ": ";
#endif

  // Initialization of S[q][i]:
  S[q][i] = S[q - 1][i - 1];
  J[q][i] = i;

  int jlow=q; // the lower end for j

  if(imin > q) {
    // jlow = std::max(jlow, (int)J[q][imin-1]);
    jlow = std::max(jlow, jmin);
  }
  jlow = std::max(jlow, (int)J[q-1][i]);

  int jhigh = i - 1; // the upper end for j
  if(imax < N-1) {
    // jhigh = std::min(jhigh, (int)J[q][imax+1]);
    jhigh = std::min(jhigh, jmax);
  }

#ifdef DEBUG
  // std::cout << "    j-=" << jlow << ", j+=" << jhigh << ": ";
#endif

  for(int j=jhigh; j>=jlow; --j) {

    // compute s(j,i)
    ldouble sji = ssq(j, i, sum_x, sum_x_sq, sum_w);

    // MS May 11, 2016 Added:
    if(sji + S[q-1][jlow-1] >= S[q][i]) break;

    // Examine the lower bound of the cluster border
    // compute s(jlow, i)
    ldouble sjlowi =
      dissimilarity(criterion, jlow, i, sum_x, sum_x_sq, sum_w, sum_w_sq);
      // ssq(jlow, i, sum_x, sum_x_sq, sum_w);

    ldouble SSQ_jlow = sjlowi + S[q-1][jlow-1];

    if(SSQ_jlow < S[q][i]) {
      // shrink the lower bound
      S[q][i] = SSQ_jlow;
      J[q][i] = jlow;
    }
    jlow ++;

    ldouble SSQ_j = sji + S[q - 1][j - 1];
    if(SSQ_j < S[q][i]) {
      S[q][i] = SSQ_j;
      J[q][i] = j;
    }
  }

#ifdef DEBUG
  //std::cout << // " q=" << q << ": " <<
  //  "\t" << S[q][i] << "\t" << J[q][i];
  //std::cout << std::endl;
#endif

  jmin = (imin > q) ? (int)J[q][imin-1] : q;
  jmax = (int)J[q][i];

  fill_row_q_log_linear(imin, i-1, q, jmin, jmax,
                        S, J, sum_x, sum_x_sq, sum_w,
                        sum_w_sq, criterion);

  jmin = (int)J[q][i];
  jmax = (imax < N-1) ? (int)J[q][imax+1] : imax;
  fill_row_q_log_linear(i+1, imax, q, jmin, jmax,
                        S, J, sum_x, sum_x_sq, sum_w,
                        sum_w_sq, criterion);

}
