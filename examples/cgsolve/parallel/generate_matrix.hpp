// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

/*
  Adapted from the Mantevo Miniapp Suite.
  https://mantevo.github.io/pdfs/MantevoOverview.pdf
*/

#ifndef GENERATE_MATRIX_HPP
#define GENERATE_MATRIX_HPP

#include <Kokkos_Core.hpp>
#include "KokkosKernels_IOUtils.hpp"
#include <mpi.h>

#define LOCAL_ORDINAL int

//#define MASK 1099511627776
#define MASK 268435456
template <class MemSpace, typename S = double> struct CrsMatrix {
  using value_type = S;

  // -------------------------------------- //
  int nlocal;
  int start_row;
  int end_row;
  // -------------------------------------- //

  Kokkos::View<int *, MemSpace> row_ptr;
  Kokkos::View<LOCAL_ORDINAL *, MemSpace> col_idx;
  Kokkos::View<S *, MemSpace> values;

  int _num_cols;
  KOKKOS_INLINE_FUNCTION
  int num_rows() const { return row_ptr.extent(0) - 1; }
  KOKKOS_INLINE_FUNCTION
  int num_cols() const { return _num_cols; }
  KOKKOS_INLINE_FUNCTION
  int nnz() const { return values.extent(0); }

  CrsMatrix() {}

  CrsMatrix(Kokkos::View<int *, MemSpace> row_ptr_,
            Kokkos::View<LOCAL_ORDINAL *, MemSpace> col_idx_,
            Kokkos::View<S *, MemSpace> values_, int num_cols_)
      : row_ptr(row_ptr_), col_idx(col_idx_), values(values_),
        _num_cols(num_cols_) {}
};

namespace Impl {
template <class GO, class S>
static void
miniFE_get_row(int *rows, S *vals, GO *cols, int rows_per_proc,
               int startrow, int endrow, int &row, int o,
               int nx1, int c1, int c2, int c3, int val,
               int &miniFE_a, int &miniFE_b, int &miniFE_c) {
  // FIXME (mfh 25 Jun 2014) Seriously, "val27"???  Who writes
  // code like this???

  bool val27 = false;
  if (c1 * c2 * c3 == 27) {
    val27 = true;
  }
  // printf("%li %li %li\n",c1,c2,c3);
  if ((row >= startrow) && (row < endrow)) {
    int offset = rows[row - startrow];
    rows[row + 1 - startrow] = offset + c1 * c2 * c3;
    for (int i = 0; i < c1; i++)
      for (int j = 0; j < c2; j++)
        for (int k = 0; k < c3; k++) {
          int m = i * c2 * c3 + j * c2 + k;
          int col_idx = o + i * nx1 * nx1 + j * nx1 + k;
          cols[offset + m] =
              (col_idx / rows_per_proc) * MASK + col_idx % rows_per_proc;
          if (val27) {
            bool doa = ((miniFE_a > 0) && (miniFE_a < nx1 - 3)) ||
                       ((miniFE_a == 0) && (m / 9 >= 1)) ||
                       ((miniFE_a == nx1 - 3) && (m / 9 < 2));
            bool dob = ((miniFE_b > 0) && (miniFE_b < nx1 - 3)) ||
                       ((miniFE_b == 0) && ((m % 9) / 3 >= 1)) ||
                       ((miniFE_b == nx1 - 3) && ((m % 9) / 3 < 2));
            bool doc = ((miniFE_c > 0) && (miniFE_c < nx1 - 3)) ||
                       ((miniFE_c == 0) && ((m % 3) >= 1)) ||
                       ((miniFE_c == nx1 - 3) && ((m % 3) < 2));
            if (doa && dob && doc) {
              if (m == 13)
                vals[offset + m] = 8.0 / 3.0 / (nx1 - 1);
              else {
                if (m % 2 == 1)
                  vals[offset + m] = -5.0e-1 / 3.0 / (nx1 - 1);
                else {
                  if ((m == 4) || (m == 22) || ((m > 9) && (m < 17)))
                    vals[offset + m] = -2.18960e-10 / (nx1 - 1);
                  else
                    vals[offset + m] = -2.5e-1 / 3.0 / (nx1 - 1);
                }
              }
            } else
              vals[offset + m] = 0.0;
          } else {
            if (val == m)
              vals[offset + m] = 1.0;
            else
              vals[offset + m] = 0.0;
          }
        }
  }
  if (c1 * c2 * c3 == 27) {
    miniFE_c++;
    if (miniFE_c > nx1 - 3) {
      miniFE_c = 0;
      miniFE_b++;
    }
    if (miniFE_b > nx1 - 3) {
      miniFE_b = 0;
      miniFE_a++;
    }
  }

  row++;
}

template <class GO, class S>
static void miniFE_get_block(int *rows, S *vals, GO *cols,
                             int rows_per_proc, int startrow,
                             int endrow, int &row, int o,
                             int nx1, int c1, int c2, int val1,
                             int val2, int val3, int &miniFE_a,
                             int &miniFE_b, int &miniFE_c) {
  miniFE_get_row(rows, vals, cols, rows_per_proc, startrow, endrow, row, o, nx1,
                 c1, c2, 2, val1, miniFE_a, miniFE_b, miniFE_c);
  for (int i = 0; i < nx1 - 2; i++)
    miniFE_get_row(rows, vals, cols, rows_per_proc, startrow, endrow, row, o++,
                   nx1, c1, c2, 3, val2, miniFE_a, miniFE_b, miniFE_c);
  miniFE_get_row(rows, vals, cols, rows_per_proc, startrow, endrow, row, o++,
                 nx1, c1, c2, 2, val3, miniFE_a, miniFE_b, miniFE_c);
}

template <class GO, class S>
static void miniFE_get_superblock(int *rows, S *vals, GO *cols,
                                  int rows_per_proc, int startrow,
                                  int endrow, int &row, int o,
                                  int nx1, int c1, int val1,
                                  int val2, int val3, int &miniFE_a,
                                  int &miniFE_b, int &miniFE_c) {
  miniFE_get_block(rows, vals, cols, rows_per_proc, startrow, endrow, row, o,
                   nx1, c1, 2, val1 + 0, val1 + val2 + 1, val1 + 1, miniFE_a,
                   miniFE_b, miniFE_c);
  for (int i = 0; i < nx1 - 2; i++) {
    miniFE_get_block(rows, vals, cols, rows_per_proc, startrow, endrow, row, o,
                     nx1, c1, 3, val1 + val2 + 3, val1 + val2 + val2 + val3 + 4,
                     val1 + val2 + 4, miniFE_a, miniFE_b, miniFE_c);
    o += nx1;
  }
  miniFE_get_block(rows, vals, cols, rows_per_proc, startrow, endrow, row, o,
                   nx1, c1, 2, val1 + 2, val1 + val2 + 3, val1 + 3, miniFE_a,
                   miniFE_b, miniFE_c);
}

template <class S = double>
static CrsMatrix<Kokkos::HostSpace, S> generate_miniFE_matrix(int nx) {

  int miniFE_a = 0;
  int miniFE_b = 0;
  int miniFE_c = 0;

  int myRank = 0;
  int numRanks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  int nx1 = nx + 1;

  int nrows_block = 1 + nx - 1 + 1;
  int nrows_superblock = (1 + nx - 1 + 1) * nrows_block;
  int nrows = (1 + (nx - 1) + 1) * nrows_superblock;

  int nnz = 0;
  nnz += 4 * (8 + (nx - 1) * 12 + 8);
  nnz += 4 * (nx - 1) * (12 + (nx - 1) * 18 + 12);
  nnz += (nx - 1) * (nx - 1) * (18 + (nx - 1) * 27 + 18);

  int rows_per_proc = (nrows + numRanks - 1) / numRanks;
  int startrow = rows_per_proc * myRank;
  int endrow = startrow + rows_per_proc;
  if (endrow > nrows)
    endrow = nrows;

  Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
      "generate_MiniFE_Matrix::rowPtr", endrow - startrow + 1);
  Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
      "generate_MiniFE_Matrix::colInd", (endrow - startrow) * 27);
  Kokkos::View<S *, Kokkos::HostSpace> values(
      "generate_MiniFE_Matrix::values", (endrow - startrow) * 27);

  int *rows = &rowPtr[0];
  S *vals = &values[0];
  LOCAL_ORDINAL *cols = &colInd[0];

  int row = 0;
  miniFE_get_superblock(rows, vals, cols, rows_per_proc, startrow, endrow, row,
                        0, nx1, 2, 0, 0, 0, miniFE_a, miniFE_b, miniFE_c);
  for (int i = 0; i < nx1 - 2; i++) {
    miniFE_get_superblock(rows, vals, cols, rows_per_proc, startrow, endrow,
                          row, i * nx1 * nx1, nx1, 3, 4, 2, 1, miniFE_a,
                          miniFE_b, miniFE_c);
  }
  miniFE_get_superblock(rows, vals, cols, rows_per_proc, startrow, endrow, row,
                        (nx1 - 2) * nx1 * nx1, nx1, 2, 4, 2, 1, miniFE_a,
                        miniFE_b, miniFE_c);

  CrsMatrix<Kokkos::HostSpace, S> matrix(rowPtr, colInd, values, nx);
  return matrix;
}

template <class S>
static void miniFE_vector_generate_block(S *vec, int nx, S a, S b, int &count,
                                         int start, int end) {
  if ((count >= start) && (count < end))
    vec[count++ - start] = 0;
  for (int i = 0; i < nx - 2; i++)
    if ((count >= start) && (count < end))
      vec[count++ - start] = a / nx / nx / nx;
  if ((count >= start) && (count < end))
    vec[count++ - start] = a / nx / nx / nx + b / nx;
  if ((count >= start) && (count < end))
    vec[count++ - start] = 1;
}

template <class S>
static void miniFE_vector_generate_superblock(S *vec, int nx, S a, S b, S c,
                                              int &count, int start, int end) {
  miniFE_vector_generate_block<S>(vec, nx, 0.0, 0.0, count, start, end);
  miniFE_vector_generate_block<S>(vec, nx, a, b, count, start, end);
  for (int i = 0; i < nx - 3; i++)
    miniFE_vector_generate_block<S>(vec, nx, a, c, count, start, end);
  miniFE_vector_generate_block<S>(vec, nx, a, b, count, start, end);
  miniFE_vector_generate_block<S>(vec, nx, 0.0, 0.0, count, start, end);
}

template <class S = double>
Kokkos::View<S *, Kokkos::HostSpace> generate_miniFE_vector(int nx) {

  int my_rank = 0;
  int num_ranks = 1;
  // MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  // MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);

  const int numRows = (nx + 1) * (nx + 1) * (nx + 1);

  int start = numRows / num_ranks * my_rank;
  int end = start + numRows / num_ranks;
  if (end > numRows)
    end = numRows;

  // Make a multivector X owned entirely by Proc 0.
  Kokkos::View<S *, Kokkos::HostSpace> X("X_host", numRows);
  S *vec = X.data();
  int count = 0;
  miniFE_vector_generate_superblock<S>(vec, nx, 0.0, 0.0, 0.0, count, start, end);
  miniFE_vector_generate_superblock<S>(vec, nx, 1.0, 5.0 / 12, 8.0 / 12, count,
                                    start, end);
  for (int i = 0; i < nx - 3; i++)
    miniFE_vector_generate_superblock<S>(vec, nx, 1.0, 8.0 / 12, 1.0, count, start,
                                      end);
  miniFE_vector_generate_superblock<S>(vec, nx, 1.0, 5.0 / 12, 8.0 / 12, count,
                                    start, end);
  miniFE_vector_generate_superblock<S>(vec, nx, 0.0, 0.0, 0.0, count, start, end);

  return X;
}



template <class S = double>
static void generate_Laplace_matrix(int nx, int ny, int nz, int **rowPtr_out, LOCAL_ORDINAL **colInd_out, S ** nzVals_out) {
  // global dimension
  int n   = (nz > 1 ? nx * ny * nz : nx * ny);
  int nnz = (nz > 1 ? 7*n : 5*n);
  //Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
  //    "generate_MiniFE_Matrix::rowPtr", n + 1);
  //Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
  //    "generate_MiniFE_Matrix::colInd", nnz);
  //Kokkos::View<S *, Kokkos::HostSpace> nzVals(
  //    "generate_MiniFE_Matrix::values", nnz);
  int *rowPtr = new int[n+1];
  LOCAL_ORDINAL *colInd = new LOCAL_ORDINAL[nnz];
  S *nzVals = new S[nnz];

  nnz = 0;
  rowPtr[0] = 0;
  for (int ii = 0; ii < n; ii++) {
    S v = -1.0;
    int i, j, k, jj;
    k = ii / (nx*ny);
    i = (ii - k*nx*ny) / nx;
    j = ii - k*nx*ny - i*nx;

    if (k > 0) {
      jj = ii - nx * ny;
      colInd[nnz] = jj;
      nzVals[nnz] = v;
      nnz++;
    }
    if (k < nz-1) {
      jj = ii + nx * ny;
      colInd[nnz] = jj;
      nzVals[nnz] = v;
      nnz++;
    }

    if (i > 0) {
      jj = ii - nx;
      colInd[nnz] = jj;
      nzVals[nnz] = v;
      nnz++;
    }
    if (i < ny-1) {
      jj = ii + nx;
      colInd[nnz] = jj;
      nzVals[nnz] = v;
      nnz++;
    }

    if (j > 0) {
      jj = ii - 1;
      colInd[nnz] = jj;
      nzVals[nnz] = v;
      nnz++;
    }
    if (j < nx-1) {
      jj = ii + 1;
      colInd[nnz] = jj;
      nzVals[nnz] = v;
      nnz++;
    }

    v = nz > 1 ? 6.0 : 4.0;
    colInd[nnz] = ii;
    nzVals[nnz] = v;
    nnz++;

    rowPtr[ii+1] = nnz;
  }
  *rowPtr_out = rowPtr;
  *colInd_out = colInd;
  *nzVals_out = nzVals;
  //CrsMatrix<Kokkos::HostSpace, S> matrix(rowPtr, colInd, nzVals, n);
  //return matrix;
}

template <class S = double>
static void sort_matrix(CrsMatrix<Kokkos::HostSpace, S> &matrix) {
  // bouble-sort col_idx in each row
  int n = matrix.row_ptr.extent(0)-1;
  for (int i = 0; i < n; i++) {
    for (int k = matrix.row_ptr(i); k < matrix.row_ptr(i+1); k++) {
      for (int k2 = k-1; k2 >= matrix.row_ptr(i); k2--) {
        int k1 = k2+1; 
        if (matrix.col_idx[k1] < matrix.col_idx[k2]) {
          int idx = matrix.col_idx[k1];
          S val = matrix.values[k1];

          matrix.values[k1] = matrix.values[k2];
          matrix.col_idx[k1] = matrix.col_idx[k2];

          matrix.values[k2] = val;
          matrix.col_idx[k2] = idx;
        } else {
          break;
        }
      }
    }
  }
}

static void sort_graph(int n, int *row_ptr, int *col_idx) {
  // bouble-sort col_idx in each row
  for (int i = 0; i < n; i++) {
    for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
      for (int k2 = k-1; k2 >= row_ptr[i]; k2--) {
        int k1 = k2+1; 
        if (col_idx[k1] < col_idx[k2]) {
          int idx = col_idx[k1];
          col_idx[k1] = col_idx[k2];
          col_idx[k2] = idx;
        } else {
          break;
        }
      }
    }
  }
}


/// ========================================================= ///
template <class scalar_type = double>
static CrsMatrix<Kokkos::HostSpace, scalar_type> generate_matrix(bool strakos, scalar_type strakos_l1, scalar_type strakos_ln, scalar_type  strakos_p,
                                                                 std::string matrixFilename, int nx, int N, bool metis, bool verbose) {

  using HAType = CrsMatrix<Kokkos::HostSpace, scalar_type>;

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  HAType h_A;
  if (strakos) {
    if (numRanks == 1) {
      h_A._num_cols = N;
      h_A.nlocal = N;
      h_A.start_row = 0;
      h_A.end_row = N;
      Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
        "Matrix::rowPtr", N+1);
      Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
        "Matrix::colInd", N);
      Kokkos::View<scalar_type *, Kokkos::HostSpace> nzVal(
        "MM_Matrix::values", N);

      rowPtr(0) = 0;
      colInd(0) = 0;
      nzVal(0)  = strakos_l1;
      rowPtr(1) = 1;
      for (int i = 1; i < N-1; i++) {
        scalar_type val1 = strakos_ln - strakos_l1;
        scalar_type val2 = ((scalar_type)(i))/((scalar_type)(N-1));
        colInd(i) =  i;
        nzVal (i) =  strakos_l1 + val1 * val2 * pow(strakos_p, N-(i+1));
        rowPtr(i+1) = i+1;
      }
      colInd(N-1) = N-1;
      nzVal (N-1) = strakos_ln;
      rowPtr(N)   = N;
      h_A = HAType (rowPtr, colInd, nzVal, N);
      if (verbose && numRanks == 1) {
        std::cout << " generate Strakos diagonal matrix( N = " << N 
                  << " l1 = " << strakos_l1 << " ln = " << strakos_ln << " p = " << strakos_p << " )" << std::endl;
      }
    }
  } else if (matrixFilename != "" || nx > 0) {

    scalar_type *values;
    int         *col_idx;
    int         *row_ptr;
    int nnz = 0;
    int n   = 0;
    if (matrixFilename != "") {
      KokkosKernels::Impl::read_matrix<int, int, scalar_type>(
        &n, &nnz, &row_ptr, &col_idx, &values, matrixFilename.c_str());
      if (verbose && numRanks == 1) {
        std::cout << " read matrix( " << matrixFilename << " )" << std::endl;
      }
    } else if (nx > 0) {
      //HAType h_G = 
      Impl::generate_Laplace_matrix<scalar_type>(nx, nx, nx, &row_ptr, &col_idx, &values);
      //values = h_G.values.data();
      //col_idx = h_G.col_idx.data();
      //row_ptr = h_G.row_ptr.data();
      n = nx * nx *nx;
      nnz = row_ptr[n];
      if (verbose && numRanks == 1) {
        std::cout << " generate Laplace 3D( nx = " << nx << " nnz = " << nnz << " )" << std::endl;
      }
    }

    if (numRanks == 1) {
      // skip partitioning
      Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
        row_ptr, n+1);
      Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
        col_idx, nnz);
      Kokkos::View<scalar_type *, Kokkos::HostSpace> nzVal(
        values, nnz);
      h_A = HAType (rowPtr, colInd, nzVal, n);
      h_A.nlocal = n;
      h_A.start_row = 0;
      h_A.end_row = n;
    } else {
      // partition
      #if defined(CGSOLVE_ENABLE_METIS)
      if (metis) {
        int *parts = new int[n];
        if (myRank == 0) {
          idx_t n_metis = n;
          idx_t nnz = row_ptr[n];

          // remove diagonal elements (and casting to METIS idx_t)
          idx_t *metis_rowptr = new idx_t[n+1];
          idx_t *metis_colind = new idx_t[nnz];

          nnz = 0;
          metis_rowptr[0] = 0;
          for (int i = 0; i < n; i++) {
            for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
              if (col_idx[k] != i) {
                metis_colind[nnz] = col_idx[k];
                nnz ++;
              }
            }
            metis_rowptr[i+1] = nnz;
          }

          // call METIS
          idx_t ncon = 1;
          idx_t nparts = numRanks;
          idx_t objval = 0;
          idx_t *metis_perm = new idx_t[n];
          idx_t *metis_part = new idx_t[n];
          std::cout << "  + calling METIS_PartGraphKway: (n=" << n << ", nnz/n=" << float(nnz)/float(n) << ") " << std::endl;
          if (METIS_OK != METIS_PartGraphKway(&n_metis, &ncon, metis_rowptr, metis_colind,
                                              NULL, NULL, NULL, &nparts, NULL, NULL, NULL,
                                              &objval, metis_part)) {
            std::cout << std::endl << "METIS_NodeND failed" << std::endl << std::endl;
          }

          for (idx_t i = 0; i < n; i++) {
            parts[i] = metis_part[i];
          }

          delete [] metis_part;
          delete [] metis_rowptr;
          delete [] metis_colind;
        }
        // bcast partition and form perm/iperm
        MPI_Bcast(parts, n, MPI_INT, 0, MPI_COMM_WORLD);
        int *perm = new int[n];
        int *iperm = new int[n];
        int *part_ptr = new int[numRanks + 1];
        // part_ptr points to the begining of each part after ND
        for (int p = 0; p <= numRanks; p++) {
          part_ptr[p] = 0;
        }
        for (int p = 0; p < n; p++) {
          part_ptr[1+parts[p]] ++;
        }
        // part_map maps row id to part id after ND
        part_map = new int[n];
        for (int p = 0; p < numRanks; p++) {
          part_ptr[p+1] += part_ptr[p];
          for (int i = part_ptr[p]; i < part_ptr[p+1]; i++) {
            part_map[i] = p;
          }
        }
        // form perm/iperm
        int nnzlocal = 0;
        for (int i = 0; i < n; i++) {
          int p = parts[i];
          perm[part_ptr[p]] = i;
          iperm[i] = part_ptr[p];

          nnzlocal += (row_ptr[i+1] - row_ptr[i]);
          part_ptr[p] ++;
        }
        for (int p = numRanks; p > 0; p--) {
          part_ptr[p] = part_ptr[p-1];
        }
        part_ptr[0] = 0;
        /*if (myRank == 0) {
          //for (int i = 0; i < n; i++) printf( " perm[%d]=%d iperm[%d]=%d, map[%d]=%d\n",i,perm[i],i,iperm[i],i,part_map[i]);
          for (int i = 0; i < n; i++) printf( " %d %d %d\n",i,perm[i],iperm[i]);
        }*/

        // form permuted local matrix
        int start_row = part_ptr[myRank];
        int end_row = part_ptr[myRank+1];
        Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
          "Matrix::rowPtr", (end_row - start_row)+1);
        Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
          "Matrix::colInd", nnzlocal);
        Kokkos::View<scalar_type *, Kokkos::HostSpace> nzVal(
          "MM_Matrix::values", nnzlocal);

        nnzlocal = 0;
        rowPtr(0) = 0;
        for (int id = start_row; id < end_row; id++) {
          int i = perm[id];
          for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
            colInd(nnzlocal) = iperm[col_idx[k]];
            nzVal(nnzlocal) =  values[k];
            nnzlocal ++;
          }
          rowPtr(id-start_row+1) = nnzlocal;
        }
        h_A = HAType (rowPtr, colInd, nzVal, end_row - start_row);
        h_A.nlocal    = end_row - start_row;
        h_A.start_row = start_row;
        h_A.end_row   = end_row;
      } else
      #endif
      {
        if (myRank == 0) {
          std::cout << "  + using regular 1D partition of matrix : (n=" << n << ", nnz=" << nnz << ") " << std::endl;
        }
        int nlocal = (n + numRanks - 1) / numRanks;
        int start_row = myRank * nlocal;
        int end_row = (myRank + 1) * nlocal;
        if (end_row > n)
          end_row = n;

        int nnzlocal = row_ptr[end_row] - row_ptr[start_row];
        Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
          "Matrix::rowPtr", (end_row - start_row)+1);
        Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
          "Matrix::colInd", nnzlocal);
        Kokkos::View<scalar_type *, Kokkos::HostSpace> nzVal(
          "MM_Matrix::values", nnzlocal);

        nnzlocal = 0;
        rowPtr(0) = 0;
        for (int i = start_row; i < end_row; i++) {
          for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
            colInd(nnzlocal) =  col_idx[k];
            nzVal(nnzlocal) =  values[k];
            nnzlocal ++;
          }
          rowPtr(i-start_row+1) = nnzlocal;
        }
        h_A = HAType (rowPtr, colInd, nzVal, end_row - start_row);
        h_A.nlocal    = nlocal;
        h_A.start_row = start_row;
        h_A.end_row   = end_row;
      }
    }
    h_A._num_cols = n;
  } else {
    // generate mini-FE matrix
    h_A = Impl::generate_miniFE_matrix<scalar_type>(N);

    // global/local dimension
    int n = h_A.num_cols();
    h_A.nlocal = (n + numRanks - 1) / numRanks;
    h_A.start_row = myRank * h_A.nlocal;
    h_A.end_row = (myRank + 1) * h_A.nlocal;
    if (h_A.end_row > n)
      h_A.end_row = n;

    // resize rowptr
    Kokkos::resize(h_A.row_ptr, (h_A.end_row - h_A.start_row)+1);

    // convert the column indexes to "standard" global indexes
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = h_A.col_idx(k) / MASK;
        int idx = h_A.col_idx(k) % MASK;
        int start_row = p * h_A.nlocal;
        h_A.col_idx(k) = start_row + idx;
      }
    }
    if (verbose) {
      std::cout << " generate miniFE( N = " << N << " )" << std::endl;
    }
  }
  return h_A;
}

} // namespace Impl
#endif
