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

#include <generate_matrix.hpp>
#include <mpi.h>

#include "KokkosKernels_IOUtils.hpp"
#include "KokkosBlas3_gemm.hpp"
#include "KokkosSparse_spmv.hpp"

#include "KokkosBlas3_gemm_rr.hpp"
#include "KokkosBlas3_gemm_rr_combined.hpp"

// more detailed timer for SpMV
//#define CGSOLVE_SPMV_TIMER

// different options for CGsolver
#define CGSOLVE_GPU_AWARE_MPI
#define CGSOLVE_ENABLE_CUBLAS
//#define CGSOLVE_ENABLE_METIS

#if defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cusparse.h>
#endif
#if defined(CGSOLVE_ENABLE_METIS)
#include "metis.h"
#endif

#define CGSOLVE_CUDA_PROFILE
#if defined(CGSOLVE_CUDA_PROFILE)
#include "cuda_profiler_api.h"
#endif
#include "cblas.h"

using host_execution_space = typename Kokkos::HostSpace;
using      execution_space = typename Kokkos::DefaultExecutionSpace;

using         memory_space = typename execution_space::memory_space;

#define USE_FLOAT
#define USE_MIXED_PRECISION
#if defined(USE_FLOAT)
 using      scalar_type = float;

 #define MPI_SCALAR      MPI_FLOAT
 #define cusparseXcsrmv  cusparseScsrmv
 #define cusparseXcsrmm  cusparseScsrmm
 #define cublasXgemm     cublasSgemm
 #define cublasXgemv     cublasSgemv

 #define cblas_rgemv     cblas_sgemv
 #define cblas_rdot      cblas_sdot
 #if !defined(USE_MIXED_PRECISION) // unit-precision
  using gram_scalar_type = float;

  #define MPI_DOT_SCALAR      MPI_FLOAT
  #define cblas_xdot          cblas_sdot
  #define cblas_xaxpy         cblas_saxpy
  #define cblas_xgemv         cblas_sgemv

  using OutputCType = Kokkos::View<gram_scalar_type**>;
  using InputAType  = Kokkos::View<     scalar_type**>;
  using InputBType  = Kokkos::View<     scalar_type**>;
  // unit precision GEMM with float input/output
  template struct KokkosBlas::Impl::DotBasedGEMM<execution_space, InputAType, InputBType, OutputCType>;
 #else // mixed-precision
  using gram_scalar_type = double;

  #define MPI_DOT_SCALAR      MPI_DOUBLE
  #define cblas_xdot          cblas_ddot
  #define cblas_xaxpy         cblas_daxpy
  #define cblas_xgemv         cblas_dgemv

  using OutputCType = Kokkos::View<gram_scalar_type**>;
  using InputAType  = Kokkos::View<     scalar_type**>;
  using InputBType  = Kokkos::View<     scalar_type**>;
  // unit precision GEMM with double input/output (input vectors are casted to double before calling GEMM)
  template struct KokkosBlas::Impl::DotBasedGEMM<execution_space, OutputCType, OutputCType, OutputCType>;
  // mixed precision GEMM with float input and double output
  template struct KokkosBlas::Impl::DotBasedGEMM<execution_space, InputAType, InputBType, OutputCType>;
 #endif
#else
 using      scalar_type = double;
 using gram_scalar_type = double;

 #define MPI_DOT_SCALAR      MPI_DOUBLE
 #define MPI_SCALAR          MPI_DOUBLE
 #define cusparseXcsrmv      cusparseDcsrmv
 #define cusparseXcsrmm      cusparseDcsrmm
 #define cublasXgemm         cublasDgemm
 #define cublasXgemv         cublasDgemv

 #define cblas_xdot          cblas_ddot
 #define cblas_xaxpy         cblas_daxpy
 #define cblas_xgemv         cblas_dgemv

 #define cblas_rgemv         cblas_dgemv
 #define cblas_rdot          cblas_ddot
 #if defined(USE_MIXED_PRECISION)
 #include "KokkosBlas3_gemm_dd.hpp"
 #include "qd/dd_real.h"
 #include "mblas_dd.h"
 #endif
#endif

// -------------------------------------------------------------
// Auxiliary function to get jth column of MM
template <class VType, class MType>
VType getCol(int j, MType M) {
  Kokkos::pair<int, int> index(j, j+1);
  auto m = Kokkos::subview(M, Kokkos::ALL (), index);
  int nloc = M.extent(0);
  return VType(m.data(), nloc);
}


// -------------------------------------------------------------
// SpMV
template <class YType, class HAType, class AType, class XType, class MType, class SpaceType>
struct cgsolve_spmv
{
  using        integer_view_t = Kokkos::View<int *>;
  using   host_integer_view_t = Kokkos::View<int *, Kokkos::HostSpace>;
  using mirror_integer_view_t = typename integer_view_t::HostMirror;

  using  buffer_view_t = Kokkos::View<scalar_type *>;

  using team_policy_type  = Kokkos::TeamPolicy<SpaceType>;
  using member_type       = typename team_policy_type::member_type;

  using crsmat_t = KokkosSparse::CrsMatrix<scalar_type, int, execution_space, void, int>;
  using graph_t = typename crsmat_t::StaticCrsGraphType;

  cgsolve_spmv(int n_, int start_row_, int end_row_, HAType h_A_, AType A_, bool time_spmv_on_) :
  n(n_),
  start_row(start_row_),
  end_row(end_row_),
  h_A(h_A_),
  A(A_),
  time_spmv_on(time_spmv_on_),
  use_stream(false)
  {
    time_comm = 0.0;
    time_spmv = 0.0;

    int numRows = h_A.row_ptr.extent(0)-1;
    graph_t static_graph (A.col_idx, A.row_ptr);
    crsmat = crsmat_t ("CrsMatrix", numRows, A.values, static_graph);
  }

  int getGlobalDim() {
    return n;
  }
  int getLocalDim() {
    return h_A.row_ptr.extent(0)-1;
  }
  int getLocalNnz() {
    return h_A.row_ptr(getLocalDim());
  }

  int getStartRow() {
    return start_row;
  }
  int getEndRow() {
    return end_row;
  }


  // -------------------------------------------------------------
  // setup P2P
  void setup(int nlocal_, int nrhs_, int *part_map_) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // global/local dimension
    nrhs = nrhs_;
    nlocal = nlocal_;
    part_map = part_map_;

    // ----------------------------------------------------------
    // find which elements to receive from which process
    const scalar_type zero (0.0);

    num_recvs = integer_view_t("num_recvs", numRanks);
    host_num_recvs = Kokkos::create_mirror_view(num_recvs);
    host_integer_view_t check("check", n);
    Kokkos::deep_copy(check, 0);
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = -1;
        int col_id = h_A.col_idx(k);
        if (nlocal > 0) {
          p = col_id / nlocal;
        } else {
          p = part_map[col_id];
        }
        if (p != myRank && check(col_id) == 0
            && h_A.values(k) != zero) { // cheat?
          host_num_recvs(p) ++;
          check(col_id) = 1;
        }
      }
    }
    int total_recvs = 0;
    int num_neighbors_recvs = 0;
    host_integer_view_t dsp_recvs("dsp_recvs", numRanks+1);
    for (int p=0; p<numRanks; p++) {
      if (host_num_recvs(p) > 0) {
        total_recvs += host_num_recvs(p);
        num_neighbors_recvs ++;
      }
      dsp_recvs(p+1) = dsp_recvs(p) + host_num_recvs(p);
    }
    host_integer_view_t map_recvs("map_recvs", numRanks);
    ngb_recvs = integer_view_t("ngb_recvs", num_neighbors_recvs);
    ptr_recvs = integer_view_t("ptr_recvs", num_neighbors_recvs+1);
    host_ngb_recvs = Kokkos::create_mirror_view(ngb_recvs);
    host_ptr_recvs = Kokkos::create_mirror_view(ptr_recvs);

    max_num_recvs = 0;
    num_neighbors_recvs = 0;
    for (int p=0; p<numRanks; p++) {
      if (host_num_recvs(p) > 0) {
        host_ptr_recvs(num_neighbors_recvs+1) = host_ptr_recvs(num_neighbors_recvs) + host_num_recvs(p);
        host_ngb_recvs(num_neighbors_recvs) = p;
        map_recvs(p) = num_neighbors_recvs;
        num_neighbors_recvs ++;

        max_num_recvs = (host_num_recvs(p) > max_num_recvs ? host_num_recvs(p) : max_num_recvs);
      }
    }
    buf_recvs = buffer_view_t ("buf_recvs", total_recvs*nrhs);
    idx_recvs = integer_view_t("idx_recvs", total_recvs);
    host_idx_recvs = Kokkos::create_mirror_view(idx_recvs);

    Kokkos::deep_copy(check, 0);
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = -1;
        int col_id = h_A.col_idx(k);
        if (nlocal > 0) {
          p = col_id / nlocal;
        } else {
          p = part_map[col_id];
        }
        if (p != myRank && check(col_id) == 0
            && h_A.values(k) != zero) {
          int owner = map_recvs(p);
          host_idx_recvs(host_ptr_recvs(owner)) = col_id;
          host_ptr_recvs(owner) ++;

          check(h_A.col_idx(k)) = 1;
        }
      }
    }
    for (int p = num_neighbors_recvs; p > 0; p--) {
      host_ptr_recvs(p) = host_ptr_recvs(p-1);
    }
    host_ptr_recvs(0) = 0;

    // sort idx to receive from each process
    Impl::sort_graph(num_neighbors_recvs, host_ptr_recvs.data(), host_idx_recvs.data()); 


    /*if (myRank == 0) {
      for (int q = 0; q < num_neighbors_recvs; q++) {
        int p = host_ngb_recvs(q);
        for (int k = host_ptr_recvs(q); k < host_ptr_recvs(q+1); k++ ) {
          printf( "%d %d %d\n",q,p,host_idx_recvs(k) );
        }
      }
    }*/

    Kokkos::deep_copy(num_recvs, host_num_recvs);
    Kokkos::deep_copy(ptr_recvs, host_ptr_recvs);
    Kokkos::deep_copy(idx_recvs, host_idx_recvs);
    Kokkos::deep_copy(ngb_recvs, host_ngb_recvs);
    requests_recvs = (MPI_Request*)malloc(num_neighbors_recvs * sizeof(MPI_Request));

    // ----------------------------------------------------------
    // find which elements to send to which process
    host_integer_view_t num_sends("num_sends", numRanks);
    host_integer_view_t dsp_sends("dsp_sends", numRanks+1);
    MPI_Alltoall(&(host_num_recvs(0)), 1, MPI_INT, &(num_sends(0)), 1, MPI_INT, MPI_COMM_WORLD);
    int total_sends = 0;
    int num_neighbors_sends = 0;
    for (int p=0; p<numRanks; p++) {
      if (num_sends(p) > 0) {
        total_sends += num_sends(p);
        num_neighbors_sends ++;
      }
      dsp_sends(p+1) = dsp_sends(p) + num_sends(p);
    }
    ngb_sends = integer_view_t("ngb_sends", num_neighbors_sends);
    ptr_sends = integer_view_t("ptr_sends", num_neighbors_sends+1);
    host_ngb_sends = Kokkos::create_mirror_view(ngb_sends);
    host_ptr_sends = Kokkos::create_mirror_view(ptr_sends);

    num_neighbors_sends = 0;
    max_num_sends = 0;
    for (int p = 0; p < numRanks; p++) {
      //printf( " > %d: num_sends(%d) = %d, num_recvs(%d) = %d\n",myRank,p,num_sends(p),p,host_num_recvs(p) );
      if (num_sends(p) > 0) {
        host_ptr_sends(num_neighbors_sends+1) = host_ptr_sends(num_neighbors_sends) + num_sends(p);
        host_ngb_sends(num_neighbors_sends) = p;
        num_neighbors_sends ++;

        max_num_sends = (num_sends(p) > max_num_sends ? num_sends(p) : max_num_sends);
      }
      dsp_sends(p+1) = dsp_sends(p) + num_sends(p);
    }
    //printf( " %d: num_sends = %d, num_recvs = %d\n",myRank,ngb_sends.extent(0),ngb_recvs.extent(0) );

    buf_sends = buffer_view_t ("buf_sends", total_sends*nrhs);
    idx_sends = integer_view_t("idx_sends", total_sends);
    host_idx_sends = Kokkos::create_mirror_view(idx_sends);
    MPI_Alltoallv(&(host_idx_recvs(0)), &(host_num_recvs(0)), &(dsp_recvs(0)), MPI_INT,
                  &(host_idx_sends(0)), &(num_sends(0)), &(dsp_sends(0)), MPI_INT,
                  MPI_COMM_WORLD);
    #if defined(CGSOLVE_SPMV_TIMER)
    if (myRank == 0) {
      for (int p = 0; p < numRanks; p ++) {
        printf( " num_sends(%d) = %d, num_recvs(%d) = %d\n",p,num_sends(p),p,host_num_recvs(p) );
      }
    }
    #endif

    // sort idx to send to each process
    Impl::sort_graph(num_neighbors_sends, host_ptr_sends.data(), host_idx_sends.data()); 

    Kokkos::deep_copy(ptr_sends, host_ptr_sends);
    Kokkos::deep_copy(idx_sends, host_idx_sends);
    Kokkos::deep_copy(ngb_sends, host_ngb_sends);
    requests_sends = (MPI_Request*)malloc(num_neighbors_sends * sizeof(MPI_Request));

    #if defined(KOKKOS_ENABLE_CUDA)
    setup_cusparse();
    #endif
  }

  #if defined(KOKKOS_ENABLE_CUDA)
  void setStream(cudaStream_t *cudaStream_) {
    if (*cudaStream_ != NULL) {
      cudaStream = cudaStream_;
      space = SpaceType (*cudaStream);
      cusparseSetStream(cusparseHandle, *cudaStream);
      use_stream = true;
    }
  }
  void unsetStream() {
    use_stream = false;
    cusparseSetStream(cusparseHandle, NULL);
  }

  void setup_cusparse() {
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  }

  #if 0
  void setup_cusparse(YType y, XType x) {
    cusparseCreate(&cusparseHandle);

    int numCols = n;
    int numRows = h_A.row_ptr.extent(0)-1;
    int nnz = h_A.row_ptr(n);
    cusparseSpMatDescr_t A_cusparse;
    cusparseIndexType_t myCusparseOffsetType = CUSPARSE_INDEX_64I;
    cusparseIndexType_t myCusparseEntryType  = CUSPARSE_INDEX_64I;
    cudaDataType        myCudaDataType       = CUDA_R_64F;
    cusparseCreateCsr(&A_cusparse, numRows, numCols, nnz,
                      (void*) A.row_ptr.data(),
                      (void*) A.col_idx.data(),
                      (void*) A.values.data(),
                       myCusparseOffsetType,
                       myCusparseEntryType,
                       CUSPARSE_INDEX_BASE_ZERO,
                       myCudaDataType);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, x.extent_int(0), (void*) x.data(), myCudaDataType);
    cusparseCreateDnVec(&vecY, y.extent_int(0), (void*) y.data(), myCudaDataType);

    double alpha = 1.0;
    double beta  = 0.0;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, A_cusparse, vecX, 
                            &beta,              vecY,
                            myCudaDataType, CUSPARSE_MV_ALG_DEFAULT,
                            &bufferSize);

    void*  dBuffer = NULL;
    cudaMalloc(&dBuffer, bufferSize);
  }
  #endif
  #endif

  // -------------------------------------------------------------
  // P2P by MPI
  void fence() {
    #if defined(KOKKOS_ENABLE_CUDA)
    if (use_stream) {
      cudaStreamSynchronize(*cudaStream);
    } else {
      Kokkos::fence();
    }
    #else
    Kokkos::fence();
    #endif
  }

  // -------------------------------------------------------------
  // P2P by MPI
  void exchange(XType x) {

    // quick return
    if (numRanks <= 1) return;

    // prepar recv on host/device
    #if defined(CGSOLVE_SPMV_TIMER)
    Kokkos::Timer timer;
    if (time_spmv_on) {
      timer.reset();
    }
    #endif
    #if !defined(CGSOLVE_GPU_AWARE_MPI)
    auto host_recvs = Kokkos::create_mirror_view(buf_recvs);
    #endif
    int num_neighbors_recvs = ngb_recvs.extent(0);
    for (int q = 0; q < num_neighbors_recvs; q++) {
      int p = host_ngb_recvs(q);
      int start = host_ptr_recvs(q);
      int count = host_num_recvs(p); //host_ptr_recvs(q+1)-start;

      #if defined(CGSOLVE_GPU_AWARE_MPI)
      scalar_type *buffer = buf_recvs.data();
      MPI_Irecv(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #else
      MPI_Irecv(&(host_recvs(start)), count, MPI_SCALAR,  p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #endif
    }

    // pack to send on device
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_mpi = timer.seconds();
      timer.reset();
    }
    #endif
    int num_neighbors_sends = ngb_sends.extent(0);
    #if 1
    using range_policy_t = Kokkos::RangePolicy<SpaceType>;
    range_policy_t send_policy (space, 0, max_num_sends);
    Kokkos::parallel_for(
      "pack-for-send", send_policy,
      KOKKOS_LAMBDA(const int & k) {
        for(int q = 0; q < num_neighbors_sends; q++) {
            int p = ngb_sends(q);
            int start = ptr_sends(q);
            int count = ptr_sends(q+1)-start;
            if(k < count) {
              buf_sends(start+k) = x(idx_sends(start+k));
            }
        }
      });
    #else
    team_policy_type send_policy (space, max_num_sends, Kokkos::AUTO);
    Kokkos::parallel_for(
      "pack-for-send", send_policy,
      KOKKOS_LAMBDA(const member_type & team) {
        int k = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_neighbors_sends),
          [&](const int q) {
            int p = ngb_sends(q);
            int start = ptr_sends(q);
            int count = ptr_sends(q+1)-start;
            if(k < count) {
              buf_sends(start+k) = x(idx_sends(start+k));
            }
          });
      });
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      fence();
      time_comm_pack = timer.seconds();
      timer.reset();
    }
    #endif

    // copy to host
    #if !defined(CGSOLVE_GPU_AWARE_MPI)
    auto host_sends = Kokkos::create_mirror_view(buf_sends);
    Kokkos::deep_copy(host_sends, buf_sends);
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_copy = timer.seconds();
      timer.reset();
    }
    #endif

    // P2P with MPI
    fence(); // synch pack
    // send on host/device
    for (int q = 0; q < num_neighbors_sends; q++) {
      int p = host_ngb_sends(q);
      int start = host_ptr_sends(q);
      int count = host_ptr_sends(q+1)-start;
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      #if !defined(CGSOLVE_GPU_AWARE_MPI)
      MPI_Isend(&(host_sends(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #else
      scalar_type *buffer = buf_sends.data();
      MPI_Isend(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #endif
    }

    // wait on recv
    MPI_Waitall(num_neighbors_recvs, requests_recvs, MPI_STATUSES_IGNORE);
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_mpi += timer.seconds();
      time_comm_wait_recv = timer.seconds();
      timer.reset();
    }
    #endif

    // copy to device
    #if !defined(CGSOLVE_GPU_AWARE_MPI)
    Kokkos::deep_copy(buf_recvs, host_recvs);
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_copy += timer.seconds();
      timer.reset();
    }
    #endif

    // unpack on device
    #if 1
    using range_policy_t = Kokkos::RangePolicy<SpaceType>;
    range_policy_t recv_policy (space, 0, max_num_recvs);
    Kokkos::parallel_for(
      "unpack-for-recv", recv_policy,
      KOKKOS_LAMBDA(const int & k) {
        for(int q = 0; q < num_neighbors_recvs; q++) {
            int p = ngb_recvs(q);
            int start = ptr_recvs(q);
            int count = num_recvs(p); //ptr_recvs(q+1)-start;
            if (k < count) {
              x(idx_recvs(start+k)) = buf_recvs(start+k);
            }
        }
      });
    #else
    team_policy_type recv_policy (space, max_num_recvs, Kokkos::AUTO);
    Kokkos::parallel_for(
      "unpack-for-recv", recv_policy,
      KOKKOS_LAMBDA(const member_type & team) {
        int k = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 0, num_neighbors_recvs),
          [&](const int q) {
            int p = ngb_recvs(q);
            int start = ptr_recvs(q);
            int count = num_recvs(p); //ptr_recvs(q+1)-start;
            if (k < count) {
              x(idx_recvs(start+k)) = buf_recvs(start+k);
            }
          });
      });
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      fence();
      time_comm_unpack = timer.seconds();
      timer.reset();
    }
    #endif

    // wait for send
    MPI_Waitall(num_neighbors_sends, requests_sends, MPI_STATUSES_IGNORE);
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_mpi += timer.seconds();
      time_comm_wait_send = timer.seconds();
    }
    #endif
  }

  // -------------------------------------------------------------
  // P2P by MPI, for multiple vectors
  void exchange(int nrhs_, MType x) {

    // quick return
    if (numRanks <= 1) return;

    // prepar recv on host/device
    #if defined(CGSOLVE_SPMV_TIMER)
    Kokkos::Timer timer;
    if (time_spmv_on) {
      timer.reset();
    }
    #endif
    #if !defined(CGSOLVE_GPU_AWARE_MPI)
    auto host_recvs = Kokkos::create_mirror_view(buf_recvs);
    #endif
    int num_neighbors_recvs = ngb_recvs.extent(0);
    for (int q = 0; q < num_neighbors_recvs; q++) {
      int p = host_ngb_recvs(q);
      int start = nrhs_*host_ptr_recvs(q);
      int count = nrhs_*host_num_recvs(p); //host_ptr_recvs(q+1)-start;

      #if defined(CGSOLVE_GPU_AWARE_MPI)
      scalar_type *buffer = buf_recvs.data();
      MPI_Irecv(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #else
      MPI_Irecv(&(host_recvs(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #endif
    }

    // pack to send on device
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_mpi = timer.seconds();
      timer.reset();
    }
    #endif
    int num_neighbors_sends = ngb_sends.extent(0);
    #if 1
    using range_policy_t = Kokkos::RangePolicy<SpaceType>;
    range_policy_t send_policy (space, 0, max_num_sends);
    Kokkos::parallel_for(
      "pack-for-send", send_policy,
      KOKKOS_LAMBDA(const int & k) {
        for(int q = 0; q < num_neighbors_sends; q++) {
            int start = ptr_sends(q);
            int count = ptr_sends(q+1)-start;
            if(k < count) {
              for (int j = 0; j < nrhs_; j++) {
                buf_sends((k+start)*nrhs_+j) = x(idx_sends(start+k), j);
              }
            }
        }
      });
    #else
    team_policy_type send_policy (space, max_num_sends, Kokkos::AUTO);
    Kokkos::parallel_for(
      "pack-for-send", send_policy,
      KOKKOS_LAMBDA(const member_type & team) {
        int k = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_neighbors_sends),
          [&](const int q) {
            //int p = ngb_sends(q);
            int start = ptr_sends(q);
            int count = ptr_sends(q+1)-start;
            if(k < count) {
              for (int j = 0; j < nrhs_; j++) {
                buf_sends((k+start)*nrhs_+j) = x(idx_sends(start+k), j);
              }
            }
          });
      });
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      fence();
      time_comm_pack = timer.seconds();
      timer.reset();
    }
    #endif

    // copy to host
    #if !defined(CGSOLVE_GPU_AWARE_MPI)
    auto host_sends = Kokkos::create_mirror_view(buf_sends);
    Kokkos::deep_copy(host_sends, buf_sends);
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_copy = timer.seconds();
      timer.reset();
    }
    #endif

    // P2P with MPI
    fence(); // synch pack
    // send on host/device
    for (int q = 0; q < num_neighbors_sends; q++) {
      int p = host_ngb_sends(q);
      int start = nrhs_*host_ptr_sends(q);
      int count = nrhs_*(host_ptr_sends(q+1)-host_ptr_sends(q));
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      #if !defined(CGSOLVE_GPU_AWARE_MPI)
      MPI_Isend(&(host_sends(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #else
      scalar_type *buffer = buf_sends.data();
      MPI_Isend(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #endif
    }

    // wait on recv
    MPI_Waitall(num_neighbors_recvs, requests_recvs, MPI_STATUSES_IGNORE);
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_mpi += timer.seconds();
      time_comm_wait_recv = timer.seconds();
      timer.reset();
    }
    #endif

    // copy to device
    #if !defined(CGSOLVE_GPU_AWARE_MPI)
    Kokkos::deep_copy(buf_recvs, host_recvs);
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_copy += timer.seconds();
      timer.reset();
    }
    #endif

    // unpack on device
    #if 1
    using range_policy_t = Kokkos::RangePolicy<SpaceType>;
    range_policy_t recv_policy (space, 0, max_num_recvs);
    Kokkos::parallel_for(
      "unpack-for-recv", recv_policy,
      KOKKOS_LAMBDA(const int & k) {
        for(int q = 0; q < num_neighbors_recvs; q++) {
            int p = ngb_recvs(q);
            int start = ptr_recvs(q);
            int count = num_recvs(p); //ptr_recvs(q+1)-start;
            if (k < count) {
              for (int j = 0; j < nrhs_; j++) {
                x(idx_recvs(start+k),j) = buf_recvs((start+k)*nrhs_+j);
              }
            }
        }
      });
    #else
    team_policy_type recv_policy (space, max_num_recvs, Kokkos::AUTO);
    Kokkos::parallel_for(
      "unpack-for-recv", recv_policy,
      KOKKOS_LAMBDA(const member_type & team) {
        int k = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 0, num_neighbors_recvs),
          [&](const int q) {
            int p = ngb_recvs(q);
            int start = ptr_recvs(q);
            int count = num_recvs(p); //ptr_recvs(q+1)-start;
            if (k < count) {
              for (int j = 0; j < nrhs_; j++) {
                x(idx_recvs(start+k),j) = buf_recvs((start+k)*nrhs_+j);
              }
            }
          });
      });
    #endif
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      fence();
      time_comm_unpack = timer.seconds();
      timer.reset();
    }
    #endif

    // wait for send
    MPI_Waitall(num_neighbors_sends, requests_sends, MPI_STATUSES_IGNORE);
    #if defined(CGSOLVE_SPMV_TIMER)
    if (time_spmv_on) {
      time_comm_mpi += timer.seconds();
      time_comm_wait_send = timer.seconds();
    }
    #endif
  }

  // -------------------------------------------------------------
  // local SpMM
  void local_apply(MType y, MType x) {
    #if defined(CGSOLVE_ENABLE_CUBLAS)
     int numCols = n;
     int numRows = h_A.row_ptr.extent(0)-1;
     int numRHSs = y.extent(1);
     int nnz = h_A.row_ptr(numRows);

     int ldy = y.extent(0);
     int ldx = x.extent(0);
     scalar_type alpha (1.0);
     scalar_type beta  (0.0);
     cusparseXcsrmm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    numRows, numRHSs, numCols, nnz,
                    &alpha, descrA,
                            A.values.data(), A.row_ptr.data(), A.col_idx.data(),
                            x.data(), ldx,
                    &beta,  y.data(), ldx); // NOTE: assuming y is subview of x
    #endif
  }

  // -------------------------------------------------------------
  // local SpMV
  void local_apply(YType y, XType x) {
    #if defined(CGSOLVE_ENABLE_CUBLAS)
     scalar_type alpha (1.0);
     scalar_type beta  (0.0);

     #if 1
     int numCols = n;
     int numRows = h_A.row_ptr.extent(0)-1;
     int nnz = h_A.row_ptr(numRows);
     cusparseXcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    numRows, numCols, nnz,
                    &alpha, descrA,
                            A.values.data(), A.row_ptr.data(), A.col_idx.data(),
                            x.data(),
                    &beta,  y.data());
     #else
     KokkosSparse::spmv (KokkosSparse::NoTranspose, alpha, crsmat, x, beta, y);
     #endif
    #else
     #if defined(KOKKOS_ENABLE_CUDA)
     int rows_per_team = 16;
     int team_size = 16;
     #else
     int rows_per_team = 512;
     int team_size = 1;
     #endif

     int vector_length = 8;

     int nrows = y.extent(0);

     auto policy =
      require(Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                                   team_size, vector_length),
              Kokkos::Experimental::WorkItemProperty::HintHeavyWeight);
     Kokkos::parallel_for(
       "spmv", policy,
       KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
         const int first_row = team.league_rank() * rows_per_team;
         const int last_row = first_row + rows_per_team < nrows
                                      ? first_row + rows_per_team
                                      : nrows;

         Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, first_row, last_row),
            [&](const int row) {
              const int row_start = A.row_ptr(row);
              const int row_length = A.row_ptr(row + 1) - row_start;

              scalar_type y_row (0.0);
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, row_length),
                  [=](const int i, scalar_type &sum) {
                    int idx = A.col_idx(i + row_start);
                    sum += A.values(i + row_start) * x(idx);
                  },
                  y_row);
              y(row) = y_row;
            });
      });
    #endif
  }

  // -------------------------------------------------------------
  // apply (exchange + local SpMV)
  void apply(MType y, MType x) {
    Kokkos::Timer timer;
    if (time_spmv_on) {
      fence();
      timer.reset();
    }
    if (numRanks > 1) {
      this->exchange(2, x);
      if (time_spmv_on) {
        fence();
        time_comm = timer.seconds();
        timer.reset();
      }
    }
    this->local_apply(y, x);
    if (time_spmv_on) {
      fence();
      time_spmv = timer.seconds();
    }
  }

  void apply(YType y, MType x) {
    Kokkos::Timer timer;
    if (time_spmv_on) {
      fence();
      timer.reset();
    }
    if (numRanks > 1) {
      this->exchange(1, x);
      if (time_spmv_on) {
        fence();
        time_comm = timer.seconds();
        timer.reset();
      }
    }
    auto x0 = getCol<YType> (0, x);
    this->local_apply(y, x0);
    if (time_spmv_on) {
      fence();
      time_spmv = timer.seconds();
    }
  }

  double time_comm;
  double time_comm_copy;
  double time_comm_pack;
  double time_comm_unpack;
  double time_comm_mpi;
  double time_comm_wait_send;
  double time_comm_wait_recv;
  double time_spmv;

private:
  AType A;
  HAType h_A;
  crsmat_t crsmat;

  int n, nlocal;
  int start_row, end_row;
  int nrhs;
  int *part_map;

  int myRank, numRanks;
  MPI_Request *requests_sends;
  MPI_Request *requests_recvs;

  int max_num_recvs;
  buffer_view_t  buf_recvs;
  integer_view_t ngb_recvs; // store proc id of neighbors
  integer_view_t num_recvs; // number of elements to send to each process
  integer_view_t ptr_recvs; // pointer to the begining of idx_recvs for each neighbor
  integer_view_t idx_recvs; // store col indices of elements to receive

  int max_num_sends;
  buffer_view_t  buf_sends;
  integer_view_t ngb_sends; // store proc id of neighbors
  integer_view_t ptr_sends; // pointer to the begining of idx_recvs for each neighbor
  integer_view_t idx_sends; // store col indices of elements to receive

  // mirrored on host
  mirror_integer_view_t host_ngb_recvs; // store proc id of neighbors
  mirror_integer_view_t host_num_recvs; // number of elements to send to each process
  mirror_integer_view_t host_ptr_recvs; // pointer to the begining of idx_recvs for each neighbor
  mirror_integer_view_t host_idx_recvs; // store col indices of elements to receive

  mirror_integer_view_t host_ngb_sends; // store proc id of neighbors
  mirror_integer_view_t host_ptr_sends; // pointer to the begining of idx_recvs for each neighbor
  mirror_integer_view_t host_idx_sends; // store col indices of elements to receive

  bool time_spmv_on;

  bool use_stream;
  #if defined(KOKKOS_ENABLE_CUDA)
  cusparseHandle_t cusparseHandle;
  cudaStream_t    *cudaStream;
  cusparseMatDescr_t descrA;

  SpaceType space;
  #endif
};


// -------------------------------------------------------------
// dot
template <class XType>
inline void dot(XType x, scalar_type &result) {
  result = 0.0;
  Kokkos::parallel_reduce(
      "DOT", x.extent(0),
      KOKKOS_LAMBDA(const int &i, scalar_type &lsum) { lsum += x(i) * x(i); },
      result);
}

template <class YType, class XType>
inline void dot(YType y, XType x, scalar_type &result) {
  result = 0.0;
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int &i, scalar_type &lsum) { lsum += y(i) * x(i); },
      result);
}


// using view to keep result on device
template <class XType, class DType>
inline void dot(XType x, DType result) {
  Kokkos::deep_copy(result, 0.0);
  Kokkos::parallel_reduce(
      "DOT", x.extent(0),
      KOKKOS_LAMBDA(const int &i, scalar_type &lsum) { lsum += x(i) * x(i); },
      result);
}

template <class YType, class XType, class DType>
inline void dot(YType y, XType x, DType result) {
  Kokkos::deep_copy(result, 0.0);
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int &i, scalar_type &lsum) { lsum += y(i) * x(i); },
      result);
}


// using view to keep result on device, and use stream for overlap
template <class XType, class DType, class SpaceType>
inline void dot_stream(XType x, DType result, SpaceType space) {
  using range_policy_t = Kokkos::RangePolicy<SpaceType>;

  Kokkos::deep_copy(space, result, 0.0);
  Kokkos::parallel_reduce(
      "dot_stream(x^T * x)",
      range_policy_t(space, 0, x.extent(0)),
      KOKKOS_LAMBDA(const int &i, scalar_type &lsum) { lsum += x(i) * x(i); },
      result);
}

template <class YType, class XType, class DType, class SpaceType>
inline void dot_stream(YType y, XType x, DType result, SpaceType space) {
  using range_policy_t = Kokkos::RangePolicy<SpaceType>;

  Kokkos::deep_copy(space, result, 0.0);
  Kokkos::parallel_reduce(
      "dot_stream(y^T * x)",
      range_policy_t(space, 0, x.extent(0)),
      KOKKOS_LAMBDA(const int &i, scalar_type &lsum) { lsum += y(i) * x(i); },
      result);
}


template <class XType, class YType, class DType>
struct dots_stream {
  using scalar_type  = typename DType::non_const_value_type;

  typedef scalar_type value_type[];
  using size_type = typename XType::size_type;

  dots_stream (XType x_, YType y_, DType result_) :
  value_count (result_.extent(0)),
  result (result_),
  x (x_),
  y (y_)
  {}

  KOKKOS_INLINE_FUNCTION void
  operator() (const size_type i, value_type sum) const {
    sum[0] += x(i) * x(i);
    sum[1] += x(i) * y(i);
  }

  KOKKOS_INLINE_FUNCTION void
  join (      volatile value_type dst,
        const volatile value_type src) const {
    dst[0] += src[0];
    dst[1] += src[1];
  }
    
  KOKKOS_INLINE_FUNCTION void init (value_type dst) const {
    const scalar_type zero (0.0);
    dst[0] = zero;
    dst[1] = zero;
  }

  size_type value_count;

  DType result;
  XType x;
  YType y;
};



// -------------------------------------------------------------
// copy
template <class XType, class YType>
void local_copy(XType x, YType y) {
  #if 1
  int n = min(x.extent(0), y.extent(0));
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { x(i) = y(i); });
  #else
  Kokkos::deep_copy(x, y);
  #endif
}

template <class XType>
void local_init(XType x, scalar_type val) {
  #if 1
  int n = x.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { x(i) = val; });
  #else
  Kokkos::deep_copy(x, val);
  #endif
}

template <class XType, class YType>
void local_mv_copy(XType x, YType y) {
  #if 1
  int m = x.extent(0);
  int n = x.extent(1);
  Kokkos::parallel_for(
      "AXPBY", m,
      KOKKOS_LAMBDA(const int &i) { for (int j=0; j<n; j++) x(i, j) = y(i, j); });
  #else
  Kokkos::deep_copy(x, y);
  #endif
}

template <class XType>
void local_mv_init(XType x, scalar_type val) {
  #if 1
  int m = x.extent(0);
  int n = x.extent(1);
  Kokkos::parallel_for(
      "AXPBY", m,
      KOKKOS_LAMBDA(const int &i) { for (int j=0; j<n; j++) x(i, j) = val; });
  #else
  Kokkos::deep_copy(x, y);
  #endif
}

template <class VType_device, class VType_host>
void copy_to_device(VType_device dst, VType_host src) {
  using copy_scalar_type = typename VType_host::value_type;
  size_t count = min(dst.extent(0), src.extent(0)) * sizeof(copy_scalar_type);
  cudaMemcpyAsync(dst.data(), src.data(), count, cudaMemcpyHostToDevice); 
}

// -------------------------------------------------------------
// axpby
template <class ZType, class YType, class XType>
void axpby(ZType z, scalar_type alpha, XType x, scalar_type beta, YType y) {
  int n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
}

template <class ZType, class YType, class XType, class SpaceType>
void axpby(ZType z, scalar_type alpha, XType x, scalar_type beta, YType y, SpaceType space) {
  using range_policy_t = Kokkos::RangePolicy<SpaceType>;
  int n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", range_policy_t(space, 0, n),
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
}

template <class ZType, class YType, class XType>
void axpby(ZType z1, XType x1, scalar_type beta1, YType y1,
           ZType z2, XType x2, scalar_type beta2, YType y2) {
  int n = z1.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) {
        z1(i) = x1(i) + beta1 * y1(i);
        z2(i) = x2(i) + beta2 * y2(i);
      });
}

template <class ZType, class YType, class XType, class SpaceType>
void axpby(ZType z1, XType x1, scalar_type beta1, YType y1,
           ZType z2, XType x2, scalar_type beta2, YType y2,
           SpaceType space) {
  using range_policy_t = Kokkos::RangePolicy<SpaceType>;

  Kokkos::parallel_for(
      "axpby(space)",
      range_policy_t(space, 0, z1.extent(0)),
      KOKKOS_LAMBDA(const int &i) {
        z1(i) = x1(i) + beta1 * y1(i);
        z2(i) = x2(i) + beta2 * y2(i);
      });
}


// =============================================================
// cg_solve
template <class VType, class OP>
int cg_solve(VType x_out, OP op, VType b,
             int max_iter, scalar_type tolerance, int s, int dot_option,
             bool replace_residual, double replace_check_tol, int replace_op, int maxNnzA, scalar_type norma, bool merge_rr_dots,
             bool verbose, bool time_spmv_on, bool time_dot_on, bool time_axpy_on) {
 
  using GMType = Kokkos::View<gram_scalar_type**>;
  using GVType = Kokkos::View<gram_scalar_type*>;
  using  MType = Kokkos::View<scalar_type**>;
  using GVType_host = typename GVType::HostMirror;
  using  VType_host = typename  VType::HostMirror;
  using  DView = Kokkos::View<scalar_type*>;

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  const scalar_type one  (1.0);
  const scalar_type zero (0.0);

  // spmv timer
  Kokkos::Timer timer_cg;
  double time_cg = 0.0;
  // spmv timer
  Kokkos::Timer timer_spmv;
  Kokkos::Timer timer_copy;
  double time_spmv = 0.0;
  double time_spmv_comm   = 0.0;
  double time_spmv_spmv   = 0.0;
  double time_spmv_copy   = 0.0;
  #if defined(CGSOLVE_SPMV_TIMER)
  double time_spmv_pack   = 0.0;
  double time_spmv_unpack = 0.0;
  double time_spmv_mpi    = 0.0;
  double time_spmv_wait_send = 0.0;
  double time_spmv_wait_recv = 0.0;
  #endif
  // idot timer
  Kokkos::Timer timer_dot;
  double time_dot = 0.0;
  double time_dot_comm = 0.0;
  double flop_dot = 0.0;
  // for rr
  double time_dot_rr = 0.0;
  double time_dot_rr_comm = 0.0;
  double flop_dot_rr = 0.0;
  // for rr
  double time_dot_rr2 = 0.0;
  double time_dot_rr2_comm = 0.0;
  double flop_dot_rr2 = 0.0;

  Kokkos::Timer timer_axpy;
  double time_axpy = 0.0;
  double flop_axpy = 0.0;

  Kokkos::Timer timer_seq;
  double time_seq = 0.0;

  gram_scalar_type normr = zero;
  gram_scalar_type new_rr = 0.0;

  // residual vector
  int nloc = x_out.extent(0);
  MType PRX("PRX", nloc, 3);
  Kokkos::pair<int, int> pr_cols(0, 2);
  auto PR = Kokkos::subview(PRX, Kokkos::ALL (), pr_cols);
  auto p = getCol<VType> (0, PRX);
  auto r = getCol<VType> (1, PRX);
  auto x = getCol<VType> (2, PRX);

  // for residual replacement
  const auto eps = std::numeric_limits<scalar_type>::epsilon();
  const scalar_type replace_tol = std::sqrt(eps);
  scalar_type d_replace = 0.0;
  scalar_type d_replace_init = 0.0;
  scalar_type d_replace_prev = 0.0;
  scalar_type d_replace_check = 0.0;
  scalar_type d_replace_check_prev = 0.0;
  Kokkos::deep_copy(x_out, zero);
  int num_rr = 0;

  int s2p1 = 2*s+1;
  using RRType = Kokkos::View<CgsolverRR_Combined::ReduceRR::rr_sum<gram_scalar_type, scalar_type> **>;
  RRType T_rr_device ("T_rr",   s2p1, s2p1);
  MType G_hat_device ("G_hat",  s2p1, s2p1);
  MType T_hat_device ("T_hat",  s2p1, s2p1);
  MType B_hat_device ("B_hat",  s2p1, s2p1);
  auto G_hat  = Kokkos::create_mirror_view(G_hat_device);
  auto T_hat  = Kokkos::create_mirror_view(T_hat_device);
  auto B_hat  = Kokkos::create_mirror_view(B_hat_device);

  scalar_type normx = zero;
  VType v_hat_device ("v_hat",  s2p1);
  auto  v_hat = Kokkos::create_mirror_view(v_hat_device);
  VType_host w_hat   ("w_hat",  s2p1);

  // basis vectors
  int         n = op.getGlobalDim();
  int start_row = op.getStartRow();
  int   end_row = op.getEndRow();
  Kokkos::pair<int, int> local_rows(start_row, end_row);
  MType  V_global("V_global",  n, s2p1);
  auto V = Kokkos::subview(V_global, local_rows, Kokkos::ALL ());
  auto V01 = Kokkos::subview(V, Kokkos::ALL (), pr_cols);

  // option to explicitly cast
  GMType V2("V2",  nloc, s2p1);
  auto p0 = getCol<VType> (0, V);
  auto p1 = getCol<VType> (1, V);

  // change-of-basis
  GMType B_device ("B",  s2p1, s2p1);
  auto B  = Kokkos::create_mirror_view(B_device);
  // Gram
  #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
   int ldg = s2p1;
   int ldb = s2p1;
   dd_real *B_dd = (dd_real*)malloc(ldb * s2p1 * sizeof(dd_real));
   dd_real *G_dd = (dd_real*)malloc(ldg * s2p1 * sizeof(dd_real));

   using DDType = Kokkos::View<CgsolverDD::ReduceDD::dd_sum**>;
   DDType DD_device("DD", s2p1, s2p1);

   GMType T_dd_device ("T_dd",  s2p1, 2*s2p1);
   Kokkos::pair<int, int> dd_hi_cols(0,    1*s2p1);
   Kokkos::pair<int, int> dd_lo_cols(s2p1, 2*s2p1);
   auto T_device    = Kokkos::subview(T_dd_device, Kokkos::ALL (), dd_hi_cols);
   auto T_lo_device = Kokkos::subview(T_dd_device, Kokkos::ALL (), dd_lo_cols);

   auto T_dd  = Kokkos::create_mirror_view(T_dd_device);
   //auto T  = Kokkos::create_mirror_view(T_device);
   //auto T_lo  = Kokkos::create_mirror_view(T_lo_device);
   auto T    = Kokkos::subview(T_dd, Kokkos::ALL (), dd_hi_cols);
   auto T_lo = Kokkos::subview(T_dd, Kokkos::ALL (), dd_lo_cols);
  #else // not with DD
   GMType T_device ("T",  s2p1, s2p1);
   auto T  = Kokkos::create_mirror_view(T_device);
  #endif
  GMType G_device ("G",  s2p1, s2p1);
  auto G  = Kokkos::create_mirror_view(G_device);
  // alpha & beta
  int ldc = s2p1;
  int ldt = s2p1;
  GMType c_device ("c",  ldc, s+1);
  GMType t_device ("t",  ldt, s+1);
  GMType y_device ("y",  ldt, s+1);
  auto c  = Kokkos::create_mirror_view(c_device);
  auto t  = Kokkos::create_mirror_view(t_device);
  auto y  = Kokkos::create_mirror_view(y_device);
  auto c_device_s = getCol<GVType> (s, c_device);
  auto c_host_s   = Kokkos::create_mirror_view(c_device_s);
  // alpha & beta
  GMType cp_device ("c2",  ldc, s+1); // device
  GMType tp_device ("t2",  ldt, s+1); // device
  GMType yp_device ("y2",  ldt, s+1); // device
  auto cp_s = getCol<GVType> (s, cp_device); // device
  auto tp_s = getCol<GVType> (s, tp_device); // device
  auto yp_s = getCol<GVType> (s, yp_device); // device
  //
  auto cp  = Kokkos::create_mirror_view(cp_device); // host
  auto tp  = Kokkos::create_mirror_view(tp_device); // host
  auto yp  = Kokkos::create_mirror_view(yp_device); // host
  auto cp_s_host = getCol<GVType_host> (s, cp); // host
  auto tp_s_host = getCol<GVType_host> (s, tp); // host
  auto yp_s_host = getCol<GVType_host> (s, yp); // host
  //
  MType CTY ("CTY",  ldc, 3);
  auto c_s = getCol<VType> (0, CTY);
  auto t_s = getCol<VType> (1, CTY);
  auto y_s = getCol<VType> (2, CTY);
  // workspace
  GVType w_device ("w",  ldt);
  GVType c2_device("c2", ldc);
  auto c2 = Kokkos::create_mirror_view(c2_device);
  auto w  = Kokkos::create_mirror_view(w_device);
  #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
  dd_real *c_dd  = (dd_real*)malloc(ldc*(s+1)*sizeof(dd_real));
  dd_real *t_dd  = (dd_real*)malloc(ldt*(s+1)*sizeof(dd_real));
  dd_real *w_dd  = (dd_real*)malloc(ldt      *sizeof(dd_real));
  dd_real *y_dd  = (dd_real*)malloc(ldt*(s+1)*sizeof(dd_real));
  dd_real *c2_dd = (dd_real*)malloc(ldc      *sizeof(dd_real));
  dd_real one_dd  = {1.0, 0.0};
  dd_real zero_dd = {0.0, 0.0};;
  #endif
  // perm to go from V = [p,r,A*p,A*r, ..] to V = [p,A*p, .., r, A*r]
  int * perm = (int*)malloc(s2p1*sizeof(int));
  int *iperm = (int*)malloc(s2p1*sizeof(int));
  for (int i = 0; i < s; i++) {
    perm[2*i + 0] = i;
    perm[2*i + 1] = 1+i + s;
  }
  perm[2*s] = s;
  for (int i = 0; i < 2*s+1; i++) iperm[perm[i]] = i;

  // compute change-of-basis
  Kokkos::deep_copy(B, zero);
  for (int i = 0;   i < s;   i++) B(i+1, i) = one;
  for (int i = s+1; i < 2*s; i++) B(i+1, i) = one;
  #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
  memset(B_dd, 0, ldb * (2*s+1) * sizeof(dd_real));
  for (int i = 0;   i < s;   i++) B_dd[i+1 + i*ldb] = one_dd;
  for (int i = s+1; i < 2*s; i++) B_dd[i+1 + i*ldb] = one_dd;
  #endif
  for (int i = 0;   i < s;   i++) B_hat(i+1, i) = one;
  for (int i = s+1; i < 2*s; i++) B_hat(i+1, i) = one;

  #if defined(KOKKOS_DEBUG_CGSOLVER)
  printf("B = [\n" );
  for (int i = 0; i < 2*s+1; i++) {
    for (int j = 0; j < 2*s+1; j++) printf("%.2e ",B(i,j));
    printf("\n");
  }
  printf("];\n");
  #endif

  // to compute true-residual with verbose on  
  Kokkos::pair<int, int> bounds(start_row, end_row);
  VType r_true  ("true_r",  nloc);
  VType Ax      ("Ax", nloc);
  MType x_global("true_r",  n, 1);
  VType x_sub = Kokkos::subview(getCol<VType>(0, x_global), bounds);
  //#define KOKKOS_DEBUG_CGSOLVER
  #if 1//defined(KOKKOS_DEBUG_CGSOLVER)
  MType V_local ("V_local", nloc, s2p1);
  auto  T_rr_host = Kokkos::create_mirror_view(T_rr_device);
  auto  V_global_host = Kokkos::create_mirror_view(V_global);
  auto  r_host   = Kokkos::create_mirror_view(r);
  auto  x_host   = Kokkos::create_mirror_view(x);
  auto  V_host   = Kokkos::create_mirror_view(V_local);
  auto  V2_host  = Kokkos::create_mirror_view(V2);
  auto  PRX_host = Kokkos::create_mirror_view(PRX);
  auto  CTY_host = Kokkos::create_mirror_view(CTY);
  auto  p0_host  = Kokkos::create_mirror_view(p0);
  auto  x_out_host  = Kokkos::create_mirror_view(x);
  auto  x_sub_host  = Kokkos::create_mirror_view(x_sub);
  auto  r_true_host = Kokkos::create_mirror_view(r_true);
  #endif
  int printRank = min(0, numRanks-1);

  #if defined(KOKKOS_ENABLE_CUDA)
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;

  if (cudaStreamCreate(&cudaStream) != cudaSuccess) {
    printf( " ** faiiled to create cudaStreams **\n" );
    return -1;
  }
  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    printf( " ** faiiled to create cublasHandle **\n" );
    return -1;
  }
  //Kokkos::Cuda cudaSpace (cudaStream);
  //op.setStream(&(cudaStream));
  //cublasSetStream(cublasHandle, cudaStream);
  #endif

  // ==============================================================================
  // r = b - A*x
  axpby(p0, zero, x, one, x);   // p = x
  op.apply(p0, V_global);       // Ap = A*p
  axpby(r, one, b, -one, p0);   // r = b-Ap
  local_copy(p0, r);            // p = r
  local_copy(p1, r);            // p1 = r for SpMM at the first iteration

  //printf("Init: x, Ax, b, r\n" );
  //Kokkos::fence();
  //for (int i=0; i<b.extent(0); i++) printf(" %e, %e, %e, %e\n",x(i),AAr(i),b(i),r(i));

  // beta = r'*r (using Kokkos and non Cuda-aware MPI)
  Kokkos::View<scalar_type> dot_result("dot_result"); 
  auto dot_host = Kokkos::create_mirror_view(dot_result);
  dot(r, dot_result);
  if (numRanks > 1) {
    Kokkos::fence();
    MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);
  }
  Kokkos::deep_copy(dot_host, dot_result);
  new_rr = *(dot_host.data());
  normr = std::sqrt(new_rr);
  tolerance *= normr;

  if (verbose && myRank == printRank) {
    std::cout << "Initial Residual = " << normr << " Max iters = " << max_iter << " Tol = " << tolerance << std::endl;
  }
  if (replace_residual) {
    //printf( " %e / %e * (%d * %e * %e + %e) = %e\n",replace_tol,eps, maxNnzA,norma, sum_x, sum_r, d_replace );

    // compute norm(x)
    if (time_dot_on) {
      timer_dot.reset();
    }
    dot(x, x, dot_result);
    if (time_dot_on) {
      Kokkos::fence();
      time_dot_rr2 += timer_dot.seconds();
      flop_dot_rr2 += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
      timer_dot.reset();
    }
    if (numRanks > 1) {
      Kokkos::fence();
      MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                    MPI_COMM_WORLD);
    }
    Kokkos::deep_copy(dot_host, dot_result);
    normx = std::sqrt(*(dot_host.data()));
    d_replace = (eps / replace_tol) * (normr + scalar_type(maxNnzA)*norma*normx);
    d_replace_check = d_replace;
    d_replace_init = d_replace;
  }

  // ---------------------------
  // Main loop
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);
  #if defined(CGSOLVE_CUDA_PROFILE)
  cudaProfilerStart();
  #endif
  timer_cg.reset();
  int num_iters = 0;
  int step = s;
  // for residual-replacement
  bool replaced = true;
  scalar_type beta0_hat = zero;
  scalar_type beta1_hat = zero;
  scalar_type beta2_hat = zero;
  scalar_type beta3_hat = zero;
  for (int k = 0; k <= max_iter && normr > tolerance; k+=step) {

    // ==============================================================================
    // Matrix-Powers Kernel
    // V = [p,A*p, r, A^2*p, A*r, ....]
    if (time_spmv_on) {
      Kokkos::fence();
      timer_spmv.reset();
    }
    // V(0) = p
    // V(1) = r
    // V(2) = A*p
    auto p2 = getCol<VType> (2, V);
    op.apply(p2, V_global);
    if (time_spmv_on) {
      Kokkos::fence();
      time_spmv_comm   += op.time_comm;
      time_spmv_spmv   += op.time_spmv;
      #if defined(CGSOLVE_SPMV_TIMER)
      time_spmv_copy   += op.time_comm_copy;
      time_spmv_pack   += op.time_comm_pack;
      time_spmv_unpack += op.time_comm_unpack;
      time_spmv_mpi    += op.time_comm_mpi;
      time_spmv_wait_send += op.time_comm_wait_send;
      time_spmv_wait_recv += op.time_comm_wait_recv;
      #endif
    }
    for (int j = 1; j < s; j++) {
      #if 0
      Kokkos::pair<int, int> index_0(2*(j-1)+1,   2*(j-1)+2);
      auto V0_global = Kokkos::subview(V_global, Kokkos::ALL (), index_0);
      auto v2 = getCol<VType> (2*(j-1)+3, V);
      op.apply(v2, V0_global);

      Kokkos::pair<int, int> index_1(2*(j-1)+2, 2*(j-1)+3);
      auto V1_global = Kokkos::subview(V_global, Kokkos::ALL (), index_1);
      auto v3 = getCol<VType> (2*(j-1)+4, V);
      op.apply(v3, V1_global);
      #else
      Kokkos::pair<int, int> index_0(2*(j-1)+1, 2*(j-1)+3);
      Kokkos::pair<int, int> index_1(2*(j-1)+3, 2*(j-1)+5);
      auto V0_global = Kokkos::subview(V_global, Kokkos::ALL (), index_0);
      auto V1        = Kokkos::subview(V,        Kokkos::ALL (), index_1);
      op.apply(V1, V0_global);
      #endif
      if (time_spmv_on) {
        time_spmv_comm   += op.time_comm;
        time_spmv_spmv   += op.time_spmv;
        #if defined(CGSOLVE_SPMV_TIMER)
        time_spmv_copy   += op.time_comm_copy;
        time_spmv_pack   += op.time_comm_pack;
        time_spmv_unpack += op.time_comm_unpack;
        time_spmv_mpi    += op.time_comm_mpi;
        time_spmv_wait_send += op.time_comm_wait_send;
        time_spmv_wait_recv += op.time_comm_wait_recv;
        #endif
      }
    }
    if (time_spmv_on) {
      Kokkos::fence();
      time_spmv += timer_spmv.seconds();
    }

    // ==============================================================================
    // update residual-replacement coefficient, if needed
    // NOTE: moved/delayed here to merge with dot-product to compute G
    if (replace_residual) {

      // ==============================================================================
      // Dot-product

      // -------------------------------------------------
      // local dot-products
      Kokkos::fence();
      if (time_dot_on) {
        timer_dot.reset();
      }
      #if defined(USE_FLOAT) | !defined(USE_MIXED_PRECISION)
       if (merge_rr_dots) {
         #if 0
          if (!replaced) {
            if (time_dot_on) {
              timer_dot.reset();
            }
            dot(x, x, dot_result);
            if (time_dot_on) {
              Kokkos::fence();
              time_dot_rr2 += timer_dot.seconds();
              flop_dot_rr2 += (2*nloc-1);
              timer_dot.reset();
            }
          }
         #endif

         // combined dot-products to compute G and G_hat, and xnorm if replaced
         CgsolverRR_Combined::
         DotBasedGEMM<execution_space, MType, MType, RRType, VType> gemm(one,  V, V, zero, T_rr_device, replaced, x);
         gemm.run();

         // extract T & T_hat
         Kokkos::parallel_for(
           "copy-DD-T", 2*s+1,
           KOKKOS_LAMBDA(const int & i) {
             for (int j = i; j < 2*s+1; j++) {
               T_device(i,j) = T_rr_device(i,j).val;
             }
             for (int j = i; j < 2*s+1; j++) {
               T_hat_device(i,j) = T_rr_device(i,j).val_hat;
             }
             if (i == 0) *(dot_result.data()) = T_rr_device(0, 0).xnorm;
           });
         if (time_dot_on) {
           Kokkos::fence();
           time_dot += timer_dot.seconds();
           flop_dot += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
           timer_dot.reset();
         }
       } else {
         // dot-products to compute G
         KokkosBlas::Impl::
         DotBasedGEMM<execution_space, MType, MType, GMType> gemm(one, V, V, zero, T_device, true);
         gemm.run(false);
         if (time_dot_on) {
           Kokkos::fence();
           time_dot += timer_dot.seconds();
           flop_dot += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
           timer_dot.reset();
         }

         // another dot to compute G_hat for residual replacement
         CgsolverRR::
         DotBasedGEMM<execution_space, MType, MType, MType> gemm_hat(one,  V, V, zero, T_hat_device, true);
         gemm_hat.run(false);
         if (time_dot_on) {
           Kokkos::fence();
           time_dot_rr += timer_dot.seconds();
           flop_dot_rr += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
           timer_dot.reset();
         }
       }
      #else
       // dot-products to compute G
       CgsolverDD::
       DotBasedGEMM_dd<execution_space, MType, MType, DDType> gemm(one,  V, V, zero, DD_device);
       gemm.run();
       if (time_dot_on) {
         Kokkos::fence();
         time_dot += timer_dot.seconds();
         flop_dot += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
         timer_dot.reset();
       }

       // another dot to compute G_hat for residual replacement
       CgsolverRR::
       DotBasedGEMM<execution_space, MType, MType, MType> gemm_hat(one,  V, V, zero, T_hat_device, true);
       gemm_hat.run(false);
       if (time_dot_on) {
         Kokkos::fence();
         time_dot_rr += timer_dot.seconds();
         flop_dot_rr += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
         timer_dot.reset();
       }

       // copying DD into T (aka MPI buffer)
       Kokkos::parallel_for(
         "copy-DD-T", 2*s+1,
         KOKKOS_LAMBDA(const int & i) {
           for (int j = i; j < 2*s+1; j++) {
             T_device(i,j) = DD_device(i,j).val_hi;
           }
           for (int j = i; j < 2*s+1; j++) {
             T_lo_device(i,j) = DD_device(i,j).val_lo;
           }
         });
      #endif // defined(USE_FLOAT) | !defined(USE_MIXED_PRECISION)

      // -------------------------------------------------
      // global reduce
      if (numRanks > 1) {
        Kokkos::fence();
        // global-reduce to form G
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        MPI_Allreduce(MPI_IN_PLACE, T_dd_device.data(), 2*(s2p1*s2p1), MPI_DOT_SCALAR, MPI_SUM, MPI_COMM_WORLD);
        #else
        MPI_Allreduce(MPI_IN_PLACE, T_device.data(), s2p1*s2p1, MPI_DOT_SCALAR, MPI_SUM, MPI_COMM_WORLD);
        #endif

        // global-reduce to form G_hat for residual replacement
        MPI_Allreduce(MPI_IN_PLACE, T_hat_device.data(), s2p1*s2p1, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);

        if (!replaced && merge_rr_dots) {
          MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                        MPI_COMM_WORLD);
        }
      }
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      Kokkos::deep_copy(T_dd, T_dd_device);
      #else
      Kokkos::deep_copy(T, T_device);
      #endif
      Kokkos::deep_copy(T_hat, T_hat_device);
      if (!replaced && merge_rr_dots) {
        // xnorm
        Kokkos::deep_copy(dot_host, dot_result);
        //Kokkos::deep_copy(dot_host, T_rr_device(0,0).xnorm);
        beta0_hat = std::sqrt(*(dot_host.data()));
        d_replace += (eps / replace_tol) * (norma * (beta0_hat + scalar_type(2 + 2*maxNnzA) * beta1_hat) + scalar_type(maxNnzA) * beta2_hat);
      }
      if (time_dot_on) {
        time_dot_rr_comm += timer_dot.seconds();
      }
    } else { // not residual-replacement

      // ==============================================================================
      // Dot-product
      Kokkos::fence();
      if (time_dot_on) {
        timer_dot.reset();
      }
      if (dot_option == 1) {
        local_mv_copy(V2, V);
        KokkosBlas::Impl::
        DotBasedGEMM<execution_space, GMType, GMType, GMType> gemm(one, V2, V2, zero, T_device, true);
        gemm.run(false);
      } else {
        // directly calling dot-based Gemm since for mixed precision, KokkosBlas::gemm ends up calling ``standard'' implementation
        #if defined(USE_FLOAT) | !defined(USE_MIXED_PRECISION)
         KokkosBlas::Impl::
         DotBasedGEMM<execution_space, MType, MType, GMType> gemm(one, V, V, zero, T_device, true);
         gemm.run(false);
        #else
         CgsolverDD::
         DotBasedGEMM_dd<execution_space, MType, MType, DDType> gemm(one,  V, V, zero, DD_device);
         gemm.run();

         // copying DD into T (aka MPI buffer)
         Kokkos::parallel_for(
           "copy-DD-T", 2*s+1,
           KOKKOS_LAMBDA(const int & i) {
             for (int j = i; j < 2*s+1; j++) {
               T_device(i,j) = DD_device(i,j).val_hi;
             }
             for (int j = i; j < 2*s+1; j++) {
               T_lo_device(i,j) = DD_device(i,j).val_lo;
             }
           });
        #endif // defined(USE_FLOAT) | !defined(USE_MIXED_PRECISION)
      }
      if (time_dot_on) {
        Kokkos::fence();
        time_dot += timer_dot.seconds();
        flop_dot += ((s2p1*s2p1+s2p1)/2)*(2*nloc-1);
        timer_dot.reset();
      }

      if (numRanks > 1) {
        Kokkos::fence();
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        MPI_Allreduce(MPI_IN_PLACE, T_dd_device.data(), 2*(s2p1*s2p1), MPI_DOT_SCALAR, MPI_SUM, MPI_COMM_WORLD);
        #else
        MPI_Allreduce(MPI_IN_PLACE, T_device.data(), s2p1*s2p1, MPI_DOT_SCALAR, MPI_SUM, MPI_COMM_WORLD);
        #endif
      }
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      Kokkos::deep_copy(T_dd, T_dd_device);
      //Kokkos::deep_copy(T, T_device);
      //Kokkos::deep_copy(T_lo, T_lo_device);
      #else
      Kokkos::deep_copy(T, T_device);
      #endif
      if (time_dot_on) {
        time_dot_comm += timer_dot.seconds();
      }
    }

    for (int i = 0; i < 2*s+1; i++) {
      for (int j = i; j < 2*s+1; j++) {
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        G_dd[perm[i] + perm[j]*ldg].x[0] = T(i, j);
        G_dd[perm[i] + perm[j]*ldg].x[1] = T_lo(i, j);;

        G_dd[perm[j] + perm[i]*ldg].x[0] = T(i, j);
        G_dd[perm[j] + perm[i]*ldg].x[1] = T_lo(i, j);;
        #else
        G(perm[i], perm[j]) = T(i, j);
        G(perm[j], perm[i]) = T(i, j);
        #endif
        if (replace_residual) {
          G_hat(perm[i], perm[j]) = T_hat(i, j);
          G_hat(perm[j], perm[i]) = T_hat(i, j);
        }
      }
    }

    #if defined(KOKKOS_DEBUG_CGSOLVER)
    if (myRank == printRank) {
      printf("\n");
      Kokkos::deep_copy(V_local, V);
      Kokkos::deep_copy(V_host, V_local);
      printf("V = [\n" );
      for (int i = 0; i < nloc; i++) {
        printf("%d ", i);
        for (int j = 0; j < 2*s+1; j++) printf("%.16e ",V_host(i,j));
        printf("\n");
      }
      printf("];\n");
      /*if (dot_option == 1) {
        Kokkos::deep_copy(V2_host, V2);
        printf("V2 = [\n" );
        for (int i = 0; i < nloc; i++) {
          for (int j = 0; j < 2*s+1; j++) printf("%.2e ",V2_host(i,j));
          printf("\n");
        }
        printf("];\n");
      }*/
      //printf("perm = [\n" );
      //for (int i = 0; i < 2*s+1; i++) printf( "%d\n",perm[i] );
      //printf("];\n");
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      printf("T = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < 2*s+1; j++) printf("%.2e+%.2e ",T(i,j),T_lo(i,j));
        printf("\n");
      }
      printf("];\n");
      #else
      printf("T = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < 2*s+1; j++) printf("%.2e ",T(i,j));
        printf("\n");
      }
      printf("];\n");
      printf("G = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < 2*s+1; j++) printf("%.2e ",G(i,j));
        printf("\n");
      }
      printf("];\n");
      #endif
    }
    #endif

    // ==============================================================================
    // Convergence check (delayed), G = V'*V, where V = [p,r,Ap,Ar,...]
    new_rr = T(1, 1);
    normr = std::sqrt(new_rr);
    if (normr <= tolerance) break;


    // ==============================================================================
    // Compute "alpha" & "beta" (local redundant compute)
    if (time_axpy_on) {
      timer_seq.reset();
    }
    #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
    for (int i = 0; i < ldc; i++) {
      t_dd[i] = zero_dd;
      c_dd[i] = zero_dd;
      y_dd[i] = zero_dd;
    }
    t_dd[s+1] = one_dd;
    c_dd[0]   = one_dd;
    #else
    auto t0 = getCol<GVType_host> (0, t);
    auto c0 = getCol<GVType_host> (0, c);
    auto y0 = getCol<GVType_host> (0, y);
    memset(t0.data(), 0, ldc*sizeof(gram_scalar_type));
    memset(c0.data(), 0, ldc*sizeof(gram_scalar_type));
    memset(y0.data(), 0, ldc*sizeof(gram_scalar_type));
    t0(s+1) = one;
    c0(0) = one;
    #endif
    for (int ii = 0; ii < 2*s+1; ii++) {
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      yp(ii, 0) = y_dd[perm[ii]].x[0] + y_dd[perm[ii]].x[1];
      cp(ii, 0) = c_dd[perm[ii]].x[0] + c_dd[perm[ii]].x[1];
      tp(ii, 0) = t_dd[perm[ii]].x[0] + t_dd[perm[ii]].x[1];
      #else
      yp(ii, 0) = y(perm[ii], 0);
      cp(ii, 0) = c(perm[ii], 0);
      tp(ii, 0) = t(perm[ii], 0);
      #endif
    }

    scalar_type rone  (1.0);
    scalar_type rzero (0.0);
    #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
    dd_real alpha  = zero_dd;
    dd_real beta   = zero_dd;
    dd_real alpha1 = zero_dd;
    dd_real alpha2 = zero_dd;
    dd_real  beta1 = zero_dd;
    #else
    gram_scalar_type cone  (1.0);
    gram_scalar_type czero (0.0);

    gram_scalar_type alpha  = czero;
    gram_scalar_type beta   = czero;
    gram_scalar_type alpha1 = czero;
    gram_scalar_type alpha2 = czero;
    gram_scalar_type  beta1 = czero;
    #endif
    step = s;
    replaced = false;
    for (int i = 0; i < s; i++) {

      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      dd_real *ti0_dd = &t_dd[(i+0)*ldt];
      dd_real *ci0_dd = &c_dd[(i+0)*ldc];
      dd_real *yi0_dd = &y_dd[(i+0)*ldt];
      dd_real *ti1_dd = &t_dd[(i+1)*ldt];
      dd_real *ci1_dd = &c_dd[(i+1)*ldc];
      dd_real *yi1_dd = &y_dd[(i+1)*ldt];
      #else
      auto ti0 = getCol<GVType_host> (i+0, t);
      auto ci0 = getCol<GVType_host> (i+0, c);
      auto yi0 = getCol<GVType_host> (i+0, y);
      auto ti1 = getCol<GVType_host> (i+1, t);
      auto ci1 = getCol<GVType_host> (i+1, c);
      auto yi1 = getCol<GVType_host> (i+1, y);
      #endif

      // c2 = B*c
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      Rgemv ("N",
             2*s+1, 2*s+1,
             one_dd,  B_dd, ldb,
                      ci0_dd, 1,
             zero_dd, c2_dd,  1);
      #else
      cblas_xgemv (CblasColMajor, CblasNoTrans,
            2*s+1, 2*s+1,
            cone,  B.data(), 2*s+1,
                   ci0.data(), 1,
            czero, c2.data(),  1);
      #endif

      // compute alpha
      // > alpha1 = t'(G*t)
      if (i == 0) {
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        Rgemv ("N",
               2*s+1, 2*s+1,
               one_dd,  G_dd,  ldg,
                        ti0_dd, 1,
               zero_dd, w_dd,   1);
        alpha1 = Rdot(2*s+1, w_dd, 1, ti0_dd, 1);
        #else
        cblas_xgemv (CblasColMajor, CblasNoTrans,
              2*s+1, 2*s+1,
              cone,  G.data(), 2*s+1,
                     ti0.data(), 1,
              czero, w.data(),   1);
        alpha1 = cblas_xdot(2*s+1, w.data(), 1, ti0.data(), 1);
        #endif
      } else {
        alpha1 = beta1;
      }
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      // > alpha2 = c'*(G*c2)
      Rgemv ("N",
             2*s+1, 2*s+1,
             one_dd,  G_dd, ldg,
                      c2_dd, 1,
             zero_dd, w_dd,  1);
      alpha2 = Rdot(2*s+1, w_dd, 1, ci0_dd, 1);

      // > alpha = alpha1/alpha2
      alpha = alpha1 / alpha2;
      //printf( " alpha1 = %.2e+%.2e, alpha2=%.2e+%.2e, alpha=%.2e+%.2e\n",alpha1.x[0],alpha1.x[1], alpha2.x[0],alpha2.x[1], alpha.x[0],alpha.x[1] );
      #else
      // > alpha2 = c'*(G*c2)
      cblas_xgemv (CblasColMajor, CblasNoTrans,
            2*s+1, 2*s+1,
            cone,  G.data(), 2*s+1,
                   c2.data(), 1,
            czero, w.data(),  1);
      alpha2 = cblas_xdot(2*s+1, w.data(), 1, ci0.data(), 1);

      // > alpha = alpha1/alpha2
      alpha = alpha1 / alpha2;
      #endif

      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      // update y = y + alpha*c
      memcpy(yi1_dd, yi0_dd, ldt*sizeof(dd_real));
      Raxpy (2*s+1, 
             alpha, ci0_dd, 1,
                    yi1_dd, 1);
      // update t = t - alpha*c2
      memcpy(ti1_dd, ti0_dd, ldt*sizeof(dd_real));
      Raxpy (2*s+1, 
            -alpha, c2_dd,  1,
                    ti1_dd, 1);
      // > beta1 = t'(G*t)
      Rgemv ("T",
             2*s+1, 2*s+1,
             one_dd,  G_dd, 2*s+1,
                      ti1_dd, 1,
             zero_dd, w_dd,   1);
      beta1 = Rdot(2*s+1, w_dd, 1, ti1_dd, 1);
      beta = beta1 / alpha1;
      //printf( " beta1 = %.2e+%.2e, beta=%.2e+%.2e\n\n",beta1.x[0],beta1.x[1], beta.x[0],beta.x[1] );
      // update c = t + beta*c
      memcpy(ci1_dd, ti1_dd, ldt*sizeof(dd_real));
      Raxpy (2*s+1, 
             beta, ci0_dd, 1,
                   ci1_dd, 1);
      #else
      // update y = y + alpha*c
      memcpy(yi1.data(), yi0.data(), ldt*sizeof(gram_scalar_type));
      cblas_xaxpy (
            2*s+1, 
            alpha, ci0.data(), 1,
                   yi1.data(), 1);
      // update t = t - alpha*c2
      memcpy(ti1.data(), ti0.data(), ldt*sizeof(gram_scalar_type));
      cblas_xaxpy (
            2*s+1, 
           -alpha, c2.data(),  1,
                   ti1.data(), 1);
      // > beta1 = t'(G*t)
      cblas_xgemv (CblasColMajor, CblasTrans,
            2*s+1, 2*s+1,
            cone,  G.data(), 2*s+1,
                   ti1.data(), 1,
            czero, w.data(),   1);
      beta1 = cblas_xdot(2*s+1, w.data(), 1, ti1.data(), 1);
      beta = beta1 / alpha1;
      //if (myRank == 0) {
      //  std::cout << " " << i << " / " << s-1 << " beta1 = " << beta1 << ", alpha1 = " << alpha1
      //            << ", beta = " << beta << std::endl << std::endl;
      //}
      // update c = t + beta*c
      memcpy(ci1.data(), ti1.data(), ldt*sizeof(gram_scalar_type));
      cblas_xaxpy (
            2*s+1, 
            beta, ci0.data(), 1,
                  ci1.data(), 1);
      #endif
      for (int ii = 0; ii < 2*s+1; ii++) {
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        yp(ii, i+1) = y_dd[perm[ii] + (i+1)*ldt].x[0] + y_dd[perm[ii] + (i+1)*ldt].x[1];
        cp(ii, i+1) = c_dd[perm[ii] + (i+1)*ldc].x[0] + c_dd[perm[ii] + (i+1)*ldc].x[1];
        tp(ii, i+1) = t_dd[perm[ii] + (i+1)*ldt].x[0] + t_dd[perm[ii] + (i+1)*ldt].x[1];
        #else
        yp(ii, i+1) = y(perm[ii], i+1);
        cp(ii, i+1) = c(perm[ii], i+1);
        tp(ii, i+1) = t(perm[ii], i+1);
        #endif
      }
      if (replace_residual) {
        // NOTE: these are done in working precision
        // || |V_k| |y| || ~ sqrt(|y|^T * G_hat * |y|)
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(yi1_dd[ii].x[0] + yi1_dd[ii].x[1]);
        #else
        for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(yi1(ii));
        #endif
        cblas_rgemv (CblasColMajor, CblasNoTrans,
              2*s+1, 2*s+1,
              rone,  G_hat.data(), 2*s+1,
                     v_hat.data(), 1,
              rzero, w_hat.data(),   1);
        beta1_hat = std::sqrt( cblas_rdot(2*s+1, w_hat.data(), 1, v_hat.data(), 1) );

        // || |V_k| |r| || ~ sqrt(|t|^T * G_hat * |t|)
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(ti1_dd[ii].x[0] + ti1_dd[ii].x[1]);
        #else
        for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(ti1(ii));
        #endif
        cblas_rgemv (CblasColMajor, CblasNoTrans,
              2*s+1, 2*s+1,
              rone,  G_hat.data(), 2*s+1,
                     v_hat.data(), 1,
              rzero, w_hat.data(),   1);
        beta2_hat = std::sqrt( cblas_rdot(2*s+1, w_hat.data(), 1, v_hat.data(), 1) );

        // || |V_k| |B| |y| || ~ sqrt(|y|^T * B_hat^T * G_hat * B_hat * |y|)
        #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
        for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(yi1_dd[ii].x[0] + yi1_dd[ii].x[1]);
        #else
        for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(yi1(ii));
        #endif
        cblas_rgemv (CblasColMajor, CblasNoTrans,
              2*s+1, 2*s+1,
              rone,  B_hat.data(), 2*s+1,
                     v_hat.data(), 1,
              rzero, w_hat.data(),   1);
        cblas_rgemv (CblasColMajor, CblasNoTrans,
              2*s+1, 2*s+1,
              rone,  G_hat.data(), 2*s+1,
                     w_hat.data(), 1,
              rzero, v_hat.data(),   1);
        beta3_hat = std::sqrt( cblas_rdot(2*s+1, w_hat.data(), 1, v_hat.data(), 1) );

        d_replace_prev = d_replace;
        d_replace += (eps / replace_tol) * (scalar_type(4 + maxNnzA) * (norma * beta1_hat + beta3_hat) + beta2_hat);
        //std::cout << " > beta_hat = " << beta0_hat << ", " << beta1_hat << ", " << beta2_hat << ", " << beta3_hat << std::endl;
        //printf( " ++ %e / %e * (%d * (%e * %e + %e) + %e = %e\n",eps,replace_tol, 4+maxNnzA,norma,beta1_hat,beta3_hat,beta2_hat,d_replace );

        // ==============================================================================
        // replace residual vector, check
        d_replace_check_prev = d_replace_check;
        d_replace_check = beta2_hat;
        if (replace_op == 0 || (d_replace > d_replace_check &&
                                d_replace_prev <= d_replace_check_prev &&
                                d_replace > replace_check_tol * d_replace_init) ) {
          if (verbose && myRank == 0) {
            std::cout << "  (replaced with step = " << step << " at iter = " << k
                      << " with d_replace = " << d_replace_prev << " -> " << d_replace
                      << " and d_replace_check = " << d_replace_check_prev << " -> " << d_replace_check;
          }
          num_rr ++;
          // --------------------------------------
          // group update
          if (time_axpy_on) {
            timer_axpy.reset();
          }
          // > update x
          auto yp_i = getCol<GVType> (i+1, yp_device); // device
          auto yp_i_host = getCol<GVType_host> (i+1, yp); // host
          //Kokkos::deep_copy(yp_i, yp_i_host); // to device
          copy_to_device (yp_i, yp_i_host); // to device
          local_copy(v_hat_device, yp_i);     // cast
          cublasXgemv(cublasHandle, CUBLAS_OP_N,
                      nloc, 2*s+1, &(one),  V.data(), nloc,
                                            v_hat_device.data(), 1,
                                   &(one),  x.data(), 1);
          // > z = z + x
          axpby(x_out, one, x_out, one, x);
          // > x = zero
          //Kokkos::deep_copy(x, zero);
          local_init(x, zero);
          if (time_axpy_on) {
            Kokkos::fence();
            time_axpy += timer_axpy.seconds();
            flop_axpy += 1*(4*s)*nloc;
          }

          // --------------------------------------
          // compute explicit residual vector
          if (time_spmv_on) {
            Kokkos::fence();
            timer_spmv.reset();
          }
          //Kokkos::deep_copy(x_sub, x_out);
          local_copy(x_sub, x_out);
          op.apply(Ax, x_global);      // Ax = A*x
          axpby(r, one, b, -one, Ax);  // r = b-Ax
          if (time_spmv_on) {
            Kokkos::fence();
            time_spmv_comm   += op.time_comm;
            time_spmv_spmv   += op.time_spmv;
            #if defined(CGSOLVE_SPMV_TIMER)
            time_spmv_copy   += op.time_comm_copy;
            time_spmv_pack   += op.time_comm_pack;
            time_spmv_unpack += op.time_comm_unpack;
            time_spmv_mpi    += op.time_comm_mpi;
            time_spmv_wait_send += op.time_comm_wait_send;
            time_spmv_wait_recv += op.time_comm_wait_recv;
            #endif
          }
          replaced = true;

          // --------------------------------------
          // compute norm(x)
          #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
          for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(ti1_dd[ii].x[0] + ti1_dd[ii].x[1]);
          #else
          for (int ii=0; ii<2*s+1; ii++) v_hat(ii) = abs(ti1(ii));
          #endif
          beta2_hat = std::sqrt( cblas_rdot(2*s+1, v_hat.data(), 1, v_hat.data(), 1) );
          if (time_dot_on) {
            timer_dot.reset();
          }
          dot(x_out, x_out, dot_result);
          if (time_dot_on) {
            Kokkos::fence();
            time_dot_rr2 += timer_dot.seconds();
            flop_dot_rr2 += (2*nloc-1);
          }
          if (numRanks > 1) {
            if (time_dot_on) {
              timer_dot.reset();
            }
            Kokkos::fence();
            MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                          MPI_COMM_WORLD);
            if (time_dot_on) {
              time_dot_rr2_comm += timer_dot.seconds();
            }
          }
          Kokkos::deep_copy(dot_host, dot_result);
          normx = std::sqrt(*(dot_host.data()));

          // --------------------------------------
          // reinitialize
          d_replace = (eps / replace_tol) * (beta2_hat + (1+2*maxNnzA)*norma*normx);
          d_replace_init = d_replace;
          step = i+1;

          if (verbose) {
            dot(r, r, dot_result);
            if (numRanks > 1) {
              Kokkos::fence();
              MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                            MPI_COMM_WORLD);
            }
            Kokkos::deep_copy(dot_host, dot_result);
            normr = std::sqrt(*(dot_host.data()));
            if (myRank == 0) {
              std::cout << "  and with resnorm = " << normr << ") with d_replace = " << eps / replace_tol
                        << " * (" << beta2_hat << " + " << (1+2*maxNnzA) << " * " << norma << " * " << normx
                        << " -> " << d_replace
                        << std::endl;
            }
          }
          break;
        }
      }
    }
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    if (myRank == printRank) {
      #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
      printf("y_hi = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",y_dd[i+j*ldt].x[0]);
        printf("\n");
      }
      printf("];\n");
      printf("c_hi = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",c_dd[i+j*ldt].x[0]);
        printf("\n");
      }
      printf("];\n");
      printf("t_hi = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",t_dd[i+j*ldt].x[0]);
        printf("\n");
      }
      printf("];\n\n");

      printf("y_lo = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",y_dd[i+j*ldt].x[1]);
        printf("\n");
      }
      printf("];\n");
      printf("c_lo = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",c_dd[i+j*ldt].x[1]);
        printf("\n");
      }
      printf("];\n");
      printf("t_lo = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",t_dd[i+j*ldt].x[1]);
        printf("\n");
      }
      printf("];\n\n");
      #else
      printf("y = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",y(i,j));
        printf("\n");
      }
      printf("];\n");
      printf("c = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",c(i,j));
        printf("\n");
      }
      printf("];\n");
      printf("t = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < s+1; j++) printf("%.2e ",t(i,j));
        printf("\n");
      }
      printf("];\n\n");
      #endif
    }
    #endif
    if (time_axpy_on) {
      time_seq += timer_seq.seconds();
    }

    // ==============================================================================
    // update local vectors
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    if (myRank == printRank) {
      //for (int i = 0; i < 2*s+1; i++) printf( " y1(%d) = %.2e\n",i,yp_s_host(i) );
      //for (int i = 0; i < 2*s+1; i++) printf( " t1(%d) = %.2e\n",i,tp_s_host(i) );
      //for (int i = 0; i < 2*s+1; i++) printf( " c1(%d) = %.2e\n",i,cp_s_host(i) );
      Kokkos::deep_copy(V_local, V);
      Kokkos::deep_copy(V_host, V_local);
      printf(" > V = [\n" );
      //for (int i = 0; i < nloc; i++)
      for (int i = 0; i < 5; i++)
      {
        printf( "%d ",i );
        for (int j = 0; j < 2*s+1; j++) printf("%.2e ",V_host(i,j));
        printf("\n");
      }
      printf("];\n");
      /*printf(" > V_global = [\n" );
      Kokkos::deep_copy(V_global_host, V_global);
      for (int i = 0; i < n; i++) {
        printf( "%d ",i );
        for (int j = 0; j < 2*s+1; j++) printf("%.2e ",V_global_host(i,j));
        printf("\n");
      }
      printf("];\n");*/
    }
    #endif
    if (time_axpy_on) {
      timer_axpy.reset();
    }
    if (replaced) {
      auto cp_i = getCol<GVType> (step, cp_device); // device
      auto cp_i_host = getCol<GVType_host> (step, cp); // host
      //Kokkos::deep_copy(cp_i, cp_i_host); // to device
      copy_to_device (cp_i, cp_i_host); // to device
      local_copy(v_hat_device, cp_i);     // cast
      cublasXgemv(cublasHandle, CUBLAS_OP_N,
                  nloc, 2*s+1, &(one),  V.data(), n,
                                        v_hat_device.data(), 1,
                               &(zero), p.data(), 1);
      if (time_axpy_on) {
        Kokkos::fence();
        time_axpy += timer_axpy.seconds();
        flop_axpy += (4*s)*nloc;
      }
    } else {
      #if defined(CGSOLVE_ENABLE_CUBLAS)
      #if 0
      Kokkos::deep_copy(yp_device, yp);
      Kokkos::deep_copy(cp_device, cp);
      Kokkos::deep_copy(tp_device, tp);
      auto y1 = getCol<VType> (s, yp_device);
      auto t1 = getCol<VType> (s, tp_device);
      auto c1 = getCol<VType> (s, cp_device);
      cublasDgemv(cublasHandle, CUBLAS_OP_N,
                  nloc, 2*s+1, &(one),   V.data(), nloc,
                                        y1.data(), 1,
                               &(one),   x.data(), 1);
      cublasDgemv(cublasHandle, CUBLAS_OP_N,
                  nloc, 2*s+1, &(one),   V.data(), nloc,
                                        t1.data(), 1,
                               &(zero),  r.data(), 1);
      cublasDgemv(cublasHandle, CUBLAS_OP_N,
                  nloc, 2*s+1, &(one),   V.data(), nloc,
                                        c1.data(), 1,
                               &(zero), p0.data(), 1);
      #else
      //Kokkos::deep_copy(yp_s, yp_s_host);
      //Kokkos::deep_copy(cp_s, cp_s_host);
      //Kokkos::deep_copy(tp_s, tp_s_host);
      copy_to_device (yp_s, yp_s_host); // to device
      copy_to_device (cp_s, cp_s_host); // to device
      copy_to_device (tp_s, tp_s_host); // to device
      local_copy(y_s, yp_s); // cast
      local_copy(c_s, cp_s); // cast
      local_copy(t_s, tp_s); // cast
      local_mv_init(PR, zero);
      cublasXgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                  nloc, 3, 2*s+1, &(one),  V.data(), n, // V is subview of V_global
                                           CTY.data(), 2*s+1,
                                  &(one),  PRX.data(), nloc);
      #endif
      #endif
      if (time_axpy_on) {
        Kokkos::fence();
        time_axpy += timer_axpy.seconds();
        flop_axpy += 3*(4*s)*nloc;
      }
      if (replace_residual && !merge_rr_dots) {
        if (time_dot_on) {
          timer_dot.reset();
        }
        dot(x, x, dot_result);
        if (time_dot_on) {
          Kokkos::fence();
          time_dot_rr2 += timer_dot.seconds();
          flop_dot_rr2 += (2*nloc-1);
          timer_dot.reset();
        }
        if (numRanks > 1) {
          Kokkos::fence();
          MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                        MPI_COMM_WORLD);
        }
        Kokkos::deep_copy(dot_host, dot_result);
        if (time_dot_on) {
          time_dot_rr2_comm += timer_dot.seconds();
        }
        beta0_hat = std::sqrt(*(dot_host.data()));
        d_replace += (eps / replace_tol) * (norma * (beta0_hat + scalar_type(2 + 2*maxNnzA) * beta1_hat) + scalar_type(maxNnzA) * beta2_hat);
      }
    }
    local_mv_copy(V01, PR);
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    if (myRank == printRank) {
      Kokkos::deep_copy(CTY_host, CTY);
      printf(" > CTY = [\n" );
      for (int i = 0; i < 2*s+1; i++) {
        for (int j = 0; j < 3; j++) printf("%.2e ",CTY_host(i,j));
        printf("\n");
      }
      printf("];\n");
      Kokkos::deep_copy(PRX_host, PRX);
      printf(" > PRX = [\n" );
      //for (int i = 0; i < nloc; i++)
      for (int i = 0; i < 5; i++)
      {
        for (int j = 0; j < 3; j++) printf("%.2e ",PRX_host(i,j));
        printf("\n");
      }
      printf("];\n");
      /*Kokkos::deep_copy(x_host, x);
      Kokkos::deep_copy(r_host, r);
      Kokkos::deep_copy(p0_host, p0);
      for (int i = 0; i < nloc; i++) printf( " + r(%d) = %.2e,\t p(%d) = %.2e,\t x(%d) = %.2e\n",i,r_host(i),i,p0_host(i),i,x_host(i) );*/
    }
    #endif

    if (verbose) {
      // r = b - A*x
      #if 1 // compute explicit residual norm
      Kokkos::deep_copy(x_sub, x);
      if (replace_residual) {
        axpby(x_sub, one, x_sub, one, x_out);
      }
      op.apply(Ax, x_global);           // Ax = A*x
      axpby(r_true, one, b, -one, Ax);  // r = b-Ax

      // explicit residual norm
      dot(r_true, r_true, dot_result);
      Kokkos::fence();
      MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                    MPI_COMM_WORLD);
      Kokkos::deep_copy(dot_host, dot_result);
      #endif

      if (myRank == printRank) {
        std::cout << "Iteration = " << k << "   Residual (delayed) = " << normr
                  << ", True Residual (current) = " << std::sqrt(*(dot_host.data()))
                  #if !defined(USE_FLOAT) & defined(USE_MIXED_PRECISION)
                  << ", beta = " << beta.x[0]+beta.x[1] << ", alpha = " << alpha.x[0]+alpha.x[1]
                  #else
                  << ", beta = " << beta << ", alpha = " << alpha
                  << ", step = " << step
                  #endif
                  << std::endl;
      }
    }
    num_iters = k;
  }
  // copy x to output
  //local_copy(x_out, x);
  axpby(x_out, one, x_out, one, x);
  #if defined(KOKKOS_DEBUG_CGSOLVER)
  Kokkos::deep_copy(x_host, x);
  Kokkos::deep_copy(x_out_host, x_out);
  for (int i = 0; i < nloc; i++) printf( " >> x(%d) = %.2e, %.2e\n",i,x_host(i), i,x_out_host(i) );
  #endif

  cudaStreamSynchronize(cudaStream);
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);
  #if defined(CGSOLVE_CUDA_PROFILE)
  cudaProfilerStop();
  #endif
  time_cg = timer_cg.seconds();
  if (myRank == 0) {
    std::cout << " > s-step CG Main loop : iter = " << num_iters << " time = " << time_cg << std::endl;
    if (normr > tolerance) {
      std::cout << " >  failed to converge with normr = " << normr << " and tol = " << tolerance << std::endl;
    } else {
      std::cout << " >  converged with normr = " << normr << " and tol = " << tolerance << std::endl;
    }
  }

  if (time_spmv_on || time_dot_on || time_axpy_on) {
    if (myRank == 0) {
      printf( "\n  -------------------------------------------\n\n" );
    }
    if (time_spmv_on) {
      double min_spmv = 0.0, max_spmv = 0.0;
      MPI_Allreduce(&time_spmv, &min_spmv, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv, &max_spmv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      double min_copy = 0.0, max_copy = 0.0;
      MPI_Allreduce(&time_spmv_copy, &min_copy, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_copy, &max_copy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      #if defined(CGSOLVE_SPMV_TIMER)
      double min_pack = 0.0, max_pack = 0.0;
      MPI_Allreduce(&time_spmv_pack, &min_pack, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_pack, &max_pack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      double min_unpack = 0.0, max_unpack = 0.0;
      MPI_Allreduce(&time_spmv_unpack, &min_unpack, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_unpack, &max_unpack, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      double min_mpi = 0.0, max_mpi = 0.0;
      MPI_Allreduce(&time_spmv_mpi, &min_mpi, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_mpi, &max_mpi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      double min_wait_send = 0.0, max_wait_send = 0.0;
      MPI_Allreduce(&time_spmv_wait_send, &min_wait_send, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_wait_send, &max_wait_send, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      double min_wait_recv = 0.0, max_wait_recv = 0.0;
      MPI_Allreduce(&time_spmv_wait_recv, &min_wait_recv, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_wait_recv, &max_wait_recv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      #endif

      double min_comp = 0.0, max_comp = 0.0;
      MPI_Allreduce(&time_spmv_spmv, &min_comp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_spmv, &max_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      double min_comm = 0.0, max_comm = 0.0;
      MPI_Allreduce(&time_spmv_comm, &min_comm, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myRank == 0) {
        printf( "   time(SpMV)            = %.2e ~ %.2e seconds\n", min_spmv,  max_spmv );
        printf( "    + time(SpMV)::comm    =  %.2e ~ %.2e seconds\n",min_comm, max_comm );
        printf( "    + time(SpMV)::copy    =  %.2e ~ %.2e seconds\n",min_copy,  max_copy );
        #if defined(CGSOLVE_SPMV_TIMER)
        printf( "     > time(SpMV)::pack    =  %.2e ~ %.2e seconds\n",min_pack,  max_pack );
        printf( "     > time(SpMV)::unpack  =  %.2e ~ %.2e seconds\n",min_unpack,max_unpack );
        printf( "     > time(SpMV)::mpi     =  %.2e ~ %.2e seconds\n",min_mpi,   max_mpi  );

        printf( "      - time(SpMV)::wait_send =  %.2e ~ %.2e seconds\n",min_wait_send, max_wait_send );
        printf( "      - time(SpMV)::wait_recv =  %.2e ~ %.2e seconds\n",min_wait_recv, max_wait_recv );
        #endif
        printf( "    + time(SpMV)::comp    =  %.2e ~ %.2e seconds\n",min_comp,  max_comp );
      }
      //printf( "    xx %d: time(SpMV)::wait    =  %.2e + %.2e seconds\n",myRank, time_spmv_wait_send,time_spmv_wait_recv );
      //printf( "    xx %d: time(SpMV)::comm    =  %.2e seconds\n",myRank, time_spmv_mpi );
      //printf( "    xx %d: time(SpMV)::comp    =  %.2e seconds (nlocal = %d, nnzlocal = %d)\n",myRank, time_spmv_spmv, 
      //        op.getLocalDim(), op.getLocalNnz());
    }
    if (time_dot_on) {
      flop_dot += flop_dot_rr;
      time_dot += time_dot_rr;
      time_dot_comm += time_dot_rr_comm;

      flop_dot += flop_dot_rr2;
      time_dot += time_dot_rr2;
      time_dot_comm += time_dot_rr2_comm;
      flop_dot /= time_dot;

      double min_dot_flop = 0.0, max_dot_flop = 0.0;
      MPI_Allreduce(&flop_dot, &min_dot_flop, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&flop_dot, &max_dot_flop, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      double min_dot_comp = 0.0, max_dot_comp = 0.0;
      MPI_Allreduce(&time_dot, &min_dot_comp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_dot, &max_dot_comp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      double min_dot_comm = 0.0, max_dot_comm = 0.0;
      MPI_Allreduce(&time_dot_comm, &min_dot_comm, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_dot_comm, &max_dot_comm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myRank == 0) {
        printf( "\n" );
        printf( "   time   (Dot)          = %.2e + %.2e = %.2e seconds\n", time_dot,time_dot_comm,
                                                                           time_dot+time_dot_comm );
        printf( "    + time(Dot)::comp     =  %.2e ~ %.2e seconds\n",min_dot_comp,max_dot_comp );
        printf( "    + time(Dot)::comm     =  %.2e ~ %.2e seconds\n",min_dot_comm,max_dot_comm );
        printf( "   Gflop/s(Dot)          = %.2e ~ %.2e\n", min_dot_flop/1e9,max_dot_flop/1e9);
        if (replace_residual) {
          printf( "    > # replacements     = %d\n", num_rr);
          printf( "      + time(Dot)::comp     =  %.2e + %.2e seconds\n",time_dot_rr, time_dot_rr2 );
          printf( "      + time(Dot)::comm     =  %.2e + %.2e seconds\n",time_dot_rr_comm, time_dot_rr2_comm );
        }
      }
    }
    if (myRank == 0) {
      if (time_axpy_on) {
        printf( "\n" );
        printf( "   time   (sequ)         = %.2e seconds\n", time_seq );
        printf( "\n" );
        printf( "   time   (axpy)         = %.2e seconds\n", time_axpy );
        printf( "   Gflop/s(axpy)         = %.2e (%.2e flops)\n", flop_axpy/(1e9*time_axpy), flop_axpy );
      }
      printf( "\n  -------------------------------------------\n" );
    }
  }
  cublasDestroy(cublasHandle);
  cudaStreamDestroy(cudaStream);
  //op.unsetStream();

  return num_iters;
}



// =============================================================
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  using  MType = Kokkos::View<scalar_type**>;
  using  VType = Kokkos::View<scalar_type *>;
  using  AType = CrsMatrix<memory_space, scalar_type>;
  using HAType = CrsMatrix<host_execution_space, scalar_type>;

  using VTypeHost = Kokkos::View<scalar_type*, Kokkos::HostSpace>;

  Kokkos::initialize(argc, argv);
  {
    int loop              = 2;

    bool         strakos  = false;
    scalar_type  strakos_l1 = 1e-3;
    scalar_type  strakos_ln = 1e+2;
    scalar_type  strakos_p  = 0.65;

    int N                 = 100;
    int nx                = 0;
    int max_iter          = 200;
    int s                 = 1;
    int dot_option        = 0; // 1: explicitly cast to output scalar type
    scalar_type tolerance = 1e-8;
    std::string matrixFilename {""};

    int replace_op = 1; // 0: replace at every step
    double replace_check_tol = 0.0;
    bool replace_residual = false;
    bool merge_rr_dots = false;

    bool metis       = false;
    bool sort_matrix = false;
    bool verbose     = false;
    bool time_dot    = false;
    bool time_spmv   = false;
    bool time_axpy   = false;
    bool check       = false;
    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i],"-loop")==0)) {
        loop = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i],"-N")==0)) {
        N = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-nx")==0)) {
        nx = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-f")==0)) {
        matrixFilename = argv[++i];
        continue;
      }
      if((strcmp(argv[i],"-iter")==0)) {
        max_iter = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-s")==0)) {
        s = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-dot")==0)) {
        dot_option = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-tol")==0)) {
        tolerance = atof(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-v")==0)) {
        verbose = true;
        continue;
      }
      if((strcmp(argv[i],"-rr")==0)) {
        replace_residual = true;
        continue;
      }
      if((strcmp(argv[i],"-merge-rr-dots")==0)) {
        merge_rr_dots = true;
        continue;
      }
      if((strcmp(argv[i],"-rtol")==0)) {
        replace_check_tol = atof(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-rop")==0)) {
        replace_op = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-time-spmv")==0)) {
        time_spmv = true;
        continue;
      }
      if((strcmp(argv[i],"-time-dot")==0)) {
        time_dot = true;
        continue;
      }
      if((strcmp(argv[i],"-time-axpy")==0)) {
        time_axpy = true;
        continue;
      }
      if((strcmp(argv[i],"-check")==0)) {
        check = true;
        continue;
      }
      if((strcmp(argv[i],"-strakos")==0)) {
        strakos = true;
        continue;
      }
      if((strcmp(argv[i],"-sort")==0)) {
        sort_matrix = true;
        continue;
      }
      if((strcmp(argv[i],"-metis")==0)) {
        metis = true;
        continue;
      }
    }

    using default_execution_space = Kokkos::DefaultExecutionSpace;
    using default_memory_space = typename default_execution_space::memory_space;
    if (myRank == 0) {
      std::cout << std::endl;
      std::cout << "Default execution space: " << default_execution_space::name () << std::endl;
      std::cout << "Default memory space   : " << default_memory_space::name () << std::endl;
      std::cout << "Number of processes    : " << numRanks << std::endl;
      std::cout << std::endl;
    }

    const scalar_type one  (1.0);
    const scalar_type zero (0.0);

    // generate matrix on host
    int *part_map = nullptr;
    int n = 0;
    int nlocal = 0;
    int start_row = 0;
    int end_row = 0;

    HAType h_A;
    VTypeHost h_b;
    {
      h_A = Impl::generate_matrix<scalar_type>(strakos, strakos_l1, strakos_ln, strakos_p,
                                               matrixFilename, nx, N, metis, verbose);
      n         = h_A.num_cols();
      nlocal    = h_A.nlocal;
      start_row = h_A.start_row;
      end_row   = h_A.end_row;
    }
    if (sort_matrix) {
      if (myRank == 0) {
        std::cout << "  + sorting matrix .." << std::endl;
      }
      Impl::sort_matrix(h_A);
    }

    // generate right-hand-side
    if (strakos || matrixFilename != "" || nx > 0) {
      h_b = VTypeHost("b_h", n);
      Kokkos::deep_copy(h_b, one);
    } else {
      h_b = Impl::generate_miniFE_vector<scalar_type>(N);
    }


    // copy the matrix to device
    // TODO: move this into op.setup
    Kokkos::View<int *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
    Kokkos::View<int *> col_idx("col_idx", h_A.col_idx.extent(0));
    Kokkos::View<scalar_type *>  values("values",  h_A.values.extent(0));
    AType A(row_ptr, col_idx, values, h_A.num_cols());
    Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
    Kokkos::deep_copy(A.col_idx, h_A.col_idx);
    Kokkos::deep_copy(A.values,  h_A.values);

    // copy rhs to device
    VType b("b", h_b.extent(0));
    Kokkos::deep_copy(b, h_b);

    /*char filename[200];
    FILE *fp;
    sprintf(filename,"A%d_%d.dat",numRanks, myRank);
    fp = fopen(filename, "w");
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        //fprintf(fp, "%d %d %e\n",i,h_A.col_idx(k), h_A.values(k) );
        //fprintf(fp, "%d %d\n",i,h_A.col_idx(k) );
        printf("%d %d %.16e\n",i,h_A.col_idx(k),h_A.values(k) );
      }
    }
    fclose(fp);*/
    /*sprintf(filename,"b%d_%d.dat",numRanks, myRank);
    fp = fopen(filename, "w");
    for (int i=0; i<h_b.extent(0); i++) {
      fprintf(fp,"%e\n",h_b(i));
    }
    fclose(fp);*/

    // local rhs on device
    Kokkos::pair<int, int> bounds(start_row, end_row);
    VType b_sub = Kokkos::subview(b, bounds);

    // setup SpMV
    int nrhs = 2;
    cgsolve_spmv<VType, HAType, AType, VType, MType, execution_space> op (n, start_row, end_row, h_A, A, time_spmv);
    op.setup(nlocal, nrhs, part_map);

    // local sol on device
    if (strakos) {
      MType x0_global("x0",  n, 1);
      VType c_sub("c", b_sub.extent(0));

      // c = A*ones
      Kokkos::deep_copy(x0_global, one);
      op.apply(c_sub, x0_global);

      // bnorm = norm(c)
      scalar_type bnorm = 0.0;
      dot(c_sub, c_sub, bnorm);
      bnorm = std::sqrt(bnorm);

      // c = c / norm(c)
      axpby(b_sub, one/bnorm, c_sub, zero, b_sub);
    }

    int maxNnzA = 0;
    scalar_type Anorm = 0.0;
    if (replace_residual) {
      // power iteration to approximate norm(A)
      VType p("p", b.extent(0));
      MType q_global("q",  n, 1);
      auto q_sub = Kokkos::subview(getCol<VType>(0, q_global), bounds);

      Anorm = n;
      Anorm = std::sqrt(Anorm);
      Kokkos::deep_copy(p, one);
      axpby(p, one/Anorm, p, zero, p);
      for (int i = 0; i < 10; i++) {
        // r = b - A*x
        axpby(q_sub, one, p, zero, p);
        op.apply(p, q_global);        // p = A*p

        // Anorm = norm(p)
        dot(p, p, Anorm);
        MPI_Allreduce(MPI_IN_PLACE, &Anorm, 1, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);
        Anorm = std::sqrt(Anorm);

        // p = p/Anorm
        axpby(p, one/Anorm, p, zero, p);
        if (verbose && myRank == 0) {
          std::cout << " norm(A) = " << Anorm << std::endl;;
        }
      }
      for (int i=0; i<n; i++) {
        int nnzRowA = h_A.row_ptr(i+1) - h_A.row_ptr(i+1);
        if (nnzRowA > maxNnzA) {
          maxNnzA = nnzRowA;
        }
      }
    }

    // call CG
    if (myRank == 0) {
      int nloc = end_row - start_row;
      int nnz = h_A.row_ptr(nloc);
      if (matrixFilename != "") {
        std::cout << std::endl << " calling cg_solve ( matrix = " << matrixFilename;
      } else if (nx > 0) {
        std::cout << std::endl << " calling cg_solve ( nx = " << nx;
      } else {
        std::cout << std::endl << " calling cg_solve ( N = " << N;
      }
      std::cout << ", n = " << n << ", nloc = " << nloc
                << ", nnz/nloc = " << double(nnz)/double(nloc)
                << ", s = " << s << ", dot-option = " << dot_option << " )"
                << std::endl;
      #if defined(USE_FLOAT) 
      std::cout << " using float as working precision";
      #else
      std::cout << " using doubl as working precision";
      #endif
      #if defined(USE_MIXED_PRECISION)
      std::cout << " (mixed precision)" << std::endl;
      #else
      std::cout << " (unit precision)" << std::endl;
      #endif
    }

    // for check
    MType p("p", n, nrhs); // global
    MType p_sub = Kokkos::subview(p, bounds, Kokkos::ALL ()); // local
    VType x_sub("x", b_sub.extent(0));
    for (int nloop = 0; nloop < loop; nloop++) {
      Kokkos::deep_copy(x_sub, zero);
      Kokkos::fence();
      MPI_Barrier(MPI_COMM_WORLD);
      Kokkos::Timer timer;
      int num_iters = cg_solve<VType> (x_sub, op, b_sub,
                                       max_iter, tolerance, s, dot_option,
                                       replace_residual, replace_check_tol, replace_op, maxNnzA, Anorm, merge_rr_dots,
                                       verbose, time_spmv, time_dot, time_axpy);
      Kokkos::fence();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = timer.seconds();
      if (check) {
        auto p0_sub = getCol<VType> (0, p_sub);
        auto p0  = getCol<VType> (0, p);
        VType r("Y", x_sub.extent(0));
        axpby(p0_sub, one, x_sub, zero, x_sub);
        op.apply(r, p);
        axpby(r, -one, r, one, b_sub);
        //for (int i = 0; i < r.extent(0); i++) printf( "%d %e %e %e\n",i,b_sub(i),x_sub(i),r(i));

        scalar_type rnorm = 0.0;
        dot(r, r, rnorm);
        MPI_Allreduce(MPI_IN_PLACE, &rnorm, 1, MPI_SCALAR, MPI_SUM,
                      MPI_COMM_WORLD);
        rnorm = std::sqrt(rnorm);
        scalar_type bnorm = 0.0;
        dot(b, b, bnorm);
        MPI_Allreduce(MPI_IN_PLACE, &bnorm, 1, MPI_SCALAR, MPI_SUM,
                      MPI_COMM_WORLD);
        bnorm = std::sqrt(bnorm);
        if (myRank == 0) {
          printf( "\n ====================================" );
          printf( "\n rnorm = %e / %e = %e\n",rnorm,bnorm,rnorm/bnorm );
        }
      } // end of check
      if (myRank == 0) {
        printf( " num_iters = %i, time = %.2lf\n\n", num_iters, time);
      }
    } // end of loop
  }
  Kokkos::finalize();
  MPI_Finalize();
  
  return 0;
}
