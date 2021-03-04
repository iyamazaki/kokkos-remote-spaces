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
//#include "kokkos_kernels_gemm.hpp"

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



// -------------------------------------------------------------
// Auxiliary function too get jth column of MM
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

  using  buffer_view_t = Kokkos::View<double *>;

  using team_policy_type  = Kokkos::TeamPolicy<SpaceType>;
  using member_type       = typename team_policy_type::member_type;

  cgsolve_spmv(int n_, int start_row_, int end_row_, HAType h_A_, AType A_, bool time_spmv_on_) :
  n(n_),
  start_row(start_row_),
  end_row(end_row_),
  h_A(h_A_),
  A(A_),
  time_spmv_on(time_spmv_on_),
  use_stream(false)
  {}

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
    const double zero = 0.0;

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
    for (int p=0; p<numRanks; p++) {
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
    if (numRanks <= 0) return;

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
      double *buffer = buf_recvs.data();
      MPI_Irecv(&buffer[start], count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #else
      MPI_Irecv(&(host_recvs(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
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
      MPI_Isend(&(host_sends(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #else
      double *buffer = buf_sends.data();
      MPI_Isend(&buffer[start], count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
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
            //int id = (k < count ? start+k : start+count-1);
            //x(idx_recvs(id)) = buf_recvs(id);
            if (k < count) {
              x(idx_recvs(start+k)) = buf_recvs(start+k);
            }
          });
      });
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
    if (numRanks <= 0) return;

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
      double *buffer = buf_recvs.data();
      MPI_Irecv(&buffer[start], count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #else
      MPI_Irecv(&(host_recvs(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
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
    team_policy_type send_policy (space, max_num_sends, Kokkos::AUTO);
    Kokkos::parallel_for(
      "pack-for-send", send_policy,
      KOKKOS_LAMBDA(const member_type & team) {
        int k = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_neighbors_sends),
          [&](const int q) {
            int p = ngb_sends(q);
            int start = nrhs_*ptr_sends(q);
            int count = ptr_sends(q+1)-start;
            if(k < count) {
              for (int j = 0; j < nrhs_; j++) {
                buf_sends(start+nrhs_*k+j) = x(idx_sends(start+k), j);
              }
            }
          });
      });
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
      int count = nrhs_*(host_ptr_sends(q+1)-start);
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      #if !defined(CGSOLVE_GPU_AWARE_MPI)
      MPI_Isend(&(host_sends(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #else
      double *buffer = buf_sends.data();
      MPI_Isend(&buffer[start], count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
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
    team_policy_type recv_policy (space, max_num_recvs, Kokkos::AUTO);
    Kokkos::parallel_for(
      "unpack-for-recv", recv_policy,
      KOKKOS_LAMBDA(const member_type & team) {
        int k = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 0, num_neighbors_recvs),
          [&](const int q) {
            int p = ngb_recvs(q);
            int start = nrhs_*ptr_recvs(q);
            int count = num_recvs(p); //ptr_recvs(q+1)-start;
            //int id = (k < count ? start+k : start+count-1);
            //x(idx_recvs(id)) = buf_recvs(id);
            if (k < count) {
              for (int j = 0; j < nrhs_; j++) {
                x(idx_recvs(start+k),j) = buf_recvs(start+nrhs_*k+j);
              }
            }
          });
      });
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
     double alpha = 1.0;
     double beta  = 0.0;
     cusparseDcsrmm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    numRows, numRHSs, numCols, nnz,
                    &alpha, descrA,
                            A.values.data(), A.row_ptr.data(), A.col_idx.data(),
                            x.data(), ldx,
                    &beta,  y.data(), ldy);
    #endif
  }

  // -------------------------------------------------------------
  // local SpMV
  void local_apply(YType y, XType x) {
    #if defined(CGSOLVE_ENABLE_CUBLAS)
     int numCols = n;
     int numRows = h_A.row_ptr.extent(0)-1;
     int nnz = h_A.row_ptr(numRows);
     double alpha = 1.0;
     double beta  = 0.0;
     cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    numRows, numCols, nnz,
                    &alpha, descrA,
                            A.values.data(), A.row_ptr.data(), A.col_idx.data(),
                            x.data(),
                    &beta,  y.data());
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

              double y_row = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, row_length),
                  [=](const int i, double &sum) {
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
    int nrhs_ = y.extent(1);
    this->exchange(nrhs_, x);
    if (time_spmv_on) {
      fence();
      time_comm = timer.seconds();
      timer.reset();
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
    int nrhs_ = 1;
    this->exchange(nrhs_, x);
    if (time_spmv_on) {
      fence();
      time_comm = timer.seconds();
      timer.reset();
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
inline void dot(XType x, double &result) {
  result = 0.0;
  Kokkos::parallel_reduce(
      "DOT", x.extent(0),
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += x(i) * x(i); },
      result);
}

template <class YType, class XType>
inline void dot(YType y, XType x, double &result) {
  result = 0.0;
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += y(i) * x(i); },
      result);
}


// using view to keep result on device
template <class XType, class DType>
inline void dot(XType x, DType result) {
  Kokkos::deep_copy(result, 0.0);
  Kokkos::parallel_reduce(
      "DOT", x.extent(0),
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += x(i) * x(i); },
      result);
}

template <class YType, class XType, class DType>
inline void dot(YType y, XType x, DType result) {
  Kokkos::deep_copy(result, 0.0);
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += y(i) * x(i); },
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
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += x(i) * x(i); },
      result);
}

template <class YType, class XType, class DType, class SpaceType>
inline void dot_stream(YType y, XType x, DType result, SpaceType space) {
  using range_policy_t = Kokkos::RangePolicy<SpaceType>;

  Kokkos::deep_copy(space, result, 0.0);
  Kokkos::parallel_reduce(
      "dot_stream(y^T * x)",
      range_policy_t(space, 0, x.extent(0)),
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += y(i) * x(i); },
      result);
}


template <class XType, class YType, class DType>
struct dots_stream {
  using scalar_type  = typename DType::non_const_value_type;

  typedef double value_type[];
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
  int n = x.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { x(i) = y(i); });
  #else
  Kokkos::deep_copy(x, y);
  #endif
}



// -------------------------------------------------------------
// axpby
template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta, YType y) {
  int n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
}

template <class ZType, class YType, class XType>
void axpby(ZType z1, XType x1, double beta1, YType y1,
           ZType z2, XType x2, double beta2, YType y2) {
  int n = z1.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) {
        z1(i) = x1(i) + beta1 * y1(i);
        z2(i) = x2(i) + beta2 * y2(i);
      });
}

template <class ZType, class YType, class XType, class SpaceType>
void axpby(ZType z1, XType x1, double beta1, YType y1,
           ZType z2, XType x2, double beta2, YType y2,
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
template <class VType, class MType, class OP>
int cg_solve(VType x, OP op, VType b,
             MType v, MType v_global,
             int max_iter, double tolerance, int s,
             bool verbose, bool time_spmv_on, bool time_dot_on, bool time_axpy_on) {
 
  using DView = Kokkos::View<double*>;
  using VType_host = typename VType::HostMirror;

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  const double one = 1.0;
  const double zero = 0.0;

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

  Kokkos::Timer timer_axpy;
  double time_axpy = 0.0;
  double flop_axpy = 0.0;

  double normr = zero;
  double alpha = zero;
  double beta  = zero;
  double new_rr = 0.0;
  Kokkos::View<double> dot_result("dot_result"); 
  auto dot_host = Kokkos::create_mirror_view(dot_result);

  // residual vector
  int nloc = v.extent(0);
  VType  r("r",  nloc);
  VType Ap("Ap", nloc);

  #if 1
  int n    = v_global.extent(0);
  int start_row = op.getStartRow();
  int   end_row = op.getEndRow();
  Kokkos::pair<int, int> local_rows(start_row, end_row);
  MType  V_global("V_global",  n, 2*s+1);
  auto V = Kokkos::subview(V_global, local_rows, Kokkos::ALL ());
  #else
  MType  V("V_global",  nloc, 2*s+1);
  #endif

  Kokkos::pair<int, int> index_P(0,   s+1);
  Kokkos::pair<int, int> index_R(s+1, 2*s);
  auto P = Kokkos::subview(V, Kokkos::ALL (), index_P);
  auto R = Kokkos::subview(V, Kokkos::ALL (), index_R);

  //#define KOKKOS_DEBUG_CGSOLVER
  #if defined(KOKKOS_DEBUG_CGSOLVER)
  auto  r_host = Kokkos::create_mirror_view(r);
  auto  v_host = Kokkos::create_mirror_view(v);
  auto  x_host = Kokkos::create_mirror_view(x);
  auto  V_host = Kokkos::create_mirror_view(V);
  #endif
  MType z ("z", nloc, 2);

  MType B_device ("B",  2*s+1, 2*s+1);
  MType G_device ("G",  2*s+1, 2*s+1);
  MType c_device ("c",  2*s+1, s+1);
  MType t_device ("t",  2*s+1, s+1);
  MType y_device ("y",  2*s+1, s+1);
  VType w_device ("w",  2*s+1);
  VType c2_device("c2", 2*s+1);
  auto B  = Kokkos::create_mirror_view(B_device);
  auto G  = Kokkos::create_mirror_view(G_device);
  auto c  = Kokkos::create_mirror_view(c_device);
  auto t  = Kokkos::create_mirror_view(t_device);
  auto y  = Kokkos::create_mirror_view(y_device);
  auto w  = Kokkos::create_mirror_view(w_device);
  auto c2 = Kokkos::create_mirror_view(c2_device);
  Kokkos::deep_copy(B, zero);
  for (int i = 0;   i < s;     i++) B(i+1, i) = one;
  for (int i = s+1; i < 2*s+1; i++) B(i+1, i) = one;
  #if defined(KOKKOS_DEBUG_CGSOLVER)
  printf("B = [\n" );
  for (int i = 0; i < 2*s+1; i++) {
    for (int j = 0; j < 2*s+1; j++) printf("%.2e ",B(i,j));
    printf("\n");
  }
  printf("];\n");
  #endif

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
  auto z0  = getCol<VType> (0, z);
  auto z1  = getCol<VType> (1, z);
  auto v0  = getCol<VType> (0, v);
  auto v1  = getCol<VType> (1, v);
  auto v0_global = getCol<VType> (0, v_global);
  auto v1_global = getCol<VType> (1, v_global);
  axpby(v0, zero, x, one, x);   // p = x
  op.apply(Ap, v_global);       // Ap = A*p
  axpby(r, one, b, -one, Ap);   // r = b-Ap

  local_copy(v0, r);     // p = r

  //printf("Init: x, Ax, b, r\n" );
  //Kokkos::fence();
  //for (int i=0; i<b.extent(0); i++) printf(" %e, %e, %e, %e\n",x(i),AAr(i),b(i),r(i));

  // beta = r'*r (using Kokkos and non Cuda-aware MPI)
  dot(r, dot_result);
  if (numRanks > 0) {
    Kokkos::fence();
    MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  Kokkos::deep_copy(dot_host, dot_result);
  new_rr = *(dot_host.data());
  normr = std::sqrt(new_rr);
  tolerance *= normr;

  if (verbose && myRank == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
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
  for (int k = 0; k <= max_iter && normr > tolerance; k+=s) {

    // ==============================================================================
    // Matrix-Powers Kernel
    if (time_spmv_on) {
      Kokkos::fence();
      timer_spmv.reset();
    }
    auto r0 = getCol<VType> (0, R);
    auto p0 = getCol<VType> (0, P);
    auto p1 = getCol<VType> (1, P);
    if (time_spmv_on) {
      Kokkos::fence();
      timer_copy.reset();
    }
    local_copy(r0, r);
    local_copy(p0, v0);
    if (time_spmv_on) {
      Kokkos::fence();
      time_spmv_copy += timer_copy.seconds();
    }
    op.apply(p1, v_global);
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
      auto rj0 = getCol<VType> (j-1, R);
      auto rj1 = getCol<VType> (j+0, R);

      auto pj0 = getCol<VType> (j+0, P);
      auto pj1 = getCol<VType> (j+1, P);

      #if 0
      Kokkos::deep_copy(v0, rj0);
      op.apply(rj1, v_global);

      Kokkos::deep_copy(v0, pj0);
      op.apply(pj1, v_global);
      #else
      if (time_spmv_on) {
        Kokkos::fence();
        timer_copy.reset();
      }
      local_copy(v0, rj0);
      local_copy(v1, pj0);
      if (time_spmv_on) {
        Kokkos::fence();
        time_spmv_copy += timer_copy.seconds();
      }
      op.apply(z, v_global);
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

      if (time_spmv_on) {
        Kokkos::fence();
        timer_copy.reset();
      }
      local_copy(rj1, z0);
      local_copy(pj1, z1);
      if (time_spmv_on) {
        Kokkos::fence();
        time_spmv_copy += timer_copy.seconds();
      }
      #endif
    }
    if (time_spmv_on) {
      Kokkos::fence();
      time_spmv += timer_spmv.seconds();
    }

    // ==============================================================================
    // Dot-product
    Kokkos::fence();
    if (time_dot_on) {
      timer_dot.reset();
    }
    #if 0//defined(CGSOLVE_ENABLE_CUBLAS)
    cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                2*s+1, 2*s+1, nloc, &(one),  V.data(), nloc,
                                             V.data(), nloc,
                                    &(zero), G_device.data(), 2*s+1);
    #else
    KokkosBlas::
    gemm("T", "N", one,  V,
                         V,
                   zero, G_device);
    Kokkos::fence();
    #endif
    if (time_dot_on) {
      time_dot += timer_dot.seconds();
      flop_dot += (2*s+1)*(2*s+1)*(2*nloc-1);
      timer_dot.reset();
    }

    Kokkos::deep_copy(G, G_device);
    Kokkos::fence();
    if (time_dot_on) {
      time_dot_comm += timer_dot.seconds();
    }

    // ==============================================================================
    // Convergence check (delayed)
    new_rr = G(s+1, s+1);
    normr = std::sqrt(new_rr);
    if (normr <= tolerance) break;

    #if defined(KOKKOS_DEBUG_CGSOLVER)
    Kokkos::deep_copy(V_host, V);
    printf("V = [\n" );
    for (int i = 0; i < nloc; i++) {
      for (int j = 0; j < 2*s+1; j++) printf("%.2e ",V_host(i,j));
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

    // ==============================================================================
    // Compute "alpha" & "beta" (local redundant compute)
    auto t0 = getCol<VType_host> (0, t);
    auto c0 = getCol<VType_host> (0, c);
    auto y0 = getCol<VType_host> (0, y);
    Kokkos::deep_copy(t0, 0);
    Kokkos::deep_copy(c0, 0);
    Kokkos::deep_copy(y0, 0);
    t0(s+1) = one;
    c0(0) = one;

    double alpha1 = zero;
    double alpha2 = zero;
    double  beta1 = zero;
    for (int i = 0; i < s; i++) {
      auto ti0 = getCol<VType_host> (i+0, t);
      auto ci0 = getCol<VType_host> (i+0, c);
      auto yi0 = getCol<VType_host> (i+0, y);
      auto ti1 = getCol<VType_host> (i+1, t);
      auto ci1 = getCol<VType_host> (i+1, c);
      auto yi1 = getCol<VType_host> (i+1, y);

      // c2 = B*c
      cblas_dgemv (CblasColMajor, CblasNoTrans,
            2*s+1, 2*s+1,
            one,  B.data(), 2*s+1,
                  ci0.data(), 1,
            zero, c2.data(),  1);
      #if defined(KOKKOS_DEBUG_CGSOLVER)
      for (int j = 0; j < 2*s+1; j++) printf( " c0(%d) = %.2e, c2(%d) = %.2e\n",j,ci0(j), j,c2(j) );
      printf("\n");
      #endif
      // compute alpha
      // > alpha1 = t'(G*t)
      if (i == 0) {
        cblas_dgemv (CblasColMajor, CblasNoTrans,
              2*s+1, 2*s+1,
              one,  G.data(), 2*s+1,
                    ti0.data(), 1,
              zero, w.data(),   1);
        #if defined(KOKKOS_DEBUG_CGSOLVER)
        for (int j = 0; j < 2*s+1; j++) printf( " t0(%d) = %.2e, w(%d) = %.2e\n",j,ti0(j), j,w(j) );
        printf("\n");
        #endif
        alpha1 = cblas_ddot(2*s+1, w.data(), 1, ti0.data(), 1);
      } else {
        alpha1 = beta1;
      }
      // > alpha2 = c'*(G*c2)
      cblas_dgemv (CblasColMajor, CblasNoTrans,
            2*s+1, 2*s+1,
            one,  G.data(), 2*s+1,
                  c2.data(), 1,
            zero, w.data(),  1);
      alpha2 = cblas_ddot(2*s+1, w.data(), 1, ci0.data(), 1);
      // > alpha = alpha1/alpha2
      alpha = alpha1 / alpha2;

      // update y = y + alpha*c
      Kokkos::deep_copy(yi1, yi0);
      cblas_daxpy (
            2*s+1, 
            alpha, ci0.data(), 1,
                   yi1.data(), 1);
      // update t = t - alpha*c2
      Kokkos::deep_copy(ti1, ti0);
      cblas_daxpy (
            2*s+1, 
           -alpha, c2.data(),  1,
                   ti1.data(), 1);
      // > beta1 = t'(G*t)
      cblas_dgemv (CblasColMajor, CblasTrans,
            2*s+1, 2*s+1,
            one,  G.data(), 2*s+1,
                  ti1.data(), 1,
            zero, w.data(),   1);
      beta1 = cblas_ddot(2*s+1, w.data(), 1, ti1.data(), 1);
      beta = beta1 / alpha1;
      #if defined(KOKKOS_DEBUG_CGSOLVER)
      for (int j = 0; j < 2*s+1; j++) printf( " y(%d) = %.2e, y(%d) = %.2e\n",j,yi1(j), j,ti1(j) );
      printf( " alpha = %.2e / %.2e = %.2e, beta = %.2e / %.2e = %.2e\n",alpha1,alpha2,alpha, beta1,alpha1,beta );
      #endif
      // update c = t + beta*c
      Kokkos::deep_copy(ci1, ti1);
      cblas_daxpy (
            2*s+1, 
            beta, ci0.data(), 1,
                  ci1.data(), 1);
    }
    #if defined(KOKKOS_DEBUG_CGSOLVER)
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

    // ==============================================================================
    // update local vectors
    Kokkos::deep_copy(y_device, y);
    Kokkos::deep_copy(c_device, c);
    Kokkos::deep_copy(t_device, t);
    #if defined(CGSOLVE_ENABLE_CUBLAS)
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    Kokkos::deep_copy(x_host, x);
    Kokkos::deep_copy(r_host, r);
    Kokkos::deep_copy(v_host, v);
    for (int i = 0; i < nloc; i++) printf( " > r(%d) = %.2e, p(%d) = %.2e, x(%d) = %.2e\n",i,r_host(i),i,p_host(i),i,x_host(i) );
    printf("\n");
    #endif
    auto y1 = getCol<VType> (s, y_device);
    auto t1 = getCol<VType> (s, t_device);
    auto c1 = getCol<VType> (s, c_device);
    if (time_axpy_on) {
      timer_axpy.reset();
    }
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
                             &(zero), v0.data(), 1);
    if (time_axpy_on) {
      Kokkos::fence();
      time_axpy += timer_axpy.seconds();
      flop_axpy += 3*(4*s)*nloc;
    }
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    Kokkos::deep_copy(V_host, V);
    printf(" > V = [\n" );
    for (int i = 0; i < nloc; i++) {
      for (int j = 0; j < 2*s+1; j++) printf("%.2e ",V_host(i,j));
      printf("\n");
    }
    printf("];\n");
    Kokkos::deep_copy(w, y1);
    for (int i = 0; i < 2*s+1; i++) printf( " y1(%d) = %.2e\n",i,w(i) );
    printf("];\n");
    Kokkos::deep_copy(w, t1);
    for (int i = 0; i < 2*s+1; i++) printf( " t1(%d) = %.2e\n",i,w(i) );
    printf("];\n");
    Kokkos::deep_copy(w, c1);
    for (int i = 0; i < 2*s+1; i++) printf( " c1(%d) = %.2e\n",i,w(i) );
    printf("];\n");
    Kokkos::deep_copy(x_host, x);
    Kokkos::deep_copy(r_host, r);
    Kokkos::deep_copy(p_host, p);
    for (int i = 0; i < nloc; i++) printf( " + r(%d) = %.2e, p(%d) = %.2e, x(%d) = %.2e\n",i,r_host(i),i,p_host(i),i,x_host(i) );
    #endif
    #endif

    if (verbose && myRank == 0) {
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << " beta = " << beta << ", alpha = " << alpha
                << std::endl;
    }

    num_iters = k;
  }
  cudaStreamSynchronize(cudaStream);
  Kokkos::fence();
  MPI_Barrier(MPI_COMM_WORLD);
  #if defined(CGSOLVE_CUDA_PROFILE)
  cudaProfilerStop();
  #endif
  time_cg = timer_cg.seconds();
  if (myRank == 0) {
    std::cout << " > CG Main loop : iter = " << num_iters << " time = " << time_cg << std::endl;
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
      }
    }
    if (myRank == 0) {
      if (time_axpy_on) {
        printf( "\n" );
        printf( "   time   ( axpy)        = %.2e seconds\n", time_axpy );
        printf( "   Gflop/s( axpy)        = %.2e (%.2e flops)\n", flop_axpy/(1e9*time_axpy), flop_axpy );
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

  using  MType = Kokkos::View<double**>;
  using  VType = Kokkos::View<double *>;
  using  AType = CrsMatrix<memory_space>;
  using HAType = CrsMatrix<host_execution_space>;

  using VTypeHost = Kokkos::View<double *, Kokkos::HostSpace>;

  Kokkos::initialize(argc, argv);
  {
    int loop         = 1;

    int N            = 100;
    int nx           = 0;
    int max_iter     = 200;
    int s            = 1;
    double tolerance = 1e-8;
    std::string matrixFilename {""};

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
      if((strcmp(argv[i],"-tol")==0)) {
        tolerance = atof(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-v")==0)) {
        verbose = true;
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
      if((strcmp(argv[i],"-sort")==0)) {
        sort_matrix = true;
        continue;
      }
      #if defined(CGSOLVE_ENABLE_METIS)
      if((strcmp(argv[i],"-metis")==0)) {
        metis = true;
        continue;
      }
      #endif
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

    const double one = 1.0;
    const double zero = 0.0;

    // generate matrix on host
    int *part_map = nullptr;
    int n = 0;
    int nlocal = 0;
    int start_row = 0;
    int end_row = 0;

    HAType h_A;
    HAType h_G;
    VTypeHost h_b;
    if (matrixFilename != "" || nx > 0) {
      double  *values;
      int *col_idx;
      int *row_ptr;
      int nnz = 0;
      if (matrixFilename != "") {
        KokkosKernels::Impl::read_matrix<int, int, double>(
          &n, &nnz, &row_ptr, &col_idx, &values, matrixFilename.c_str());
      } else if (nx > 0) {
        h_G = Impl::generate_Laplace_matrix(nx, nx, nx);
        values = h_G.values.data();
        col_idx = h_G.col_idx.data();
        row_ptr = h_G.row_ptr.data();
        n = nx * nx *nx;
        nnz = row_ptr[n];
      }

      if (numRanks == 1) {
        // skip partitioning
        nlocal = n;
        start_row = 0;
        end_row = n;
        Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
          row_ptr, n+1);
        Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
          col_idx, nnz);
        Kokkos::View<double *, Kokkos::HostSpace> nzVal(
          values, nnz);
        h_A = HAType (rowPtr, colInd, nzVal, n);
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
          start_row = part_ptr[myRank];
          end_row = part_ptr[myRank+1];
          Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
            "Matrix::rowPtr", (end_row - start_row)+1);
          Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
            "Matrix::colInd", nnzlocal);
          Kokkos::View<double *, Kokkos::HostSpace> nzVal(
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
        } else
        #endif
        {
          if (myRank == 0) {
            std::cout << "  + using regular 1D partition of matrix : (n=" << n << ", nnz=" << nnz << ") " << std::endl;
          }
          nlocal = (n + numRanks - 1) / numRanks;
          start_row = myRank * nlocal;
          end_row = (myRank + 1) * nlocal;
          if (end_row > n)
          end_row = n;

          int nnzlocal = row_ptr[end_row] - row_ptr[start_row];
          Kokkos::View<int *, Kokkos::HostSpace> rowPtr(
            "Matrix::rowPtr", (end_row - start_row)+1);
          Kokkos::View<LOCAL_ORDINAL *, Kokkos::HostSpace> colInd(
            "Matrix::colInd", nnzlocal);
          Kokkos::View<double *, Kokkos::HostSpace> nzVal(
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
        }
      }
      h_b = VTypeHost("b_h", n);
      Kokkos::deep_copy(h_b, one);
    } else {
      h_A = Impl::generate_miniFE_matrix(N);
      // generate rhs
      h_b = Impl::generate_miniFE_vector(N);

      // global/local dimension
      n = h_b.extent(0);
      nlocal = (n + numRanks - 1) / numRanks;
      start_row = myRank * nlocal;
      end_row = (myRank + 1) * nlocal;
      if (end_row > n)
        end_row = n;

      // resize rowptr
      Kokkos::resize(h_A.row_ptr, (end_row - start_row)+1);

      // convert the column indexes to "standard" global indexes
      for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
        for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
          int p = h_A.col_idx(k) / MASK;
          int idx = h_A.col_idx(k) % MASK;
          int start_row = p * nlocal;
          h_A.col_idx(k) = start_row + idx;
        }
      }
    }
    if (sort_matrix) {
      if (myRank == 0) {
        std::cout << "  + sorting matrix .." << std::endl;
      }
      Impl::sort_matrix(h_A);
    }

    // copy the matrix to device
    // TODO: move this into op.setup
    Kokkos::View<int *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
    Kokkos::View<int *> col_idx("col_idx", h_A.col_idx.extent(0));
    Kokkos::View<double  *>  values("values",  h_A.values.extent(0));
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
        printf("%d %d %e\n",i,h_A.col_idx(k),h_A.values(k) );
      }
    }
    fclose(fp);
    sprintf(filename,"b%d_%d.dat",numRanks, myRank);
    fp = fopen(filename, "w");
    for (int i=0; i<h_b.extent(0); i++) {
      printf("%e\n",h_b(i));
    }
    fclose(fp);*/

    // local rhs on device
    Kokkos::pair<int, int> bounds(start_row, end_row);
    VType b_sub = Kokkos::subview(b, bounds);

    // input vector for SpMV
    int nrhs = 2;
    MType p("p", n, nrhs); // global
    MType p_sub = Kokkos::subview(p, bounds, Kokkos::ALL ()); // local

    // setup SpMV
    cgsolve_spmv<VType, HAType, AType, VType, MType, execution_space> op (n, start_row, end_row, h_A, A, time_spmv);
    op.setup(nlocal, nrhs, part_map);

    // local sol on device
    VType x_sub("x", b_sub.extent(0));

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
                << ", s = " << s
                << std::endl;
    }

    for (int nloop = 0; nloop < loop; nloop++) {
      Kokkos::deep_copy(x_sub, zero);
      Kokkos::fence();
      MPI_Barrier(MPI_COMM_WORLD);
      Kokkos::Timer timer;
      int num_iters = cg_solve(x_sub, op, b_sub,
                               p_sub, p,
                               max_iter, tolerance, s,
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

        double rnorm = 0.0;
        dot(r, r, rnorm);
        MPI_Allreduce(MPI_IN_PLACE, &rnorm, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        rnorm = std::sqrt(rnorm);
        double bnorm = 0.0;
        dot(b, b, bnorm);
        MPI_Allreduce(MPI_IN_PLACE, &bnorm, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        bnorm = std::sqrt(bnorm);
        if (myRank == 0) {
          printf( "\n ====================================" );
          printf( "\n rnorm = %e / %e = %e\n",rnorm,bnorm,rnorm/bnorm );
          printf( " num_iters = %i, time = %.2lf\n\n", num_iters, time);
        }
      } // end of check
    } // end of loop
  }
  Kokkos::finalize();
  MPI_Finalize();
  
  return 0;
}
