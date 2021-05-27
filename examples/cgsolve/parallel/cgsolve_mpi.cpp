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

using host_execution_space = typename Kokkos::HostSpace;
using      execution_space = typename Kokkos::DefaultExecutionSpace;

using memory_space = typename execution_space::memory_space;

//#define USE_FLOAT
#if defined(USE_FLOAT)
 using scalar_type = float;

 #define MPI_SCALAR        MPI_FLOAT
 #define cusparseXcsrmv    cusparseScsrmv
 #define cublasXaxpy       cublasSaxpy
 #define cublasXscal       cublasSscal
 #define cublasXdot        cublasSdot

 //#define USE_MIXED_PRECISION
 #if defined(USE_MIXED_PRECISION)
  using solution_type = double;
 #else
  using solution_type = float;
 #endif
#else
 using scalar_type = double;
 using solution_type = double;

 #define MPI_SCALAR        MPI_DOUBLE
 #define cusparseXcsrmv    cusparseDcsrmv
 #define cublasXaxpy       cublasDaxpy
 #define cublasXscal       cublasDscal
 #define cublasXdot        cublasDdot
#endif

// -------------------------------------------------------------
// SpMV
template <class YType, class HAType, class AType, class XType, class SpaceType>
struct cgsolve_spmv
{
  using        integer_view_t = Kokkos::View<int *>;
  using   host_integer_view_t = Kokkos::View<int *, Kokkos::HostSpace>;
  using mirror_integer_view_t = typename integer_view_t::HostMirror;

  using  value_type = typename HAType::value_type;
  using  buffer_view_t = Kokkos::View<value_type *>;

  using team_policy_type  = Kokkos::TeamPolicy<SpaceType>;
  using member_type       = typename team_policy_type::member_type;

  cgsolve_spmv(int n_, HAType h_A_, AType A_, bool time_spmv_on_) :
  n(n_),
  h_A(h_A_),
  A(A_),
  time_spmv_on(time_spmv_on_),
  use_stream(false)
  {
    time_comm = 0.0;
    time_spmv = 0.0;
  }

  int getLocalDim() {
    return h_A.row_ptr.extent(0)-1;
  }
  int getLocalNnz() {
    return h_A.row_ptr(getLocalDim());
  }

  // -------------------------------------------------------------
  // setup P2P
  void setup(int nlocal_, int *part_map_) {
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // global/local dimension
    nlocal = nlocal_;
    part_map = part_map_;

    // ----------------------------------------------------------
    // find which elements to receive from which process
    const value_type zero (0.0);

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
    buf_recvs = buffer_view_t ("buf_recvs", total_recvs);
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
//#define USE_PERSISTENT_MPI
#ifdef  USE_PERSISTENT_MPI
    for (int q = 0; q < num_neighbors_recvs; q++) {
      int p = host_ngb_recvs(q);
      int start = host_ptr_recvs(q);
      int count = host_num_recvs(p); //host_ptr_recvs(q+1)-start;

      #if defined(CGSOLVE_GPU_AWARE_MPI)
      value_type *buffer = buf_recvs.data();
      MPI_Recv_init(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #else
      MPI_Recv_init(&(host_recvs(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #endif
    }
#endif

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

    buf_sends = buffer_view_t ("buf_send", total_sends);
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
#ifdef  USE_PERSISTENT_MPI
    for (int q = 0; q < num_neighbors_sends; q++) {
      int p = host_ngb_sends(q);
      int start = host_ptr_sends(q);
      int count = host_ptr_sends(q+1)-start;
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      #if !defined(CGSOLVE_GPU_AWARE_MPI)
      MPI_Send_init(&(host_sends(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #else
      value_type *buffer = buf_sends.data();
      MPI_Send_init(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #endif
    }
#endif

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
#ifdef  USE_PERSISTENT_MPI
      MPI_Start(&requests_recvs[q]);
#else
      int p = host_ngb_recvs(q);
      int start = host_ptr_recvs(q);
      int count = host_num_recvs(p); //host_ptr_recvs(q+1)-start;

      #if defined(CGSOLVE_GPU_AWARE_MPI)
      value_type *buffer = buf_recvs.data();
      MPI_Irecv(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #else
      MPI_Irecv(&(host_recvs(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_recvs[q]);
      #endif
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
        for (int q = 0; q < num_neighbors_sends; q++ ) {
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
#ifdef  USE_PERSISTENT_MPI
      MPI_Start(&requests_sends[q]);
#else
      int p = host_ngb_sends(q);
      int start = host_ptr_sends(q);
      int count = host_ptr_sends(q+1)-start;
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      #if !defined(CGSOLVE_GPU_AWARE_MPI)
      MPI_Isend(&(host_sends(start)), count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #else
      value_type *buffer = buf_sends.data();
      MPI_Isend(&buffer[start], count, MPI_SCALAR, p, 0, MPI_COMM_WORLD, &requests_sends[q]);
      #endif
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
            //int id = (k < count ? start+k : start+count-1);
            //x(idx_recvs(id)) = buf_recvs(id);
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
  // local SpMV
  void local_apply(YType y, XType x) {
    #if defined(CGSOLVE_ENABLE_CUBLAS)
     int numCols = n;
     int numRows = h_A.row_ptr.extent(0)-1;
     int nnz = h_A.row_ptr(numRows);
     value_type alpha (1.0);
     value_type beta  (0.0);
     cusparseXcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    numRows, numCols, nnz,
                    &alpha, descrA,
                            A.values.data(), A.row_ptr.data(), A.col_idx.data(),
                            x.data(),
                    &beta, y.data());
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

              value_type y_row = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, row_length),
                  [=](const int i, value_type &sum) {
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
  void apply(YType y, XType x) {
    Kokkos::Timer timer;
    if (time_spmv_on) {
      fence();
      timer.reset();
    }
    if (numRanks > 1) {
      this->exchange(x);
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

  // -------------------------------------------------------------
  // print
  void print() {
    auto  values_host = Kokkos::create_mirror_view(A.values);
    auto  rowptr_host = Kokkos::create_mirror_view(A.row_ptr);
    auto  colind_host = Kokkos::create_mirror_view(A.col_idx);
    Kokkos::deep_copy(values_host, A.values);
    Kokkos::deep_copy(rowptr_host, A.row_ptr);
    Kokkos::deep_copy(colind_host, A.col_idx);

    printf( "A=[\n" );
    for (int i = 0; i < nlocal; i++) {
      for (int k = (int)rowptr_host(i); k < (int)rowptr_host(i+1); k++) {
        printf( "%d %d %e\n",i,colind_host(k),values_host(k) );
      }
    }
    printf("];\n");
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
// axpby
template <class ZType, class YType, class XType>
void axpby(ZType z, scalar_type alpha, XType x, scalar_type beta, YType y) {
  int n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
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
template <class VType, class OP1, class OP2>
int cg_solve(VType x, VType b, OP1 op1, OP2 op2,
             int n, int start_row, int end_row,
             int max_iter, scalar_type tolerance,
             bool replace_residual, int replace_op, int maxNnzA, scalar_type norma,
             bool verbose, bool time_spmv_on, bool time_dot_on, bool time_axpy_on) {

  using value_type1 = typename OP1::value_type; // working precision
  using value_type2 = typename OP2::value_type; // inner lower precision
  using VType2 = Kokkos::View<value_type2 *>;

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
  double time_spmv = 0.0;
  double time_spmv_comm   = 0.0;
  double time_spmv_spmv   = 0.0;
  #if defined(CGSOLVE_SPMV_TIMER)
  double time_spmv_copy   = 0.0;
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
  double time_axpby = 0.0;
  double flop_axpby = 0.0;

  value_type2 normr = 0.0;
  value_type2 alpha = 0.0;
  value_type2 beta = 0.0;
  value_type2 old_rr = 0.0;
  value_type2 new_rr = 0.0;
  value_type2 pAp = 0.0;
  Kokkos::View<value_type2> dot_result("dot_result"); 
  auto dot_host = Kokkos::create_mirror_view(dot_result);

  // input vector for SpMV
  Kokkos::pair<int, int> bounds(start_row, end_row);
  VType2 p_global("p", n); // global
  VType2 p = Kokkos::subview(p_global, bounds); // local

  // residual vector
  int nloc = x.extent(0);
  VType2  r("r",  nloc);
  VType2 Ap("Ap", nloc);
  //#define KOKKOS_DEBUG_CGSOLVER
  #if defined(KOKKOS_DEBUG_CGSOLVER)
  auto  x_host = Kokkos::create_mirror_view(x);
  auto  b_host = Kokkos::create_mirror_view(b);
  auto  r_host = Kokkos::create_mirror_view(r);
  auto  p_host = Kokkos::create_mirror_view(p);
  auto Ap_host = Kokkos::create_mirror_view(Ap);
  #endif

  // for residual replacement
  const auto eps = std::numeric_limits<value_type2>::epsilon();
  const value_type2 replace_tol = std::sqrt(eps);
  value_type2 normx = 0.0;
  value_type2 d_replace = 0.0;
  value_type2 normr_prev = 0.0;
  value_type2 d_replace_prev = 0.0;
  value_type2 sum_x = 0.0;
  value_type2 sum_r = 0.0;
  VType z("z",  nloc); // store "group" update
  VType w("w",  nloc); // workspace for computing explicit residual norm (w = x + z)
  Kokkos::deep_copy(z, zero);

  // to compute true-residual with verbose on
  VType r_true  ("true_r",  nloc);
  VType Ax      ("Ax", nloc);
  VType x_global("true_r",  n);
  VType x_sub = Kokkos::subview(x_global, bounds);

  #if defined(KOKKOS_ENABLE_CUDA)
  cudaStream_t cudaStream[2];
  cublasHandle_t cublasHandle;

  if (cudaStreamCreate(&cudaStream[0]) != cudaSuccess ||
      cudaStreamCreate(&cudaStream[1]) != cudaSuccess) {
    printf( " ** faiiled to create cudaStreams **\n" );
    return -1;
  }
  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
    printf( " ** faiiled to create cublasHandle **\n" );
    return -1;
  }
  Kokkos::Cuda cudaSpace0 (cudaStream[0]);
  Kokkos::Cuda cudaSpace1 (cudaStream[1]);

  /*if (idot_option == 2) {
    op.setStream(&(cudaStream[0]));
    cublasSetStream(cublasHandle, cudaStream[0]);
  }*/
  #endif

  // ==============================================================================
  // r = b - A*x
  axpby(p, zero, x, one, x);    // p = x
  op2.apply(Ap, p_global);      // Ap = A*p
  axpby(r, one, b, -one, Ap);   // r = b-Ap

  #if defined(KOKKOS_DEBUG_CGSOLVER)
  printf("Init: x, Ax, b, r\n" );
  Kokkos::deep_copy( x_host, x);
  Kokkos::deep_copy( b_host, b);
  Kokkos::deep_copy( r_host, r);
  Kokkos::deep_copy(Ap_host, Ap);
  for (int i=0; i<b.extent(0); i++) printf(" %e, %e, %e, %e\n",x_host(i),Ap_host(i),b_host(i),r_host(i));
  #endif

  // beta = r'*r (using Kokkos and non Cuda-aware MPI)
  dot(r, dot_result);
  if (numRanks > 0) {
    Kokkos::fence();
    MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD);
  }
  Kokkos::deep_copy(dot_host, dot_result);
  beta = *(dot_host.data());
  normr = std::sqrt(beta);
  tolerance *= normr;
  if (replace_residual) {
    sum_r = normr;
    d_replace = (replace_tol/eps) * (maxNnzA * norma * sum_x + sum_r);
  }

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
  for (int k = 0; k <= max_iter && normr > tolerance; ++k) {

    // ==============================================================================
    // compute new search direction: p = r + beta*p
    if (k == 0) {
      new_rr = beta;
      axpby(p, one, r, zero, r);
    } else {
      // compute rr = r'*r
      old_rr = new_rr;
      if (time_dot_on) {
        Kokkos::fence();
        timer_dot.reset();
      }
      #if defined(CGSOLVE_ENABLE_CUBLAS)
      cublasXdot(cublasHandle, nloc, r.data(), 1, r.data(), 1, dot_result.data());
      #else
      dot(r, dot_result);
      #endif
      if (time_dot_on) {
        Kokkos::fence();
        time_dot += timer_dot.seconds();
        flop_dot += 2*nloc-1;
        timer_dot.reset();
      }
      if (numRanks > 1) {
        MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                      MPI_COMM_WORLD);
      }
      Kokkos::deep_copy(dot_host, dot_result);
      new_rr = *(dot_host.data());
      if (time_dot_on) {
        time_dot_comm += timer_dot.seconds();
      }

      // compute beta
      beta = new_rr / old_rr;

      // p = r + beta*p
      if (time_axpy_on) {
        Kokkos::fence();
        timer_axpy.reset();
      }
      #if 0//defined(CGSOLVE_ENABLE_CUBLAS)
      cublasXscal(cublasHandle, nloc, &(beta), p.data(), 1);
      cublasXaxpy(cublasHandle, nloc, &(one),  r.data(), 1, p.data(), 1);
      #else
      axpby(p, one, r, beta, p);
      #endif
      if (time_axpy_on) {
        Kokkos::fence();
        time_axpby += timer_axpy.seconds();
        flop_axpby += 2*nloc;
      }

      // compute various scalars on host
      normr_prev = normr;
      normr = std::sqrt(new_rr);
    }

    // ==============================================================================
    // Ap = A*p
    if (time_spmv_on) {
      timer_spmv.reset();
    }
    op2.apply(Ap, p_global);
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    printf(" %d: p, Ap\n", k);
    Kokkos::deep_copy( p_host, p);
    Kokkos::deep_copy(Ap_host, Ap);
    for (int i=0; i<b.extent(0); i++) printf(" %e, %e\n",p_host(i),Ap_host(i));
    #endif

    if (time_spmv_on) {
      Kokkos::fence();
      time_spmv += timer_spmv.seconds();
      time_spmv_comm   += op2.time_comm;
      time_spmv_spmv   += op2.time_spmv;
      #if defined(CGSOLVE_SPMV_TIMER)
      time_spmv_copy   += op2.time_comm_copy;
      time_spmv_pack   += op2.time_comm_pack;
      time_spmv_unpack += op2.time_comm_unpack;
      time_spmv_mpi    += op2.time_comm_mpi;
      time_spmv_wait_send += op2.time_comm_wait_send;
      time_spmv_wait_recv += op2.time_comm_wait_recv;
      #endif
    }

    // compute p'*A*p
    if (time_dot_on) {
      Kokkos::fence();
      timer_dot.reset();
    }
    #if defined(CGSOLVE_ENABLE_CUBLAS)
    cublasXdot(cublasHandle, nloc, Ap.data(), 1, p.data(), 1, dot_result.data());
    #else
    dot(Ap, p, dot_result);
    #endif
    if (time_dot_on) {
      time_dot += timer_dot.seconds();
      flop_dot += 2*nloc-1;
      timer_dot.reset();
    }
    if (numRanks > 1) {
      MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                    MPI_COMM_WORLD);
    }
    Kokkos::deep_copy(dot_host, dot_result);
    pAp = *(dot_host.data());
    if (time_dot_on) {
      time_dot_comm += timer_dot.seconds();
    }

    // compute alpha
    alpha = new_rr / pAp;
    if (verbose) {
      // r = b - A*x
      if (replace_residual) {
        axpby(w, one, x, one, z);  // w = x + z
        Kokkos::deep_copy(x_sub, w);
      } else {
        Kokkos::deep_copy(x_sub, x);
      }
      op1.apply(Ax, x_global);           // Ax = A*x
      axpby(r_true, one, b, -one, Ax);   // r = b-Ax

      // explicit residual norm
      dot(r_true, r_true, dot_result);
      Kokkos::deep_copy(dot_host, dot_result);
      MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                    MPI_COMM_WORLD);
      if (myRank == 0) {
        std::cout << "Iteration = " << k << "   Residual = " << normr << ", " << std::sqrt(*(dot_host.data()))
                  << ", beta = " << beta << ", alpha = " << alpha
                  << std::endl;
      }
    }

    // ==============================================================================
    // update local vectors
    if (time_axpy_on) {
      timer_axpy.reset();
    }

    // x = x + alpha*p
    // r = r - alpha*Ap
    #if defined(CGSOLVE_ENABLE_CUBLAS)
    scalar_type malpha = -alpha;
    cublasXaxpy(cublasHandle, nloc, &( alpha),  p.data(), 1, x.data(), 1);
    cublasXaxpy(cublasHandle, nloc, &(malpha), Ap.data(), 1, r.data(), 1);
    #else
    axpby(x, one, x,  alpha,  p);
    axpby(r, one, r, -alpha, Ap);
    #endif
    if (time_axpy_on) {
      Kokkos::fence();
      time_axpy += timer_axpy.seconds();
      flop_axpy += 4*nloc;
    }
    #if defined(KOKKOS_DEBUG_CGSOLVER)
    Kokkos::deep_copy(Ap_host, Ap);
    for (int i = 0; i < nloc; i++) printf( " Ap(%d) = %e\n",i,Ap_host(i) );
    Kokkos::deep_copy( p_host,  p);
    Kokkos::deep_copy( x_host,  x);
    Kokkos::deep_copy( r_host,  r);
    for (int i = 0; i < nloc; i++) printf( "%d %e %e %e\n",i,p_host(i),x_host(i),r_host(i) );
    #endif

    // ==============================================================================
    // replace residual vector, check
    if (replace_residual) {
      // compute norm(x) for now
      dot(x, x, dot_result);
      if (numRanks > 1) {
        MPI_Allreduce(MPI_IN_PLACE, dot_result.data(), 1, MPI_SCALAR, MPI_SUM,
                      MPI_COMM_WORLD);
      }
      Kokkos::deep_copy(dot_host, dot_result);
      normx = std::sqrt(*(dot_host.data()));

      // accumulate
      sum_x += normx;
      sum_r += normr;

      d_replace_prev = d_replace;
      d_replace = (eps / replace_tol) * (maxNnzA * norma * sum_x + sum_r);
      if (verbose) {
        if (myRank == 0) {
          std::cout << "  >> replace residual at Iteration = " << k << ", d = " << d_replace;
        }
      }
      if (replace_op == 0 || (d_replace > normr && d_replace_prev <= normr_prev) ) {
        // z = z + x
        axpby(z, one, z, one, x);

        // r = b - A*z
        Kokkos::deep_copy(x_sub, z);
        op1.apply(Ax, x_global);          // Ax = A*x
        axpby(r_true, one, b, -one, Ax);  // r = b-Ax

        // cast-copy to inner r
        axpby(r, one, r_true, zero, r_true); //

        // x = zero
        Kokkos::deep_copy(x, zero);

        // update data
        sum_x = 0.0;
        sum_r = normr;
        d_replace = (replace_tol/eps) * (maxNnzA * norma * sum_x + sum_r);
        if (verbose && myRank == 0) {
          std::cout << "  (replaced)" << std::endl;
        }
      } else {
        if (verbose && myRank == 0) {
          std::cout << "  (not replaced)" << std::endl;
        }
      }
    }

    num_iters = k;
  }

  // ==============================================================================
  // replace residual vector, check
  if (replace_residual) {
    axpby(x, one, x, one, z);    // x = x+z
  }

  cudaStreamSynchronize(cudaStream[0]);
  cudaStreamSynchronize(cudaStream[1]);
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

      #if defined(CGSOLVE_SPMV_TIMER)
      double min_copy = 0.0, max_copy = 0.0;
      MPI_Allreduce(&time_spmv_copy, &min_copy, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&time_spmv_copy, &max_copy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
        #if defined(CGSOLVE_SPMV_TIMER)
        printf( "     > time(SpMV)::copy    =  %.2e ~ %.2e seconds\n",min_copy,  max_copy );
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
        printf( "   time   ( axpy)         = %.2e seconds\n", time_axpy );
        printf( "   time   (axpby)         = %.2e seconds\n", time_axpby );
        printf( "   time   (total)         = %.2e seconds\n", time_axpy+time_axpby );
        printf( "   Gflop/s( axpy)         = %.2e (%.2e flops)\n", flop_axpy/(1e9*time_axpy), flop_axpy );
        printf( "   Gflop/s(axpby)         = %.2e (%.2e flops)\n", flop_axpby/(1e9*time_axpby), flop_axpby );
        printf( "   Gflop/s(total)         = %.2e (%.2e flops)\n", (flop_axpy+flop_axpby)/(1e9*(time_axpy+time_axpby)), flop_axpy+flop_axpby );
      }
      printf( "\n  -------------------------------------------\n" );
    }
  }
  #if defined(KOKKOS_ENABLE_CUDA)
  cublasDestroy(cublasHandle);
  cudaStreamDestroy(cudaStream[0]);
  cudaStreamDestroy(cudaStream[1]);
  /*if (idot_option == 2) {
    op.unsetStream();
  }*/
  #endif

  return num_iters;
}


// cg_solve (unit precision)
template <class VType, class OP>
int cg_solve(VType x, VType b, OP op,
             int n, int start_row, int end_row,
             int max_iter, scalar_type tolerance,
             bool replace_residual, int replace_op, int maxNnzA, scalar_type norma,
             bool verbose, bool time_spmv_on, bool time_dot_on, bool time_axpy_on) {

  return cg_solve(x, b, op, op, n, start_row, end_row,
                  max_iter, tolerance, replace_residual, replace_op, maxNnzA, norma,
                  verbose, time_spmv_on, time_dot_on, time_axpy_on);
}

// =============================================================
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  using  VType = Kokkos::View<solution_type *>;
  using  AType = CrsMatrix<memory_space, solution_type>;
  using HAType = CrsMatrix<host_execution_space, solution_type>;

  using VTypeHost = Kokkos::View<solution_type *, Kokkos::HostSpace>;

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
    scalar_type tolerance = 1e-8;
    std::string matrixFilename {""};

    int replace_op = 0; // 0: replace at every step
    bool replace_residual = false;

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
    Kokkos::View<scalar_type *> values("values",  h_A.values.extent(0));
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
    fprintf(fp,"%d %d %d\n",n,n,h_A.row_ptr(n));
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      fprintf(fp, " > %d: %d %d\n",i,h_A.row_ptr(i),h_A.row_ptr(i+1));
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        fprintf(fp, "%d %d %e\n",i,h_A.col_idx(k), h_A.values(k) );
        //fprintf(fp, "%d %d\n",i,h_A.col_idx(k) );
        //printf("%d %d %e\n",i,h_A.col_idx(k),h_A.values(k) );
      }
    }
    fclose(fp);
    sprintf(filename,"b%d_%d.dat",numRanks, myRank);
    fp = fopen(filename, "w");
    for (int i=0; i<h_b.extent(0); i++) {
      fprintf(fp,"%e\n",h_b(i));
    }
    fclose(fp);*/

    // local rhs on device
    Kokkos::pair<int, int> bounds(start_row, end_row);
    VType b_sub = Kokkos::subview(b, bounds);

    // setup SpMV
    cgsolve_spmv<VType, HAType, AType, VType, execution_space> op (n, h_A, A, time_spmv);
    op.setup(nlocal, part_map);
    //op.print();

    // local sol on device
    VType x_sub("x", b_sub.extent(0));
    if (strakos) {
      VType x0_global("x0", n);
      VType c_sub("c", b_sub.extent(0));

      Kokkos::deep_copy(x0_global, one);
      op.apply(c_sub, x0_global);

      scalar_type bnorm = 0.0;
      dot(c_sub, c_sub, bnorm);
      bnorm = std::sqrt(bnorm);

      axpby(b_sub, one/bnorm, c_sub, zero, b_sub);
    }

    int maxNnzA = 0;
    scalar_type Anorm = 0.0;
    if (replace_residual) {
      VType p("p", b.extent(0));
      VType q_global("q",  n);
      VType q_sub = Kokkos::subview(q_global, bounds);

      Anorm = n;
      Anorm = std::sqrt(Anorm);
      Kokkos::deep_copy(p, one);
      axpby(p, one/Anorm, p, zero, p);
      for (int i = 0; i < 10; i++) {
        // r = b - A*x
        Kokkos::deep_copy(q_sub, p);
        op.apply(p, q_global);        // p = A*p

        // Anorm = norm(p)
        dot(p, p, Anorm);
        Anorm = std::sqrt(Anorm);

        // p = p/Anorm
        axpby(p, one/Anorm, p, zero, p);
        if (verbose && myRank == 0) {
          std::cout << " norm(A) = " << Anorm << std::endl;;
        }
      }
      if (myRank == 0) {
        for (int i=0; i<n; i++) {
          int nnzRowA = h_A.row_ptr(i+1) - h_A.row_ptr(i+1);
          if (nnzRowA > maxNnzA) {
            maxNnzA = nnzRowA;
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &maxNnzA, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
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
                << " )" << std::endl;
    }

    for (int nloop = 0; nloop < loop; nloop++) {
      Kokkos::deep_copy(x_sub, zero);
      Kokkos::fence();
      MPI_Barrier(MPI_COMM_WORLD);
      Kokkos::Timer timer;
      int num_iters = cg_solve(x_sub, b_sub, op,
                               n, start_row, end_row,
                               max_iter, tolerance,
                               replace_residual, replace_op, maxNnzA, Anorm,
                               verbose, time_spmv, time_dot, time_axpy);
      Kokkos::fence();
      MPI_Barrier(MPI_COMM_WORLD);
      double time = timer.seconds();
      if (check) {
        VType p("p", n); // global
        VType p_sub = Kokkos::subview(p, bounds); // local
        VType r("Y", x_sub.extent(0));

        axpby(p_sub, one, x_sub, zero, x_sub);
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
