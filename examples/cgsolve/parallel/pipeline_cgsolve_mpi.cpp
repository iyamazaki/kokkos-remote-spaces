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

#define CGSOLVE_TIMER

// -------------------------------------------------------------
// SpMV
template <class YType, class HAType, class AType, class XType>
struct cgsolve_spmv
{
  using        integer_view_t = Kokkos::View<int *>;
  using   host_integer_view_t = Kokkos::View<int *, Kokkos::HostSpace>;
  using mirror_integer_view_t = typename integer_view_t::HostMirror;

  using  buffer_view_t = Kokkos::View<double *>;

  using team_policy_type = Kokkos::TeamPolicy<>;
  using member_type      = typename team_policy_type::member_type;

  cgsolve_spmv(int n_, HAType h_A_, AType A_) :
  n(n_),
  h_A(h_A_),
  A(A_)
  {}

  void setup() {
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // global/local dimension
    nlocal = (n + numRanks - 1) / numRanks;

    // TODO: need to know if GPU-aware, and integer_view_t should be on device/host
    // TODO: if GPU-aware, do setup on mirror view and deep-copy at the end?
    // ----------------------------------------------------------
    // find which elements to receive from which process
    host_integer_view_t num_recvs("num_recvs", numRanks);
    host_integer_view_t dsp_recvs("dsp_recvs", numRanks+1);
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = h_A.col_idx(k) / nlocal;
        if (p != myRank) {
          num_recvs(p) ++;
        }
      }
    }
    int total_recvs = 0;
    int num_neighbors = 0;
    for (int p=0; p<numRanks; p++) {
      if (num_recvs(p) > 0) {
        total_recvs += num_recvs(p);
        num_neighbors ++;
      }
      dsp_recvs(p+1) = dsp_recvs(p) + num_recvs(p);
    }
    host_integer_view_t map_recvs("map_recvs", numRanks);
    ngb_recvs = integer_view_t("ngb_recvs", num_neighbors);
    ptr_recvs = integer_view_t("ptr_recvs", num_neighbors+1);
    host_ngb_recvs = Kokkos::create_mirror_view(ngb_recvs);
    host_ptr_recvs = Kokkos::create_mirror_view(ptr_recvs);

    num_neighbors = 0;
    for (int p=0; p<numRanks; p++) {
      if (num_recvs(p) > 0) {
        host_ptr_recvs(num_neighbors+1) = host_ptr_recvs(num_neighbors) + num_recvs(p);
        host_ngb_recvs(num_neighbors) = p;
        map_recvs(p) = num_neighbors;
        num_neighbors ++;
      }
    }
    buf_recvs = buffer_view_t ("buf_recvs", total_recvs);
    idx_recvs = integer_view_t("idx_recvs", total_recvs);
    host_idx_recvs = Kokkos::create_mirror_view(idx_recvs);

    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = h_A.col_idx(k) / nlocal;
        if (p != myRank) {
          int owner = map_recvs(p);
          host_idx_recvs(host_ptr_recvs(owner)) = h_A.col_idx(k);
          host_ptr_recvs(owner) ++;
        }
      }
    }
    for (int p=num_neighbors; p > 0; p--) {
      host_ptr_recvs(p) = host_ptr_recvs(p-1);
    }
    host_ptr_recvs(0) = 0;
    Kokkos::deep_copy(ptr_recvs, host_ptr_recvs);
    Kokkos::deep_copy(idx_recvs, host_idx_recvs);
    Kokkos::deep_copy(ngb_recvs, host_ngb_recvs);

    // ----------------------------------------------------------
    // find which elements to send to which process
    host_integer_view_t num_sends("num_sends", numRanks);
    host_integer_view_t dsp_sends("dsp_sends", numRanks+1);
    MPI_Alltoall(&(num_recvs(0)), 1, MPI_INT, &(num_sends(0)), 1, MPI_INT, MPI_COMM_WORLD);
    int total_sends = 0;
    num_neighbors = 0;
    for (int p=0; p<numRanks; p++) {
      if (num_sends(p) > 0) {
        total_sends += num_sends(p);
        num_neighbors ++;
      }
      dsp_sends(p+1) = dsp_sends(p) + num_sends(p);
    }
    ngb_sends = integer_view_t("ngb_sends", num_neighbors);
    ptr_sends = integer_view_t("ptr_sends", num_neighbors+1);
    host_ngb_sends = Kokkos::create_mirror_view(ngb_sends);
    host_ptr_sends = Kokkos::create_mirror_view(ptr_sends);

    num_neighbors = 0;
    for (int p=0; p<numRanks; p++) {
      //printf( " > %d: num_sends(%d) = %d, num_recvs(%d) = %d\n",myRank,p,num_sends(p),p,num_recvs(p) );
      if (num_sends(p) > 0) {
        host_ptr_sends(num_neighbors+1) = host_ptr_sends(num_neighbors) + num_sends(p);
        host_ngb_sends(num_neighbors) = p;
        num_neighbors ++;
      }
      dsp_sends(p+1) = dsp_sends(p) + num_sends(p);
    }
    //printf( " %d: num_sends = %d, num_recvs = %d\n",myRank,ngb_sends.extent(0),ngb_recvs.extent(0) );

    buf_sends = buffer_view_t ("buf_recvs", total_sends);
    idx_sends = integer_view_t("idx_sends", total_sends);
    host_idx_sends = Kokkos::create_mirror_view(idx_sends);
    MPI_Alltoallv(&(host_idx_recvs(0)), &(num_recvs(0)), &(dsp_recvs(0)), MPI_INT,
                  &(host_idx_sends(0)), &(num_sends(0)), &(dsp_sends(0)), MPI_INT,
                  MPI_COMM_WORLD);

    Kokkos::deep_copy(ptr_sends, host_ptr_sends);
    Kokkos::deep_copy(idx_sends, host_idx_sends);
    Kokkos::deep_copy(ngb_sends, host_ngb_sends);
    requests = (MPI_Request*)malloc(num_neighbors * sizeof(MPI_Request));
  }

  void exchange(XType x) {
    // TODO: need to know if GPU-aware
    #ifdef CGSOLVE_TIMER
    Kokkos::Timer timer;
    #endif

    // pack
    int num_sends = ngb_sends.extent(0);
    auto host_sends = Kokkos::create_mirror_view(buf_sends);
    #if 1
    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    Kokkos::parallel_for(team_policy_type(num_sends, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type & team) {
        int q = team.league_rank();
        int p = ngb_sends(q);
        int start = ptr_sends(q);
        int count = ptr_sends(q+1)-start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 0, count),
          [&](const int k) {
            buf_sends(start+k) = x(idx_sends(start+k));
          });
      });
    Kokkos::fence();
    #ifdef CGSOLVE_TIMER
    time_comm_pack = timer.seconds();
    #endif

    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    Kokkos::deep_copy(host_sends, buf_sends);
    #ifdef CGSOLVE_TIMER
    time_comm_copy = timer.seconds();
    #endif
    #else
    timer.reset();
    auto host_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(host_x, x);
    time_comm_copy = timer.seconds();

    for (int q=0; q<num_sends; q++) {
      int p = ngb_sends(q);
      int start = ptr_sends(q);
      int count = ptr_sends(q+1)-start;
      for (int k=start; k<start+count; k++) {
        host_sends(k) = host_x(idx_sends(k));
      }
    }
    #endif

    // send
    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    for (int q=0; q<num_sends; q++) {
      int p = host_ngb_sends(q);
      int start = host_ptr_sends(q);
      int count = host_ptr_sends(q+1)-start;
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      MPI_Isend(&(host_sends(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests[q]);
    }

    // recv
    auto host_recvs = Kokkos::create_mirror_view(buf_recvs);
    for (int q=0; q<ngb_recvs.extent(0); q++) {
      int p = host_ngb_recvs(q);
      int start = host_ptr_recvs(q);
      int count = host_ptr_recvs(q+1)-start;

      MPI_Status stat;
      //printf( " %d: MPI_Irecv(count = %d, p = %d)\n",myRank,count,p );
      MPI_Recv(&(host_recvs(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &stat);
    }
    #ifdef CGSOLVE_TIMER
    time_comm_mpi = timer.seconds();
    #endif

    // unpack
    int num_recvs = ngb_recvs.extent(0);
    #if 1
    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    Kokkos::deep_copy(buf_recvs, host_recvs);
    #ifdef CGSOLVE_TIMER
    time_comm_copy += timer.seconds();
    #endif

    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    Kokkos::parallel_for(team_policy_type(num_recvs, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type & team) {
        int q = team.league_rank();
        int p = ngb_recvs(q);
        int start = ptr_recvs(q);
        int count = ptr_recvs(q+1)-start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, 0, count),
          [&](const int k) {
            x(idx_recvs(start+k)) = buf_recvs(start+k);
          });
      });
    Kokkos::fence();
    #ifdef CGSOLVE_TIMER
    time_comm_unpack = timer.seconds();
    #endif
    #else
    for (int q=0; q<num_recvs; q++) {
      int p = ngb_recvs(q);
      int start = ptr_recvs(q);
      int count = ptr_recvs(q+1)-start;
      for (int k=start; k<start+count; k++) {
        host_x(idx_recvs(k)) = host_recvs(k);
      }
    }
    timer.reset();
    Kokkos::deep_copy(x, host_x);
    time_comm_copy += timer.seconds();
    #endif

    // wait for send
    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    MPI_Waitall(num_sends, requests, MPI_STATUSES_IGNORE);
    #ifdef CGSOLVE_TIMER
    time_comm_mpi += timer.seconds();
    #endif
  }

  void local_apply(YType y, XType x) {
    #ifdef KOKKOS_ENABLE_CUDA
    int rows_per_team = 16;
    int team_size = 16;
    #else
    int rows_per_team = 512;
    int team_size = 1;
    #endif

    int vector_length = 8;

    int64_t nrows = y.extent(0);

    auto policy =
      require(Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                                   team_size, vector_length),
              Kokkos::Experimental::WorkItemProperty::HintHeavyWeight);
    Kokkos::parallel_for(
      "spmv", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        const int64_t first_row = team.league_rank() * rows_per_team;
        const int64_t last_row = first_row + rows_per_team < nrows
                                     ? first_row + rows_per_team
                                     : nrows;

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, first_row, last_row),
            [&](const int64_t row) {
              const int64_t row_start = A.row_ptr(row);
              const int64_t row_length = A.row_ptr(row + 1) - row_start;

              double y_row = 0.0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team, row_length),
                  [=](const int64_t i, double &sum) {
                    int64_t idx = A.col_idx(i + row_start);
                    //if (myRank == 0) {
                    //  printf( " (row=%d, col=%d): %e * %e (idx=%d)\n",
                    //         (int)row,(int)idx,A.values(i + row_start), x(idx), (int)idx );
                    //}
                    sum += A.values(i + row_start) * x(idx);
                  },
                  y_row);
                  //if (myRank == 0) {
                  //  printf( " -> y(%d) = %e\n\n",row,y_row );
                  //}
              y(row) = y_row;
            });
      });
  }

  void apply(YType y, XType x) {
    #ifdef CGSOLVE_TIMER
    Kokkos::Timer timer;
    #endif

    #ifdef CGSOLVE_TIMER
    timer.reset();
    #endif
    this->exchange(x);
    #ifdef CGSOLVE_TIMER
    time_comm = timer.seconds();

    timer.reset();
    #endif
    this->local_apply(y, x);
    #ifdef CGSOLVE_TIMER
    time_spmv = timer.seconds();
    #endif
  }

  double time_comm;
  double time_comm_copy;
  double time_comm_pack;
  double time_comm_unpack;
  double time_comm_mpi;
  double time_spmv;

private:
  int n, nlocal;
  AType A;
  HAType h_A;

  int myRank, numRanks;
  MPI_Request *requests;

  buffer_view_t  buf_recvs;
  integer_view_t ngb_recvs; // store proc id of neighbors
  integer_view_t ptr_recvs; // pointer to the begining of idx_recvs for each neighbor
  integer_view_t idx_recvs; // store col indices of elements to receive

  buffer_view_t  buf_sends;
  integer_view_t ngb_sends; // store proc id of neighbors
  integer_view_t ptr_sends; // pointer to the begining of idx_recvs for each neighbor
  integer_view_t idx_sends; // store col indices of elements to receive

  // mirrored on host
  mirror_integer_view_t host_ngb_recvs; // store proc id of neighbors
  mirror_integer_view_t host_ptr_recvs; // pointer to the begining of idx_recvs for each neighbor
  mirror_integer_view_t host_idx_recvs; // store col indices of elements to receive

  mirror_integer_view_t host_ngb_sends; // store proc id of neighbors
  mirror_integer_view_t host_ptr_sends; // pointer to the begining of idx_recvs for each neighbor
  mirror_integer_view_t host_idx_sends; // store col indices of elements to receive
};


// -------------------------------------------------------------
// dot
template <class YType, class XType> double dot(YType y, XType x) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int64_t &i, double &lsum) { lsum += y(i) * x(i); },
      result);
  return result;
}


// -------------------------------------------------------------
// axpby
template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta, YType y) {
  int64_t n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
}


// -------------------------------------------------------------
// cg_solve
template <class VType, class OP>
int cg_solve(VType x, OP op, VType b,
             VType Ar, VType Ar_global,
             int max_iter, double tolerance, bool verbose) {
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  const double one = 1.0;
  const double zero = 0.0;

  #ifdef CGSOLVE_TIMER
  Kokkos::Timer timer_spmv;
  Kokkos::Timer timer_idot;
  double time_spmv = 0.0;
  double time_spmv_comm   = 0.0;
  double time_spmv_copy   = 0.0;
  double time_spmv_pack   = 0.0;
  double time_spmv_unpack = 0.0;
  double time_spmv_mpi    = 0.0;
  double time_spmv_spmv   = 0.0;
  double time_idot = 0.0;
  double time_idot_comm = 0.0;
  #endif

  double normr = 0.0;
  double alpha = 0.0;
  double beta = 0.0;
  double old_rr = 0.0;
  double new_rr = 0.0;
  double rAr = 0.0;
  double pAp = 0.0;
  double dots_out[2];

  int64_t print_freq = max_iter / 10;
  print_freq = 1;
  if (print_freq > 50)
    print_freq = 50;
  if (print_freq < 1)
    print_freq = 1;
  VType r("r", x.extent(0));
  VType p("p", x.extent(0));
  VType Ap("Ap", x.extent(0));

  // extra vectors needed for pipeline
  VType AAp("AAp", x.extent(0));
  VType AAr("AAr", x.extent(0));

  // r = b - A*x
  axpby(Ar, zero, x, one, x);   // Ar = x
  #ifdef CGSOLVE_TIMER
  timer_spmv.reset();
  #endif
  op.apply(AAr, Ar_global);     // AAr = A*Ar
  #ifdef CGSOLVE_TIMER
  time_spmv += timer_spmv.seconds();
  time_spmv_comm   += op.time_comm;
  time_spmv_copy   += op.time_comm_copy;
  time_spmv_pack   += op.time_comm_pack;
  time_spmv_unpack += op.time_comm_unpack;
  time_spmv_mpi    += op.time_comm_mpi;
  time_spmv_spmv   += op.time_spmv;
  #endif
  axpby(r, one, b, -one, AAr);  // r = b-AAr

  //printf("Init: x, Ax, b, r\n" );
  //Kokkos::fence();
  //for (int i=0; i<b.extent(0); i++) printf(" %e, %e, %e, %e\n",x(i),AAr(i),b(i),r(i));

  // beta = r'*r
  beta = dot(r, r);
  MPI_Allreduce(MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // normr = sqrt(beta)
  normr = std::sqrt(beta);

  if (verbose && myRank == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }
  //double brkdown_tol = std::numeric_limits<double>::epsilon();

  // Ar = A*r 
  axpby(Ar, one, r, zero, r);      // Ar = r
  #ifdef CGSOLVE_TIMER
  timer_spmv.reset();
  #endif
  op.apply(AAr, Ar_global);        // AAr = A*Ar
  #ifdef CGSOLVE_TIMER
  time_spmv += timer_spmv.seconds();
  time_spmv_comm   += op.time_comm;
  time_spmv_copy   += op.time_comm_copy;
  time_spmv_pack   += op.time_comm_pack;
  time_spmv_unpack += op.time_comm_unpack;
  time_spmv_mpi    += op.time_comm_mpi;
  time_spmv_spmv   += op.time_spmv;
  #endif
  axpby(Ar, one, AAr, zero, AAr);  // Ar = AAr

  int num_iters = 0;
  for (int64_t k = 1; k <= max_iter && normr > tolerance; ++k) {
    // beta = r'*r
    #ifdef CGSOLVE_TIMER
    timer_idot.reset();
    #endif
    new_rr = dot(r, r);
    #ifdef CGSOLVE_TIMER
    time_idot += timer_idot.seconds();
    #endif


    // rAr = r'*Ar
    #ifdef CGSOLVE_TIMER
    timer_idot.reset();
    #endif
    rAr = dot(r, Ar);
    #ifdef CGSOLVE_TIMER
    time_idot += timer_idot.seconds();
    #endif

    #ifdef CGSOLVE_TIMER
    timer_idot.reset();
    #endif
    dots_out[0] = new_rr;
    dots_out[1] = rAr;
    MPI_Allreduce(MPI_IN_PLACE, dots_out, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    new_rr = dots_out[0];
    rAr = dots_out[1];
    #ifdef CGSOLVE_TIMER
    time_idot_comm += timer_idot.seconds();
    #endif

    // AAr = A*Ar
    #ifdef CGSOLVE_TIMER
    timer_spmv.reset();
    #endif
    op.apply(AAr, Ar_global);
    #ifdef CGSOLVE_TIMER
    time_spmv += timer_spmv.seconds();
    time_spmv_comm   += op.time_comm;
    time_spmv_copy   += op.time_comm_copy;
    time_spmv_pack   += op.time_comm_pack;
    time_spmv_unpack += op.time_comm_unpack;
    time_spmv_mpi    += op.time_comm_mpi;
    time_spmv_spmv   += op.time_spmv;
    #endif

    // synch dots
    // ...

    // normr = sqrt(rtrans)
    normr = std::sqrt(new_rr);
    if (verbose && myRank == 0) {
      std::cout << "Iteration = " << k << "   Residual = " << normr
                << std::endl;
    }

    // compute beta and alpha
    if (k == 1) {
      alpha = new_rr / rAr;
      //printf( " > alpha = %e / %e = %e\n",new_rr,rAr,alpha );
      beta = zero;
    } else {
      #if 0
      pAp = new_rr - beta*beta*alpha;
      beta = new_rr / old_rr;
      printf( " > pap = %e - %e * %e * %e = %e\n",new_rr, beta,beta,alpha, pAp);
      #else
      beta = new_rr / old_rr;
      pAp = rAr - new_rr * (beta / alpha);
      //printf( " > pap = %e - %e * (%e / %e) = %e\n",rAr,new_rr,beta,alpha, rAr - new_rr * (beta / alpha) );
      #endif
      alpha = new_rr / pAp;
      //printf( " %d:%d: > beta = %e / %e = %e\n",myRank,k,new_rr,old_rr,beta );
      //printf( " %d:%d: > alpha = %e / %e = %e\n",myRank,k,new_rr,pAp,alpha );
    }
    old_rr = new_rr;

    // p = r + beta*p
    axpby( p, one,  r, beta,  p);
    // Ap = Ar + beta * Ap
    axpby(Ap, one, Ar, beta, Ap);

    // x = x + alpha*p
    axpby(x, one, x,  alpha,  p);
    // r = r - alpha*Ap
    axpby(r, one, r, -alpha, Ap);

    // AAp = AAr + beta*AAp (since p = r + beta*p)
    axpby(AAp, one, AAr, beta, AAp);
    // Ar = Ar - alpha*AAp (since Ar = A*(r - alpha*Ap))
    axpby(Ar, one, Ar, -alpha, AAp);
    //printf( " %d: p Ap, x r, Ar AAp\n",k );
    //Kokkos::fence();
    //for (int i=0; i<b.extent(0); i++) printf(" %e %e, %e %e, %e %e\n",p(i),Ap(i), x(i),r(i), Ar(i),AAp(i));
    //printf("\n");

    num_iters = k;
  }
  #ifdef CGSOLVE_TIMER
  double min_spmv = 0.0, max_spmv = 0.0;
  MPI_Allreduce(&time_spmv, &min_spmv, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&time_spmv, &max_spmv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (myRank == 0) {
    printf( "\n  -------------------------------------------\n" );
    printf( "   time(SpMV)             = %.2e -- %.2e seconds\n", min_spmv,max_spmv );
    printf( "    + time(SpMV)::comm    =  %.2e seconds\n",time_spmv_comm );
    printf( "     > time(SpMV)::copy   =  %.2e seconds\n",time_spmv_copy );
    printf( "     > time(SpMV)::pack   =  %.2e seconds\n",time_spmv_pack );
    printf( "     > time(SpMV)::unpack =  %.2e seconds\n",time_spmv_unpack );
    printf( "     > time(SpMV)::mpi    =  %.2e seconds\n",time_spmv_mpi  );
    printf( "    + time(SpMV)::spmv    =  %.2e seconds\n",time_spmv_spmv );
    printf( "   time(iDot)           = %.2e seconds\n", time_idot+time_idot_comm );
    printf( "    + time(iDot)::comm  =  %.2e seconds\n",time_idot_comm );
    printf( "  -------------------------------------------\n" );
  }
  #endif
  return num_iters;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  using  VType = Kokkos::View<double *>;
  using  AType = CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space>;
  using HAType = CrsMatrix<Kokkos::HostSpace>;

  Kokkos::initialize(argc, argv);
  {
    int N            = 100;
    int max_iter     = 200;
    double tolerance = 1e-7;
    bool verbose     = false;
    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i],"-N")==0)) {
        N = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-iter")==0)) {
        max_iter = atoi(argv[++i]);
        continue;
      }
      if((strcmp(argv[i],"-v")==0)) {
        verbose = true;
        i++;
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

    // generate matrix on host
    HAType h_A = Impl::generate_miniFE_matrix(N);
    // generate rhs
    Kokkos::View<double *, Kokkos::HostSpace> h_b =
        Impl::generate_miniFE_vector(N);
    // global/local dimension
    int n = h_b.extent(0);
    int nlocal = (n + numRanks - 1) / numRanks;

    // convert the column indexes to "standard" global indexes
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = h_A.col_idx(k) / MASK;
        int idx = h_A.col_idx(k) % MASK;
        int64_t start_row = p * nlocal;
        h_A.col_idx(k) = start_row + idx;
      }
    }

    // copy the matrix to device
    // TODO: move this into op.setup
    Kokkos::View<int64_t *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
    Kokkos::View<int64_t *> col_idx("col_idx", h_A.col_idx.extent(0));
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
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++)
        fprintf(fp, "%d %d %.16e %d\n",i,h_A.col_idx(k), h_A.values(k), k );
    }
    fclose(fp);*/

    // global
    int64_t start_row = myRank * nlocal;
    int64_t end_row = (myRank + 1) * nlocal;
    if (end_row > n)
      end_row = n;

    // local rhs on device
    Kokkos::pair<int64_t, int64_t> bounds(start_row, end_row);
    VType b_sub = Kokkos::subview(b, bounds);
    // local sol on device
    VType x_sub("x", b_sub.extent(0));

    // input vector for SpMV
    VType p("p", n); // global
    VType p_sub = Kokkos::subview(p, bounds); // local

    // setup SpMV
    cgsolve_spmv<VType, HAType, AType, VType> op (n, h_A, A);
    op.setup();

    // call CG
    if (myRank == 0) {
      std::cout << " calling cg_solve ( N = " << N << ")" << std::endl;
    }
    Kokkos::fence();
    Kokkos::Timer timer;
    int num_iters = cg_solve(x_sub, op, b_sub,
                             p_sub, p,
                             max_iter, tolerance, verbose);
    double time = timer.seconds();

    {
      const double one = 1.0;
      const double zero = 0.0;
      VType r("Y", x_sub.extent(0));
      axpby(p_sub, one, x_sub, zero, x_sub);
      op.apply(r, p);
      axpby(r, -one, r, one, b_sub);
      //for (int i = 0; i < r.extent(0); i++) printf( "%d %e %e %e\n",i,b_sub(i),x_sub(i),r(i));

      double rnorm = dot(r, r);
      MPI_Allreduce(MPI_IN_PLACE, &rnorm, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      rnorm = std::sqrt(rnorm);
      if (myRank == 0) {
        printf( "\n ====================================" );
        printf( "\n rnorm=%e\n",rnorm );
      }
    }

    // Compute Bytes and Flops
    double spmv_bytes  = A.num_rows() * sizeof(int64_t) +   // A.row_ptr
                         A.nnz()      * sizeof(int64_t) +   // A.col_idx
                         A.nnz()      * sizeof(double)  +   // A.values
                         A.nnz()      * sizeof(double)  +   // input vector
                         A.num_rows() * sizeof(double);     // output vector
    double dot_bytes   = A.num_rows() * sizeof(double) * 2;
    double axpby_bytes = A.num_rows() * sizeof(double) * 3;

    double spmv_flops  = A.nnz()      * 2;
    double dot_flops   = A.num_rows() * 2;
    double axpby_flops = A.num_rows() * 3;

    int spmv_calls  = 1 + num_iters;
    int dot_calls   = num_iters;
    int axpby_calls = 2 + num_iters * 3;

    double total_flops = spmv_flops  * spmv_calls + 
                         dot_flops   * dot_calls  +
                         axpby_flops * axpby_calls;

    double total_bytes = spmv_bytes  * spmv_calls + 
                         dot_bytes   * dot_calls  +
                         axpby_bytes * axpby_calls;

    MPI_Allreduce(MPI_IN_PLACE, &total_flops, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_bytes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double GFlops = 1e-9 * total_flops / time;
    double GBs = (1.0 / 1024 / 1024 / 1024) * total_bytes / time;

    if (myRank == 0) {
      printf(
        " N = %i, n = %i, num_iters = %i, total_flops = %.2e, time = %.2lf, GFlops = %.2lf, GBs = %.2lf\n\n", 
        N, n, num_iters, total_flops, time, GFlops, GBs
      );
    }
  }
  
  Kokkos::finalize();
  MPI_Finalize();
  
  return 0;
}
