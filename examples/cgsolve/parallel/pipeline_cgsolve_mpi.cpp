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


// -------------------------------------------------------------
// SpMV
template <class YType, class HAType, class AType, class XType>
struct cgsolve_spmv
{
  using integer_view_t = Kokkos::View<int *>;
  using  scalar_view_t = Kokkos::View<double *>;

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
    integer_view_t num_recvs("num_recvs", numRanks);
    integer_view_t dsp_recvs("dsp_recvs", numRanks+1);
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
    integer_view_t map_recvs("map_recvs", numRanks);
    ngb_recvs = integer_view_t("ngb_recvs", num_neighbors);
    ptr_recvs = integer_view_t("ptr_recvs", num_neighbors+1);

    num_neighbors = 0;
    for (int p=0; p<numRanks; p++) {
      if (num_recvs(p) > 0) {
        ptr_recvs(num_neighbors+1) = ptr_recvs(num_neighbors) + num_recvs(p);
        ngb_recvs(num_neighbors) = p;
        map_recvs(p) = num_neighbors;
        num_neighbors ++;
      }
    }
    idx_recvs = integer_view_t("idx_recvs", total_recvs);
    buf_recvs = scalar_view_t("buf_recvs", total_recvs);
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++) {
        int p = h_A.col_idx(k) / nlocal;
        if (p != myRank) {
          int owner = map_recvs(p);
          idx_recvs(ptr_recvs(owner)) = h_A.col_idx(k);
          ptr_recvs(owner) ++;
        }
      }
    }
    for (int p=num_neighbors; p > 0; p--) {
      ptr_recvs(p) = ptr_recvs(p-1);
    }
    ptr_recvs(0) = 0;

    // ----------------------------------------------------------
    // find which elements to send to which process
    integer_view_t num_sends("num_sends", numRanks);
    integer_view_t dsp_sends("dsp_sends", numRanks+1);
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
    num_neighbors = 0;
    for (int p=0; p<numRanks; p++) {
      //printf( " > %d: num_sends(%d) = %d, num_recvs(%d) = %d\n",myRank,p,num_sends(p),p,num_recvs(p) );
      if (num_sends(p) > 0) {
        ptr_sends(num_neighbors+1) = ptr_sends(num_neighbors) + num_sends(p);
        ngb_sends(num_neighbors) = p;
        num_neighbors ++;
      }
      dsp_sends(p+1) = dsp_sends(p) + num_sends(p);
    }
    //printf( " %d: num_sends = %d, num_recvs = %d\n",myRank,ngb_sends.extent(0),ngb_recvs.extent(0) );

    idx_sends = integer_view_t("idx_sends", total_sends);
    buf_sends = scalar_view_t("buf_recvs", total_sends);
    MPI_Alltoallv(&(idx_recvs(0)), &(num_recvs(0)), &(dsp_recvs(0)), MPI_INT,
                  &(idx_sends(0)), &(num_sends(0)), &(dsp_sends(0)), MPI_INT,
                  MPI_COMM_WORLD);
    requests = (MPI_Request*)malloc(num_neighbors * sizeof(MPI_Request));

  }

  void exchange(XType x) {
    // TODO: need to know if GPU-aware
    int num_sends = ngb_sends.extent(0);
    for (int q=0; q<num_sends; q++) {
      int p = ngb_sends(q);
      int start = ptr_sends(q);
      int count = ptr_sends(q+1)-start;
      for (int k=start; k<start+count; k++) {
        buf_sends(k) = x(idx_sends(k));
      }
      //printf( " %d: MPI_Isend(count = %d, p = %d)\n",myRank,count,p );
      MPI_Isend(&(buf_sends(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &requests[q]);
    }

    for (int q=0; q<ngb_recvs.extent(0); q++) {
      int p = ngb_recvs(q);
      int start = ptr_recvs(q);
      int count = ptr_recvs(q+1)-start;

      MPI_Status stat;
      //printf( " %d: MPI_Irecv(count = %d, p = %d)\n",myRank,count,p );
      MPI_Recv(&(buf_recvs(start)), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &stat);

      for (int k=start; k<start+count; k++) {
        x(idx_recvs(k)) = buf_recvs(k);
      }
    }
    MPI_Waitall(num_sends, requests, MPI_STATUSES_IGNORE);
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
    this->exchange(x);
    this->local_apply(y, x);
  }

private:
  int n, nlocal;
  AType A;
  HAType h_A;

  int myRank, numRanks;
  MPI_Request *requests;

  integer_view_t ngb_recvs; // store proc id of neighbors
  integer_view_t ptr_recvs; // pointer to the begining of idx_recvs for each neighbor
  integer_view_t idx_recvs; // store col indices of elements to receive
  scalar_view_t  buf_recvs;

  integer_view_t ngb_sends; // store proc id of neighbors
  integer_view_t ptr_sends; // pointer to the begining of idx_recvs for each neighbor
  integer_view_t idx_sends; // store col indices of elements to receive
  scalar_view_t  buf_sends;
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
             int max_iter, double tolerance) {
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  int num_iters = 0;

  const double one = 1.0;
  const double zero = 0.0;

  double normr = 0.0;
  double alpha = 0.0;
  double beta = 0.0;
  double old_rr = 0.0;
  double new_rr = 0.0;
  double rAr = 0.0;
  double pAp = 0.0;

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
  op.apply(AAr, Ar_global);     // AAr = A*Ar
  axpby(r, one, b, -one, AAr);  // r = b-AAr

  //printf("Init: x, Ax, b, r\n" );
  //Kokkos::fence();
  //for (int i=0; i<b.extent(0); i++) printf(" %e, %e, %e, %e\n",x(i),AAr(i),b(i),r(i));

  // beta = r'*r
  beta = dot(r, r);
  MPI_Allreduce(MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // normr = sqrt(beta)
  normr = std::sqrt(beta);

  bool verbose = true;
  if (verbose && myRank == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }
  double brkdown_tol = std::numeric_limits<double>::epsilon();

  // Ar = A*r 
  axpby(Ar, one, r, zero, r);      // Ar = r
  op.apply(AAr, Ar_global);        // AAr = A*Ar
  axpby(Ar, one, AAr, zero, AAr);  // Ar = AAr

  for (int64_t k = 1; k <= max_iter && normr > tolerance; ++k) {
    // beta = r'*r
    new_rr = dot(r, r);
    MPI_Allreduce(MPI_IN_PLACE, &new_rr, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    // rAr = r'*Ar
    rAr = dot(r, Ar);
    MPI_Allreduce(MPI_IN_PLACE, &rAr, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // AAr = A*Ar
    op.apply(AAr, Ar_global);

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
  return num_iters;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

#ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_init();
#endif
#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  using  VType = Kokkos::View<double *>;
  using  AType = CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space>;
  using HAType = CrsMatrix<Kokkos::HostSpace>;

  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 100;
    int max_iter = argc > 2 ? atoi(argv[2]) : 200;
    double tolerance = argc > 3 ? atoi(argv[3]) : 1e-7;

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

    //Kokkos::fence();
    char filename[200];
    FILE *fp;
    sprintf(filename,"A%d_%d.dat",numRanks, myRank);
    fp = fopen(filename, "w");
    for (int i=0; i<h_A.row_ptr.extent(0)-1; i++) {
      for (int k=h_A.row_ptr(i); k<h_A.row_ptr(i+1); k++)
        fprintf(fp, "%d %d %.16e %d\n",i,h_A.col_idx(k), h_A.values(k), k );
    }
    fclose(fp);

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
    Kokkos::Timer timer;
    int num_iters = cg_solve(x_sub, op, b_sub,
                             p_sub, p,
                             max_iter, tolerance);
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
  #ifdef KOKKOS_ENABLE_SHMEMSPACE
  shmem_finalize();
  #endif
  #ifdef KOKKOS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
  #endif
  MPI_Finalize();
  
  return 0;
}
