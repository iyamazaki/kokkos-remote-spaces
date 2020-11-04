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

#include <Kokkos_RemoteSpaces.hpp>
#include <generate_matrix.hpp>
#include <mpi.h>

typedef Kokkos::Experimental::DefaultRemoteMemorySpace RemoteMemSpace_t;
typedef Kokkos::View<double **, RemoteMemSpace_t> RemoteView_t;

template <class YType, class AType, class XType>
void spmv(YType y, AType A, XType x) {

#ifdef KOKKOS_ENABLE_CUDA
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
                    int pid = idx / MASK;
                    int offset = idx % MASK;
//printf( " (row=%d, col=%d): %e * %e (pid=%d, offset=%d)\n",row,idx,A.values(i + row_start), x(pid, offset),pid,offset );
                    sum += A.values(i + row_start) * x(pid, offset);
                  },
                  y_row);
//printf( " -> y(%d) = %e\n\n",row,y_row );
              y(row) = y_row;
            });
      });

  RemoteMemSpace_t().fence();
}

template <class YType, class XType> double dot(YType y, XType x) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "DOT", y.extent(0),
      KOKKOS_LAMBDA(const int &i, double &lsum) { lsum += y(i) * x(i); },
      result);
  return result;
}

template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta, YType y) {
  int n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
}

template <class VType> void print_vector(int label, VType v) {
  std::cout << "\n\nPRINT " << v.label() << std::endl << std::endl;

  int myRank = 0;
  Kokkos::parallel_for(
      v.extent(0), KOKKOS_LAMBDA(const int i) {
        printf("%i %i %i %lf\n", label, myRank, i, v(i));
      });
  Kokkos::fence();
  std::cout << "\n\nPRINT DONE " << v.label() << std::endl << std::endl;
}

template <class VType, class AType, class PType>
int cg_solve(VType x, AType A, VType b, PType Ar_global, int max_iter,
             double tolerance) {
  int myproc = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
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

  int print_freq = max_iter / 10;
  print_freq = 1;
  if (print_freq > 50)
    print_freq = 50;
  if (print_freq < 1)
    print_freq = 1;
  VType r("r", x.extent(0));
  VType p("p", x.extent(0));
  VType Ap("Ap", x.extent(0));

  // extra vectors needed for pipeline
  VType Ar(Ar_global.data(), x.extent(0)); // Globally accessible data
  VType AAp("AAp", x.extent(0));
  VType AAr("AAr", x.extent(0));

  // r = b - A*x
  axpby(Ar, zero, x, one, x);   // Ar = x
  spmv(AAr, A, Ar_global);      // AAr = A*Ar
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
  if (verbose && myproc == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }
  //double brkdown_tol = std::numeric_limits<double>::epsilon();

  // Ar = A*r 
  axpby(Ar, one, r, zero, r);      // Ar = r
  spmv(AAr, A, Ar_global);         // AAr = A*Ar
  axpby(Ar, one, AAr, zero, AAr);  // Ar = AAr

  for (int k = 1; k <= max_iter && normr > tolerance; ++k) {
    // beta = r'*r
    new_rr = dot(r, r);
    MPI_Allreduce(MPI_IN_PLACE, &new_rr, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    // rAr = r'*Ar
    rAr = dot(r, Ar);
    MPI_Allreduce(MPI_IN_PLACE, &rAr, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // t = A*w (t is AAR)
    spmv(AAr, A, Ar_global);

    // synch dots
    // ...

    // normr = sqrt(rtrans)
    normr = std::sqrt(new_rr);
    if (verbose && myproc == 0) {
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
      //printf( " > beta = %e / %e = %e\n",new_rr,old_rr,beta );
      //printf( " > alpha = %e / %e = %e\n",new_rr,pAp,alpha );
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

  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 100;
    int max_iter = argc > 2 ? atoi(argv[2]) : 200;
    double tolerance = argc > 3 ? atoi(argv[3]) : 1e-7;
    CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
    Kokkos::View<double *, Kokkos::HostSpace> h_b =
        Impl::generate_miniFE_vector(N);

    Kokkos::View<int *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
    Kokkos::View<int *> col_idx("col_idx", h_A.col_idx.extent(0));
    Kokkos::View<double *> values("values", h_A.values.extent(0));
    CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A(
        row_ptr, col_idx, values, h_A.num_cols());
    Kokkos::View<double *> b("b", h_b.extent(0));

    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
    Kokkos::deep_copy(A.col_idx, h_A.col_idx);
    Kokkos::deep_copy(A.values, h_A.values);

    // Remote View
    RemoteView_t p = Kokkos::Experimental::allocate_symmetric_remote_view<RemoteView_t>(
        "MyView", numRanks, (h_b.extent(0) + numRanks - 1) / numRanks);

    int start_row = myRank * p.extent(1);
    int end_row = (myRank + 1) * p.extent(1);
    if (end_row > h_b.extent(0))
      end_row = h_b.extent(0);

    // CG
    Kokkos::pair<int, int> bounds(start_row, end_row);
    Kokkos::View<double *> b_sub = Kokkos::subview(b, bounds);
    Kokkos::View<double *> x_sub("x", b_sub.extent(0));

    Kokkos::Timer timer;
    int num_iters = cg_solve(x_sub, A, b_sub, p, max_iter, tolerance);
    double time = timer.seconds();

    {
      const double one = 1.0;
      const double zero = 0.0;
      Kokkos::View<double *> r("Y", x_sub.extent(0));
      Kokkos::View<double *> q(p.data(), x_sub.extent(0));
      axpby(q, one, x_sub, zero, x_sub);
      spmv(r, A, p);
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
    double spmv_bytes  = A.num_rows() * sizeof(int) +   // A.row_ptr
                         A.nnz()      * sizeof(int) +   // A.col_idx
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
        " N = %i, num_iters = %i, total_flops = %.2e, time = %.2lf, GFlops = %.2lf, GBs = %.2lf\n", 
        N, num_iters, total_flops, time, GFlops, GBs
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
