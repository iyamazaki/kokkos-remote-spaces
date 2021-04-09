/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "ops_dd.h"

#ifndef KOKKOSBLAS3_GEMM_DD_HPP_
#define KOKKOSBLAS3_GEMM_DD_HPP_

// DotBasedGEMM implements the optimization for C = beta*C + alpha*A^TB 
// with A and B matrices both being tall and skinny. C matrix is assumably 
// small, so, each entry of C is computed by performing the dot product of 
// respective columns of A and B matrices. Note that the dot products are
// performed on very long vectors, so, each dot product is distributed among
// numDivPerDot teams.     

/* ------------------------------------------------------------------------------------ */
namespace ReduceDD {
   struct dd_sum {
     double val_hi;
     double val_lo;

     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     dd_sum() {
       val_hi = 0.0;
       val_lo = 0.0;
     }

     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     dd_sum(const dd_sum & src) { 
       val_hi = src.val_hi;
       val_lo = src.val_lo;
     }

     KOKKOS_INLINE_FUNCTION   // add operator
     dd_sum& operator += (const dd_sum& src) {
       #if 0
       val_hi += src.val_hi;
       #else
       dd_add(src.val_hi, src.val_lo,
                  val_hi,     val_lo);
       #endif

       return *this;
     } 
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator += (const volatile dd_sum& src) volatile {
       #if 0
       val_hi += src.val_hi;
       #else
       dd_add(src.val_hi, src.val_lo,
                  val_hi,     val_lo);
       #endif
     }

     KOKKOS_INLINE_FUNCTION   // add operator
     dd_sum& operator + (const dd_sum& src) {
       dd_add(src.val_hi, src.val_lo,
                  val_hi,     val_lo);

       return *this;
     }
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator + (const volatile dd_sum& src) volatile {
       dd_add(src.val_hi, src.val_lo,
                  val_hi,     val_lo);
     }

    KOKKOS_INLINE_FUNCTION
    void operator=(const volatile dd_sum &src) volatile {
      val_hi = src.val_hi;
      val_lo = src.val_lo;
    }
    KOKKOS_INLINE_FUNCTION
    dd_sum&  operator=(const dd_sum &src) {
      this->val_hi = src.val_hi;
      this->val_lo = src.val_lo;
      return *this;
    }
   };
  

  struct dd_adder {
    double *hi;
    double *lo;

    KOKKOS_INLINE_FUNCTION
    dd_adder () {
      //hi = 0.0;
      //lo = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    dd_adder (double* _hi, double* _lo) {
      hi = _hi;
      lo =_lo;
    }

    KOKKOS_INLINE_FUNCTION
    void  operator+ (const volatile dd_adder &a) volatile {
      dd_add(*(a.hi), *(a.lo), *(this->hi), *(this->lo));
    }
    KOKKOS_INLINE_FUNCTION
    dd_adder& operator+ (const dd_adder &a) {
#if 0
      dd_add(*(a.hi), *(a.lo), *(this->hi), *(this->lo));
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile dd_adder &a) volatile {
      dd_add(*(a.hi), *(a.lo), *(this->hi), *(this->lo));
    }
    KOKKOS_INLINE_FUNCTION
    dd_adder& operator+=(const dd_adder &a) {
      dd_add(*(a.hi), *(a.lo), *(this->hi), *(this->lo));
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(const volatile dd_adder &a) volatile {
      *(this->hi) = *(a.hi);
      *(this->lo) = *(a.lo);
    }
    KOKKOS_INLINE_FUNCTION
    dd_adder&  operator=(const dd_adder &a) {
#if 0
      this->hi = a.hi;
      this->lo = a.lo;
#endif
      return *this;
    }
  };

}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity< ReduceDD::dd_sum > {
      KOKKOS_FORCEINLINE_FUNCTION static ReduceDD::dd_sum sum() {
         return ReduceDD::dd_sum();
      }
   };
}
/* ------------------------------------------------------------------------------------ */


struct TagZero{};      // The init tag for beta=0 
struct TagInit{};      // The init tag for beta!=0 and beta !=1 
struct TagInitCheck{}; // The init tag for checks
struct TagMult{};      // The multiplication tag for transposed A
template<class ExecSpace, class AV, class BV, class CV, class DD, class IV>
struct DotBasedGEMM_dd{

  const AV A;
  const BV B;
  CV C_hi;
  CV C_lo;
  DD C;
  IV checks;

  using scalar_A = typename AV::non_const_value_type;
  using size_A = typename AV::size_type;
  using scalar_C = typename CV::non_const_value_type;
  using size_C = typename CV::size_type;
  using AVT = Kokkos::Details::ArithTraits<scalar_A>;
  using CVT = Kokkos::Details::ArithTraits<scalar_C>;

  const scalar_A alpha;
  const scalar_C beta;

  // The following types (especially dotSize) could have simply been int,
  const size_C numCrows;           
  const size_C numCcols;

  size_C numDivPerDot;   // number of teams collectively performing a dot product
  size_C numTeams;       // total number of teams
  
  const size_A dotSize;  // the length of the vectors in the dot products
  size_A chunkSize;      // the local length of each team's share on the dot product  
  

  DotBasedGEMM_dd(const scalar_A& alpha_, const AV& A_,
                                          const BV& B_,
                  const scalar_C& beta_,  const CV& C_hi_, const CV& C_lo_, const DD& C_,
                                          const IV& checks_) :
  A(A_),
  B(B_),
  C_hi(C_hi_),
  C_lo(C_lo_),
  C(C_),
  alpha(alpha_),
  beta(beta_),
  checks(checks_),
  numCrows(C_hi.extent(0)),
  numCcols(C_hi.extent(1)),
  dotSize(A.extent(0))
  { }

  void run() {

    constexpr size_C workPerTeam = 4096;                   // Amount of work per team
    const size_C ndots = numCrows * numCcols;              // Number of dot products
    size_C appxNumTeams = (dotSize * ndots) / workPerTeam; // Estimation for appxNumTeams

    // Adjust appxNumTeams in case it is too small or too large
    if(appxNumTeams < 1)
      appxNumTeams = 1;
    if(appxNumTeams > 1024)
      appxNumTeams = 1024;

    #if 0
    // debug: forcing one team per C(i,j), so no reduction among teams at the end
    numTeams = ndots;
    numDivPerDot = 1;
    #else
    // If there are more dot products than the number of teams,
    // then set the number of teams to be number of dot products
    // and each team will perform only one dot product.
    // We don't want a team to perform more than one dot product.
    if(ndots >= appxNumTeams) {
      numTeams = ndots;
      numDivPerDot = 1;
    }
    // If there are more teams than dot products, each dot product can
    // potentially be performed by multiple teams. First, compute 
    // numDivPerDot as an integer (take the floor, not ceiling), then,
    // compute actual number of teams by using this factor.
    else{
      numDivPerDot = appxNumTeams / ndots;
      numTeams = ndots * numDivPerDot;
    }
    #endif

    // Determine the local length for the dot product
    chunkSize = dotSize / numDivPerDot;
    if(numDivPerDot > 1)
      chunkSize++;

    // Initialize C matrix if beta != 1
    if(beta == CVT::zero()) {
      Kokkos::MDRangePolicy<TagZero, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {numCrows, numCcols});
      Kokkos::parallel_for("Initialize C for Dot Product Based GEMM", policyInit, *this);
    }
    else if(beta != CVT::one()) {
      Kokkos::MDRangePolicy<TagInit, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {numCrows, numCcols});
      Kokkos::parallel_for("Initialize C for Dot Product Based GEMM", policyInit, *this);
    }

    // Initialize checks
    {
      Kokkos::MDRangePolicy<TagInitCheck, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {numCrows, numCcols});
      Kokkos::parallel_for("Initialize checks for Dot Product Based GEMM", policyInit, *this);
    }

    // Multiply alpha*A^TB and add it to beta*C
    Kokkos::TeamPolicy<TagMult, ExecSpace> policyMult(numTeams, Kokkos::AUTO);
    Kokkos::parallel_for("Perform Dot Product Based GEMM", policyMult, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagZero&, const size_C &rowId, const size_C &colId ) const {
    C_hi(rowId, colId) = CVT::zero(); 
    C_lo(rowId, colId) = CVT::zero();

    C(rowId, colId).val_hi = CVT::zero(); 
    C(rowId, colId).val_lo = CVT::zero();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagInit&, const size_C &rowId, const size_C &colId ) const {
    C_hi(rowId, colId) = beta * C_hi(rowId, colId);
    C_lo(rowId, colId) = beta * C_lo(rowId, colId);

    C(rowId, colId).val_hi = beta * C(rowId, colId).val_hi;
    C(rowId, colId).val_lo = beta * C(rowId, colId).val_lo;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagInitCheck&, const size_C &rowId, const size_C &colId ) const {
    checks(rowId, colId) = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagMult&, const typename Kokkos::TeamPolicy<>::member_type& teamMember) const {

    const size_C globalRank = teamMember.league_rank();
    const size_C localRank = globalRank % numDivPerDot;
    const size_C i = globalRank / numDivPerDot;
    const size_C rowId = i / numCcols;
    const size_C colId = i % numCcols;
    
    const size_A baseInd = chunkSize*localRank; 

    #if 1
    // inputs vectors 
    const size_A endInd = (baseInd+chunkSize > dotSize ? dotSize : baseInd+chunkSize); 
    Kokkos::pair<int, int> bounds(baseInd, endInd);
    Kokkos::pair<int, int> col(colId, colId+1);
    Kokkos::pair<int, int> row(rowId, rowId+1);
    auto lAview = Kokkos::subview(A, bounds, col);
    auto lBview = Kokkos::subview(B, bounds, row);
    auto lA = Kokkos::View<double*>(lAview.data(), endInd-baseInd);
    auto lB = Kokkos::View<double*>(lBview.data(), endInd-baseInd);

    // ----------------------------------------------------------------------------
    // paralllel-reduce among threads to form local accumulation into "result"
    // Question: does only one thread execute dd_mad at a time, or just write to result?
    #if 0
    // debug: just doing reduction with double
    double result_d = 0.0;
    Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, double &update ) {
	if(baseInd + k < dotSize)
	  update += alpha * A(baseInd+k, rowId) * B(baseInd+k, colId);
      }, result_d );
    teamMember.team_barrier ();
    ReduceDD::dd_sum result;
    result.val_hi = result_d;
    result.val_lo = 0.0;
    #else
    ReduceDD::dd_sum result;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, chunkSize), [&](const size_A k, ReduceDD::dd_sum &update) {
        if(baseInd + k < dotSize) {
          double a = alpha * A(baseInd+k, rowId);
          double b =         B(baseInd+k, colId);
          #if 0
           update.val_hi += alpha * A(baseInd+k, rowId) * B(baseInd+k, colId);
          #else
           dd_mad(a, b, update.val_hi, update.val_lo);
          #endif
        }
     }, Kokkos::Sum<ReduceDD::dd_sum>(result) );
    teamMember.team_barrier ();
    #endif

    //const int team_rank = teamMember.team_rank ();
    //if (rowId == 0 && colId == 0 && team_rank == 0) printf( " %d %e, %e\n",globalRank,result.val_hi,result.val_lo );

    // ----------------------------------------------------------------------------
    // parallel-reduce of local "result" among teams
    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
#if 1
      //ReduceDD::dd_adder cij (&(C_hi(rowId, colId)), &(C_lo(rowId, colId)));
      //ReduceDD::dd_adder lij (&(result.val_hi),      &(result.val_lo));
      //Kokkos::atomic_add(&cij, lij);
      Kokkos::atomic_add(&C(rowId, colId), result);
#else
      const int free_val = 0;
      const int busy_val = 1;
      #if 0
      double local_result = result.val.x[0] + result.val.x[1];
      Kokkos::atomic_add(&C(rowId, colId), local_result);
      #else
      // busy wait
      while (!Kokkos::atomic_compare_exchange_strong<int>(&checks(rowId, colId), free_val, busy_val)) {}

      // atomic-add
      #if 0
      dd_add(result.val_hi, result.val_lo,
             C_hi(rowId, colId), C_lo(rowId, colId));
      #else
       //double local_result = result.val_hi + result.val_lo;
       //Kokkos::atomic_add(&C_hi(rowId, colId), local_result);
       //C_hi(rowId, colId) += local_result;

       // compute: result += local
       #if 1
//if (rowId == 0 && colId == 0) printf( " - %d: %e + %e, %e + %e\n",globalRank, C_hi(rowId, colId),C_lo(rowId, colId), result.val_hi,result.val_lo );
#if 1
       double local_hi = C_hi(rowId, colId);
       double local_lo = C_lo(rowId, colId);
//if (rowId == 0 && colId == 0) printf( " - %d: %e + %e, %e + %e\n",globalRank, local_hi,local_lo, result.val_hi,result.val_lo );
       dd_add(result.val_hi, result.val_lo,
              local_hi, local_lo);
       C_hi(rowId, colId) = local_hi;
       C_lo(rowId, colId) = local_lo;
       //Kokkos::Impl::atomic_store<double>(&C_hi(rowId, colId), local_hi);
       //Kokkos::Impl::atomic_store<double>(&C_lo(rowId, colId), local_lo);
#else
       double local_hi = 0.0; //C_hi(rowId, colId);
       double local_lo = 0.0; //C_lo(rowId, colId);
       dd_add(result.val_hi, result.val_lo,
              local_hi, local_lo);
       Kokkos::atomic_add(&C_hi(rowId, colId), local_hi);
       Kokkos::atomic_add(&C_lo(rowId, colId), local_lo);
#endif
//if (rowId == 0 && colId == 0) printf( " + %d: %e + %e\n",globalRank, C_hi(rowId, colId),C_lo(rowId, colId) );
       #else
       local_hi += result.val_hi;
       local_lo += result.val_lo;
       #endif

       // storing into C(i, j)
       #if 1
       //Kokkos::Impl::atomic_store<double>(&C_hi(rowId, colId), local_hi);
       //Kokkos::Impl::atomic_store<double>(&C_lo(rowId, colId), local_lo);
       //Kokkos::atomic_add(&C_hi(rowId, colId), local_hi);
       //Kokkos::atomic_add(&C_lo(rowId, colId), local_lo);
       #else
       C_hi(rowId, colId) = local_hi;
       C_lo(rowId, colId) = local_lo;
       #endif
      #endif

      // free atomic
      Kokkos::Impl::atomic_store<int>(&checks(rowId, colId), free_val);
      #endif
#endif
      });
    //if (rowId == 0 && colId == 0 && team_rank == 0) printf( " %d -> %e\n",globalRank,C(rowId,colId) );
    #else
    scalar_C result = CVT::zero();
    Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, chunkSize), [&]( const size_A k, scalar_C &update ) {
	if(baseInd + k < dotSize)
	  update += alpha * A(baseInd+k, rowId) * B(baseInd+k, colId);
      }, result );

    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () { 
      Kokkos::atomic_add(&C(rowId, colId), result);
      });
    #endif
  }
};

#endif
