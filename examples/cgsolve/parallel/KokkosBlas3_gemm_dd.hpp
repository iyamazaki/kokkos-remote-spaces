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

#ifndef KOKKOSBLAS3_GEMM_DD_HPP_
#define KOKKOSBLAS3_GEMM_DD_HPP_

// DotBasedGEMM implements the optimization for C = beta*C + alpha*A^TB 
// with A and B matrices both being tall and skinny. C matrix is assumably 
// small, so, each entry of C is computed by performing the dot product of 
// respective columns of A and B matrices. Note that the dot products are
// performed on very long vectors, so, each dot product is distributed among
// numDivPerDot teams.     

#ifdef USE_REDUCE_DD_FUNCTOR
class ReduceDD {
  public:
  // the reduction result is an double-double, i.e., array of double
  typedef double value_type[];
  typedef Kokkos::View<double*>::size_type size_type;

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  size_type value_count;
   
  // input is two double vectoors 
  Kokkos::View<double*> X_;
  Kokkos::View<double*> Y_;

  // Be sure to set value_count in the constructor.
  ReduceDD (const Kokkos::View<double*>& X,
            const Kokkos::View<double*>& Y) :
        value_count (2), // output is DD
        X_ (X),
        Y_ (Y)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_type i, value_type sum) const {
    sum[0] += X_(i) * Y_(i);
  }

  KOKKOS_INLINE_FUNCTION
  void join (      volatile value_type dst,
             const volatile value_type src) const {
      dst[0] += src[0];
      dst[1] += src[1];
  }

  KOKKOS_INLINE_FUNCTION
  void init (value_type sum) const {
    sum[0] = 0.0;
    sum[1] = 0.0;
  }
};
#else
namespace ReduceDD {
   struct dd_add {
     double val[2];
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     dd_add() { 
       val[0] = 0.0;
       val[1] = 0.0;
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     dd_add(const dd_add & rhs) { 
       val[0] = rhs.val[0];
       val[1] = rhs.val[1];
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     dd_add& operator += (const dd_add& src) {
       val[0] += src.val[0];
       return *this;
     } 
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator += (const volatile dd_add& src) volatile {
       val[0] += src.val[0];
     }
   };
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity< ReduceDD::dd_add > {
      KOKKOS_FORCEINLINE_FUNCTION static ReduceDD::dd_add sum() {
         return ReduceDD::dd_add();
      }
   };
}
#endif

struct TagZero{};   // The init tag for beta=0 
struct TagInit{};   // The init tag for beta!=0 and beta !=1 
struct TagMult{};   // The multiplication tag for transposed A
template<class ExecSpace, class AV, class BV, class CV>
struct DotBasedGEMM_dd{

  const AV A;
  const BV B;
  CV C;

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
  

  DotBasedGEMM_dd(const scalar_A& alpha_, const AV& A_, const BV& B_, const scalar_C& beta_, const CV& C_):A(A_),B(B_),C(C_),alpha(alpha_),beta(beta_),numCrows(C.extent(0)),numCcols(C.extent(1)),dotSize(A.extent(0))
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
    
    // Multiply alpha*A^TB and add it to beta*C
    Kokkos::TeamPolicy<TagMult, ExecSpace> policyMult(numTeams, Kokkos::AUTO);
    Kokkos::parallel_for("Perform Dot Product Based GEMM", policyMult, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagZero&, const size_C &rowId, const size_C &colId ) const {
    C(rowId, colId) = CVT::zero(); 
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagInit&, const size_C &rowId, const size_C &colId ) const {
    C(rowId, colId) = beta * C(rowId, colId);
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

    #ifdef USE_REDUCE_DD_FUNCTOR
    // output in DD
    //scalar_C result[2];
    //result[0] = CVT::zero();
    //result[1] = CVT::zero();
    auto result = Kokkos::View<double*>("result", 2);
    ReduceDD reducer(lA, lB);
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, chunkSize), reducer, result);

    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () { 
      double local_result = result[0] + result[1];
      Kokkos::atomic_add(&C(rowId, colId), local_result);
      });
    #else
    ReduceDD::dd_add result;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, chunkSize), [&](const size_A k, ReduceDD::dd_add &update ) {
        if(baseInd + k < dotSize) {
          update.val[0] += alpha * A(baseInd+k, rowId) * B(baseInd+k, colId);
        }
     }, Kokkos::Sum<ReduceDD::dd_add>(result) );

    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () { 
      double local_result = result.val[0] + result.val[1];
      Kokkos::atomic_add(&C(rowId, colId), local_result);
      });
    #endif
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
