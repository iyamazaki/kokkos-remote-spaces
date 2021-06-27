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

#ifndef KOKKOSBLAS3_GEMM_RR_COMBINED_HPP_
#define KOKKOSBLAS3_GEMM_RR_COMBINED_HPP_

namespace CgsolverRR_Combined {

// DotBasedGEMM implements the optimization for C = beta*C + alpha*A^TB 
// with A and B matrices both being tall and skinny. C matrix is assumably 
// small, so, each entry of C is computed by performing the dot product of 
// respective columns of A and B matrices. Note that the dot products are
// performed on very long vectors, so, each dot product is distributed among
// numDivPerDot teams.     

/* ------------------------------------------------------------------------------------ */
namespace ReduceRR {
   template<typename scalarC_type, typename scalar_type>
   struct rr_sum {
     using non_const_scalar_type = typename std::remove_const<scalar_type>::type;
     using non_const_scalarC_type = typename std::remove_const<scalarC_type>::type;

     scalarC_type val;

     scalar_type val_hat;
     scalar_type xnorm;

     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     rr_sum() {
       val     = 0.0;
       val_hat = 0.0;

       xnorm = 0.0;
     }

     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     rr_sum(const rr_sum & src) { 
       val     = src.val;
       val_hat = src.val_hat;

       xnorm = src.xnorm;
     }

     KOKKOS_INLINE_FUNCTION   // add operator
     rr_sum& operator += (const rr_sum& src) {
       val     += src.val;
       val_hat += src.val_hat;

       xnorm += src.xnorm;

       return *this;
     } 
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator += (const volatile rr_sum& src) volatile {
       val     += src.val;
       val_hat += src.val_hat;

       xnorm += src.xnorm;
     }

     KOKKOS_INLINE_FUNCTION   // add operator
     rr_sum& operator + (const rr_sum& src) {
       val     += src.val;
       val_hat += src.val_hat;

       xnorm += src.xnorm;
       return *this;
     }
     KOKKOS_INLINE_FUNCTION   // volatile add operator 
     void operator + (const volatile rr_sum& src) volatile {
       val     += src.val;
       val_hat += src.val_hat;

       xnorm += src.xnorm;
     }

     KOKKOS_INLINE_FUNCTION
     void operator=(const volatile rr_sum &src) volatile {
       val     = src.val;
       val_hat = src.val_hat;

       xnorm = src.xnorm;
     }

     KOKKOS_INLINE_FUNCTION
     rr_sum&  operator=(const rr_sum &src) {
       this->val     = src.val;
       this->val_hat = src.val_hat;

       this->xnorm = src.xnorm;
       return *this;
     }
   };
}

/* ------------------------------------------------------------------------------------ */


struct TagZero{};      // The init tag for beta=0 
struct TagInit{};      // The init tag for beta!=0 and beta !=1 
struct TagMult{};      // The multiplication tag for transposed A
template<class ExecSpace, class AV, class BV, class CV, class VType>
struct DotBasedGEMM{

  const AV A;
  const BV B;
        CV C;

  const bool replaced;
  const VType x;

  using scalar_type = typename AV::non_const_value_type;
  using size_type   = typename AV::size_type;
  using STS = Kokkos::Details::ArithTraits<scalar_type>;

  using scalarC_type = typename CV::non_const_value_type::non_const_scalarC_type;
  using STSC = Kokkos::Details::ArithTraits<scalarC_type>;

  const scalar_type alpha;
  const scalar_type beta;

  // The following types (especially dotSize) could have simply been int,
  const size_type numCrows;           
  const size_type numCcols;

  size_type numDivPerDot;   // number of teams collectively performing a dot product
  size_type numTeams;       // total number of teams
  
  const size_type dotSize;  // the length of the vectors in the dot products
  size_type chunkSize;      // the local length of each team's share on the dot product  
  

  DotBasedGEMM(const scalar_type& alpha_, const AV& A_,
                                          const BV& B_,
               const scalar_type& beta_,  const CV& C_,
                          bool replaced_, const VType x_) :
  A(A_),
  B(B_),
  C(C_),
  replaced(replaced_),
  x(x_),
  alpha(alpha_),
  beta(beta_),
  numCrows(C.extent(0)),
  numCcols(C.extent(1)),
  dotSize(A.extent(0))
  { }

  void run() {

    constexpr size_type workPerTeam = 4096;                   // Amount of work per team
    const size_type ndots = numCrows * numCcols;              // Number of dot products
    size_type appxNumTeams = (dotSize * ndots) / workPerTeam; // Estimation for appxNumTeams

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
    if(beta == STS::zero()) {
      Kokkos::MDRangePolicy<TagZero, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {numCrows, numCcols});
      Kokkos::parallel_for("Initialize C for Dot Product Based GEMM", policyInit, *this);
    }
    else if(beta != STS::one()) {
      Kokkos::MDRangePolicy<TagInit, ExecSpace, Kokkos::Rank<2>> policyInit({0,0}, {numCrows, numCcols});
      Kokkos::parallel_for("Initialize C for Dot Product Based GEMM", policyInit, *this);
    }

    // Multiply alpha*A^TB and add it to beta*C
    Kokkos::TeamPolicy<TagMult, ExecSpace> policyMult(numTeams, Kokkos::AUTO);
    Kokkos::parallel_for("Perform Dot Product Based GEMM", policyMult, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagZero&, const size_type &rowId, const size_type &colId ) const {
    C(rowId, colId).val     = STSC::zero(); 
    C(rowId, colId).val_hat = STSC::zero();

    C(rowId, colId).xnorm = STSC::zero();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagInit&, const size_type &rowId, const size_type &colId ) const {
    C(rowId, colId).val     = beta * C(rowId, colId).val;
    C(rowId, colId).val_hat = beta * C(rowId, colId).val_hat;

    C(rowId, colId).xnorm = STSC::zero();
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagMult&, const typename Kokkos::TeamPolicy<>::member_type& teamMember) const {

    const size_type globalRank = teamMember.league_rank();
    const size_type localRank = globalRank % numDivPerDot;
    const size_type i = globalRank / numDivPerDot;
    const size_type rowId = i / numCcols;
    const size_type colId = i % numCcols;
    
    const size_type baseInd = chunkSize*localRank; 

    // inputs vectors 
    const size_type endInd = (baseInd+chunkSize > dotSize ? dotSize : baseInd+chunkSize); 
    Kokkos::pair<int, int> bounds(baseInd, endInd);
    Kokkos::pair<int, int> col(colId, colId+1);
    Kokkos::pair<int, int> row(rowId, rowId+1);
    auto lAview = Kokkos::subview(A, bounds, col);
    auto lBview = Kokkos::subview(B, bounds, row);
    auto lA = Kokkos::View<scalar_type*>(lAview.data(), endInd-baseInd);
    auto lB = Kokkos::View<scalar_type*>(lBview.data(), endInd-baseInd);


    if (rowId > colId) return;

    // ----------------------------------------------------------------------------
    // paralllel-reduce among threads to form local accumulation into "result"
    using rr_type = ReduceRR::rr_sum<scalarC_type, scalar_type>;
    rr_type result;
    if (replaced) {
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, chunkSize), [&](const size_type k, rr_type &update) {
          if(baseInd + k < dotSize) {
            scalarC_type a = A(baseInd+k, rowId); //alpha * A(baseInd+k, rowId);
            scalarC_type b = B(baseInd+k, colId); //        B(baseInd+k, colId);
            scalarC_type v = a * b;

            update.val     += v;
            update.val_hat += STSC::abs(v);
          }
       }, Kokkos::Sum<rr_type>(result) );
    } else {
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, chunkSize), [&](const size_type k, rr_type &update) {
          if(baseInd + k < dotSize) {
            scalarC_type a = A(baseInd+k, rowId); //alpha * A(baseInd+k, rowId);
            scalarC_type b = B(baseInd+k, colId); //        B(baseInd+k, colId);
            scalarC_type v = a * b;

            update.val     += v;
            update.val_hat += STSC::abs(v);

            if (rowId == 0 && colId == 0) {
              update.xnorm += x(baseInd+k) * x(baseInd+k);
            }
          }
       }, Kokkos::Sum<rr_type>(result) );
    }
    teamMember.team_barrier ();

    // ----------------------------------------------------------------------------
    // parallel-reduce of local "result" among teams (+, =, += operators are defined in rr_sum)
    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
      Kokkos::atomic_add(&C(rowId, colId), result);
      });
  }
};

} // CgsolverRR namespace

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<typename scalarC_type, typename scalar_type>
   struct reduction_identity< CgsolverRR_Combined::ReduceRR::rr_sum<scalarC_type, scalar_type> > {

      using T = CgsolverRR_Combined::ReduceRR::rr_sum<scalarC_type, scalar_type>;
      KOKKOS_FORCEINLINE_FUNCTION static T sum() {
         return T ();
      }
   };
}

#endif
