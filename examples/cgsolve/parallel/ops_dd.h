/*
 * These subroutines are based on (but modified to test with Kokkos)
 * from QD library for real double-double and quad-double arithmetic 
 * which is based on modified BSD license.
 *
 * The original copyright is:
 * 
 * This work was supported by the Director, Office of Science, Division
 * of Mathematical, Information, and Computational Sciences of the
 * U.S. Department of Energy under contract numbers DE-AC03-76SF00098 and
 * DE-AC02-05CH11231.
 * 
 * Copyright (c) 2003-2009, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from U.S. Dept. of Energy) All rights reserved. 
 * 
 * By downloading or using this software you are agreeing to the modified
 * BSD license that is in file "BSD-LBNL-License.doc" in the main ARPREC
 * directory. If you wish to use the software for commercial purposes
 * please contact the Technology Transfer Department at TTD@lbl.gov or
 * call 510-286-6457.
*/

#ifndef OPS_DD_H
#define OPS_DD_H

#ifdef __cplusplus
extern "C" {
#endif

inline __host__ __device__
double dd_two_sum(volatile double a, volatile double b, volatile double &err)
{
  double s = a + b;
  double bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

inline __host__ __device__ 
double dd_quick_two_sum(volatile double a, volatile double b, volatile double &err)
{
  double s = a + b;
  err = b - (s - a);
  return s;
}

#define IEEE_754
#ifdef  IEEE_754
inline __host__ __device__
void dd_add(const volatile double &a_hi, const volatile double &a_lo,
                  volatile double &b_hi,       volatile double &b_lo)
{
    // Two-Sum(a_hi,b_hi)->(s1,s2)
    double s1,s2;
    s1 = dd_two_sum(a_hi, b_hi, s2);
    // Two-Sum(a_lo,b_lo)->(t1,t2)
    double t1,t2;
    t1 = dd_two_sum(a_lo, b_lo, t2);

    // s2+=t1
    s2 += t1;

    // Two-Sum(s1,s2)->(s1,s2)
    s1 = dd_quick_two_sum(s1, s2, s2);

    // u2+=t2
    s2 += t2;

    // Two-Sum(s1,s2)->(b_hi, b_lo)
    b_hi = dd_quick_two_sum(s1, s2, b_lo);
}
#else
inline __host__ __device__
void dd_add(const volatile double &a_hi, const volatile double &a_lo,
                  volatile double &b_hi,       volatile double &b_lo)
{
#if 0
    //s = qd::two_sum(a.x[0], b.x[0], e);
    double s1,s2;
    s1 = dd_two_sum(a_hi, b_hi, s2);

    //e += (a.x[1] + b.x[1]);
    s2 = s2 + a_lo;

    //s = qd::quick_two_sum(s, e, e);
    b_hi = dd_quick_two_sum(s1, s2, b_lo);
#else
    double s1,s2;
    double v;
    double t1,t2;

    //s = qd::two_sum(a.x[0], b.x[0], e);
    s1 = a_hi + b_hi;
    v = s1 - a_hi;
    s2 = ((b_hi - v) + (a_hi - (s1 - v)));

    //e += (a.x[1] + b.x[1]);
    s2 = s2 + (a_lo + b_lo);

    //s = qd::quick_two_sum(s, e, e);
    //return dd_real(s, e);
    t1 = s1 + s2;
    t2 = s2 - (t1 - s1);

    b_hi = t1;
    b_lo = t2;
#endif
}
#endif


#if 1
#define _QD_SPLITTER 134217729.0               // = 2^27 + 1
#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996
inline __host__ __device__
void dd_mul_split(volatile double a, volatile double &hi, volatile double &lo)
{
  double temp;
  if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH) {
    a *= 3.7252902984619140625e-09;  // 2^-28
    temp = _QD_SPLITTER * a;
    hi = temp - (temp - a);
    lo = a - hi;
    hi *= 268435456.0;          // 2^28
    lo *= 268435456.0;          // 2^28
  } else {
    temp = _QD_SPLITTER * a;
    hi = temp - (temp - a);
    lo = a - hi;
  }
}

inline __host__ __device__
void dd2_mul(const volatile double &a,
             const volatile double &b,
                   volatile double &c_hi, volatile double &c_lo)
{
  double a_hi, a_lo, b_hi, b_lo;
  c_hi = a * b;
  dd_mul_split(a, a_hi, a_lo);
  dd_mul_split(b, b_hi, b_lo);
  c_lo = ((a_hi * b_hi - c_hi) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}
#else
inline __host__ __device__
void dd2_mul(const volatile double &a,
             const volatile double &b,
                   volatile double &c_hi, volatile double &c_lo)
{
  double p;
  double e;

  #if 0
  p = __dmul_rn(a, b);
  e = __fma_rn(a, b, -1.0 * p);
  #else
  p = a * b;
  e = a * b - p;
  #endif

  c_hi = p + e;
  c_lo = e - (c_hi - p);
}
#endif

inline __host__ __device__
void dd_mad(const volatile double &a,
            const volatile double &b,
            volatile double &c_hi, volatile double &c_lo)
{
#if 0
    c_hi += a* b;
#else
    double t_hi; 
    double t_lo; 
    dd2_mul(a, b, t_hi, t_lo);
    dd_add (t_hi, t_lo,
            c_hi, c_lo);
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* OPS_DD_H */
