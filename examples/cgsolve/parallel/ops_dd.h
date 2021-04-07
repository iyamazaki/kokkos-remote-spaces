#ifndef OPS_DD_H
#define OPS_DD_H

#ifdef __cplusplus
extern "C" {
#endif

//struct dd_real {
//  double x[2];
//};

#define IEEE_754
#ifdef  IEEE_754
inline __host__ __device__ void dd_add(const volatile double &a_hi, const volatile double &a_lo,
                                       const volatile double &b_hi, const volatile double &b_lo,
                                             volatile double &c_hi,       volatile double &c_lo)
{
    double s1,s2;
    double v;
    double t1,t2;

    // Two-Sum(a_hi,b_hi)->(s1,s2)
    s1 = a_hi + b_hi;
    v = s1 - a_hi;
    s2 = ((b_hi - v) + (a_hi - (s1 - v)));
    // Two-Sum(a_lo,b_lo)->(t1,t2)
    t1 = a_lo + b_lo;
    v = t1 - a_lo;
    t2 = ((b_lo - v) + (a_lo - (t1 - v)));

    s2 = s2 + t1;

    t1 = s1 + s2;
    s2 = s2 - (t1 - s1);

    s2 = t2 + s2;

    c_hi = t1 + t2;
    c_lo = t2 - (c_hi - t1);
}
#else
inline __host__ __device__ void dd_add(const volatile double &a_hi, const volatile double &a_lo,
                                       const volatile double &b_hi, const volatile double &b_lo,
                                             volatile double &c_hi,       volatile double &c_lo)
{
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

    c_hi = t1;
    c_lo = t2;
}
#endif

#if 1
#define _QD_SPLITTER 134217729.0               // = 2^27 + 1
#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996
inline __host__ __device__ void dd_mul_split(double a, double &hi, double &lo) {
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

inline __host__ __device__ void dd2_mul(const double &a,
                                        const double &b,
                                              double &c_hi, double &c_lo)
{
  double a_hi, a_lo, b_hi, b_lo;
  c_hi = a * b;
  dd_mul_split(a, a_hi, a_lo);
  dd_mul_split(b, b_hi, b_lo);
  c_lo = ((a_hi * b_hi - c_hi) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}
#else
inline __host__ __device__ void dd2_mul(const double &a,
                                        const double &b,
                                              double &c_hi, double &c_lo)
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

inline __host__ __device__ void dd_mad(      double &c_hi, double &c_lo,
                                       const double &a,
                                       const double &b)
{
    double t_hi; 
    double t_lo; 
    dd2_mul(a, b, t_hi, t_lo);
    dd_add (t_hi, t_lo,
            c_hi, c_lo,
            c_hi, c_lo);
}

#ifdef __cplusplus
}
#endif

#endif /* OPS_DD_H */
