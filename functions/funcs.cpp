#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <xsimd/xsimd.hpp>

using namespace std;
using namespace xsimd;

void myratio(float* num, float* den, float* out, int len)
{
    using dtype = simd_type<float>;
    size_t inc = dtype::size;
    size_t vec_size = len-len%inc;
    dtype num_vec;
    dtype den_vec;
    dtype res_vec;
    dtype zero_vec = (dtype)xsimd::set_simd(0.0f);
    //xsimd::batch_bool<bool, inc> cond;
    for(size_t i=0; i<vec_size; i+=inc)
    {
        num_vec.load_unaligned(&num[i]);
        den_vec.load_unaligned(&den[i]);
        auto cond = den_vec != zero_vec;
        res_vec = xsimd::select(cond, num_vec/den_vec, zero_vec);
        xsimd::store_unaligned(&out[i], res_vec);
    }

    for(size_t i=vec_size; i<len; i++)
    {
        if (den[i]==0){
            out[i] =  0;
        }
        else {
            out[i] =  num[i]/den[i];
        }
    }

}

void deltaR2(float* eta1, float* phi1, float* eta2, float* phi2, float* out, int len)
{
    using dtype = xsimd::simd_type<float>;
    size_t inc = dtype::size;
    dtype eta1_vec;
    dtype phi1_vec;
    dtype eta2_vec;
    dtype phi2_vec;
    size_t vec_size = len-len%inc;
    dtype deta; dtype dphi, res_vec;
    for(size_t i=0; i<vec_size; i+=inc)
    {
        eta1_vec.load_unaligned(&eta1[i]);
        phi1_vec.load_unaligned(&phi1[i]);
        eta2_vec.load_unaligned(&eta2[i]);
        phi2_vec.load_unaligned(&phi2[i]);
        deta = xsimd::abs(eta1_vec-eta2_vec);
        res_vec = deta;
        xsimd::store_unaligned(&out[i], res_vec);
    }

}

void deltaPhi(float* phi1, float* phi2,float* result, int len) {
    for (int i=0; i<len; i++)
    {
        result[i] = phi1[i] - phi2[i];

    while (result[i] > (float)(M_PI)) result[i] -= (float)(2*M_PI);
    while (result[i] <= -(float)(M_PI)) result[i] += (float)(2*M_PI);
    }

}

