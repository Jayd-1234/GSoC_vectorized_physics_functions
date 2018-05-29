#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <xsimd/xsimd.hpp>

using namespace std;
using namespace xsimd;
using dtype_f = xsimd::simd_type<float>;

void myratio(float* num, float* den, float* out, int len)
{
	using dtype = simd_type<float>;
	size_t inc = dtype::size;
	size_t vec_size = len - len%inc;
	dtype num_vec;
	dtype den_vec;
	dtype res_vec;
	dtype zero_vec = (dtype)xsimd::set_simd(0.0f);
	//xsimd::batch_bool<bool, inc> cond;
	for (size_t i = 0; i<vec_size; i += inc)
	{
		num_vec.load_unaligned(&num[i]);
		den_vec.load_unaligned(&den[i]);
		auto cond = den_vec != zero_vec;
		res_vec = xsimd::select(cond, num_vec / den_vec, zero_vec);
		xsimd::store_unaligned(&out[i], res_vec);
	}

	for (size_t i = vec_size; i<len; i++)
	{
		if (den[i] == 0) {
			out[i] = 0;
		}
		else {
			out[i] = num[i] / den[i];
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
	size_t vec_size = len - len%inc;
	dtype deta; dtype dphi, res_vec;
	for (size_t i = 0; i<vec_size; i += inc)
	{
		eta1_vec.load_unaligned(&eta1[i]);
		phi1_vec.load_unaligned(&phi1[i]);
		eta2_vec.load_unaligned(&eta2[i]);
		phi2_vec.load_unaligned(&phi2[i]);
		deta = xsimd::abs(eta1_vec - eta2_vec);
		res_vec = deta;
		xsimd::store_unaligned(&out[i], res_vec);
	}

}

/* No need to explicitly vectorize deltaphi(). It is autovectorizable.
* */



// Need to check this once
inline dtype_f deltaPhi(dtype_f phi1, dtype_f phi2)
{
	dtype_f res_vec = phi1 - phi2;
	dtype_f pi_vec = (dtype_f)xsimd::set_simd((float)(M_PI));
	auto cond1 = res_vec > pi_vec;
	auto cond2 = res_vec <= -pi_vec;
	auto res_vec1 = xsimd::select(cond1, res_vec - (xsimd::floor((res_vec / pi_vec - 1) / 2) + 1) * 2 * pi_vec, res_vec);
	auto res_vec2 = xsimd::select(cond2, res_vec1 + (xsimd::floor((xsimd::abs(res_vec1) / pi_vec - 1) / 2) + 1) * 2 * pi_vec, res_vec1);
	return res_vec2;
}



inline dtype_f deltaR2(dtype_f eta1, dtype_f eta2, dtype_f phi1, dtype_f phi2)
{
	auto deta = xsimd::abs(eta1 - eta2);
	auto detp = deltaPhi(phi1, phi2);
	return xsimd::hypot(deta, detp);
}



inline float deltaPhi(float phi1, float phi2) {
	float result = phi1 - phi2;
	while (result > float(M_PI)) result -= float(2 * M_PI);
	while (result <= -float(M_PI)) result += float(2 * M_PI);
	return result;
}



inline float deltaR2(float eta1, float phi1, float eta2, float phi2) {
	float deta = std::abs(eta1 - eta2);
	float dphi = deltaPhi(phi1, phi2);
	return deta*deta + dphi*dphi;
}



void deltaR(float* eta1, float* phi1, float* eta2, float* phi2, float* out, int len)
{
	using dtype = xsimd::simd_type<float>;
	size_t inc = dtype::size;
	dtype eta1_vec;
	dtype phi1_vec;
	dtype eta2_vec;
	dtype phi2_vec;
	size_t vec_size = len - len%inc;
	for (size_t i = 0; i < vec_size; i+=inc)
	{
		eta1_vec = xsimd::load_unaligned(&eta1[i]);
		phi1_vec = xsimd::load_unaligned(&phi1[i]);
		eta2_vec = xsimd::load_unaligned(&eta2[i]);
		phi2_vec = xsimd::load_unaligned(&phi2[i]);
		auto res_vec = xsimd::sqrt(deltaR2(eta1_vec, eta2_vec, phi1_vec, phi2_vec));
		xsimd::store_unaligned(&out[i], res_vec);
	}

	for (size_t i = vec_size; i < len; i++)
	{
		out[i] = deltaR2(eta1[i], phi1[i], eta2[i], phi2[i]);
	}

}



