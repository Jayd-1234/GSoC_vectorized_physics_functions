#include <iostream>
#include <cstddef>
#include <xsimd\xsimd.hpp>
#include <immintrin.h>
#include <vector>
#include <cstdlib>

using namespace std;

using simd_type_float = xsimd::simd_type<float>;
using simd_type_double = xsimd::simd_type<double>;
using simd_type_int = xsimd::simd_type<int>;
using float_vector = vector<float>;
using double_vector = vector<double>;

class TVector3Methods
{
private:
	float_vector fX;
	float_vector fY;
	float_vector fZ;
	size_t length, inc, vec_length;


public:
	TVector3Methods();
	TVector3Methods(float* xi, float * yi, float * zi, size_t len, size_t increment, size_t vector_length);
	TVector3Methods(float_vector xi, float_vector yi, float_vector zi, size_t len, size_t increment, size_t vector_length);
	TVector3Methods fromSphericalCoordinates(float* r,float* theta, float* phi, int len);

	TVector3Methods fromCylindricalCoordinates(float * rho, float * phi, float * z, int len);
	float_vector setx(float_vector x);
	float_vector getx();
	
	float_vector sety(float_vector y);
	float_vector gety();

	float_vector getz();
	float_vector setz(float_vector z);

	float_vector rho();

	float_vector r();

};

TVector3Methods::TVector3Methods()
{
	length = 32;
	inc = simd_type_float::size;
	vec_length = length - length%inc;
	fX.resize(length, 0);
	fY.resize(length, 0);
	fZ.resize(length, 0);
}

TVector3Methods::TVector3Methods(float* xi, float* yi, float* zi, size_t len, size_t increment, size_t vector_length)
{
	length = len;
	inc = increment;
	vec_length = vector_length;
	fX.reserve(length);
	fY.reserve(length);
	fZ.reserve(length);
	std::copy(&xi[0], &xi[len - 1], fX.begin());
	std::copy(&yi[0], &yi[len - 1], fY.begin());
	std::copy(&zi[0], &zi[len - 1], fZ.begin());
}

TVector3Methods::TVector3Methods(float_vector xi, float_vector yi, float_vector zi, size_t len, size_t increment, size_t vector_length)
{
	length = len;
	inc = increment;
	vec_length = vector_length;
	fX.reserve(length);
	fY.reserve(length);
	fZ.reserve(length);
	std::copy(xi.begin(), xi.end(), fX.begin());
	std::copy(yi.begin(), yi.end(), fY.begin());
	std::copy(zi.begin(), zi.end(), fZ.begin());
}

TVector3Methods TVector3Methods::fromSphericalCoordinates(float* r, float* theta, float* phi, int len)
{
	auto length = len;
	simd_type_float r_vec, theta_vec, phi_vec, x_vec, y_vec, z_vec;
	float_vector fx = float_vector(length);
	float_vector fy = float_vector(length);
	float_vector fz = float_vector(length);
	inc = simd_type_float::size;
	vec_length = length - length%inc;
	for (size_t i = 0; i < vec_length; i+=inc)
	{
		r_vec.load_unaligned(&r[i]);
		phi_vec.load_unaligned(&phi[i]);
		theta_vec.load_unaligned(&theta[i]);
		x_vec = r_vec*xsimd::sin(theta_vec)*xsimd::cos(phi_vec);
		y_vec = r_vec*xsimd::sin(theta_vec)*xsimd::sin(phi_vec);
		z_vec = r_vec*xsimd::cos(theta_vec);
		xsimd::store_aligned(&fx[i], x_vec);
		xsimd::store_aligned(&fy[i], y_vec);
		xsimd::store_aligned(&fz[i], z_vec);
	}

	for (size_t i = vec_length; i < length; i++)
	{
		fx[i] = r[i] * sin(theta[i])*cos(phi[i]);
		fy[i] = r[i] * sin(theta[i])*sin(phi[i]);
		fz[i] = r[i] * cos(theta[i]);
	}
	return TVector3Methods(fx, fy, fz, length, inc, vec_length);
}

TVector3Methods TVector3Methods::fromCylindricalCoordinates(float* rho, float* phi, float* z, int len)
{
	auto length = len;
	simd_type_float r_vec, theta_vec, phi_vec, x_vec, y_vec, z_vec;
	float_vector fx = float_vector(length);
	float_vector fy = float_vector(length);
	float_vector fz = float_vector(length);
	inc = simd_type_float::size;
	vec_length = length - length%inc;
	for (size_t i = 0; i < vec_length; i += inc)
	{
		r_vec.load_unaligned(&rho[i]);
		phi_vec.load_unaligned(&phi[i]);
		theta_vec.load_unaligned(&z[i]);
		x_vec = r_vec*xsimd::cos(phi_vec);
		y_vec = r_vec*xsimd::sin(phi_vec);
		z_vec = theta_vec;
		xsimd::store_aligned(&fx[i], x_vec);
		xsimd::store_aligned(&fy[i], y_vec);
		xsimd::store_aligned(&fz[i], z_vec);
	}

	for (size_t i = vec_length; i < length; i++)
	{
		fx[i] = rho[i] * cos(phi[i]);
		fy[i] = rho[i] * sin(phi[i]);
		fz[i] = z[i];
	}
	return TVector3Methods(fx, fy, fz, length, inc, vec_length);
}

float_vector TVector3Methods::getx()
{
	return fX;
}

float_vector TVector3Methods::gety()
{
	return fY;
}

float_vector TVector3Methods::getz()
{
	return fZ;
}

float_vector TVector3Methods::setx(float_vector x)
{
	std::copy(x.begin(), x.end(), fX.begin());
}

float_vector TVector3Methods::sety(float_vector y)
{
	std::copy(y.begin(), y.end(), fY.begin());
}

float_vector TVector3Methods::setz(float_vector z)
{
	std::copy(z.begin(), z.end(), fZ.begin());
}

float_vector TVector3Methods::rho()
{
	float_vector rho = float_vector(length);
	simd_type_float rho_vec, x_vec, y_vec;
	for (size_t i = 0; i < vec_length; i+=inc)
	{
		x_vec = xsimd::load_aligned(&fX[i]);
		y_vec = xsimd::load_aligned(&fY[i]);
		rho_vec = xsimd::hypot(x_vec, y_vec);
		xsimd::store_aligned(&rho[i], rho_vec);
	}

	for (size_t i = vec_length; i< length; i++)
	{
		rho[i] = sqrt(fX[i] * fX[i] + fY[i] * fY[i]);
	}

	return rho;
}

float_vector TVector3Methods::r()
{
	float_vector rho = float_vector(length);
	simd_type_float rho_vec, x_vec, y_vec, z_vec;
	auto power = xsimd::set_simd(2);
	for (size_t i = 0; i < vec_length; i += inc)
	{
		x_vec = xsimd::load_aligned(&fX[i]);
		y_vec = xsimd::load_aligned(&fY[i]);
		z_vec = xsimd::load_aligned(&fZ[i]);
		rho_vec = xsimd::sqrt(x_vec*x_vec+ y_vec*y_vec+ z_vec*z_vec);
		xsimd::store_aligned(&rho[i], rho_vec);
	}

	for (size_t i = vec_length; i< length; i++)
	{
		rho[i] = sqrt(fX[i] * fX[i] + fY[i] * fY[i] + fZ[i]*fZ[i]);
	}

	return rho;
}

