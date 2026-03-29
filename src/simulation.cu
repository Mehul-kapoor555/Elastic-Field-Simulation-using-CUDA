 #include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/complex.h> // Use thrust::complex<float>
#include <cufftXt.h> // For complex types


// Grid dimensions
const int Nx = 32;
const int Ny = 32;
const int Nz = 32;
const int N = Nx * Ny * Nz;


// Material constants (for cubic symmetry)
__constant__ float C11;
__constant__ float C12;
__constant__ float C44;
__constant__ float Q11;
__constant__ float Q12;
__constant__ float Q44;

// Host function to initialize constants on the device
void initializeConstants() {
    float h_C11 = 1.746e11f;
    float h_C12 = 7.937e10f;
    float h_C44 = 1.111e11f;
    float h_Q11 = 0.089f;
    float h_Q12 = -0.026f;
    float h_Q44 = 0.03375f;

    // Copy constants to constant memory on device
    cudaMemcpyToSymbol(C11, &h_C11, sizeof(float));
    cudaMemcpyToSymbol(C12, &h_C12, sizeof(float));
    cudaMemcpyToSymbol(C44, &h_C44, sizeof(float));
    cudaMemcpyToSymbol(Q11, &h_Q11, sizeof(float));
    cudaMemcpyToSymbol(Q12, &h_Q12, sizeof(float));
    cudaMemcpyToSymbol(Q44, &h_Q44, sizeof(float));
}

// cuFFT helper: Error checking macro (optional, but recommended)
#define CUFFT_CHECK(call) \     // CUFFT_CHECK
    { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            printf("cuFFT Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }


//----Vec3
struct Vec3 {
	float x, y, z;
	__host__ __device__ Vec3() : x(0), y(0), z(0) {}
	__host__ __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

//----the Polarizations
struct polarization_initialization {
	int Nx, Ny, Nz;
	__host__ __device__ polarization_initialization(int Nx_, int Ny_, int Nz_) : Nx(Nx_), Ny(Ny_), Nz(Nz_) {}
	__host__ __device__ Vec3 operator()(const int idx) const {
		int i = idx % Nx;

		if (i < Nx / 2) {
			return Vec3(1.0f, 1.0f, 0.0f);  // Left half: strong diagonal in xy
		}
		else {
			return Vec3(0.5f, 0.5f, 0.0f);  // Right half: weaker diagonal
		}
	}
};

//----Spontaneous Strain Field Calculation, 6 component for each grid point : Structure row vector
struct spontaneous_strain {
	const Vec3* P;
	int Nx, Ny, Nz;

	__host__ __device__ spontaneous_strain
	(const Vec3* P_, int Nx_, int Ny_, int Nz_)
		: P(P_), Nx(Nx_), Ny(Ny_), Nz(Nz_) {
	}

	__host__ __device__ float operator()(int idx) const {
		int Np = Nx * Ny * Nz;

		int e_zero_id = idx % 6; // 0-5
		int pid = idx / 6;       // 0 to N-1

		float Px = P[pid].x, Py = P[pid].y, Pz = P[pid].z;

		if (e_zero_id == 0) { // e_11
			return Q11 * Px * Px + Q12 * (Py * Py + Pz * Pz);
		}
		else if (e_zero_id == 1) { // e_22
			return Q11 * Py * Py + Q12 * (Px * Px + Pz * Pz);
		}
		else if (e_zero_id == 2) { // e_33
			return Q11 * Pz * Pz + Q12 * (Px * Px + Py * Py);
		}
		else if (e_zero_id == 3) { // e_12
			return Q44 * Px * Py;
		}
		else if (e_zero_id == 4) { // e_13
			return Q44 * Px * Pz;
		}
		else { // e_23
			return Q44 * Py * Pz;
		}
	}
};


//----------------Phase two:----------------------------
// Prepration to enter the Fourier Space for calculating the displacmets Feild
// Splitting the spontaneous strains into separate fields - creating Fourior space(k vectors)

	//---- Functor to split spontaneous strains into separate fields : 6device vector for e_ij : 6 Fucntor calls
struct extract_single_strain_component {
	const float* e_zero; // Original concatenated spontaneous strains
	int component_id;    // 0 for e11, 1 for e22, ..., 5 for e23
	int N;               // Total number of grid points

	__host__ __device__
		extract_single_strain_component(const float* e_zero_, int component_id_, int N_)
		: e_zero(e_zero_), component_id(component_id_), N(N_) {
	}

	__host__ __device__
		float operator()(const int idx) const {
		return e_zero[idx * 6 + component_id];
	}
};

struct KVectorFunctor {
    int Nx, Ny, Nz;
    float Lx, Ly, Lz; // Physical lengths
    float two_pi;

    // Constructor takes grid dimensions and physical lengths
    __host__ __device__
    KVectorFunctor(int _Nx, int _Ny, int _Nz, float _Lx, float _Ly, float _Lz)
        : Nx(_Nx), Ny(_Ny), Nz(_Nz), Lx(_Lx), Ly(_Ly), Lz(_Lz), two_pi(2.0f * M_PI) {}

    __device__
    Vec3 operator()(int idx) const {
        // Compute (i, j, k) indices in the complex frequency grid
        int i = idx / (Ny * (Nz / 2 + 1));
        int j = (idx / (Nz / 2 + 1)) % Ny;
        int k = idx % (Nz / 2 + 1);

        // Convert to real-space indices (apply frequency shifting for FFT)
        int real_i = (i <= Nx / 2) ? i : i - Nx;
        int real_j = (j <= Ny / 2) ? j : j - Ny;
        int real_k = k;

        // Compute k-space vector components (in physical units)
        float kx = two_pi * real_i / Lx;  // kx = 2π * i / Lx
        float ky = two_pi * real_j / Ly;  // ky = 2π * j / Ly
        float kz = two_pi * real_k / Lz;  // kz = 2π * k / Lz

        // Return the 3D wave vector as a Vec3
        return Vec3(kx, ky, kz);
    }
};

struct build_A_and_b_complex {
	const Vec3* kvec;    // Real-valued Fourier-space k-vectors
	const thrust::complex<float>* e0_fft; // FFTed spontaneous strain fields (concatenated)

	int Nx, Ny, Nz;
	int NzComplex; // (Nz/2 + 1) for cufft output

	__host__ __device__
		build_A_and_b_complex(const Vec3* kvec_,
			const thrust::complex<float>* e0_fft_,
			int Nx_, int Ny_, int Nz_, int NzComplex_
			)
		: kvec(kvec_), e0_fft(e0_fft_),
		Nx(Nx_), Ny(Ny_), Nz(Nz_), NzComplex(NzComplex_) {
	}

	struct output {
		float A[9]; // 3x3 matrix (row-major)
		float b[3]; // Right-hand side vector
	};

	__host__ __device__
		output operator()(int idx) const {
		//int i = idx % Nx;
		//int j = (idx / Nx) % Ny;
		//int k = (idx / (Nx * Ny)) % NzComplex; // Notice: NzComplex, not Nz!

		float kx_val = kvec[idx].x;
        float ky_val = kvec[idx].y;
        float kz_val = kvec[idx].z;

		float kx2 = kx_val * kx_val;
		float ky2 = ky_val * ky_val;
		float kz2 = kz_val * kz_val;

		//--------- Build matrix A
		output out;

		out.A[0] = -(C11 * kx2 + 0.5f * C44 * (ky2 + kz2)); // A11
		out.A[1] = -(C12 * ky2 + 0.5f * C44 * kx_val * ky_val); // A12
		out.A[2] = -(C12 * kz2 + 0.5f * C44 * kx_val * kz_val); // A13

		out.A[3] = -(C12 * kx2 + 0.5f * C44 * kx_val * ky_val); // A21
		out.A[4] = -(C11 * ky2 + 0.5f * C44 * (kx2 + kz2)); // A22
		out.A[5] = -(C12 * kz2 + 0.5f * C44 * ky_val * kz_val); // A23

		out.A[6] = -(C12 * kx2 + 0.5f * C44 * kx_val * kz_val); // A31
		out.A[7] = -(C12 * ky2 + 0.5f * C44 * ky_val * kz_val); // A32
		out.A[8] = -(C11 * kz2 + 0.5f * C44 * (kx2 + ky2)); // A33

		//--------- Build source vector b
		int plane_size = Nx * Ny;

		int e11_idx = (0 * plane_size) + idx;
		int e22_idx = (1 * plane_size) + idx;
		int e33_idx = (2 * plane_size) + idx;
		int e12_idx = (3 * plane_size) + idx;
		int e13_idx = (4 * plane_size) + idx;
		int e23_idx = (5 * plane_size) + idx;

		thrust::complex<float> e11 = e0_fft[e11_idx];
		thrust::complex<float> e22 = e0_fft[e22_idx];
		thrust::complex<float> e33 = e0_fft[e33_idx];
		thrust::complex<float> e12 = e0_fft[e12_idx];
		thrust::complex<float> e13 = e0_fft[e13_idx];
		thrust::complex<float> e23 = e0_fft[e23_idx];

		// Multiply by -i, meaning swap real/imag parts and change sign
		auto minus_imag = [](thrust::complex<float> z) {
			return thrust::complex<float>(z.imag(), -z.real());
			};

		thrust::complex<float> e11_mi = minus_imag(e11);
		thrust::complex<float> e22_mi = minus_imag(e22);
		thrust::complex<float> e33_mi = minus_imag(e33);
		thrust::complex<float> e12_mi = minus_imag(e12);
		thrust::complex<float> e13_mi = minus_imag(e13);
		thrust::complex<float> e23_mi = minus_imag(e23);

		out.b[0] = C11 * kx_val * e11_mi.imag() +
			C12 * ky_val * e22_mi.imag() +
			C12 * kz_val * e33_mi.imag() +
			C44 * ky_val * e12_mi.imag() +
			C44 * kz_val * e13_mi.imag();

		out.b[1] = C12 * kx_val * e11_mi.imag() +
			C11 * ky_val * e22_mi.imag() +
			C12 * kz_val * e33_mi.imag() +
			C44 * kx_val * e12_mi.imag() +
			C44 * kz_val * e23_mi.imag();

		out.b[2] = C12 * kx_val * e11_mi.imag() +
			C12 * ky_val * e22_mi.imag() +
			C11 * kz_val * e33_mi.imag() +
			C44 * kx_val * e13_mi.imag() +
			C44 * ky_val * e23_mi.imag();

		return out;
	}
};


//---- Functor to solve 3x3 linear system A * u = b using Cramer's Rule
struct solve_3x3_system {
	const build_A_and_b_complex::output* A_b_data;

	__host__ __device__
		solve_3x3_system(const build_A_and_b_complex::output* A_b_data_)
		: A_b_data(A_b_data_) {
	}

	struct output {
		float u[3]; // Solved displacement components in Fourier space
	};

	__host__ __device__
		output operator()(int idx) const {
		build_A_and_b_complex::output system = A_b_data[idx];

		float A11 = system.A[0], A12 = system.A[1], A13 = system.A[2];
		float A21 = system.A[3], A22 = system.A[4], A23 = system.A[5];
		float A31 = system.A[6], A32 = system.A[7], A33 = system.A[8];

		float b1 = system.b[0];
		float b2 = system.b[1];
		float b3 = system.b[2];

		// Calculate determinant of A
		float detA = A11 * (A22 * A33 - A23 * A32) - A12 * (A21 * A33 - A23 * A31) + A13 * (A21 * A32 - A22 * A31);

		output out;

		if (fabsf(detA) > 1e-8f) { // Non-singular
			// Solve using Cramer's Rule
			float detU1 = b1 * (A22 * A33 - A23 * A32) - A12 * (b2 * A33 - A23 * b3) + A13 * (b2 * A32 - A22 * b3);
			float detU2 = A11 * (b2 * A33 - A23 * b3) - b1 * (A21 * A33 - A23 * A31) + A13 * (A21 * b3 - b2 * A31);
			float detU3 = A11 * (A22 * b3 - b2 * A32) - A12 * (A21 * b3 - b2 * A31) + b1 * (A21 * A32 - A22 * A31);

			out.u[0] = detU1 / detA;
			out.u[1] = detU2 / detA;
			out.u[2] = detU3 / detA;
		}
		else {
			// Matrix is singular: set displacement to zero
			out.u[0] = 0.0f;
			out.u[1] = 0.0f;
			out.u[2] = 0.0f;
			//printf("zero condition\n");

		}

		return out;
	}
};

//---- Functor to split u_hat into three separate fields for inverse FFT
struct split_u_hat_to_fields {
	const solve_3x3_system::output* u_hat_data;
	int field_id; // 0 for u_x, 1 for u_y, 2 for u_z

	__host__ __device__
		split_u_hat_to_fields(const solve_3x3_system::output* u_hat_data_, int field_id_)
		: u_hat_data(u_hat_data_), field_id(field_id_) {
	}

	__host__ __device__
		thrust::complex<float> operator()(int idx) const {
		// u_hat is purely real at this point (small numerical imaginary parts can exist but we treat them as 0)
		return thrust::complex<float>(u_hat_data[idx].u[field_id], 0.0f);
	}
};

struct convert_to_cufftComplex {
    __host__ __device__
    cufftComplex operator()(const thrust::complex<float>& val) const {
      cufftComplex result;
      result.x = val.real();
      result.y = val.imag();
      return result;
    }
};

struct StrainFromDisplacementFFT {
    const cufftComplex* u0;
    const cufftComplex* u1;
    const cufftComplex* u2;
    cufftComplex* strain_fft;  // Flattened: [e11, e22, e33, e23, e13, e12]
    
    const Vec3* kvec;  // Pass k-space vectors directly
    int complex_size;  // The total number of complex points in the Fourier grid
    float two_pi;

    __host__ __device__
    StrainFromDisplacementFFT(const cufftComplex* _u0,
                              const cufftComplex* _u1,
                              const cufftComplex* _u2,
                              cufftComplex* _strain_fft,
                              const Vec3* _kvec,
                              int _complex_size)
        : u0(_u0), u1(_u1), u2(_u2),
          strain_fft(_strain_fft), kvec(_kvec),
          complex_size(_complex_size), two_pi(2.0f * M_PI) {}

    __device__
    void operator()(int idx) const {
        if (idx >= complex_size) return;

        // Access k-space vector directly from kvec
        float kx = kvec[idx].x;
        float ky = kvec[idx].y;
        float kz = kvec[idx].z;

        float fx = two_pi * kx;
        float fy = two_pi * ky;
        float fz = two_pi * kz;

        // Load displacement components
        cufftComplex u0_ = u0[idx];
        cufftComplex u1_ = u1[idx];
        cufftComplex u2_ = u2[idx];

        // Compute derivatives (∂u_i/∂x_j = i * k_j * u_i in Fourier domain)
        // Real and imaginary parts follow the rule:
        // i * k * (a + ib) = -k*b + i*k*a

        // Normal strains (∂u_i/∂x_i)
        cufftComplex e11 = make_cuFloatComplex(-fx * u0_.y, fx * u0_.x);
        cufftComplex e22 = make_cuFloatComplex(-fy * u1_.y, fy * u1_.x);
        cufftComplex e33 = make_cuFloatComplex(-fz * u2_.y, fz * u2_.x);

        // Shear strains (symmetrized)
        cufftComplex e12 = make_cuFloatComplex(
            0.5f * (-fy * u0_.y - fx * u1_.y),
            0.5f * ( fy * u0_.x + fx * u1_.x)
        );
        cufftComplex e13 = make_cuFloatComplex(
            0.5f * (-fz * u0_.y - fx * u2_.y),
            0.5f * ( fz * u0_.x + fx * u2_.x)
        );
        cufftComplex e23 = make_cuFloatComplex(
            0.5f * (-fz * u1_.y - fy * u2_.y),
            0.5f * ( fz * u1_.x + fy * u2_.x)
        );

        // Store all 6 strain components at proper offsets
        strain_fft[0 * complex_size + idx] = e11;
        strain_fft[1 * complex_size + idx] = e22;
        strain_fft[2 * complex_size + idx] = e33;
        strain_fft[3 * complex_size + idx] = e12;
        strain_fft[4 * complex_size + idx] = e13;
        strain_fft[5 * complex_size + idx] = e23;
    }
};


struct compute_stress_tensor {
    const cufftComplex* e;     // pointer to total strain tensor (6 * complex_size)
    const cufftComplex* e0;    // pointer to spontaneous strain tensor (6 * complex_size)
    int comp;                  // which stress component (0 to 5)
    size_t complex_size;

    __host__ __device__
    compute_stress_tensor(const cufftComplex* e_,
                          const cufftComplex* e0_,
                          int comp_,
                          size_t complex_size_)
        : e(e_), e0(e0_), comp(comp_), complex_size(complex_size_) {}

    __host__ __device__
    cufftComplex operator()(int idx) const {
        cufftComplex sig = {0.0f, 0.0f};
        cufftComplex diff[6];

        // Compute difference between total and spontaneous strain for all 6 components at idx

        for (int i = 0; i < 6; ++i) {
            int global_idx = i * complex_size + idx;
            diff[i].x = e[global_idx].x - e0[global_idx].x;
            diff[i].y = e[global_idx].y - e0[global_idx].y;
        }

        // Compute stress component
        if (comp == 0) { // sigma11
            sig.x = C11 * diff[0].x + C12 * (diff[1].x + diff[2].x);
            sig.y = C11 * diff[0].y + C12 * (diff[1].y + diff[2].y);
        } else if (comp == 1) { // sigma22
            sig.x = C11 * diff[1].x + C12 * (diff[0].x + diff[2].x);
            sig.y = C11 * diff[1].y + C12 * (diff[0].y + diff[2].y);
        } else if (comp == 2) { // sigma33
            sig.x = C11 * diff[2].x + C12 * (diff[0].x + diff[1].x);
            sig.y = C11 * diff[2].y + C12 * (diff[0].y + diff[1].y);
        } else if (comp == 3) { // sigma12
            sig.x = C44 * diff[3].x;
            sig.y = C44 * diff[3].y;
        } else if (comp == 4) { // sigma13
            sig.x = C44 * diff[4].x;
            sig.y = C44 * diff[4].y;
        } else if (comp == 5) { // sigma23
            sig.x = C44 * diff[5].x;
            sig.y = C44 * diff[5].y;
        }

        return sig;
    }
};

struct ComplexVec3 {
    cufftComplex x, y, z;

    __host__ __device__
    ComplexVec3() {
        x = make_cuFloatComplex(0.0f, 0.0f);
        y = make_cuFloatComplex(0.0f, 0.0f);
        z = make_cuFloatComplex(0.0f, 0.0f);
    }

    __host__ __device__
    ComplexVec3(cufftComplex a, cufftComplex b, cufftComplex c)
        : x(a), y(b), z(c) {}
};



struct ElasticFieldFromStressAllComponents {
    const cufftComplex* sigma_fft;  // Packed [σ11, σ22, ..., σ23]
    const Vec3* kvec;    // Real-valued Fourier-space k-vectors
    size_t complex_size;

    __host__ __device__
    ElasticFieldFromStressAllComponents(const cufftComplex* sigma_fft_,
                                        const Vec3* kvec_,
                                        size_t complex_size_)
        : sigma_fft(sigma_fft_), kvec(kvec_), complex_size(complex_size_) {}

    __host__ __device__
    ComplexVec3 operator()(int idx) const {
        float kx = kvec[idx].x;
        float ky = kvec[idx].y;
        float kz = kvec[idx].z;
        
        cufftComplex s11 = sigma_fft[0 * complex_size + idx]; // stress 
        cufftComplex s22 = sigma_fft[1 * complex_size + idx];
        cufftComplex s33 = sigma_fft[2 * complex_size + idx];
        cufftComplex s12 = sigma_fft[3 * complex_size + idx];
        cufftComplex s13 = sigma_fft[4 * complex_size + idx];
        cufftComplex s23 = sigma_fft[5 * complex_size + idx];

        // Elastic force in Fourier space: -i * k_j * σ_ij
        cufftComplex fx, fy, fz;

        // fx = -i(kx*σ11 + ky*σ12 + kz*σ13)
        fx.x =  (kx * s11.y + ky * s12.y + kz * s13.y);  // Real part = -Imag(...)
        fx.y = -(kx * s11.x + ky * s12.x + kz * s13.x);  // Imag part = -Real(...)

        // fy = -i(kx*σ12 + ky*σ22 + kz*σ23)
        fy.x =  (kx * s12.y + ky * s22.y + kz * s23.y);
        fy.y = -(kx * s12.x + ky * s22.x + kz * s23.x);

        // fz = -i(kx*σ13 + ky*σ23 + kz*σ33)
        fz.x =  (kx * s13.y + ky * s23.y + kz * s33.y);
        fz.y = -(kx * s13.x + ky * s23.x + kz * s33.x);

        // Pack into one ComplexVec3 to store real and imag parts
        return ComplexVec3(fx, fy, fz); 
    }
};

struct ExtractForceComponent {
    int component;  // 0 = fx, 1 = fy, 2 = fz

    __host__ __device__
    ExtractForceComponent(int component_) : component(component_) {}

    __host__ __device__
    cufftComplex operator()(const ComplexVec3& v) const {
        if (component < 0 || component > 2) {
            // Return an error code as a cufftComplex (e.g., NaN)
            return cufftComplex{std::numeric_limits<float>::quiet_NaN(), 0.0f};  // NaN value
        }

        switch (component) {
            case 0: 
                return v.x;  // Extract fx
            case 1: 
                return v.y;  // Extract fy
            case 2: 
                return v.z;  // Extract fz
            default: 
                // Should never be hit due to previous validation
                return cufftComplex{0.0f, 0.0f};   
        }
    }
};

//Normalization of the results
struct normalize_by_N {
	float N_total;

	__host__ __device__
		normalize_by_N(float N_) : N_total(N_) {}

	__host__ __device__
		float operator()(float val) const {
		return val / N_total;
	}
};

struct pack_real_forces { 
    const float* fx_real;  // Force component in x
    const float* fy_real;  // Force component in y
    const float* fz_real;  // Force component in z

    __host__ __device__
    pack_real_forces(const float* fx_real_, const float* fy_real_, const float* fz_real_)
        : fx_real(fx_real_), fy_real(fy_real_), fz_real(fz_real_) {
    }

    __host__ __device__
    Vec3 operator()(int idx) const {
        // Packing the real force components (fx, fy, fz) into Vec3
        return Vec3(fx_real[idx], fy_real[idx], fz_real[idx]);
    }
};


int main(){
    initializeConstants();

    //Polarization initialization
	thrust::device_vector<Vec3> P(N);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		P.begin(),
		polarization_initialization(Nx, Ny, Nz)
	);

	//Spontaneous strains
	thrust::device_vector<float> e_zero(N * 6);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N * 6),
		e_zero.begin(),
		spontaneous_strain(
			thrust::raw_pointer_cast(P.data()),
			Nx, Ny, Nz
		)
	);



	//----------------Phase two:----------------------------

	//1.Extracting the sponteneous starin to 6 device vector : eij------------------------------------------------------
	thrust::device_vector<float> e11(N);
	thrust::device_vector<float> e22(N);
	thrust::device_vector<float> e33(N);
	thrust::device_vector<float> e12(N);
	thrust::device_vector<float> e13(N);
	thrust::device_vector<float> e23(N);

	//extract each field separately
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		e11.begin(),
		extract_single_strain_component(
			thrust::raw_pointer_cast(e_zero.data()),
			0, N
		)
	);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		e22.begin(),
		extract_single_strain_component(
			thrust::raw_pointer_cast(e_zero.data()),
			1, N
		)
	);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		e33.begin(),
		extract_single_strain_component(
			thrust::raw_pointer_cast(e_zero.data()),
			2, N
		)
	);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		e12.begin(),
		extract_single_strain_component(
			thrust::raw_pointer_cast(e_zero.data()),
			3, N
		)
	);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		e13.begin(),
		extract_single_strain_component(
			thrust::raw_pointer_cast(e_zero.data()),
			4, N
		)
	);
	thrust::transform(
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(N),
		e23.begin(),
		extract_single_strain_component(
			thrust::raw_pointer_cast(e_zero.data()),
			5, N
		)
	);


	//2. FFT the spontanues straines from splited onces------------------------------------------------------
	int NzComplex = Nz / 2 + 1; // For real-to-complex transform cuFFT rule
	size_t complex_size = Nx * Ny * NzComplex; // Complex points

	// Allocate complex output arrays as thrust::complex<float>
	thrust::device_vector<thrust::complex<float>> e11_fft(complex_size);
	thrust::device_vector<thrust::complex<float>> e22_fft(complex_size);
	thrust::device_vector<thrust::complex<float>> e33_fft(complex_size);
	thrust::device_vector<thrust::complex<float>> e12_fft(complex_size);
	thrust::device_vector<thrust::complex<float>> e13_fft(complex_size);
	thrust::device_vector<thrust::complex<float>> e23_fft(complex_size);

	// cuFFT Plan
	cufftHandle plan;
	CUFFT_CHECK(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_R2C));

	// Execute FFT for each strain component
	CUFFT_CHECK(cufftExecR2C(
		plan,
		thrust::raw_pointer_cast(e11.data()),
		reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(e11_fft.data()))
	));
	CUFFT_CHECK(cufftExecR2C(
		plan,
		thrust::raw_pointer_cast(e22.data()),
		reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(e22_fft.data()))
	));
	CUFFT_CHECK(cufftExecR2C(
		plan,
		thrust::raw_pointer_cast(e33.data()),
		reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(e33_fft.data()))
	));
	CUFFT_CHECK(cufftExecR2C(
		plan,
		thrust::raw_pointer_cast(e12.data()),
		reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(e12_fft.data()))
	));
	CUFFT_CHECK(cufftExecR2C(
		plan,
		thrust::raw_pointer_cast(e13.data()),
		reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(e13_fft.data()))
	));
	CUFFT_CHECK(cufftExecR2C(
		plan,
		thrust::raw_pointer_cast(e23.data()),
		reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(e23_fft.data()))
	));

	// Destroy FFT plan after all transforms
	CUFFT_CHECK(cufftDestroy(plan));

	//3. Pack all FFT fields into a single big concatenated array------------------------------------------------------
	thrust::device_vector<thrust::complex<float>> e0_fft_concat(6 * complex_size);

	// Copy each field into its correct block
	cudaMemcpy(
		thrust::raw_pointer_cast(e0_fft_concat.data()) + 0 * complex_size,
		thrust::raw_pointer_cast(e11_fft.data()),
		complex_size * sizeof(thrust::complex<float>),
		cudaMemcpyDeviceToDevice
	);
	cudaMemcpy(
		thrust::raw_pointer_cast(e0_fft_concat.data()) + 1 * complex_size,
		thrust::raw_pointer_cast(e22_fft.data()),
		complex_size * sizeof(thrust::complex<float>),
		cudaMemcpyDeviceToDevice
	);
	cudaMemcpy(
		thrust::raw_pointer_cast(e0_fft_concat.data()) + 2 * complex_size,
		thrust::raw_pointer_cast(e33_fft.data()),
		complex_size * sizeof(thrust::complex<float>),
		cudaMemcpyDeviceToDevice
	);
	cudaMemcpy(
		thrust::raw_pointer_cast(e0_fft_concat.data()) + 3 * complex_size,
		thrust::raw_pointer_cast(e12_fft.data()),
		complex_size * sizeof(thrust::complex<float>),
		cudaMemcpyDeviceToDevice
	);
	cudaMemcpy(
		thrust::raw_pointer_cast(e0_fft_concat.data()) + 4 * complex_size,
		thrust::raw_pointer_cast(e13_fft.data()),
		complex_size * sizeof(thrust::complex<float>),
		cudaMemcpyDeviceToDevice
	);
	cudaMemcpy(
		thrust::raw_pointer_cast(e0_fft_concat.data()) + 5 * complex_size,
		thrust::raw_pointer_cast(e23_fft.data()),
		complex_size * sizeof(thrust::complex<float>),
		cudaMemcpyDeviceToDevice
	);

	//4. Building the Fourior space :(kx, ky, kz)
	thrust::device_vector<Vec3> kvec(complex_size);

    float Lx = Nx, Ly = Ny, Lz = Nz; // physical size in x, y, z
    
    // Call the functor to fill the k_vectors
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(complex_size),
        kvec.begin(),
        KVectorFunctor (Nx, Ny, Nz, Lx, Ly, Lz)
    );

    
	//----------------Phase three:----------------------------

	//1.Building mechanical equilibrium matrix A and vector b------------------------------------------------------
	thrust::device_vector<build_A_and_b_complex::output> A_b(complex_size);

	thrust::transform(
		thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(complex_size),
		A_b.begin(),
		build_A_and_b_complex(
			thrust::raw_pointer_cast(kvec.data()),
			thrust::raw_pointer_cast(e0_fft_concat.data()), // now correctly packed
			Nx, Ny, Nz, NzComplex
		)
	);

	//2.Solving 3*3 matrix multiplications linear system result is displacments in Fourior space------------------------------------------------------
	thrust::device_vector<solve_3x3_system::output> u_hat(complex_size);

	thrust::transform(
		thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(complex_size),
		u_hat.begin(),
		solve_3x3_system(
			thrust::raw_pointer_cast(A_b.data())
		)
	);

    	//----------------Phase four:----------------------------

	//1.Functor to split u_hat into three separate fields for inverse FFT------------------------------------------------------

	// Allocate device vectors for u1_hat, u2_hat, u3_hat (complex fields)
	thrust::device_vector<thrust::complex<float>> u1_hat(complex_size);
	thrust::device_vector<thrust::complex<float>> u2_hat(complex_size);
	thrust::device_vector<thrust::complex<float>> u3_hat(complex_size);

	// Fill u1_hat
	thrust::transform(
		thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(complex_size),
		u1_hat.begin(),
		split_u_hat_to_fields(
			thrust::raw_pointer_cast(u_hat.data()),
			0 // field_id = 0 for u_x
		)
	);
	// Fill u2_hat
	thrust::transform(
		thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(complex_size),
		u2_hat.begin(),
		split_u_hat_to_fields(
			thrust::raw_pointer_cast(u_hat.data()),
			1 // field_id = 1 for u_y
		)
	);
	// Fill u3_hat
	thrust::transform(
		thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(complex_size),
		u3_hat.begin(),
		split_u_hat_to_fields(
			thrust::raw_pointer_cast(u_hat.data()),
			2 // field_id = 2 for u_z
		)
	);
    thrust::device_vector<cufftComplex> u1_hat_fft(complex_size);
	thrust::device_vector<cufftComplex> u2_hat_fft(complex_size);
	thrust::device_vector<cufftComplex> u3_hat_fft(complex_size);

    thrust::transform(
        u1_hat.begin(),
        u1_hat.end(),
        u1_hat_fft.begin(),
        convert_to_cufftComplex()
    );

    thrust::transform(
        u2_hat.begin(),
        u2_hat.end(),
        u2_hat_fft.begin(),
        convert_to_cufftComplex()
    );

    thrust::transform(
        u3_hat.begin(),
        u3_hat.end(),
        u3_hat_fft.begin(),
        convert_to_cufftComplex()
    );

    // Output strain: 6 components per point
    thrust::device_vector<cufftComplex> strain_fft(6 * complex_size); 

    // Launch functor over all complex grid points
    thrust::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(complex_size),
        StrainFromDisplacementFFT(
            thrust::raw_pointer_cast(u1_hat_fft.data()),
            thrust::raw_pointer_cast(u2_hat_fft.data()),
            thrust::raw_pointer_cast(u3_hat_fft.data()),
            thrust::raw_pointer_cast(strain_fft.data()),
            thrust::raw_pointer_cast(kvec.data()),
            complex_size
        )
    );
    
    thrust::device_vector<cufftComplex> e0_fft(6 * complex_size);  // 6 total strain values
    
    thrust::transform(
        e0_fft_concat.begin(),
        e0_fft_concat.end(),
        e0_fft.begin(),
        convert_to_cufftComplex()
    );

    // Allocate output vector for stress in Fourier space
    thrust::device_vector<cufftComplex> sigma_fft(6 * complex_size); // 6 stress components

    // For each of the 6 stress components (σ11 to σ23)
    for (int comp = 0; comp < 6; ++comp) {
        thrust::transform(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(complex_size),
            sigma_fft.begin() + comp * complex_size,
            compute_stress_tensor(
                thrust::raw_pointer_cast(strain_fft.data()),     // total strain
                thrust::raw_pointer_cast(e0_fft.data()),         // spontaneous strain
                comp,
                complex_size
            )
        );
    }

    //----Calculating elastic field
    thrust::device_vector<ComplexVec3> elastic_field_fft(3 * complex_size);  
    
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(complex_size),  // make int----size_t
        elastic_field_fft.begin(),
        ElasticFieldFromStressAllComponents(
            thrust::raw_pointer_cast(sigma_fft.data()),
            thrust::raw_pointer_cast(kvec.data()),
            complex_size
        )
    );
    
    //-------------Lastly perform Inverse FFT---------------

    // Define the device vectors to store the results
    thrust::device_vector<cufftComplex> fx_fft(complex_size);  
    thrust::device_vector<cufftComplex> fy_fft(complex_size);
    thrust::device_vector<cufftComplex> fz_fft(complex_size);

    // Apply the functor for extracting fx, fy, fz components
    thrust::transform(elastic_field_fft.begin(), elastic_field_fft.end(), fx_fft.begin(), ExtractForceComponent(0));
    thrust::transform(elastic_field_fft.begin(), elastic_field_fft.end(), fy_fft.begin(), ExtractForceComponent(1));
    thrust::transform(elastic_field_fft.begin(), elastic_field_fft.end(), fz_fft.begin(), ExtractForceComponent(2));

    // Allocate real output arrays for inverse FFT
	thrust::device_vector<float> fx_real(N);
	thrust::device_vector<float> fy_real(N);
	thrust::device_vector<float> fz_real(N);

    // Inverse FFT setup
    cufftHandle planC2R;
    CUFFT_CHECK(cufftPlan3d(&planC2R, Nx, Ny, Nz, CUFFT_C2R));

    CUFFT_CHECK(cufftExecC2R(planC2R,
        thrust::raw_pointer_cast(fx_fft.data()),
        thrust::raw_pointer_cast(fx_real.data())
    ));
    
    CUFFT_CHECK(cufftExecC2R(planC2R,
        thrust::raw_pointer_cast(fy_fft.data()),
        thrust::raw_pointer_cast(fy_real.data())
    ));
    
    CUFFT_CHECK(cufftExecC2R(planC2R,
        thrust::raw_pointer_cast(fz_fft.data()),
        thrust::raw_pointer_cast(fz_real.data())
    ));
    
    // Destroy the inverse plan
    CUFFT_CHECK(cufftDestroy(planC2R));

    //Normalization
	float N_total = static_cast<float>(Nx * Ny * Nz);

	thrust::transform(
		fx_real.begin(), fx_real.end(),
		fx_real.begin(),
		normalize_by_N(N_total)
	);
	thrust::transform(
		fy_real.begin(), fy_real.end(),
		fy_real.begin(),
		normalize_by_N(N_total)
	);
	thrust::transform(
		fz_real.begin(), fz_real.end(),
		fz_real.begin(),
		normalize_by_N(N_total)
	);

    // Define a device vector to store the packed force field
    thrust::device_vector<Vec3> force_real_space(N);  // Real space force field

    // Apply the functor to pack the force components (fx, fy, fz) into the force_real_space field
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        force_real_space.begin(),
        pack_real_forces(
            thrust::raw_pointer_cast(fx_real.data()),
            thrust::raw_pointer_cast(fy_real.data()),
            thrust::raw_pointer_cast(fz_real.data())
        )
    );

    // clearing the allocated memory like ---- cudaFree(device_vector);

}
