
extern "C" __global__ void a4c_addf64(const double* x, const double* y, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void a4c_subf64(const double* x, const double* y, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void a4c_mulf64(const double* x, const double* y, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] * y[i];
    }
}

extern "C" __global__ void a4c_divf64(const double* x, const double* y, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] / y[i];
    }
}

extern "C" __global__ void a4c_cosf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cosf(x[i]);
    }
}

extern "C" __global__ void a4c_sinf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sin(x[i]);
    }
}

extern "C" __global__ void a4c_tanf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = tan(x[i]);
    }
}

extern "C" __global__ void a4c_acosf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = acos(x[i]);
    }
}

extern "C" __global__ void a4c_asinf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = asin(x[i]);
    }
}

extern "C" __global__ void a4c_atanf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atan(x[i]);
    }
}

extern "C" __global__ void a4c_coshf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cosh(x[i]);
    }
}

extern "C" __global__ void a4c_sinhf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sinh(x[i]);
    }
}

extern "C" __global__ void a4c_tanhf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = tanh(x[i]);
    }
}

extern "C" __global__ void a4c_acoshf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = acosh(x[i]);
    }
}

extern "C" __global__ void a4c_asinhf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = asinh(x[i]);
    }
}

extern "C" __global__ void a4c_atanhf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atanh(x[i]);
    }
}

extern "C" __global__ void a4c_lnf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log(x[i]);
    }
}
extern "C" __global__ void a4c_ln_1pf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log1p(x[i]);
    }
}
extern "C" __global__ void a4c_log10f64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log10(x[i]);
    }
}
extern "C" __global__ void a4c_log2f64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log2(x[i]);
    }
}

extern "C" __global__ void a4c_expf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = exp(x[i]);
    }
}
extern "C" __global__ void a4c_exp2f64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = exp2(x[i]);
    }
}
extern "C" __global__ void a4c_exp_m1f64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = expm1(x[i]);
    }
}

extern "C" __global__ void a4c_sqrtf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sqrt(x[i]);
    }
}
extern "C" __global__ void a4c_cbrtf64(const double* x, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cbrt(x[i]);
    }
}

extern "C" __global__ void a4c_powff64(const double* x, const double* y, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = pow(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_hypotf64(const double* x, const double* y, double* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = hypot(x[i], y[i]);
    }
}