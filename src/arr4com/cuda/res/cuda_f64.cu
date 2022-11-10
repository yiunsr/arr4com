
extern "C" __global__ void a4c_addf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void a4c_subf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void a4c_mulf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] * y[i];
    }
}

extern "C" __global__ void a4c_divf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] / y[i];
    }
}

extern "C" __global__ void a4c_mul_addf64(double* out, int count, const double* x, const double* y, const double* z) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fma(x[i], y[i], z[i]);
    }
}

extern "C" __global__ void a4c_gtff64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] > y[i];
    }
}
extern "C" __global__ void a4c_gteff64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] >= y[i];
    }
}
extern "C" __global__ void a4c_ltff64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] < y[i];
    }
}
extern "C" __global__ void a4c_lteff64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] <= y[i];
    }
}

extern "C" __global__ void a4c_ceilf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = ceil(x[i]);
    }
}
extern "C" __global__ void a4c_floorf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = floor(x[i]);
    }
}
extern "C" __global__ void a4c_roundf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = round(x[i]);
    }
}
extern "C" __global__ void a4c_truncf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = trunc(x[i]);
    }
}
extern "C" __global__ void a4c_absf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fabs(x[i]);
    }
}
extern "C" __global__ void a4c_maxf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fmax(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_minf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fmin(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_copysignf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = copysign(x[i], y[i]);
    }
}


extern "C" __global__ void a4c_cosf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cosf(x[i]);
    }
}
extern "C" __global__ void a4c_sinf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sin(x[i]);
    }
}

extern "C" __global__ void a4c_tanf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = tan(x[i]);
    }
}

extern "C" __global__ void a4c_acosf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = acos(x[i]);
    }
}

extern "C" __global__ void a4c_asinf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = asin(x[i]);
    }
}

extern "C" __global__ void a4c_atanf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atan(x[i]);
    }
}

extern "C" __global__ void a4c_atan2f64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atan2(x[i], y[i]);
    }
}

extern "C" __global__ void a4c_coshf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cosh(x[i]);
    }
}

extern "C" __global__ void a4c_sinhf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sinh(x[i]);
    }
}

extern "C" __global__ void a4c_tanhf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = tanh(x[i]);
    }
}

extern "C" __global__ void a4c_acoshf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = acosh(x[i]);
    }
}

extern "C" __global__ void a4c_asinhf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = asinh(x[i]);
    }
}

extern "C" __global__ void a4c_atanhf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atanh(x[i]);
    }
}

extern "C" __global__ void a4c_lnf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log(x[i]);
    }
}
extern "C" __global__ void a4c_ln_1pf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log1p(x[i]);
    }
}
extern "C" __global__ void a4c_log10f64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log10(x[i]);
    }
}
extern "C" __global__ void a4c_log2f64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log2(x[i]);
    }
}

extern "C" __global__ void a4c_expf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = exp(x[i]);
    }
}
extern "C" __global__ void a4c_exp2f64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = exp2(x[i]);
    }
}
extern "C" __global__ void a4c_exp_m1f64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = expm1(x[i]);
    }
}

extern "C" __global__ void a4c_sqrtf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sqrt(x[i]);
    }
}
extern "C" __global__ void a4c_cbrtf64(double* out, int count, const double* x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cbrt(x[i]);
    }
}

extern "C" __global__ void a4c_powff64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = pow(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_hypotf64(double* out, int count, const double* x, const double* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = hypot(x[i], y[i]);
    }
}