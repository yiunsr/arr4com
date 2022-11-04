
extern "C" __global__ void a4c_addf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void a4c_subf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] - y[i];
    }
}

extern "C" __global__ void a4c_mulf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] * y[i];
    }
}

extern "C" __global__ void a4c_divf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] / y[i];
    }
}

extern "C" __global__ void a4c_mul_addf32(const float* x, const float* y, const float* z, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fmaf(x[i], y[i], z[i]);
    }
}

extern "C" __global__ void a4c_gtff32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] > y[i];
    }
}
extern "C" __global__ void a4c_gteff32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] >= y[i];
    }
}
extern "C" __global__ void a4c_ltff32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] < y[i];
    }
}
extern "C" __global__ void a4c_lteff32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = x[i] <= y[i];
    }
}

extern "C" __global__ void a4c_ceilf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = ceilf(x[i]);
    }
}
extern "C" __global__ void a4c_floorf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = floorf(x[i]);
    }
}
extern "C" __global__ void a4c_roundf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = roundf (x[i]);
    }
}
extern "C" __global__ void a4c_truncf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = truncf(x[i]);
    }
}
extern "C" __global__ void a4c_absf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fabsf(x[i]);
    }
}
extern "C" __global__ void a4c_maxf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fmaxf(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_minf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = fminf(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_copysignf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = copysignf(x[i], y[i]);
    }
}

extern "C" __global__ void a4c_cosf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cosf(x[i]);
    }
}

extern "C" __global__ void a4c_sinf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sinf(x[i]);
    }
}

extern "C" __global__ void a4c_tanf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = tanf(x[i]);
    }
}

extern "C" __global__ void a4c_acosf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = acosf(x[i]);
    }
}

extern "C" __global__ void a4c_asinf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = asinf(x[i]);
    }
}

extern "C" __global__ void a4c_atanf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atanf(x[i]);
    }
}

extern "C" __global__ void a4c_atan2f32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atan2f(x[i], y[i]);
    }
}

extern "C" __global__ void a4c_coshf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = coshf(x[i]);
    }
}

extern "C" __global__ void a4c_sinhf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sinhf(x[i]);
    }
}

extern "C" __global__ void a4c_tanhf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = tanhf(x[i]);
    }
}

extern "C" __global__ void a4c_acoshf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = acoshf(x[i]);
    }
}

extern "C" __global__ void a4c_asinhf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = asinhf(x[i]);
    }
}

extern "C" __global__ void a4c_atanhf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = atanhf(x[i]);
    }
}

extern "C" __global__ void a4c_lnf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = logf(x[i]);
    }
}
extern "C" __global__ void a4c_ln_1pf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log1pf(x[i]);
    }
}
extern "C" __global__ void a4c_log10f32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log10f(x[i]);
    }
}
extern "C" __global__ void a4c_log2f32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = log2f(x[i]);
    }
}

extern "C" __global__ void a4c_expf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = expf (x[i]);
    }
}
extern "C" __global__ void a4c_exp2f32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = exp2f(x[i]);
    }
}
extern "C" __global__ void a4c_exp_m1f32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = expm1f(x[i]);
    }
}

extern "C" __global__ void a4c_sqrtf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = sqrtf (x[i]);
    }
}
extern "C" __global__ void a4c_cbrtf32(const float* x, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = cbrtf (x[i]);
    }
}

extern "C" __global__ void a4c_powff32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = powf(x[i], y[i]);
    }
}
extern "C" __global__ void a4c_hypotf32(const float* x, const float* y, float* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        out[i] = hypotf(x[i], y[i]);
    }
}