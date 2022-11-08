
__kernel void a4c_addf32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] + y[idx];
}

__kernel void a4c_subf32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] - y[idx];
}

__kernel void a4c_mulf32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] * y[idx];
}

__kernel void a4c_divf32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] / y[idx];
}

__kernel void a4c_mul_addf32(__global float* out, __global float* x, __global float* y, __global float* z){
    int idx = get_global_id(0);
    out[idx] = fma(x[idx], y[idx], z[idx]);
}

__kernel void a4c_gtff32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] > y[idx];
}

__kernel void a4c_gteff32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] >= y[idx];
}

__kernel void a4c_ltff32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] < y[idx];
}

__kernel void a4c_lteff32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] <= y[idx];
}

__kernel void a4c_ceilf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = ceil(x[idx]);
}

__kernel void a4c_floorf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = floor(x[idx]);
}

__kernel void a4c_roundf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = round(x[idx]);
}

__kernel void a4c_truncf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = trunc(x[idx]);
}

__kernel void a4c_absf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = fabs(x[idx]);
}

__kernel void a4c_maxf32(__global float* out, __global float* x,  __global float* y){
    int idx = get_global_id(0);
    out[idx] = fmax(x[idx], y[idx]);
}

__kernel void a4c_minf32(__global float* out, __global float* x,  __global float* y){
    int idx = get_global_id(0);
    out[idx] = fmin(x[idx], y[idx]);
}

__kernel void a4c_copysignf32(__global float* out, __global float* x,  __global float* y){
    int idx = get_global_id(0);
    out[idx] = copysign(x[idx], y[idx]);
}

__kernel void a4c_cosf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = cos(x[idx]);
}

__kernel void a4c_sinf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = sin(x[idx]);
}

__kernel void a4c_tanf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = tan(x[idx]);
}

__kernel void a4c_acosf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = acos(x[idx]);
}

__kernel void a4c_asinf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = asin(x[idx]);
}

__kernel void a4c_atanf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = atan(x[idx]);
}

__kernel void a4c_atan2f32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = atan2(x[idx], y[idx]);
}

__kernel void a4c_coshf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = cosh(x[idx]);
}

__kernel void a4c_sinhf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = sinh(x[idx]);
}

__kernel void a4c_tanhf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = tanh(x[idx]);
}

__kernel void a4c_acoshf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = acosh(x[idx]);
}

__kernel void a4c_asinhf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = asinh(x[idx]);
}

__kernel void a4c_atanhf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = atanh(x[idx]);
}

__kernel void a4c_lnf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = log(x[idx]);
}

__kernel void a4c_ln_1pf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = log1p(x[idx]);
}

__kernel void a4c_log10f32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = log10(x[idx]);
}

__kernel void a4c_log2f32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = log2(x[idx]);
}

__kernel void a4c_expf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = exp(x[idx]);
}

__kernel void a4c_exp2f32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = exp2(x[idx]);
}

__kernel void a4c_exp_m1f32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = expm1(x[idx]);
}

__kernel void a4c_sqrtf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = sqrt(x[idx]);
}

__kernel void a4c_cbrtf32(__global float* out, __global float* x){
    int idx = get_global_id(0);
    out[idx] = cbrt(x[idx]);
}

__kernel void a4c_powff32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = pow(x[idx], y[idx]);
}

__kernel void a4c_hypotf32(__global float* out, __global float* x, __global float* y){
    int idx = get_global_id(0);
    out[idx] = hypot(x[idx],  y[idx]);
}