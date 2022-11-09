
__kernel void a4c_addf64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] + y[idx];
}

__kernel void a4c_subf64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] - y[idx];
}

__kernel void a4c_mulf64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] * y[idx];
}

__kernel void a4c_divf64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] / y[idx];
}

__kernel void a4c_mul_addf64(__global double* out, __global double* x, __global double* y, __global double* z){
    int idx = get_global_id(0);
    out[idx] = fma(x[idx], y[idx], z[idx]);
}

__kernel void a4c_gtff64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] > y[idx];
}

__kernel void a4c_gteff64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] >= y[idx];
}

__kernel void a4c_ltff64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] < y[idx];
}

__kernel void a4c_lteff64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = x[idx] <= y[idx];
}

__kernel void a4c_ceilf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = ceil(x[idx]);
}

__kernel void a4c_floorf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = floor(x[idx]);
}

__kernel void a4c_roundf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = round(x[idx]);
}

__kernel void a4c_truncf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = trunc(x[idx]);
}

__kernel void a4c_absf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = fabs(x[idx]);
}

__kernel void a4c_maxf64(__global double* out, __global double* x,  __global double* y){
    int idx = get_global_id(0);
    out[idx] = fmax(x[idx], y[idx]);
}

__kernel void a4c_minf64(__global double* out, __global double* x,  __global double* y){
    int idx = get_global_id(0);
    out[idx] = fmin(x[idx], y[idx]);
}

__kernel void a4c_copysignf64(__global double* out, __global double* x,  __global double* y){
    int idx = get_global_id(0);
    out[idx] = copysign(x[idx], y[idx]);
}

__kernel void a4c_cosf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = cos(x[idx]);
}

__kernel void a4c_sinf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = sin(x[idx]);
}

__kernel void a4c_tanf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = tan(x[idx]);
}

__kernel void a4c_acosf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = acos(x[idx]);
}

__kernel void a4c_asinf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = asin(x[idx]);
}

__kernel void a4c_atanf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = atan(x[idx]);
}

__kernel void a4c_atan2f64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = atan2(x[idx], y[idx]);
}

__kernel void a4c_coshf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = cosh(x[idx]);
}

__kernel void a4c_sinhf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = sinh(x[idx]);
}

__kernel void a4c_tanhf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = tanh(x[idx]);
}

__kernel void a4c_acoshf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = acosh(x[idx]);
}

__kernel void a4c_asinhf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = asinh(x[idx]);
}

__kernel void a4c_atanhf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = atanh(x[idx]);
}

__kernel void a4c_lnf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = log(x[idx]);
}

__kernel void a4c_ln_1pf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = log1p(x[idx]);
}

__kernel void a4c_log10f64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = log10(x[idx]);
}

__kernel void a4c_log2f64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = log2(x[idx]);
}

__kernel void a4c_expf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = exp(x[idx]);
}

__kernel void a4c_exp2f64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = exp2(x[idx]);
}

__kernel void a4c_exp_m1f64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = expm1(x[idx]);
}

__kernel void a4c_sqrtf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = sqrt(x[idx]);
}

__kernel void a4c_cbrtf64(__global double* out, __global double* x){
    int idx = get_global_id(0);
    out[idx] = cbrt(x[idx]);
}

__kernel void a4c_powff64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = pow(x[idx], y[idx]);
}

__kernel void a4c_hypotf64(__global double* out, __global double* x, __global double* y){
    int idx = get_global_id(0);
    out[idx] = hypot(x[idx],  y[idx]);
}