
__kernel void a4c_addf32(__global float* out, __global float* x, __global float* y){
            int idx = get_global_id(0);
            out[idx] = x[idx] + y[idx];
}

