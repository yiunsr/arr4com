// pub mod arr4float;
mod legacy;
mod avx2;
mod cuda;
mod sleef;
pub mod arr_f32;

#[derive(Clone, Copy)]
pub enum OpTarget {
    LEGACY,
    AVX2,
    CUDA,
}

pub struct Arr4Com<T, const DLEN: usize>{
    op_target: OpTarget,
    #[allow(dead_code)]
    dlen: usize,
    legacy_com: Option<legacy::LegacyArr4Float<T, DLEN>>,
    avx2_com: Option<avx2::Avx2Arr4Float<T, DLEN>>,
    cuda_com: Option<cuda::CudaArr4Float<T, DLEN>>
}

pub trait Arr4ComAL<T, const DLEN: usize>{
    fn add(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn sub(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn mul(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn div(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);

    fn cos(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn sin(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn tan(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn acos(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn asin(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn atan(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn cosh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn sinh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn tanh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn acosh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn asinh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn atanh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);

    fn ln(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);  // https://en.wikipedia.org/wiki/Natural_logarithm
    fn ln_1p(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]); // ln(x+1)
    fn log10(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn log2(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);

    fn exp(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);   // e (https://en.wikipedia.org/wiki/Exponential_function)
    fn exp2(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);  // 2^lhs 
    fn exp_m1(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);  // e^lhs - 1

    fn sqrt(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn cbrt(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);

    fn powf(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn hypot(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);  // sqrt(lhs^2 + rhs^2)
    
    
}
