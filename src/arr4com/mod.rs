// pub mod arr4float;
mod legacy_type;
mod legacy_f32;
mod legacy_f64;
mod avx2_type;
mod avx2_f32;
mod avx2_f64;
mod cuda_type;
mod cuda_f32;
mod cuda_f64;

mod sleef;
pub mod arr_f32;
pub mod arr_f64;



#[derive(Clone, Copy)]
pub enum OpTarget {
    LEGACY,
    AVX2,
    CUDA,
}

pub type Bool = i8;

pub struct Arr4Com<T, const DLEN: usize>{
    op_target: OpTarget,
    #[allow(dead_code)]
    dlen: usize,
    legacy_com: Option<legacy_type::LegacyArr4Float<T, DLEN>>,
    avx2_com: Option<avx2_type::Avx2Arr4Float<T, DLEN>>,
    cuda_com: Option<cuda_type::CudaArr4Float<T, DLEN>>
}

pub trait Arr4ComAL<T, const DLEN: usize>{
    fn add(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);
    fn sub(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);
    fn mul(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);
    fn div(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);
    // ret = opr1 * opr2 + opr3
    fn mul_add(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN], opr3: [T;DLEN]);

    // fn gt(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // opr2 > opr1
    // fn gte(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // opr2 >= opr1
    // fn lt(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // opr2 < opr1
    // fn lte(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // opr2 <= opr1

    fn ceil(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn floor(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn round(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn trunc(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);

    // fn abs(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    // fn max(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    // fn min(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    // fn copysign(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);

    fn cos(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn sin(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn tan(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn acos(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn asin(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn atan(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn atan2(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);
    fn cosh(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn sinh(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn tanh(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn acosh(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn asinh(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn atanh(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);

    fn ln(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // https://en.wikipedia.org/wiki/Natural_logarithm
    fn ln_1p(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]); // ln(x+1)
    fn log10(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn log2(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);

    fn exp(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);   // e (https://en.wikipedia.org/wiki/Exponential_function)
    fn exp2(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // 2^opr1 
    fn exp_m1(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);  // e^opr1 - 1

    fn sqrt(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);
    fn cbrt(&self, ret: &mut [T;DLEN], opr1: [T;DLEN]);

    fn powf(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);
    fn hypot(&self, ret: &mut [T;DLEN], opr1: [T;DLEN], opr2: [T;DLEN]);  // sqrt(opr1^2 + opr2^2)
    
    
}
