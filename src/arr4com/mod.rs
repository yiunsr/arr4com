// pub mod arr4float;
mod legacy;
mod avx2;
mod cuda;
mod opencl;
mod sleef;
pub mod arr_f32;
pub mod arr_f64;

use legacy::legacy_type;
use avx2::avx2_type;
use cuda::cuda_type;
use opencl::opencl_type;

#[derive(Clone, Copy)]
pub enum OpTarget {
    LEGACY,
    AVX2,
    CUDA,
    OPENCL,
}

pub type Bool = i8;

const BLOCK_ALIGN:usize = 64;

pub struct Arr4Com<T>{
    op_target: OpTarget,
    #[allow(dead_code)]
    dlen: usize,
    legacy_com: Option<legacy_type::LegacyArr4Float<T>>,
    avx2_com: Option<avx2_type::Avx2Arr4Float<T>>,
    cuda_com: Option<cuda_type::CudaArr4Float<T>>,
    opencl_com: Option<opencl_type::OpenclArr4Float<T>>,
}

pub trait Arr4ComInt<T>{
    fn add(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn sub(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn mul(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn div(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    // ret = opr1 * opr2 + opr3
    fn mul_add(&self, ret: &mut [T], opr1: &[T], opr2: &[T], opr3: &[T]);

    fn gti(&self, ret: &mut [T], opr1: &[T]);  // opr2 > opr1
    fn gtei(&self, ret: &mut [T], opr1: &[T]);  // opr2 >= opr1
    fn lti(&self, ret: &mut [T], opr1: &[T]);  // opr2 < opr1
    fn ltei(&self, ret: &mut [T], opr1: &[T]);  // opr2 <= opr1

    fn abs(&self, ret: &mut [T], opr1: &[T]);
    fn max(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn min(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn copysign(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);

    fn pow(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn hypot(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);  // sqrt(opr1^2 + opr2^2)
}


pub trait Arr4ComFloat<T>{
    fn add(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn sub(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn mul(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn div(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    // ret = opr1 * opr2 + opr3
    fn mul_add(&self, ret: &mut [T], opr1: &[T], opr2: &[T], opr3: &[T]);

    fn gtf(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);  // opr2 > opr1   ==>> result float
    fn gtef(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);  // opr2 >= opr1   ==>> result float 
    fn ltf(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);  // opr2 < opr1  ==>> result float
    fn ltef(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);  // opr2 <= opr1  ==>> result float

    fn ceil(&self, ret: &mut [T], opr1: &[T]);
    fn floor(&self, ret: &mut [T], opr1: &[T]);
    fn round(&self, ret: &mut [T], opr1: &[T]);
    fn trunc(&self, ret: &mut [T], opr1: &[T]);

    fn abs(&self, ret: &mut [T], opr1: &[T]);
    fn max(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn min(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn copysign(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);

    fn cos(&self, ret: &mut [T], opr1: &[T]);
    fn sin(&self, ret: &mut [T], opr1: &[T]);
    fn tan(&self, ret: &mut [T], opr1: &[T]);
    fn acos(&self, ret: &mut [T], opr1: &[T]);
    fn asin(&self, ret: &mut [T], opr1: &[T]);
    fn atan(&self, ret: &mut [T], opr1: &[T]);
    fn atan2(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn cosh(&self, ret: &mut [T], opr1: &[T]);
    fn sinh(&self, ret: &mut [T], opr1: &[T]);
    fn tanh(&self, ret: &mut [T], opr1: &[T]);
    fn acosh(&self, ret: &mut [T], opr1: &[T]);
    fn asinh(&self, ret: &mut [T], opr1: &[T]);
    fn atanh(&self, ret: &mut [T], opr1: &[T]);

    fn ln(&self, ret: &mut [T], opr1: &[T]);  // https://en.wikipedia.org/wiki/Natural_logarithm
    fn ln_1p(&self, ret: &mut [T], opr1: &[T]); // ln(x+1)
    fn log10(&self, ret: &mut [T], opr1: &[T]);
    fn log2(&self, ret: &mut [T], opr1: &[T]);

    fn exp(&self, ret: &mut [T], opr1: &[T]);   // e (https://en.wikipedia.org/wiki/Exponential_function)
    fn exp2(&self, ret: &mut [T], opr1: &[T]);  // 2^opr1 
    fn exp_m1(&self, ret: &mut [T], opr1: &[T]);  // e^opr1 - 1

    fn sqrt(&self, ret: &mut [T], opr1: &[T]);
    fn cbrt(&self, ret: &mut [T], opr1: &[T]);

    fn powf(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);
    fn hypot(&self, ret: &mut [T], opr1: &[T], opr2: &[T]);  // sqrt(opr1^2 + opr2^2)
}

