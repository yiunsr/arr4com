use core::{arch::x86_64::*};
use crate::arr4com::Arr4ComFloat;
use crate::arr4com::sleef::simddp;
use crate::arr4com::avx2_type::Avx2Arr4Float;

type Float = f64;

macro_rules! InterLoop{
    ($ret:ident, $opr1:ident,  $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_pd(&$opr1[index * bs]);
                let result = $F(opr1);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_pd(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_pd(&$opr2[index * bs]);
                let result = $F(opr1, opr2);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_pd(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_pd(&$opr2[index * bs]);
                let opr3 = _mm256_loadu_pd(&$opr3[index * bs]);
                let result = $F(opr1, opr2, opr3);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };
}

macro_rules! InterLoopSleef{
    ($ret:ident, $opr1:ident, $F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_pd(&$opr1[index * bs]);
                let result = simddp::$F(opr1);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_pd(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_pd(&$opr2[index * bs]);
                let result = simddp::$F(opr1, opr2);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_pd(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_pd(&$opr2[index * bs]);
                let opr3 = _mm256_loadu_pd(&$opr3[index * bs]);
                let result = simddp::$F(opr1, opr2, opr3);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };
}

impl<const DLEN: usize> Avx2Arr4Float<f64, DLEN>{
    pub fn newf64() -> Self{
        Avx2Arr4Float {
            nerver_use: 0f64,
        }
    }
}

fn trunc(a:__m256d)->__m256d{
    unsafe{
        //  _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
        const ROUND_MODE:i32 = 0x08|0x03;
        _mm256_round_pd::<ROUND_MODE>(a)
    }
} 

type F64Avx<const DLEN: usize> = Avx2Arr4Float<f64, DLEN>;

impl<const DLEN: usize> Arr4ComFloat<f64, DLEN> for F64Avx<DLEN>{
    fn add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        let dlen = DLEN;
        println!("dlen : {}", dlen);
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&opr1[index * bs]);
                let right = _mm256_loadu_pd(&opr2[index * bs]);
                let result = _mm256_add_pd(left, right);
                _mm256_storeu_pd(&mut ret[index * bs], result);
            }
        }
    }

    fn sub(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, _mm256_sub_pd);
    }

    fn mul(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, _mm256_mul_pd);
    }

    fn div(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, _mm256_div_pd);
    }

    fn mul_add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN], opr3: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, opr3, _mm256_fmadd_pd);
    }

    fn ceil(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoop!(ret, opr1, _mm256_ceil_pd);
    }
    fn floor(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoop!(ret, opr1, _mm256_floor_pd);
    }
    fn round(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xround);
    }
    fn trunc(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoop!(ret, opr1, trunc);
    }
    fn abs(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xfabs);
    }
    fn max(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xfmax);
    }
    fn min(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xfmin);
    }
    fn copysign(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xcopysign);
    }

    fn cos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&opr1[index * bs]);
                let result = simddp::xcos_u1(left);
                _mm256_storeu_pd(&mut ret[index * bs], result);
            }
        }
    }
    fn sin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xsin_u1);
    }
    fn tan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xtan_u1);
    }
    fn asin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xasin_u1);
    }
    fn acos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xacos_u1);
    }
    fn atan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xatan_u1);
    }
    fn atan2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xatan2_u1);
    }
    fn sinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xsinh);
    }
    fn cosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xcosh);
    }
    fn tanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xtanh);
    }
    fn asinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xasinh);
    }
    fn acosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xacosh);
    }
    fn atanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xatanh);
    }
    fn ln(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog_u1);
    }
    fn ln_1p(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog1p);
    }
    fn log10(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog10);
    }
    fn log2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog2);
    }

    fn exp(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xexp);
    }
    fn exp2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xexp2);
    }
    fn exp_m1(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xexpm1);
    }

    fn sqrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xsqrt);
    }
    fn cbrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xcbrt_u1);
    }
    fn powf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xpow);
    }
    fn hypot(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xhypot_u05);
    }

}

