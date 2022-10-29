use core::{arch::x86_64::*};
use crate::arr4com::Arr4ComAL;
use crate::arr4com::sleef::simddp;
use crate::arr4com::avx2_type::Avx2Arr4Float;

macro_rules! InterLoopSleef1f64{
    ($ret:ident, $lhs:ident,$F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&$lhs[index * bs]);
                let result = simddp::$F(left);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };
}
macro_rules! InterLoopSleef2f64{
    ($ret:ident, $lhs:ident, $rhs:ident,$F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&$lhs[index * bs]);
                let right = _mm256_loadu_pd(&$rhs[index * bs]);
                let result = simddp::$F(left, right);
                _mm256_storeu_pd(&mut $ret[index * bs], result);
            }
        }
    };
}

macro_rules! InterLoop2f64{
    ($ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&$lhs[index * bs]);
                let right = _mm256_loadu_pd(&$rhs[index * bs]);
                let result = $F(left, right);
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

type F64Avx<const DLEN: usize> = Avx2Arr4Float<f64, DLEN>;

impl<const DLEN: usize> Arr4ComAL<f64, DLEN> for F64Avx<DLEN>{
    fn add(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        let dlen = DLEN;
        println!("dlen : {}", dlen);
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&lhs[index * bs]);
                let right = _mm256_loadu_pd(&rhs[index * bs]);
                let result = _mm256_add_pd(left, right);
                _mm256_storeu_pd(&mut ret[index * bs], result);
            }
        }
    }

    fn sub(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterLoop2f64!(ret, lhs, rhs, _mm256_sub_pd);
    }

    fn mul(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterLoop2f64!(ret, lhs, rhs, _mm256_mul_pd);
    }

    fn div(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterLoop2f64!(ret, lhs, rhs, _mm256_div_pd);
    }

    fn cos(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_pd(&lhs[index * bs]);
                let result = simddp::xcos_u1(left);
                _mm256_storeu_pd(&mut ret[index * bs], result);
            }
        }
    }
    fn sin(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xsin_u1);
    }
    fn tan(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xtan_u1);
    }
    fn asin(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xasin_u1);
    }
    fn acos(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xacos_u1);
    }
    fn atan(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xatan_u1);
    }
    fn sinh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xsinh);
    }
    fn cosh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xcosh);
    }
    fn tanh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xtanh);
    }
    fn asinh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xasinh);
    }
    fn acosh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xacosh);
    }
    fn atanh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xatanh);
    }
    fn ln(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xlog_u1);
    }
    fn ln_1p(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xlog1p);
    }
    fn log10(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xlog10);
    }
    fn log2(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xlog2);
    }

    fn exp(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xexp);
    }
    fn exp2(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xexp2);
    }
    fn exp_m1(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xexpm1);
    }

    fn sqrt(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xsqrt);
    }
    fn cbrt(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        InterLoopSleef1f64!(ret, lhs, xcbrt_u1);
    }
    fn powf(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterLoopSleef2f64!(ret, lhs, rhs, xpow);
    }
    fn hypot(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterLoopSleef2f64!(ret, lhs, rhs, xhypot_u05);
    }

}

