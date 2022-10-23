use core::{arch::x86_64::*};
use crate::arr4com::Arr4ComAL;
use crate::arr4com::sleef::simdsp;


macro_rules! InterLoop1f32{
    ($ret:ident, $lhs:ident,$F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&$lhs[index * bs]);
                let result = simdsp::$F(left);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };
}

macro_rules! InterLoop2f32{
    ($ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&$lhs[index * bs]);
                let right = _mm256_loadu_ps(&$rhs[index * bs]);
                let result = $F(left, right);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };
}

pub struct Avx2Arr4Float<T, const DLEN: usize>{
    #[allow(dead_code)]
    nerver_use:T
}
impl<const DLEN: usize> Avx2Arr4Float<f32, DLEN>{
    pub fn newf32() -> Self{
        Avx2Arr4Float {
            nerver_use: 0f32,
        }
    }
}

type F32Avx<const DLEN: usize> = Avx2Arr4Float<f32, DLEN>;

impl<const DLEN: usize> Arr4ComAL<f32, DLEN> for F32Avx<DLEN>{
    fn add(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        let dlen = DLEN;
        println!("dlen : {}", dlen);
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&lhs[index * bs]);
                let right = _mm256_loadu_ps(&rhs[index * bs]);
                let result = _mm256_add_ps(left, right);
                _mm256_storeu_ps(&mut ret[index * bs], result);
            }
        }
    }

    // fn add(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
    //     InterLoop2f32!(ret, lhs, rhs, _mm256_add_ps);
    // }

    fn sub(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterLoop2f32!(ret, lhs, rhs, _mm256_sub_ps);
    }

    fn mul(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterLoop2f32!(ret, lhs, rhs, _mm256_mul_ps);
    }

    fn div(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterLoop2f32!(ret, lhs, rhs, _mm256_div_ps);
    }

    fn cos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&lhs[index * bs]);
                let result = simdsp::xcosf_u1(left);
                _mm256_storeu_ps(&mut ret[index * bs], result);
            }
        }
    }
    fn sin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xsinf_u1);
    }
    fn tan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xtanf_u1);
    }
    fn asin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xasinf_u1);
    }
    fn acos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xacosf_u1);
    }
    fn atan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xatanf_u1);
    }
    fn sinh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xsinhf);
    }
    fn cosh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xcoshf);
    }
    fn tanh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xtanhf);
    }
    fn asinh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xasinhf);
    }
    fn acosh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xacoshf);
    }
    fn atanh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterLoop1f32!(ret, lhs, xatanhf);
    }

}

