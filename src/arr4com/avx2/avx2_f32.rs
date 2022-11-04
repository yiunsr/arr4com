use core::{arch::x86_64::*};
use crate::arr4com::Arr4ComFloat;
use crate::arr4com::sleef::simdsp;
use crate::arr4com::avx2_type::Avx2Arr4Float;

type Float = f32;

macro_rules! InterLoop{
    ($ret:ident, $opr1:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_ps(&$opr1[index * bs]);
                let result = $F(opr1);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_ps(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_ps(&$opr2[index * bs]);
                let result = $F(opr1, opr2);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        //let dlen = DLEN;
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_ps(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_ps(&$opr2[index * bs]);
                let opr3 = _mm256_loadu_ps(&$opr3[index * bs]);
                let result = $F(opr1, opr2, opr3);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };
}

macro_rules! InterLoopSleef{
    
    ($ret:ident, $opr1:ident,$F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_ps(&$opr1[index * bs]);
                let result = simdsp::$F(opr1);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident,$F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let opr1 = _mm256_loadu_ps(&$opr1[index * bs]);
                let opr2 = _mm256_loadu_ps(&$opr2[index * bs]);
                let result = simdsp::$F(opr1, opr2);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };

    ($ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&$opr1[index * bs]);
                let result = simdsp::$F(left);
                _mm256_storeu_ps(&mut $ret[index * bs], result);
            }
        }
    };
}


impl<const DLEN: usize> Avx2Arr4Float<f32, DLEN>{
    pub fn newf32() -> Self{
        Avx2Arr4Float {
            nerver_use: 0f32,
        }
    }
}


fn trunc(a:__m256)->__m256{
    unsafe{
        //  _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
        const ROUND_MODE:i32 = 0x08|0x03;
        _mm256_round_ps::<ROUND_MODE>(a)
    }
}

fn gtf(a:__m256, b:__m256)->__m256{
    unsafe{
        //  _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
        const CMP_MODE:i32 = _CMP_LT_OS;
        // a > b ==>>  b < a
        let ret_mask = _mm256_cmp_ps::<CMP_MODE>(b, a);
        let ret_mask = _mm256_castps_si256(ret_mask);
        let f32_one = _mm256_set1_epi32(0x3F800000);
        let ret_si256 = _mm256_and_si256(ret_mask, f32_one);
        _mm256_castsi256_ps(ret_si256)
    }
}
fn gtef(a:__m256, b:__m256)->__m256{
    unsafe{
        //  _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
        const CMP_MODE:i32 = _CMP_LE_OS;
        // a >= b ==>>  b <= a
        let ret_mask = _mm256_cmp_ps::<CMP_MODE>(b, a);
        let ret_mask = _mm256_castps_si256(ret_mask);
        let f32_one = _mm256_set1_epi32(0x3F800000);
        let ret_si256 = _mm256_and_si256(ret_mask, f32_one);
        _mm256_castsi256_ps(ret_si256)
    }
}
fn ltf(a:__m256, b:__m256)->__m256{
    unsafe{
        //  _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
        const CMP_MODE:i32 = _CMP_LT_OS;
        // a < b 
        let ret_mask = _mm256_cmp_ps::<CMP_MODE>(a, b);
        let ret_mask = _mm256_castps_si256(ret_mask);
        let f32_one = _mm256_set1_epi32(0x3F800000);
        let ret_si256 = _mm256_and_si256(ret_mask, f32_one);
        _mm256_castsi256_ps(ret_si256)
    }
}
fn ltef(a:__m256, b:__m256)->__m256{
    unsafe{
        //  _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
        const CMP_MODE:i32 = _CMP_LE_OS;
        // a <= b 
        let ret_mask = _mm256_cmp_ps::<CMP_MODE>(a, b);
        let ret_mask = _mm256_castps_si256(ret_mask);
        let f32_one = _mm256_set1_epi32(0x3F800000);
        let ret_si256 = _mm256_and_si256(ret_mask, f32_one);
        _mm256_castsi256_ps(ret_si256)
    }
}


type F32Avx<const DLEN: usize> = Avx2Arr4Float<f32, DLEN>;

impl<const DLEN: usize> Arr4ComFloat<f32, DLEN> for F32Avx<DLEN>{
    fn add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        let dlen = DLEN;
        println!("dlen : {}", dlen);
        let bs = 8;
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&opr1[index * bs]);
                let right = _mm256_loadu_ps(&opr2[index * bs]);
                let result = _mm256_add_ps(left, right);
                _mm256_storeu_ps(&mut ret[index * bs], result);
            }
        }
    }

    fn sub(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, _mm256_sub_ps);
    }

    fn mul(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, _mm256_mul_ps);
    }

    fn div(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, _mm256_div_ps);
    }

    fn mul_add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN], opr3: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, opr3, _mm256_fmadd_ps);
    }

    fn gtf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, gtf);
    }
    fn gtef(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, gtef);
    }
    fn ltf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, ltf);
    }
    fn ltef(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoop!(ret, opr1, opr2, ltef);
    }
    
    fn ceil(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoop!(ret, opr1, _mm256_ceil_ps);
    }
    fn floor(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoop!(ret, opr1, _mm256_floor_ps);
    }
    fn round(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xroundf);
    }
    fn trunc(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoop!(ret, opr1, trunc);
    }
    fn abs(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xfabsf);
    }
    fn max(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xfmaxf);
    }
    fn min(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xfminf);
    }
    fn copysign(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xcopysignf);
    }

    fn cos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&opr1[index * bs]);
                let result = simdsp::xcosf_u1(left);
                _mm256_storeu_ps(&mut ret[index * bs], result);
            }
        }
    }
    fn sin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xsinf_u1);
    }
    fn tan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xtanf_u1);
    }
    fn asin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xasinf_u1);
    }
    fn acos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xacosf_u1);
    }
    fn atan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xatanf_u1);
    }
    fn atan2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xatan2f_u1);
    }

    fn sinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xsinhf);
    }
    fn cosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xcoshf);
    }
    fn tanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xtanhf);
    }
    fn asinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xasinhf);
    }
    fn acosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xacoshf);
    }
    fn atanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xatanhf);
    }
    fn ln(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlogf_u1);
    }
    fn ln_1p(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog1pf);
    }
    fn log10(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog10f);
    }
    fn log2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xlog2f);
    }

    fn exp(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xexpf);
    }
    fn exp2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xexp2f);
    }
    fn exp_m1(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xexpm1f);
    }

    fn sqrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xsqrtf);
    }
    fn cbrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, xcbrtf_u1);
    }
    fn powf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xpowf);
    }
    fn hypot(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterLoopSleef!(ret, opr1, opr2, xhypotf_u05);
    }

}

