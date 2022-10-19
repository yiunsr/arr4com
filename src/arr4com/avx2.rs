use core::{arch::x86_64::*};
use crate::arr4com::Arr4ComAL;
use crate::arr4com::sleef::simdsp;

macro_rules! InterLoop2{
    ($ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        let dlen = DLEN;
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
    nerver_use:T
}
type F32Avx<const DLEN: usize> = Avx2Arr4Float<f32, DLEN>;

impl<const DLEN: usize> Arr4ComAL<f32, DLEN> for F32Avx<DLEN>{
    fn add(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
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
    //     InterLoop2!(ret, lhs, rhs, _mm256_add_ps);
    // }

    fn sub(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterLoop2!(ret, lhs, rhs, _mm256_sub_ps);
    }

    fn mul(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterLoop2!(ret, lhs, rhs, _mm256_mul_ps);
    }

    fn div(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterLoop2!(ret, lhs, rhs, _mm256_div_ps);
    }

    fn sin(ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        let bs = 8;
        let block = DLEN as usize / 8;

        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&lhs[index * bs]);
                let result = simdsp::xsinf_u1(left);
                _mm256_storeu_ps(&mut ret[index * bs], result);
            }
        }
    }

}

