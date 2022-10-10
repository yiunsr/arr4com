use core::{arch::x86_64::*};
use crate::arr4com::OpTarget;
use crate::arr4com::Arr4F32;
use crate::arr4com::Arr4ComF32;

macro_rules! InterLoop2{
    ($ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        let aligend_n = DLEN as usize % 8;
        for index in 0..aligend_n{
            unsafe{
                let left = _mm256_loadu_ps(&$lhs.data[index * aligend_n]);
                let right = _mm256_loadu_ps(&$rhs.data[index * aligend_n]);
                let result = $F(left, right);
                _mm256_storeu_ps(&mut $ret.data[index * aligend_n], result);
            }
        }
    };
}

pub struct Avx2Arr4F32<const DLEN: usize>{
}

type F32Avx<const DLEN: usize> = Avx2Arr4F32<DLEN>;

impl<const DLEN: usize> Arr4ComF32<DLEN> for F32Avx<DLEN>{
    fn add(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        let dlen = DLEN;
        println!("dlen : {}", dlen);
        let block = DLEN as usize / 8;
        for index in 0..block{
            unsafe{
                let left = _mm256_loadu_ps(&lhs.data[index * block]);
                let right = _mm256_loadu_ps(&rhs.data[index * block]);
                let result = _mm256_add_ps(left, right);
                _mm256_storeu_ps(&mut ret.data[index * block], result);
            }
        }
    }

    // fn add(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
    //     InterLoop2!(ret, lhs, rhs, _mm256_add_ps);
    // }

    fn sub(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        InterLoop2!(ret, lhs, rhs, _mm256_sub_ps);
    }

    fn mul(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        InterLoop2!(ret, lhs, rhs, _mm256_mul_ps);
    }

    fn div(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        InterLoop2!(ret, lhs, rhs, _mm256_div_ps);
    }
}
