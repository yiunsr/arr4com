use core::{arch::x86_64::*};
use arrayvec::ArrayVec;

use crate::arr4com::OpTarget;
use crate::arr4com::Arr4FloatCom;

macro_rules! InterLoop{
    ($self:ident, $rhs:ident, $F:ident) => {
        let aligend_n = DLEN as usize % 8;
        for index in 0..aligend_n{
            unsafe{
                let left = _mm256_loadu_ps(&$self.data[index * aligend_n]);
                let right = _mm256_loadu_ps(&$rhs.data[index * aligend_n]);
                let result = $F(left, right);
                _mm256_storeu_ps(&mut $self.data[index * aligend_n], result);
            }
        }
    };
}

pub struct Avx2Arr4F32<const DLEN: usize>{
    pub op_target: OpTarget,
    pub data: ArrayVec<f32, DLEN>,
}

type F32Avx<const DLEN: usize> = Avx2Arr4F32<DLEN>;

impl<const DLEN: usize> Arr4FloatCom<f32, DLEN> for F32Avx<DLEN>{
    fn new() -> Self{
        Avx2Arr4F32{
            op_target: OpTarget::LEGACY,
            data: ArrayVec::<f32, DLEN>::new()
        }
    }

    fn add(&mut self, rhs: &F32Avx<DLEN>){
        InterLoop!(self, rhs, _mm256_add_ps);
    }

    fn sub(&mut self, rhs: &F32Avx<DLEN>){
        InterLoop!(self, rhs, _mm256_sub_ps);
    }

    fn mul(&mut self, rhs: &F32Avx<DLEN>){
        InterLoop!(self, rhs, _mm256_mul_ps);
    }

    fn div(&mut self, rhs: &F32Avx<DLEN>){
        InterLoop!(self, rhs, _mm256_div_ps);
    }
}
