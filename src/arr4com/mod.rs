// pub mod arr4float;
mod avx2;
mod legacy;
use arrayvec::ArrayVec;


pub enum OpTarget {
    AVX2,
    // CUDA,
    LEGACY,
}


pub trait Arr4FloatCom<T, const DLEN: usize>{
    fn new()-> Self;
    fn add(&mut self, rhs: &Self);
    fn sub(&mut self, rhs: &Self);
    fn mul(&mut self, rhs: &Self);
    fn div(&mut self, rhs: &Self);
}

pub struct Arr4F32<const DLEN: usize>{
    pub op_target: OpTarget,
    pub imp_avx2 : Option<avx2::Avx2Arr4F32<DLEN>>,
    pub imp_leg : Option<legacy::LegacyArr4F32<DLEN>>
}

type F32Arr<const DLEN: usize> = Arr4F32<DLEN>;
pub fn new_arr4f32<const DLEN: usize>(op_target: OpTarget) -> F32Arr<DLEN>{
    match op_target {
        AVX2 => {
            let arr = avx2::Avx2Arr4F32::<DLEN>::new();
            Arr4F32{op_target: AVX2,  imp_avx2: Some(arr), imp_leg: None}
        },
        LEGACY => {
            let arr = legacy::LegacyArr4F32::<DLEN>::new();
            Arr4F32{op_target: LEGACY,  imp_avx2: None, imp_leg: Some(arr)}
        }
    }
}


impl<const DLEN: usize> Arr4F32<DLEN> for Arr4FloatCom<f32, DLEN:>{

}
