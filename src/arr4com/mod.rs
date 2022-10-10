// pub mod arr4float;
mod avx2;
mod legacy;

#[derive(Clone, Copy)]
pub enum OpTarget {
    LEGACY,
    AVX2,
    // CUDA,
}

pub struct Arr4Com<const DLEN: usize>{
    pub op_target: OpTarget,
    pub dlen: usize,
}

pub trait Arr4ComAL<T, const DLEN: usize>{
    fn add(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn sub(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn mul(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn div(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
}


impl<const DLEN: usize> Arr4Com<DLEN>{
    pub fn new(op_target: OpTarget) -> Self{
        match op_target {
            OpTarget::AVX2 => {
                Arr4Com{op_target: OpTarget::AVX2, dlen: DLEN}
            },
            OpTarget::LEGACY => {
                Arr4Com{op_target: OpTarget::LEGACY, dlen: DLEN}
            }
        }
    }
}

impl<const DLEN: usize> Arr4Com<DLEN>{
    pub fn add(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::add(ret, lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::add(ret, lhs, rhs);
            }
        }
    }

    pub fn sub(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::sub(ret, lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::sub(ret, lhs, rhs);
            }
        }
    }

    pub fn mul(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::mul(ret, lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::mul(ret, lhs, rhs);
            }
        }
    }

    pub fn div(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::div(ret, lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::div(ret, lhs, rhs);
            }
        }
    }
}

