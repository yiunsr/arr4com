// pub mod arr4float;
mod avx2;
mod legacy;

#[derive(Clone, Copy)]
pub enum OpTarget {
    AVX2,
    // CUDA,
    LEGACY,
}

pub struct Arr4F32<const DLEN: usize>{
    pub op_target: OpTarget,
    pub data: [f32; DLEN],
}

pub trait Arr4ComF32<const DLEN: usize>{
    fn add(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>);
    fn sub(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>);
    fn mul(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>);
    fn div(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>);
}


impl<const DLEN: usize> Arr4F32<DLEN>{
    pub fn new(op_target: OpTarget) -> Self{
        match op_target {
            OpTarget::AVX2 => {
                Arr4F32{op_target: OpTarget::AVX2,  data: [0f32; DLEN]}
            },
            OpTarget::LEGACY => {
                Arr4F32{op_target: OpTarget::LEGACY,  data: [0f32; DLEN]}
            }
        }
    }
    pub fn from(op_target: OpTarget, array: [f32; DLEN]) -> Self{
        match op_target {
            OpTarget::AVX2 => {
                Arr4F32{op_target: OpTarget::AVX2,  data: array}
            },
            OpTarget::LEGACY => {
                Arr4F32{op_target: OpTarget::LEGACY,  data: array}
            }
        }
    }

    pub fn at(self, index:usize)->f32{
        self.data[index]
    }

    pub fn add(&mut self, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4F32::add(self, &lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4F32::add(self, &lhs, rhs);
            }
        }
    }

    pub fn sub(&mut self, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4F32::add(self, &lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4F32::add(self, &lhs, rhs);
            }
        }
    }

    pub fn mul(&mut self, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4F32::add(self, &lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4F32::add(self, &lhs, rhs);
            }
        }
    }

    pub fn div(&mut self, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        match self.op_target {
            OpTarget::AVX2 => {
                avx2::Avx2Arr4F32::add(self, &lhs, rhs);
            },
            OpTarget::LEGACY => {
                legacy::LegacyArr4F32::add(self, &lhs, rhs);
            }
        }
    }
}

