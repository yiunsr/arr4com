// pub mod arr4float;
mod legacy;
mod avx2;
mod cuda;
mod sleef;

#[derive(Clone, Copy)]
pub enum OpTarget {
    LEGACY,
    AVX2,
    CUDA,
}

pub struct Arr4Com<T, const DLEN: usize>{
    op_target: OpTarget,
    dlen: usize,
    cuda_com: Option<cuda::CudaArr4Float<T, DLEN>>
}

pub trait Arr4ComAL<T, const DLEN: usize>{
    fn add(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn sub(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn mul(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn div(ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);

    fn sin(ret: &mut [T;DLEN], lhs: [T;DLEN]);
}


impl<const DLEN: usize> Arr4Com<f32, DLEN>{
    pub fn new(op_target: OpTarget) -> Self{
        match op_target {
            OpTarget::LEGACY => {
                Arr4Com{op_target: OpTarget::LEGACY, dlen: DLEN, cuda_com: None}
            }
            OpTarget::AVX2 => {
                Arr4Com{op_target: OpTarget::AVX2, dlen: DLEN, cuda_com: None}
            },
            OpTarget::CUDA => {
                let cuda_com = Some(cuda::CudaArr4Float::newf32());
                Arr4Com{op_target: OpTarget::CUDA, dlen: DLEN, cuda_com: cuda_com}
            }
        }
    }
}

impl<const DLEN: usize> Arr4Com<f32, DLEN>{
    pub fn add(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::add(ret, lhs, rhs);
            }
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::add(ret, lhs, rhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.unwrap();
                cuda_com.add(ret, lhs, rhs);
            },
            
        }
    }

    pub fn sub(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::sub(ret, lhs, rhs);
            },
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::sub(ret, lhs, rhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.unwrap();
                cuda_com.sub(ret, lhs, rhs);
            },
        }
    }

    pub fn mul(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::mul(ret, lhs, rhs);
            },
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::mul(ret, lhs, rhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.unwrap();
                cuda_com.mul(ret, lhs, rhs);
            },
        }
    }

    pub fn div(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::div(ret, lhs, rhs);
            },
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::div(ret, lhs, rhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.unwrap();
                cuda_com.div(ret, lhs, rhs);
            },
        }
    }

    pub fn sin(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                legacy::LegacyArr4Float::sin(ret, lhs);
            },
            OpTarget::AVX2 => {
                avx2::Avx2Arr4Float::sin(ret, lhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.unwrap();
                cuda_com.sin(ret, lhs);
            },
        }
    }

    // pub fn sort(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
    //     let cuda_com = self.cuda_com.unwrap();
    //     cuda_com.sort(ret, lhs);
    // }


}

