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
    #[allow(dead_code)]
    dlen: usize,
    legacy_com: Option<legacy::LegacyArr4Float<T, DLEN>>,
    avx2_com: Option<avx2::Avx2Arr4Float<T, DLEN>>,
    cuda_com: Option<cuda::CudaArr4Float<T, DLEN>>
}

pub trait Arr4ComAL<T, const DLEN: usize>{
    fn add(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn sub(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn mul(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);
    fn div(&self, ret: &mut [T;DLEN], lhs: [T;DLEN], rhs: [T;DLEN]);

    fn cos(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn sin(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn tan(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn acos(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn asin(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn atan(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn cosh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn sinh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn tanh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn acosh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn asinh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
    fn atanh(&self, ret: &mut [T;DLEN], lhs: [T;DLEN]);
}


macro_rules! InterCall1f32{
    ($self:ident, $ret:ident, $lhs:ident, $F:ident) => {
        match $self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = $self.legacy_com.as_ref().unwrap();
                legacy_com.$F($ret, $lhs);
            }
            OpTarget::AVX2 => {
                let avx2_com = $self.avx2_com.as_ref().unwrap();
                avx2_com.$F($ret, $lhs);
            },
            OpTarget::CUDA => {
                let cuda_com = $self.cuda_com.as_ref().unwrap();
                cuda_com.$F($ret, $lhs);
            },
            
        }
    };
}


macro_rules! InterCall2f32{
    ($self:ident, $ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        match $self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = $self.legacy_com.as_ref().unwrap();
                legacy_com.$F($ret, $lhs, $rhs);
            }
            OpTarget::AVX2 => {
                let avx2_com = $self.avx2_com.as_ref().unwrap();
                avx2_com.$F($ret, $lhs, $rhs);
            },
            OpTarget::CUDA => {
                let cuda_com = $self.cuda_com.as_ref().unwrap();
                cuda_com.$F($ret, $lhs, $rhs);
            },
            
        }
    };
}

impl<const DLEN: usize> Arr4Com<f32, DLEN>{
    pub fn new(op_target: OpTarget) -> Self{
        match op_target {
            OpTarget::LEGACY => {
                let legacy_com = Some(legacy::LegacyArr4Float::newf32());
                Arr4Com{op_target: OpTarget::LEGACY, dlen: DLEN, legacy_com, avx2_com: None, cuda_com: None}
            }
            OpTarget::AVX2 => {
                let avx2_com = Some(avx2::Avx2Arr4Float::newf32());
                Arr4Com{op_target: OpTarget::AVX2, dlen: DLEN, legacy_com:None, avx2_com, cuda_com: None}
            },
            OpTarget::CUDA => {
                let cuda_com = Some(cuda::CudaArr4Float::newf32());
                Arr4Com{op_target: OpTarget::CUDA, dlen: DLEN, legacy_com:None, avx2_com: None, cuda_com}
            }
        }
    }
}

impl<const DLEN: usize> Arr4Com<f32, DLEN>{
    pub fn add(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = self.legacy_com.as_ref().unwrap();
                legacy_com.add(ret, lhs, rhs);
            }
            OpTarget::AVX2 => {
                let avx2_com = self.avx2_com.as_ref().unwrap();
                avx2_com.add(ret, lhs, rhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.as_ref().unwrap();
                cuda_com.add(ret, lhs, rhs);
            },
            
        }
    }

    pub fn sub(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCall2f32!(self, ret, lhs, rhs, sub);
    }

    pub fn mul(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCall2f32!(self, ret, lhs, rhs, mul);
    }

    pub fn div(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCall2f32!(self, ret, lhs, rhs, div);
    }

    pub fn cos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = self.legacy_com.as_ref().unwrap();
                legacy_com.cos(ret, lhs);
            },
            OpTarget::AVX2 => {
                let avx2_com = self.avx2_com.as_ref().unwrap();
                avx2_com.cos(ret, lhs);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.as_ref().unwrap();
                cuda_com.cos(ret, lhs);
            },
        }
    }
    pub fn sin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, sin);
    }
    pub fn tan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, tan);
    }
    pub fn acos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, acos);
    }
    pub fn asin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, asin);
    }
    pub fn atan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, atan);
    }
    pub fn cosh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, cosh);
    }
    pub fn sinh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, sinh);
    }
    pub fn tanh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, tanh);
    }
    pub fn acosh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, acosh);
    }
    pub fn asinh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, asinh);
    }
    pub fn atanh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, atanh);
    }

    // pub fn sort(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
    //     let cuda_com = self.cuda_com.unwrap();
    //     cuda_com.sort(ret, lhs);
    // }


}

