use crate::arr4com::Arr4ComAL;
use crate::arr4com::*;

type Float = f32;

macro_rules! InterCall{
    ($self:ident, $ret:ident, $opr1:ident, $F:ident) => {
        match $self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = $self.legacy_com.as_ref().unwrap();
                legacy_com.$F($ret, $opr1);
            }
            OpTarget::AVX2 => {
                let avx2_com = $self.avx2_com.as_ref().unwrap();
                avx2_com.$F($ret, $opr1);
            },
            OpTarget::CUDA => {
                let cuda_com = $self.cuda_com.as_ref().unwrap();
                cuda_com.$F($ret, $opr1);
            },
            
        }
    };
    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $F:ident) => {
        match $self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = $self.legacy_com.as_ref().unwrap();
                legacy_com.$F($ret, $opr1, $opr2);
            }
            OpTarget::AVX2 => {
                let avx2_com = $self.avx2_com.as_ref().unwrap();
                avx2_com.$F($ret, $opr1, $opr2);
            },
            OpTarget::CUDA => {
                let cuda_com = $self.cuda_com.as_ref().unwrap();
                cuda_com.$F($ret, $opr1, $opr2);
            },
            
        }
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        match $self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = $self.legacy_com.as_ref().unwrap();
                legacy_com.$F($ret, $opr1, $opr2, $opr3);
            }
            OpTarget::AVX2 => {
                let avx2_com = $self.avx2_com.as_ref().unwrap();
                avx2_com.$F($ret, $opr1, $opr2, $opr3);
            },
            OpTarget::CUDA => {
                let cuda_com = $self.cuda_com.as_ref().unwrap();
                cuda_com.$F($ret, $opr1, $opr2, $opr3);
            },
            
        }
    };
}

impl<const DLEN: usize> Arr4Com<Float, DLEN>{
    pub fn newf32(op_target: OpTarget) -> Self{
        match op_target {
            OpTarget::LEGACY => {
                let legacy_com = Some(legacy_type::LegacyArr4Float::newf32());
                Arr4Com{op_target: OpTarget::LEGACY, dlen: DLEN, legacy_com, avx2_com: None, cuda_com: None}
            }
            OpTarget::AVX2 => {
                let avx2_com = Some(avx2_type::Avx2Arr4Float::newf32());
                Arr4Com{op_target: OpTarget::AVX2, dlen: DLEN, legacy_com:None, avx2_com, cuda_com: None}
            },
            OpTarget::CUDA => {
                let cuda_com = Some(cuda_type::CudaArr4Float::newf32());
                Arr4Com{op_target: OpTarget::CUDA, dlen: DLEN, legacy_com:None, avx2_com: None, cuda_com}
            }
        }
    }
}

impl<const DLEN: usize> Arr4Com<Float, DLEN>{
    pub fn add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = self.legacy_com.as_ref().unwrap();
                legacy_com.add(ret, opr1, opr2);
            }
            OpTarget::AVX2 => {
                let avx2_com = self.avx2_com.as_ref().unwrap();
                avx2_com.add(ret, opr1, opr2);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.as_ref().unwrap();
                cuda_com.add(ret, opr1, opr2);
            },
            
        }
    }

    pub fn sub(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, sub);
    }

    pub fn mul(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, mul);
    }

    pub fn div(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, div);
    }

    pub fn mul_add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN], opr3: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, opr3, mul_add);
    }

    pub fn ceil(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, ceil);
    }
    pub fn floor(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, floor);
    }
    pub fn round(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, round);
    }
    pub fn trunc(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, trunc);
    }

    pub fn cos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        match self.op_target {
            OpTarget::LEGACY => {
                let legacy_com = self.legacy_com.as_ref().unwrap();
                legacy_com.cos(ret, opr1);
            },
            OpTarget::AVX2 => {
                let avx2_com = self.avx2_com.as_ref().unwrap();
                avx2_com.cos(ret, opr1);
            },
            OpTarget::CUDA => {
                let cuda_com = self.cuda_com.as_ref().unwrap();
                cuda_com.cos(ret, opr1);
            },
        }
    }
    pub fn sin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, sin);
    }
    pub fn tan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, tan);
    }
    pub fn acos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, acos);
    }
    pub fn asin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, asin);
    }
    pub fn atan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, atan);
    }
    pub fn cosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, cosh);
    }
    pub fn sinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, sinh);
    }
    pub fn tanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, tanh);
    }
    pub fn acosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, acosh);
    }
    pub fn asinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, asinh);
    }
    pub fn atanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, atanh);
    }
    pub fn ln(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, ln);
    }
    pub fn ln_1p(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, ln_1p);
    }
    pub fn log10(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, log10);
    }
    pub fn log2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, log2);
    }

    pub fn exp(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, exp);
    }
    pub fn exp2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, exp2);
    }
    pub fn exp_m1(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, exp_m1);
    }

    pub fn sqrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, sqrt);
    }
    pub fn cbrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCall!(self, ret, opr1, cbrt);
    }
    pub fn atan2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, atan2);
    }
    pub fn powf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, powf);
    }
    pub fn hypot(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCall!(self, ret, opr1, opr2, hypot);
    }

    // pub fn sort(self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
    //     let cuda_com = self.cuda_com.unwrap();
    //     cuda_com.sort(ret, opr1);
    // }


}
