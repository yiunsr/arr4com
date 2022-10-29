use crate::arr4com::Arr4ComAL;
use crate::arr4com::legacy_f32;
use crate::arr4com::avx2_f32;
use crate::arr4com::cuda_f32;
use crate::arr4com::*;



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
    pub fn ln(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, ln);
    }
    pub fn ln_1p(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, ln_1p);
    }
    pub fn log10(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, log10);
    }
    pub fn log2(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, log2);
    }

    pub fn exp(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, exp);
    }
    pub fn exp2(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, exp2);
    }
    pub fn exp_m1(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, exp_m1);
    }

    pub fn sqrt(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, sqrt);
    }
    pub fn cbrt(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCall1f32!(self, ret, lhs, cbrt);
    }
    pub fn powf(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCall2f32!(self, ret, lhs, rhs, powf);
    }
    pub fn hypot(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCall2f32!(self, ret, lhs, rhs, hypot);
    }

    // pub fn sort(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
    //     let cuda_com = self.cuda_com.unwrap();
    //     cuda_com.sort(ret, lhs);
    // }


}
