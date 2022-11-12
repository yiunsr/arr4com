use crate::arr4com::Arr4ComFloat;
use crate::arr4com::*;

type Float = f64;

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
            OpTarget::OPENCL => {
                let opencl_com = $self.opencl_com.as_ref().unwrap();
                opencl_com.$F($ret, $opr1);
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
            OpTarget::OPENCL => {
                let opencl_com = $self.opencl_com.as_ref().unwrap();
                opencl_com.$F($ret, $opr1, $opr2);
            },
        }
    };

    ($self:ident,  $ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
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
            OpTarget::OPENCL => {
                let opencl_com = $self.opencl_com.as_ref().unwrap();
                opencl_com.$F($ret, $opr1, $opr2, $opr3);
            },
        }
    };
}

impl Arr4Com<f64>{
    pub fn newf64(op_target: OpTarget, dlen: usize) -> Self{
        // For speed optimization, block size can only be a multiple of 64
        if dlen % BLOCK_ALIGN  != 0{
            panic!("Only multiples of 64 are possible.")
        }
        match op_target {
            OpTarget::LEGACY => {
                let legacy_com = Some(legacy_type::LegacyArr4Float::newf64(dlen));
                Arr4Com{op_target: OpTarget::LEGACY, dlen, legacy_com, avx2_com: None, 
                    cuda_com: None, opencl_com:None}
            }
            OpTarget::AVX2 => {
                let avx2_com = Some(avx2_type::Avx2Arr4Float::newf64(dlen));
                Arr4Com{op_target: OpTarget::AVX2, dlen, legacy_com:None, avx2_com,
                    cuda_com: None, opencl_com: None}
            },
            OpTarget::CUDA => {
                let cuda_com = Some(cuda_type::CudaArr4Float::newf64(dlen));
                Arr4Com{op_target: OpTarget::CUDA, dlen, legacy_com:None, avx2_com: None, 
                    cuda_com, opencl_com: None}
            },
            OpTarget::OPENCL => {
                let opencl_com = Some(opencl_type::OpenclArr4Float::newf64(dlen));
                Arr4Com{op_target: OpTarget::OPENCL, dlen, legacy_com:None, avx2_com: None,
                    cuda_com: None, opencl_com}
            }
        }
    }
}

impl Arr4Com<Float>{
    pub fn add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
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
            OpTarget::OPENCL => {
                let opencl_com = self.opencl_com.as_ref().unwrap();
                opencl_com.add(ret, opr1, opr2);
            },
        }
    }

    pub fn sub(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, sub);
    }

    pub fn mul(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, mul);
    }

    pub fn div(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, div);
    }

    pub fn mul_add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float], opr3: &[Float]){
        InterCall!(self, ret, opr1, opr2, opr3, mul_add);
    }

    pub fn gtf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, gtf);
    }
    pub fn gtef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, gtef);
    }
    pub fn ltf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, ltf);
    }
    pub fn ltef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, ltef);
    }

    pub fn ceil(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, ceil);
    }
    pub fn floor(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, floor);
    }
    pub fn round(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, round);
    }
    pub fn trunc(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, trunc);
    }
    
    pub fn abs(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, abs);
    }
    pub fn max(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, max);
    }
    pub fn min(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, min);
    }
    pub fn copysign(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, copysign);
    }

    pub fn cos(&self, ret: &mut [Float], opr1: &[Float]){
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
            OpTarget::OPENCL => {
                let opencl_com = self.opencl_com.as_ref().unwrap();
                opencl_com.cos(ret, opr1);
            },
        }
    }
    pub fn sin(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, sin);
    }
    pub fn tan(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, tan);
    }
    pub fn acos(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, acos);
    }
    pub fn asin(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, asin);
    }
    pub fn atan(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, atan);
    }
    pub fn atan2(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, atan2);
    }
    pub fn cosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, cosh);
    }
    pub fn sinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, sinh);
    }
    pub fn tanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, tanh);
    }
    pub fn acosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, acosh);
    }
    pub fn asinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, asinh);
    }
    pub fn atanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, atanh);
    }
    pub fn ln(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, ln);
    }
    pub fn ln_1p(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, ln_1p);
    }
    pub fn log10(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, log10);
    }
    pub fn log2(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, log2);
    }

    pub fn exp(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, exp);
    }
    pub fn exp2(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, exp2);
    }
    pub fn exp_m1(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, exp_m1);
    }

    pub fn sqrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, sqrt);
    }
    pub fn cbrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterCall!(self, ret, opr1, cbrt);
    }
    pub fn powf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, powf);
    }
    pub fn hypot(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCall!(self, ret, opr1, opr2, hypot);
    }

    // pub fn sort(self, ret: &mut [Float], opr1: &[Float]){
    //     let cuda_com = self.cuda_com.unwrap();
    //     cuda_com.sort(ret, opr1);
    // }


}
