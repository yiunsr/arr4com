use crate::arr4com::Arr4ComAL;
use crate::arr4com::legacy_type::LegacyArr4Float;

type Float = f32;

impl<const DLEN: usize> LegacyArr4Float<Float, DLEN>{
    pub fn newf32() -> Self{
        LegacyArr4Float {
            nerver_use: 0f32,
        }
    }
}

impl<const DLEN: usize> Arr4ComAL<Float, DLEN> for LegacyArr4Float<Float, DLEN>{

    fn add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index] + opr2[index];
        }
    }
    
    fn sub(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index] - opr2[index];
        }
    }

    fn mul(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index] * opr2[index];
        }
    }

    fn div(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index] / opr2[index];
        }
    }

    fn mul_add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN], opr3: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index] * opr2[index] + opr3[index];
        }
    }
    fn ceil(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].ceil();
        }
    }
    fn floor(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].floor();
        }
    }
    fn round(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].round();
        }
    }
    fn trunc(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].trunc();
        }
    }

    fn abs(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].abs();
        }
    }
    fn max(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].max(opr2[index]);
        }
    }
    fn min(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].min(opr2[index]);
        }
    }
    fn copysign(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].copysign(opr2[index]);
        }
    }

    fn sin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].sin();
        }
    }
    fn cos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].cos();
        }
    }
    fn tan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].tan();
        }
    }
    fn asin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].asin();
        }
    }
    fn acos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].acos();
        }
    }
    fn atan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].atan();
        }
    }
    fn atan2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].atan2(opr2[index]);
        }
    }
    fn sinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].sinh();
        }
    }
    fn cosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].cosh();
        }
    }
    fn tanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].tanh();
        }
    }
    fn asinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].asinh();
        }
    }
    fn acosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].acosh();
        }
    }
    fn atanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].atanh();
        }
    }

    fn ln(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].ln();
        }
    }
    fn ln_1p(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].ln_1p();
        }
    }
    fn log10(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].log10();
        }
    }
    fn log2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].log2();
        }
    }

    fn exp(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].exp();
        }
    }
    fn exp2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].exp2();
        }
    }
    fn exp_m1(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].exp_m1();
        }
    }
    fn sqrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].sqrt();
        }
    }
    fn cbrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].cbrt();
        }
    }

    fn powf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].powf(opr2[index]);
        }
    }
    fn hypot(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        for index in 0..DLEN{
            ret[index] = opr1[index].hypot(opr2[index]);
        }
    }

}

