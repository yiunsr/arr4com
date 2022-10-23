use crate::arr4com::Arr4ComAL;

pub struct LegacyArr4Float<T, const DLEN: usize>{
    #[allow(dead_code)]
    nerver_use:T
}
impl<const DLEN: usize> LegacyArr4Float<f32, DLEN>{
    pub fn newf32() -> Self{
        LegacyArr4Float {
            nerver_use: 0f32,
        }
    }
}

impl<const DLEN: usize> Arr4ComAL<f32, DLEN> for LegacyArr4Float<f32, DLEN>{

    fn add(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] + rhs[index];
        }
    }
    
    fn sub(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] - rhs[index];
        }
    }

    fn mul(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] * rhs[index];
        }
    }

    fn div(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] / rhs[index];
        }
    }

    fn sin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].sin();
        }
    }
    fn cos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].cos();
        }
    }
    fn tan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].tan();
        }
    }
    fn asin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].asin();
        }
    }
    fn acos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].acos();
        }
    }
    fn atan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].atan();
        }
    }
    fn sinh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].sinh();
        }
    }
    fn cosh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].cosh();
        }
    }
    fn tanh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].tanh();
        }
    }
    fn asinh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].asinh();
        }
    }
    fn acosh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].acosh();
        }
    }
    fn atanh(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].atanh();
        }
    }

    fn ln(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].ln();
        }
    }
    fn ln_1p(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].ln_1p();
        }
    }
    fn log10(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].log10();
        }
    }
    fn log2(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].log2();
        }
    }

    fn exp(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].exp();
        }
    }
    fn exp2(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].exp2();
        }
    }
    fn exp_m1(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].exp_m1();
        }
    }
    fn sqrt(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].sqrt();
        }
    }
    fn cbrt(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].cbrt();
        }
    }

    fn powf(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].powf(rhs[index]);
        }
    }
    fn hypot(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].hypot(rhs[index]);
        }
    }

}

