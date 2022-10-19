use crate::arr4com::Arr4ComAL;

pub struct LegacyArr4Float<T, const DLEN: usize>{
    nerver_use:T
}


impl<const DLEN: usize> Arr4ComAL<f32, DLEN> for LegacyArr4Float<f32, DLEN>{

    fn add(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] + rhs[index];
        }
    }
    
    fn sub(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] - rhs[index];
        }
    }

    fn mul(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] * rhs[index];
        }
    }

    fn div(ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index] / rhs[index];
        }
    }

    fn sin(ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        for index in 0..DLEN{
            ret[index] = lhs[index].sin();
        }
    }
}
