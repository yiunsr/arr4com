use crate::arr4com::OpTarget;
use crate::arr4com::Arr4F32;
use crate::arr4com::Arr4ComF32;

pub struct LegacyArr4F32<const DLEN: usize>{
}


type F32Leg<const DLEN: usize> = LegacyArr4F32<DLEN>;
impl<const DLEN: usize> Arr4ComF32<DLEN> for F32Leg<DLEN>{

    fn add(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        for index in 0..DLEN{
            ret.data[index] = lhs.data[index] + rhs.data[index];
        }
    }
    
    fn sub(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        for index in 0..DLEN{
            ret.data[index] = lhs.data[index] - rhs.data[index];
        }
    }

    fn mul(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        for index in 0..DLEN{
            ret.data[index] = lhs.data[index] * rhs.data[index];
        }
    }

    fn div(ret: &mut Arr4F32<DLEN>, lhs: &Arr4F32<DLEN>, rhs: &Arr4F32<DLEN>){
        for index in 0..DLEN{
            ret.data[index] = lhs.data[index] / rhs.data[index];
        }
    }
}
