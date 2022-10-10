use arrayvec::ArrayVec;
use crate::arr4com::OpTarget;
use crate::arr4com::Arr4FloatCom;

pub struct LegacyArr4F32<const DLEN: usize>{
    pub op_target: OpTarget,
    pub data: ArrayVec<f32, DLEN>,
}


type F32Leg<const DLEN: usize> = LegacyArr4F32<DLEN>;
impl<const DLEN: usize> Arr4FloatCom<f32, DLEN> for F32Leg<DLEN>{
    fn new() -> Self{
        LegacyArr4F32{
            op_target: OpTarget::LEGACY,
            data: ArrayVec::<f32, DLEN>::new()
        }
    }

    fn add(&mut self, rhs:&F32Leg<DLEN>){
        for index in 0..DLEN{
            self.data[index] += rhs.data[index];
        }
    }
    fn sub(&mut self, rhs: &F32Leg<DLEN>){
        for index in 0..DLEN{
            self.data[index] -= rhs.data[index];
        }
    }

    fn mul(&mut self, rhs: &F32Leg<DLEN>){
        for index in 0..DLEN{
            self.data[index] *= rhs.data[index];
        }
    }

    fn div(&mut self, rhs: &F32Leg<DLEN>){
        for index in 0..DLEN{
            self.data[index] /= rhs.data[index];
        }
    }
}
