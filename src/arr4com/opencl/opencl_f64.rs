use ocl::ProQue;
use crate::arr4com::Arr4ComFloat;
use crate::arr4com::opencl_type::OpenclArr4Float;

type Float = f64;

macro_rules! InterOpencl{
    ($self:ident, $ret:ident, $opr1:ident,  $F:tt) => {
        let out = $self.pro_que.create_buffer::<f64>().unwrap();
        let x = $self.pro_que.create_buffer::<f64>().unwrap();
        x.write(&$opr1[..]).enq().unwrap();
        //x.write(&opr1).enq().unwrap();
        let kernel = $self.pro_que.kernel_builder($F)
            .arg(&out)
            .arg(&x)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        out.read(&mut $ret[..]).enq().unwrap();
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $F:tt) => {
        let out = $self.pro_que.create_buffer::<f64>().unwrap();
        let x = $self.pro_que.create_buffer::<f64>().unwrap();
        x.write(&$opr1[..]).enq().unwrap();

        let y = $self.pro_que.create_buffer::<f64>().unwrap();
        y.write(&$opr2[..]).enq().unwrap();
        let kernel = $self.pro_que.kernel_builder($F)
            .arg(&out)
            .arg(&x)
            .arg(&y)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        out.read(&mut $ret[..]).enq().unwrap();
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:tt) => {
        let out = $self.pro_que.create_buffer::<f64>().unwrap();
        let x = $self.pro_que.create_buffer::<f64>().unwrap();
        x.write(&$opr1[..]).enq().unwrap();

        let y = $self.pro_que.create_buffer::<f64>().unwrap();
        y.write(&$opr2[..]).enq().unwrap();

        let z = $self.pro_que.create_buffer::<f64>().unwrap();
        z.write(&$opr3[..]).enq().unwrap();
        let kernel = $self.pro_que.kernel_builder($F)
            .arg(&out)
            .arg(&x)
            .arg(&y)
            .arg(&z)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        out.read(&mut $ret[..]).enq().unwrap();
    };
}



impl<const DLEN: usize> OpenclArr4Float<f64, DLEN>{
    pub fn newf64() -> Self{
        let src = include_str!("./res/opencl_f64.cl");
        let pro_que =  ProQue::builder()
        .src(src)
        .dims(DLEN)
        .build().unwrap();
        OpenclArr4Float {
            dlen: DLEN,
            nerver_use: 0f64, pro_que,
        }
    }
}


type F32Opencl<const DLEN: usize> = OpenclArr4Float<f64, DLEN>;

impl<const DLEN: usize> Arr4ComFloat<f64, DLEN> for F32Opencl<DLEN>{
    fn add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        let out = self.pro_que.create_buffer::<f64>().unwrap();
        let x = self.pro_que.create_buffer::<f64>().unwrap();
        x.write(&opr1[..]).enq().unwrap();

        let y = self.pro_que.create_buffer::<f64>().unwrap();
        y.write(&opr2[..]).enq().unwrap();
        let kernel = self.pro_que.kernel_builder("a4c_addf64")
            .arg(&out)
            .arg(&x)
            .arg(&y)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        out.read(&mut ret[..]).enq().unwrap();
    }

    fn sub(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_subf64");
    }

    fn mul(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_mulf64");
    }

    fn div(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_divf64");
    }

    fn mul_add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN], opr3: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, opr3, "a4c_mul_addf64");
    }
    fn gtf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_gtff64");
    }
    fn gtef(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_gteff64");
    }
    fn ltf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_ltff64");
    }
    fn ltef(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_lteff64");
    }

    fn ceil(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_ceilf64");
    }
    fn floor(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_floorf64");
    }
    fn round(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_roundf64");
    }
    fn trunc(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_truncf64");
    }
    fn abs(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_absf64");
    }
    fn max(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_maxf64");
    }
    fn min(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_minf64");
    }
    fn copysign(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_copysignf64");
    }

    fn cos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_cosf64");
    }
    fn sin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_sinf64");
    }
    fn tan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_tanf64");
    }
    fn acos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_acosf64");
    }
    fn asin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_asinf64");
    }
    fn atan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_atanf64");
    }
    fn atan2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_atan2f64");
    }
    fn cosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_coshf64");
    }
    fn sinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_sinhf64");
    }
    fn tanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_tanhf64");
    }
    fn acosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_acoshf64");
    }
    fn asinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_asinhf64");
    }
    fn atanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_atanhf64");
    }

    fn ln(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_lnf64");
    }
    fn ln_1p(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_ln_1pf64");
    }
    fn log10(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_log10f64");
    }
    fn log2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_log2f64");
    }

    fn exp(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_expf64");
    }
    fn exp2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_exp2f64");
    }
    fn exp_m1(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_exp_m1f64");
    }

    fn sqrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_sqrtf64");
    }
    fn cbrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, "a4c_cbrtf64");
    }

    fn powf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_powff64");
    }
    fn hypot(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_hypotf64");
    }
    
}
