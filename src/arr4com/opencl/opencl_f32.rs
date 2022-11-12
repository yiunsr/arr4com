use ocl::ProQue;
use crate::arr4com::Arr4ComFloat;
use crate::arr4com::opencl_type::OpenclArr4Float;

type Float = f32;

macro_rules! InterOpencl{
    ($self:ident, $ret:ident, $opr1:ident,  $F:tt) => {
        let out = $self.pro_que.create_buffer::<f32>().unwrap();
        let x = $self.pro_que.create_buffer::<f32>().unwrap();
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
        let out = $self.pro_que.create_buffer::<f32>().unwrap();
        let x = $self.pro_que.create_buffer::<f32>().unwrap();
        x.write(&$opr1[..]).enq().unwrap();

        let y = $self.pro_que.create_buffer::<f32>().unwrap();
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
        let out = $self.pro_que.create_buffer::<f32>().unwrap();
        let x = $self.pro_que.create_buffer::<f32>().unwrap();
        x.write(&$opr1[..]).enq().unwrap();

        let y = $self.pro_que.create_buffer::<f32>().unwrap();
        y.write(&$opr2[..]).enq().unwrap();

        let z = $self.pro_que.create_buffer::<f32>().unwrap();
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



impl OpenclArr4Float<f32>{
    pub fn newf32(dlen: usize) -> Self{
        let src = include_str!("./res/opencl_f32.cl");
        let pro_que =  ProQue::builder()
        .src(src)
        .dims(dlen)
        .build().unwrap();
        OpenclArr4Float {
            dlen,
            nerver_use: 0f32, pro_que,
        }
    }
}


type F32Opencl = OpenclArr4Float<f32>;

impl Arr4ComFloat<f32> for F32Opencl{
    fn add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        let out = self.pro_que.create_buffer::<f32>().unwrap();
        let x = self.pro_que.create_buffer::<f32>().unwrap();
        x.write(&opr1[..]).enq().unwrap();

        let y = self.pro_que.create_buffer::<f32>().unwrap();
        y.write(&opr2[..]).enq().unwrap();
        let kernel = self.pro_que.kernel_builder("a4c_addf32")
            .arg(&out)
            .arg(&x)
            .arg(&y)
            .build().unwrap();

        unsafe { kernel.enq().unwrap(); }
        out.read(&mut ret[..]).enq().unwrap();
    }

    fn sub(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_subf32");
    }

    fn mul(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_mulf32");
    }

    fn div(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_divf32");
    }

    fn mul_add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float], opr3: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, opr3, "a4c_mul_addf32");
    }
    fn gtf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_gtff32");
    }
    fn gtef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_gteff32");
    }
    fn ltf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_ltff32");
    }
    fn ltef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_lteff32");
    }

    fn ceil(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_ceilf32");
    }
    fn floor(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_floorf32");
    }
    fn round(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_roundf32");
    }
    fn trunc(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_truncf32");
    }
    fn abs(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_absf32");
    }
    fn max(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_maxf32");
    }
    fn min(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_minf32");
    }
    fn copysign(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_copysignf32");
    }

    fn cos(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_cosf32");
    }
    fn sin(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_sinf32");
    }
    fn tan(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_tanf32");
    }
    fn acos(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_acosf32");
    }
    fn asin(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_asinf32");
    }
    fn atan(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_atanf32");
    }
    fn atan2(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_atan2f32");
    }
    fn cosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_coshf32");
    }
    fn sinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_sinhf32");
    }
    fn tanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_tanhf32");
    }
    fn acosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_acoshf32");
    }
    fn asinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_asinhf32");
    }
    fn atanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_atanhf32");
    }

    fn ln(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_lnf32");
    }
    fn ln_1p(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_ln_1pf32");
    }
    fn log10(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_log10f32");
    }
    fn log2(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_log2f32");
    }

    fn exp(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_expf32");
    }
    fn exp2(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_exp2f32");
    }
    fn exp_m1(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_exp_m1f32");
    }

    fn sqrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_sqrtf32");
    }
    fn cbrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterOpencl!(self, ret, opr1, "a4c_cbrtf32");
    }

    fn powf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_powff32");
    }
    fn hypot(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterOpencl!(self, ret, opr1, opr2, "a4c_hypotf32");
    }
    
}
