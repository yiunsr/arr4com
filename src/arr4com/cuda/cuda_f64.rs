use rustacuda::prelude::*;
use rustacuda::launch;
use std::ffi::CString;
use crate::arr4com::Arr4ComFloat;
use crate::arr4com::cuda_type::CudaArr4Float;

type Float = f64;

macro_rules! InterCuda{
    ($self:ident, $ret:ident, $opr1:ident,  $F:ident) => {
        let mut x = DeviceBuffer::from_slice(&$opr1).unwrap();
        let mut r = unsafe { DeviceBuffer::uninitialized($self.dlen).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./res/cuda_f64.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.$F<<<1, 256, 0, stream>>>(
                r.as_device_ptr(),
                $self.dlen, // Length
                x.as_device_ptr()
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to($ret).unwrap();
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $F:ident) => {
        let mut x = DeviceBuffer::from_slice(&$opr1).unwrap();
        let mut y = DeviceBuffer::from_slice(&$opr2).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized($self.dlen).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./res/cuda_f64.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.$F<<<1, 256, 0, stream>>>(
                r.as_device_ptr(),
                $self.dlen, // Length
                x.as_device_ptr(),
                y.as_device_ptr()
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to($ret).unwrap();
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        let mut x = DeviceBuffer::from_slice(&$opr1).unwrap();
        let mut y = DeviceBuffer::from_slice(&$opr2).unwrap();
        let mut z = DeviceBuffer::from_slice(&$opr3).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized($self.dlen).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./res/cuda_f64.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.$F<<<1, 256, 0, stream>>>(
                r.as_device_ptr(),
                $self.dlen, // Length
                x.as_device_ptr(),
                y.as_device_ptr(),
                z.as_device_ptr()
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to($ret).unwrap();
    };
}


fn init_module() -> Context {
    // Get the first device
    let _result = rustacuda::init(CudaFlags::empty());
    let device = Device::get_device(0).unwrap();
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();

    context
    //let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    //(context, stream)
}


impl CudaArr4Float<f64>{
    pub fn newf64(dlen: usize) -> Self{
        let c = init_module();
        CudaArr4Float {
            dlen,
            nerver_use: 0f64, ctx:c,
        }
    }
}

type F64Cuda = CudaArr4Float<f64>;

impl Arr4ComFloat<f64> for F64Cuda{
    fn add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        let mut x = DeviceBuffer::from_slice(&opr1).unwrap();
        let mut y = DeviceBuffer::from_slice(&opr2).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(self.dlen).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./res/cuda_f64.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.a4c_addf64<<<1, 256, 0, stream>>>(
                r.as_device_ptr(),
                self.dlen, // Length
                x.as_device_ptr(),
                y.as_device_ptr()
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to(ret).unwrap();
    }

    fn sub(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_subf64);
    }

    fn mul(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_mulf64);
    }

    fn mul_add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float], opr3: &[Float]){
        InterCuda!(self, ret, opr1, opr2, opr3, a4c_mul_addf64);
    }
    fn gtf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_gtff64);
    }
    fn gtef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_gteff64);
    }
    fn ltf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_ltff64);
    }
    fn ltef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_lteff64);
    }
    fn ceil(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_ceilf64);
    }
    fn floor(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_floorf64);
    }
    fn round(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_roundf64);
    }
    fn trunc(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_truncf64);
    }

    fn abs(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_absf64);
    }
    fn max(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_maxf64);
    }
    fn min(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_minf64);
    }
    fn copysign(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_copysignf64);
    }

    fn div(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_divf64);
    }

    fn cos(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_cosf64);
    }
    fn sin(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_sinf64);
    }
    fn tan(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_tanf64);
    }
    fn acos(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_acosf64);
    }
    fn asin(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_asinf64);
    }
    fn atan(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_atanf64);
    }
    fn atan2(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_atan2f64);
    }
    fn cosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_coshf64);
    }
    fn sinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_sinhf64);
    }
    fn tanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_tanhf64);
    }
    fn acosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_acoshf64);
    }
    fn asinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_asinhf64);
    }
    fn atanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_atanhf64);
    }

    fn ln(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_lnf64);
    }
    fn ln_1p(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_ln_1pf64);
    }
    fn log10(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_log10f64);
    }
    fn log2(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_log2f64);
    }

    fn exp(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_expf64);
    }
    fn exp2(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_exp2f64);
    }
    fn exp_m1(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_exp_m1f64);
    }

    fn sqrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_sqrtf64);
    }
    fn cbrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_cbrtf64);
    }

    fn powf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_powff64);
    }
    fn hypot(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_hypotf64);
    }

    // fn sort(self, ret: &mut [Float], opr1: &[Float]){
    //     let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
    //     x.copy_from(&opr1).unwrap();

    //     let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };

    //     let module_data = CString::new(include_str!("./cuda/al.ptx")).unwrap();
    //     let module = Module::load_from_string(&module_data).unwrap();
    //     let stream = self.stream;

    //     unsafe {
    //         launch!(module.arr4com_sort<<<1, 256, 0, stream>>>(
    //             x.as_device_ptr(),
    //             r.as_device_ptr(),
    //             DLEN // Length
    //         )).unwrap();
    //     }
    
    //     stream.synchronize().unwrap();
    //     r.copy_to(ret).unwrap();
    // }
}
