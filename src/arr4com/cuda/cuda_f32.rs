use rustacuda::prelude::*;
use rustacuda::launch;
use std::ffi::CString;
use crate::arr4com::Arr4ComFloat;
use crate::arr4com::cuda_type::CudaArr4Float;

type Float = f32;

macro_rules! InterCuda{
    ($self:ident, $ret:ident, $opr1:ident,  $F:ident) => {
        let mut x = DeviceBuffer::from_slice(&$opr1).unwrap();
        let mut r = unsafe { DeviceBuffer::uninitialized($self.dlen).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./res/cuda_f32.ptx")).unwrap();
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

        let module_data = CString::new(include_str!("./res/cuda_f32.ptx")).unwrap();
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

        let module_data = CString::new(include_str!("./res/cuda_f32.ptx")).unwrap();
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


pub fn init_module() -> Context {
    // Get the first device
    let _result = rustacuda::init(CudaFlags::empty());
    let device = Device::get_device(0).unwrap();
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();

    context
    //let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    //(context, stream)
}

impl CudaArr4Float<f32>{
    pub fn newf32(dlen: usize) -> Self{
        let c = init_module();
        CudaArr4Float {
            dlen,
            nerver_use: 0f32, ctx:c,
        }
    }
}


type F32Cuda= CudaArr4Float<f32>;

impl Arr4ComFloat<f32> for F32Cuda{
    fn add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        let mut x = DeviceBuffer::from_slice(&opr1).unwrap();
        let mut y = DeviceBuffer::from_slice(&opr2).unwrap();
        let mut r = unsafe { DeviceBuffer::uninitialized(self.dlen).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./res/cuda_f32.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.a4c_addf32<<<1, 256, 0, stream>>>(
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
        InterCuda!(self, ret, opr1, opr2, a4c_subf32);
    }

    fn mul(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_mulf32);
    }

    fn div(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_divf32);
    }

    fn mul_add(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float], opr3: &[Float]){
        InterCuda!(self, ret, opr1, opr2, opr3, a4c_mul_addf32);
    }
    fn gtf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_gtff32);
    }
    fn gtef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_gteff32);
    }
    fn ltf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_ltff32);
    }
    fn ltef(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_lteff32);
    }

    fn ceil(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_ceilf32);
    }
    fn floor(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_floorf32);
    }
    fn round(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_roundf32);
    }
    fn trunc(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_truncf32);
    }
    fn abs(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_absf32);
    }
    fn max(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_maxf32);
    }
    fn min(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_minf32);
    }
    fn copysign(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_copysignf32);
    }

    fn cos(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_cosf32);
    }
    fn sin(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_sinf32);
    }
    fn tan(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_tanf32);
    }
    fn acos(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_acosf32);
    }
    fn asin(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_asinf32);
    }
    fn atan(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_atanf32);
    }
    fn atan2(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_atan2f32);
    }
    fn cosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_coshf32);
    }
    fn sinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_sinhf32);
    }
    fn tanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_tanhf32);
    }
    fn acosh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_acoshf32);
    }
    fn asinh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_asinhf32);
    }
    fn atanh(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_atanhf32);
    }

    fn ln(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_lnf32);
    }
    fn ln_1p(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_ln_1pf32);
    }
    fn log10(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_log10f32);
    }
    fn log2(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_log2f32);
    }

    fn exp(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_expf32);
    }
    fn exp2(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_exp2f32);
    }
    fn exp_m1(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_exp_m1f32);
    }

    fn sqrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_sqrtf32);
    }
    fn cbrt(&self, ret: &mut [Float], opr1: &[Float]){
        InterCuda!(self, ret, opr1, a4c_cbrtf32);
    }

    fn powf(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_powff32);
    }
    fn hypot(&self, ret: &mut [Float], opr1: &[Float], opr2: &[Float]){
        InterCuda!(self, ret, opr1, opr2, a4c_hypotf32);
    }

    // pub fn sort(self, ret: &mut [Float], opr1: &[Float]){
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
