use rustacuda::prelude::*;
use rustacuda::launch;
use std::ffi::CString;
use crate::arr4com::cuda_type::CudaArr4Float;

type Float = f32;

macro_rules! InterCuda{
    ($self:ident, $ret:ident, $opr1:ident,  $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$opr1).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f32.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.$F<<<1, 256, 0, stream>>>(
                x.as_device_ptr(),
                r.as_device_ptr(),
                DLEN // Length
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to($ret).unwrap();
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$opr1).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&$opr2).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f32.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.$F<<<1, 256, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                r.as_device_ptr(),
                DLEN // Length
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to($ret).unwrap();
    };

    ($self:ident, $ret:ident, $opr1:ident, $opr2:ident, $opr3:ident, $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$opr1).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&$opr2).unwrap();

        let mut z = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        z.copy_from(&$opr3).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f32.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.$F<<<1, 256, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                z.as_device_ptr(),
                r.as_device_ptr(),
                DLEN // Length
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

impl<const DLEN: usize> CudaArr4Float<f32, DLEN>{
    pub fn newf32() -> Self{
        let c = init_module();
        CudaArr4Float {
            nerver_use: 0f32, ctx:c,
        }
    }
}


type F32Cuda<const DLEN: usize> = CudaArr4Float<f32, DLEN>;

impl<const DLEN: usize> F32Cuda<DLEN>{
    pub fn add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&opr1).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&opr2).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f32.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.a4c_addf32<<<1, 256, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                r.as_device_ptr(),
                DLEN // Length
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to(ret).unwrap();
    }

    pub fn sub(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, a4c_subf32);
    }

    pub fn mul(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, a4c_mulf32);
    }

    pub fn div(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, a4c_divf32);
    }

    pub fn mul_add(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN], opr3: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, opr3, a4c_mul_addf32);
    }
    pub fn ceil(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_ceilf32);
    }
    pub fn floor(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_floorf32);
    }
    pub fn round(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_roundf32);
    }
    pub fn trunc(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_truncf32);
    }

    pub fn cos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_cosf32);
    }
    pub fn sin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_sinf32);
    }
    pub fn tan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_tanf32);
    }
    pub fn acos(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_acosf32);
    }
    pub fn asin(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_asinf32);
    }
    pub fn atan(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_atanf32);
    }
    pub fn atan2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, a4c_atan2f32);
    }
    pub fn cosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_coshf32);
    }
    pub fn sinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_sinhf32);
    }
    pub fn tanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_tanhf32);
    }
    pub fn acosh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_acoshf32);
    }
    pub fn asinh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_asinhf32);
    }
    pub fn atanh(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_atanhf32);
    }

    pub fn ln(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_lnf32);
    }
    pub fn ln_1p(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_ln_1pf32);
    }
    pub fn log10(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_log10f32);
    }
    pub fn log2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_log2f32);
    }

    pub fn exp(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_expf32);
    }
    pub fn exp2(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_exp2f32);
    }
    pub fn exp_m1(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_exp_m1f32);
    }

    pub fn sqrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_sqrtf32);
    }
    pub fn cbrt(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
        InterCuda!(self, ret, opr1, a4c_cbrtf32);
    }

    pub fn powf(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, a4c_powff32);
    }
    pub fn hypot(&self, ret: &mut [Float;DLEN], opr1: [Float;DLEN], opr2: [Float;DLEN]){
        InterCuda!(self, ret, opr1, opr2, a4c_hypotf32);
    }

    // pub fn sort(self, ret: &mut [Float;DLEN], opr1: [Float;DLEN]){
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
