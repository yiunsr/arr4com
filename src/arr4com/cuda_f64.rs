use rustacuda::prelude::*;
use rustacuda::launch;
use std::ffi::CString;
use crate::arr4com::cuda_type::CudaArr4Float;

macro_rules! IntInterCuda1f64{
    ($self:ident, $ret:ident, $lhs:ident,  $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$lhs).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f64.ptx")).unwrap();
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
}

macro_rules! InterCuda2f64{
    ($self:ident, $ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$lhs).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&$rhs).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f64.ptx")).unwrap();
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


impl<const DLEN: usize> CudaArr4Float<f64, DLEN>{
    #[allow(dead_code)]
    pub fn newf64() -> Self{
        let c = init_module();
        CudaArr4Float {
            nerver_use: 0f64, ctx:c,
        }
    }
}

type F64Cuda<const DLEN: usize> = CudaArr4Float<f64, DLEN>;

impl<const DLEN: usize> F64Cuda<DLEN>{
    pub fn add(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&lhs).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&rhs).unwrap();

        let mut r = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        // r.copy_from(&ret).unwrap();

        let module_data = CString::new(include_str!("./cuda/al_f64.ptx")).unwrap();
        let module = Module::load_from_string(&module_data).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        unsafe {
            // Launch the `arr4com_add` function with one block containing one thread on the given stream.
            launch!(module.a4c_addf64<<<1, 256, 0, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                r.as_device_ptr(),
                DLEN // Length
            )).unwrap();
        }
    
        stream.synchronize().unwrap();
        r.copy_to(ret).unwrap();
    }

    pub fn sub(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterCuda2f64!(self, ret, lhs, rhs, a4c_subf64);
    }

    pub fn mul(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterCuda2f64!(self, ret, lhs, rhs, a4c_mulf64);
    }

    pub fn div(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterCuda2f64!(self, ret, lhs, rhs, a4c_divf64);
    }

    pub fn cos(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_cosf64);
    }
    pub fn sin(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_sinf64);
    }
    pub fn tan(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_tanf64);
    }
    pub fn acos(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_acosf64);
    }
    pub fn asin(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_asinf64);
    }
    pub fn atan(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_atanf64);
    }
    pub fn cosh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_coshf64);
    }
    pub fn sinh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_sinhf64);
    }
    pub fn tanh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_tanhf64);
    }
    pub fn acosh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_acoshf64);
    }
    pub fn asinh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_asinhf64);
    }
    pub fn atanh(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_atanhf64);
    }

    pub fn ln(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_lnf64);
    }
    pub fn ln_1p(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_ln_1pf64);
    }
    pub fn log10(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_log10f64);
    }
    pub fn log2(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_log2f64);
    }

    pub fn exp(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_expf64);
    }
    pub fn exp2(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_exp2f64);
    }
    pub fn exp_m1(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_exp_m1f64);
    }

    pub fn sqrt(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_sqrtf64);
    }
    pub fn cbrt(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
        IntInterCuda1f64!(self, ret, lhs, a4c_cbrtf64);
    }

    pub fn powf(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterCuda2f64!(self, ret, lhs, rhs, a4c_powff64);
    }
    pub fn hypot(&self, ret: &mut [f64;DLEN], lhs: [f64;DLEN], rhs: [f64;DLEN]){
        InterCuda2f64!(self, ret, lhs, rhs, a4c_hypotf64);
    }

    // pub fn sort(self, ret: &mut [f64;DLEN], lhs: [f64;DLEN]){
    //     let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
    //     x.copy_from(&lhs).unwrap();

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
