use rustacuda::prelude::*;
use rustacuda::launch;
use std::ffi::CString;

macro_rules! InterCuda1f32{
    ($self:ident, $ret:ident, $lhs:ident,  $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$lhs).unwrap();

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
}

macro_rules! InterCuda2f32{
    ($self:ident, $ret:ident, $lhs:ident, $rhs:ident, $F:ident) => {
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&$lhs).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&$rhs).unwrap();

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

pub struct CudaArr4Float<T, const DLEN: usize>{
    #[allow(dead_code)]
    nerver_use:T,
    #[allow(dead_code)]
    ctx: Context,  // never user but need this
    //stream: Stream,
}
impl<const DLEN: usize> CudaArr4Float<f32, DLEN>{
    pub fn newf32() -> Self{
        let c = init_module();
        CudaArr4Float {
            nerver_use: 0f32, ctx:c,
        }
    }
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

type F32Cuda<const DLEN: usize> = CudaArr4Float<f32, DLEN>;

impl<const DLEN: usize> F32Cuda<DLEN>{
    pub fn add(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        let mut x = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        x.copy_from(&lhs).unwrap();

        let mut y = unsafe { DeviceBuffer::uninitialized(DLEN).unwrap() };
        y.copy_from(&rhs).unwrap();

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

    pub fn sub(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCuda2f32!(self, ret, lhs, rhs, a4c_subf32);
    }

    pub fn mul(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCuda2f32!(self, ret, lhs, rhs, a4c_mulf32);
    }

    pub fn div(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN], rhs: [f32;DLEN]){
        InterCuda2f32!(self, ret, lhs, rhs, a4c_divf32);
    }

    pub fn sin(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCuda1f32!(self, ret, lhs, a4c_sinf32);
    }

    pub fn cos(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCuda1f32!(self, ret, lhs, a4c_cosf32);
    }

    pub fn tan(&self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
        InterCuda1f32!(self, ret, lhs, a4c_tanf32);
    }

    // pub fn sort(self, ret: &mut [f32;DLEN], lhs: [f32;DLEN]){
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
