use rustacuda::prelude::*;

pub struct CudaArr4Float<T>{
    pub dlen: usize,
    #[allow(dead_code)]
    pub nerver_use:T,
    #[allow(dead_code)]
    pub ctx: Context,  // never user but need this
    //stream: Stream,
}
