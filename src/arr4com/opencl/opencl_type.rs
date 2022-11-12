use ocl::ProQue;

pub struct OpenclArr4Float<T>{
    pub dlen: usize,
    pub pro_que: ProQue,
    #[allow(dead_code)]
    pub nerver_use:T
}