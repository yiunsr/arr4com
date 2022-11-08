use ocl::ProQue;

// pub struct OpenclArr4Float<T, const DLEN: usize>{
//     #[allow(dead_code)]
//     pub nerver_use:T,
//     #[allow(dead_code)]
//     pub ctx: Context,  // never user but need this
//     //stream: Stream,
// }


pub struct OpenclArr4Float<T, const DLEN: usize>{
    pub dlen: usize,
    pub pro_que: ProQue,
    #[allow(dead_code)]
    pub nerver_use:T
}