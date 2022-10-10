use arr4com::arr4com::Arr4FloatCom;
use arr4com::arr4com::OpTarget;

fn main() {
    //arr4com::arr4com::OpTarget 
    //let arr = arr4com::arr4com::arr4float::new();
    let arr = arr4com::arr4com::new_arr4f32::<256>(OpTarget::AVX2);
    for i in 0..256{
        arr.imp_avx2.unwrap().data[i] = i as f32;
    }
    println!("Hello, world!");
}
