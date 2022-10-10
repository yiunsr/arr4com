use arr4com::arr4com::Arr4F32;
use arr4com::arr4com::Arr4ComF32;
use arr4com::arr4com::OpTarget;

fn main() {
    //arr4com::arr4com::OpTarget 
    //let arr = arr4com::arr4com::arr4float::new();
    let mut ret:Arr4F32<256> = Arr4F32::new(OpTarget::AVX2);
    let mut arrLhs:Arr4F32<256> = Arr4F32::new(OpTarget::AVX2);
    let mut arrRhs:Arr4F32<256> = Arr4F32::new(OpTarget::AVX2);
    for i in 0..256{
        arrLhs.data[i] = (i as f32) * 2f32;
        arrRhs.data[i] = i as f32;
    }
    ret.add(&arrLhs, &arrRhs);

    println!("ret: {}", ret.data[0..10][0]);
    println!("ret: {}", ret.data[0..10][1]);
    println!("ret: {}", ret.data[0..10][2]);
    
}
