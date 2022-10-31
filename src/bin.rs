#![allow(
    dead_code
)]

use arr4com::arr4com::Arr4Com;
use arr4com::arr4com::OpTarget;

const BLOCK_SIZE: usize = 256;


fn main01() {
    //arr4com::arr4com::OpTarget 
    //let arr = arr4com::arr4com::arr4float::new();
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::newf32(OpTarget::LEGACY);
    let mut result = [0f32;BLOCK_SIZE];
    let mut opr1 = [0f32;BLOCK_SIZE];
    let mut opr2 = [0f32;BLOCK_SIZE];
    for i in 0..BLOCK_SIZE{
        opr1[i] = (i as f32) * 2f32;
        opr2[i] = i as f32;
    }
    compute.add(&mut result, opr1, opr2);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main02(){
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::newf32(OpTarget::AVX2);
    let mut result = [0f32;BLOCK_SIZE];
    let mut opr1:Vec<f32> = Vec::with_capacity(BLOCK_SIZE);
    let mut opr2:Vec<f32> = Vec::with_capacity(BLOCK_SIZE);
    for i in 0..BLOCK_SIZE{
        opr1.push((i as f32) * 2f32);
        opr2.push(i as f32);
    }
    let opr1:[f32; BLOCK_SIZE] = opr1[..].try_into().unwrap();
    let opr2:[f32; BLOCK_SIZE] = opr2[..].try_into().unwrap();

    compute.add(&mut result, opr1, opr2);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main03(){
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::newf32(OpTarget::CUDA);
    let mut result = [0f32;BLOCK_SIZE];
    let mut opr1 = [0f32;BLOCK_SIZE];
    let mut opr2 = [0f32;BLOCK_SIZE];
    for i in 0..BLOCK_SIZE{
        opr1[i] = (i as f32) * 3f32;
        opr2[i] = i as f32;
    }
    compute.sub(&mut result, opr1, opr2);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main04(){
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::newf32(OpTarget::AVX2);
    let mut result = [0f32;BLOCK_SIZE];
    let mut opr1 = [0f32;BLOCK_SIZE];
    for i in 0..BLOCK_SIZE{
        opr1[i] = i as f32;
    }
    compute.sin(&mut result, opr1);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main() {
    //main01();
    //main02();
    //main03();
    main04();
}


#[cfg(test)]
mod tests;
