use arr4com::arr4com::Arr4Com;
use arr4com::arr4com::OpTarget;

const BLOCK_SIZE: usize = 256;


fn main01() {
    //arr4com::arr4com::OpTarget 
    //let arr = arr4com::arr4com::arr4float::new();
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::LEGACY);
    let mut result = [0f32;BLOCK_SIZE];
    let mut lhs = [0f32;BLOCK_SIZE];
    let mut rhs = [0f32;BLOCK_SIZE];
    for i in 0..BLOCK_SIZE{
        lhs[i] = (i as f32) * 2f32;
        rhs[i] = i as f32;
    }
    compute.add(&mut result, lhs, rhs);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main02(){
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::AVX2);
    let mut result = [0f32;BLOCK_SIZE];
    let mut lhs:Vec<f32> = Vec::with_capacity(BLOCK_SIZE);
    let mut rhs:Vec<f32> = Vec::with_capacity(BLOCK_SIZE);
    for i in 0..BLOCK_SIZE{
        lhs.push((i as f32) * 2f32);
        rhs.push(i as f32);
    }
    let lhs:[f32; BLOCK_SIZE] = lhs[..].try_into().unwrap();
    let rhs:[f32; BLOCK_SIZE] = rhs[..].try_into().unwrap();

    compute.add(&mut result, lhs, rhs);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main03(){
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::CUDA);
    let mut result = [0f32;BLOCK_SIZE];
    let mut lhs = [0f32;BLOCK_SIZE];
    let mut rhs = [0f32;BLOCK_SIZE];
    for i in 0..BLOCK_SIZE{
        lhs[i] = (i as f32) * 3f32;
        rhs[i] = i as f32;
    }
    compute.sub(&mut result, lhs, rhs);

    println!("ret: {}", result[0]);
    println!("ret: {}", result[1]);
    println!("ret: {}", result[2]);
}

fn main04(){
    let compute:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::AVX2);
    let mut result = [0f32;BLOCK_SIZE];
    let mut lhs = [0f32;BLOCK_SIZE];
    for i in 0..BLOCK_SIZE{
        lhs[i] = i as f32;
    }
    compute.sin(&mut result, lhs);

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
