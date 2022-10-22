macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d || $y - $x < $d) { panic!(); }
    }
}

macro_rules! assert_eq_f32 {
    ($x:expr, $y:expr) => {
        if !($x - $y < f32::EPSILON || $y - $x < f32::EPSILON) { panic!(); }
    }
}

macro_rules! assert_eq_f32_array256 {
    ($arr:expr, $y0:expr, $y1:expr, $y2:expr, $y100:expr, $y255:expr) => {
        let epsilonx:f32 = f32::EPSILON;

        if !($arr[0] - $y0 < epsilonx || $arr[0] - $y0 < epsilonx) { println!("index 0 error");panic!(); }
        if !($arr[1] - $y1 < epsilonx || $arr[1] - $y0 < epsilonx) { println!("index 1 error");panic!(); }
        if !($arr[2] - $y2 < epsilonx || $arr[2] - $y2 < epsilonx) { println!("index 2 error");panic!(); }
        if !($arr[100] - $y100 < epsilonx || $arr[100] - $y100 < epsilonx) { println!("index 100 error");panic!(); }
        if !($arr[255] - $y255 < epsilonx || $arr[255] - $y255 < epsilonx) { println!("index 255 error");panic!(); }
    }
}

macro_rules! assert_eq_f32_array256_2 {
    ($arr:expr, $y0:expr, $y1:expr, $y2:expr, $y100:expr, $y255:expr) => {
        let epsilonx:f32 = f32::EPSILON * 1.001f32;

        if !($arr[0] - $y0 < epsilonx || $arr[0] - $y0 < epsilonx) { println!("index 0 error");panic!(); }
        if !($arr[1] - $y1 < epsilonx || $arr[1] - $y0 < epsilonx) { println!("index 1 error");panic!(); }
        if !($arr[2] - $y2 < epsilonx || $arr[2] - $y2 < epsilonx) { println!("index 2 error");panic!(); }
        if !($arr[100] - $y100 < epsilonx || $arr[100] - $y100 < epsilonx) { println!("index 100 error");panic!(); }
        if !($arr[255] - $y255 < epsilonx || $arr[255] - $y255 < epsilonx) { println!("index 255 error");panic!(); }
    }
}

#[cfg(test)]
mod float_tests {
    

    use arr4com::arr4com::Arr4Com;
    use arr4com::arr4com::OpTarget;
    const BLOCK_SIZE: usize = 256;

    #[test]
    fn test_0001_01_arithmetic32() {
        println!("==== test_0001_01_arithmetic32 start ====");
        let legacy:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::LEGACY);
        let avx2:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::AVX2);
        let cuda:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::CUDA);

        let mut result = [0f32;BLOCK_SIZE];
        let mut lhs = [0f32;BLOCK_SIZE];
        let mut rhs = [0f32;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            lhs[i] = (i as f32) * 2f32;
            rhs[i] = i as f32;
        }
        legacy.add(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 3f32, 6f32, 300f32, 765f32);
        avx2.add(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 3f32, 6f32, 300f32, 765f32);
        cuda.add(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 3f32, 6f32, 300f32, 765f32);

        legacy.sub(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 1f32, 2f32, 100f32, 255f32);
        avx2.sub(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 1f32, 2f32, 100f32, 255f32);
        cuda.sub(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 1f32, 2f32, 100f32, 255f32);

        legacy.mul(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 2f32, 8f32, 20000f32, 130050f32);
        avx2.mul(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 2f32, 8f32, 20000f32, 130050f32);
        cuda.mul(&mut result, lhs, rhs);
        assert_eq_f32_array256!(result, 0f32, 2f32, 8f32, 20000f32, 130050f32);

        legacy.div(&mut result, lhs, rhs);
        assert_eq_f32!(result[1], 2f32);  assert_eq_f32!(result[2], 2f32);
        assert_eq_f32!(result[100], 2f32);  assert_eq_f32!(result[255], 2f32);
        avx2.div(&mut result, lhs, rhs);
        assert_eq_f32!(result[1], 2f32);  assert_eq_f32!(result[2], 2f32);
        assert_eq_f32!(result[100], 2f32);  assert_eq_f32!(result[255], 2f32);
        cuda.div(&mut result, lhs, rhs);
        assert_eq_f32!(result[1], 2f32);  assert_eq_f32!(result[2], 2f32);
        assert_eq_f32!(result[100], 2f32);  assert_eq_f32!(result[255], 2f32);

        println!("==== test_0001_01_arithmetic32 end ====");
    }

    #[test]
    fn test_0002_01_trigonometric() {
        println!("==== test_0002_01_trigonometric start ====");
        let legacy:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::LEGACY);
        let avx2:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::AVX2);
        let cuda:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::CUDA);

        let mut result = [0f32;BLOCK_SIZE];
        let mut lhs = [0f32;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            lhs[i] = i as f32;
        }
        legacy.sin(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 0.8414709848078965f32, 0.9092974268256817f32,
            -0.5063656411097588f32, -0.5063916349244909f32);
        avx2.sin(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 0.8414709848078965f32, 0.9092974268256817f32,
            -0.5063656411097588f32, -0.5063916349244909f32);
        cuda.sin(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 0.8414709848078965f32, 0.9092974268256817f32,
            -0.5063656411097588f32, -0.5063916349244909f32);

        legacy.cos(&mut result, lhs);
        assert_eq_f32_array256!(result, 1f32, 0.5403023058681398f32, -0.4161468365471424f32,
            0.8623188722876839f32, -0.8623036078310824f32);
        avx2.cos(&mut result, lhs);
        assert_eq_f32_array256!(result, 1f32, 0.5403023058681398f32, -0.4161468365471424f32,
            0.8623188722876839f32, -0.8623036078310824f32);
        cuda.cos(&mut result, lhs);
        assert_eq_f32_array256!(result, 1f32, 0.5403023058681398f32, -0.4161468365471424f32,
            0.8623188722876839f32, -0.8623036078310824f32);
        
        legacy.tan(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 1.5574077246549023f32, -2.185039863261519f32,
            -0.5872139151569291f32, 0.5872544546093196f32);
        avx2.tan(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 1.5574077246549023f32, -2.185039863261519f32,
            -0.5872139151569291f32, 0.5872544546093196f32);
        cuda.tan(&mut result, lhs);
        // 오차가 조금 더 있다.
        assert_eq_f32_array256_2!(result, 0f32, 1.5574077246549023f32, -2.185039863261519f32,
            -0.5872139151569291f32, 0.5872544546093196f32);

        println!("==== test_0002_01_trigonometric end ====");
    }
}
