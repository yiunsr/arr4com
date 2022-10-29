macro_rules! assert_eq_f64 {
    ($x:expr, $y:expr) => {
        if !($x.is_infinite() && $y.is_infinite() && $x.signum() == $y.signum()){
            let epsilonx:f64 = f64::EPSILON;
            if !($x - $y <= epsilonx && $y - $x <= epsilonx) { panic!(); }
        }
    }
}

macro_rules! assert_eq_f64_percent{
    ($x:expr, $y:expr) => {
        let diff = ($x - $y).abs();
        if diff > f64::EPSILON{
            let percent = diff / $x;
            let base_per = 0.000001f64;
            if !(percent <= base_per && percent <= base_per) { panic!(); }
        }
    }
}


macro_rules! assert_eq_f64_array256 {
    ($arr:expr, $y0:expr, $y1:expr, $y2:expr, $y100:expr, $y255:expr) => {
        let epsilonx:f64 = f64::EPSILON;

        if !($arr[0] - $y0 <= epsilonx && $arr[0] - $y0 <= epsilonx) { println!("index 0 error");panic!(); }
        if !($arr[1] - $y1 <= epsilonx && $arr[1] - $y1 <= epsilonx) { println!("index 1 error");panic!(); }
        if !($arr[2] - $y2 <= epsilonx && $arr[2] - $y2 <= epsilonx) { println!("index 2 error");panic!(); }
        if !($arr[100] - $y100 <= epsilonx && $arr[100] - $y100 <= epsilonx) { println!("index 100 error");panic!(); }
        if !($arr[255] - $y255 <= epsilonx && $arr[255] - $y255 <= epsilonx) { println!("index 255 error");panic!(); }
    }
}



#[cfg(test)]
mod f64_tests{
    use arr4com::arr4com::Arr4Com;
    use arr4com::arr4com::OpTarget;
    const BLOCK_SIZE: usize = 256;
    }

    #[test]
    fn test_0001_01_arithmetic64() {
        println!("==== test_0001_01_arithmetic32 start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::new(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::new(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::new(OpTarget::CUDA);

        let mut result = [0f64;BLOCK_SIZE];
        let mut lhs = [0f64;BLOCK_SIZE];
        let mut rhs = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            lhs[i] = (i as f64) * 2f64;
            rhs[i] = i as f64;
        }
        legacy.add(&mut result, lhs, rhs);
        assert_eq_f64_array256!(&result, 0f32, 3f32, 6f32, 300f32, 765f32);
        avx2.add(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 3f32, 6f32, 300f32, 765f32);
        cuda.add(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 3f32, 6f32, 300f32, 765f32);

        legacy.sub(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 1f32, 2f32, 100f32, 255f32);
        avx2.sub(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 1f32, 2f32, 100f32, 255f32);
        cuda.sub(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 1f32, 2f32, 100f32, 255f32);

        legacy.mul(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 2f32, 8f32, 20000f32, 130050f32);
        avx2.mul(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 2f32, 8f32, 20000f32, 130050f32);
        cuda.mul(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f32, 2f32, 8f32, 20000f32, 130050f32);

        legacy.div(&mut result, lhs, rhs);
        assert_eq_f64!(result[1], 2f32);  assert_eq_f64!(result[2], 2f32);
        assert_eq_f64!(result[100], 2f32);  assert_eq_f64!(result[255], 2f32);
        avx2.div(&mut result, lhs, rhs);
        assert_eq_f64!(result[1], 2f32);  assert_eq_f64!(result[2], 2f32);
        assert_eq_f64!(result[100], 2f32);  assert_eq_f64!(result[255], 2f32);
        cuda.div(&mut result, lhs, rhs);
        assert_eq_f64!(result[1], 2f32);  assert_eq_f64!(result[2], 2f32);
        assert_eq_f64!(result[100], 2f32);  assert_eq_f64!(result[255], 2f32);

        println!("==== test_0001_01_arithmetic32 end ====");
    }
