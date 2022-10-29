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
            let base_per = 0.0000001f64;
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

    #[test]
    fn test_0001_01_arithmetic64() {
        println!("==== test_0001_01_arithmetic32 start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);

        let mut result = [0f64;BLOCK_SIZE];
        let mut lhs = [0f64;BLOCK_SIZE];
        let mut rhs = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            lhs[i] = (i as f64) * 2f64;
            rhs[i] = i as f64;
        }
        legacy.add(&mut result, lhs, rhs);
        assert_eq_f64_array256!(&result, 0f64, 3f64, 6f64, 300f64, 765f64);
        avx2.add(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 3f64, 6f64, 300f64, 765f64);
        cuda.add(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 3f64, 6f64, 300f64, 765f64);

        legacy.sub(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 1f64, 2f64, 100f64, 255f64);
        avx2.sub(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 1f64, 2f64, 100f64, 255f64);
        cuda.sub(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 1f64, 2f64, 100f64, 255f64);

        legacy.mul(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 2f64, 8f64, 20000f64, 130050f64);
        avx2.mul(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 2f64, 8f64, 20000f64, 130050f64);
        cuda.mul(&mut result, lhs, rhs);
        assert_eq_f64_array256!(result, 0f64, 2f64, 8f64, 20000f64, 130050f64);

        legacy.div(&mut result, lhs, rhs);
        assert_eq_f64!(result[1], 2f64);  assert_eq_f64!(result[2], 2f64);
        assert_eq_f64!(result[100], 2f64);  assert_eq_f64!(result[255], 2f64);
        avx2.div(&mut result, lhs, rhs);
        assert_eq_f64!(result[1], 2f64);  assert_eq_f64!(result[2], 2f64);
        assert_eq_f64!(result[100], 2f64);  assert_eq_f64!(result[255], 2f64);
        cuda.div(&mut result, lhs, rhs);
        assert_eq_f64!(result[1], 2f64);  assert_eq_f64!(result[2], 2f64);
        assert_eq_f64!(result[100], 2f64);  assert_eq_f64!(result[255], 2f64);

        println!("==== test_0001_01_arithmetic32 end ====");
    }

    #[test]
    fn test_0002_01_trigonometric() {
        println!("==== test_0002_01_trigonometric start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);

        let mut result = [0f64;BLOCK_SIZE];
        let mut lhs = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            lhs[i] = i as f64;
        }

        //////// cos 
        legacy.cos(&mut result, lhs);
        assert_eq_f64_array256!(result, 1f64, 0.5403023058681398f64, -0.4161468365471424f64,
            0.8623188722876839f64, -0.8623036078310824f64);
        avx2.cos(&mut result, lhs);
        assert_eq_f64_array256!(result, 1f64, 0.5403023058681398f64, -0.4161468365471424f64,
            0.8623188722876839f64, -0.8623036078310824f64);
        cuda.cos(&mut result, lhs);
        for i in 0..6{
            //// 연산시 절대오차가 좀 크다.
            assert_eq_f64_percent!(result[i], lhs[i].cos());
        }

        //////// sin
        legacy.sin(&mut result, lhs);
        assert_eq_f64_array256!(result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        avx2.sin(&mut result, lhs);
        assert_eq_f64_array256!(result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        cuda.sin(&mut result, lhs);
        assert_eq_f64_array256!(result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        
        //////// tan
        legacy.tan(&mut result, lhs);
        assert_eq_f64_array256!(result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
        avx2.tan(&mut result, lhs);
        assert_eq_f64_array256!(result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
        cuda.tan(&mut result, lhs);
        assert_eq_f64_array256!(result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
        
        //////// acos
        lhs[0] = -1.001f64;   lhs[1] = -1.0f64;   lhs[2] = -0.9f64;   lhs[3] = -0.5f64;
        lhs[4] = -0.1f64;   lhs[5] = 0.0f64;   lhs[6] = 0.1f64;    lhs[7] = 0.5f64;
        lhs[8] = 0.9f64;    lhs[9] = 1.0f64;    lhs[10] = 1.001f64;
        legacy.acos(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(result[i], lhs[i].acos());
        }
        assert!(result[10].is_nan());

        avx2.acos(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(result[i], lhs[i].acos());
        }
        assert!(result[10].is_nan());
        
        cuda.acos(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(result[i], lhs[i].acos());
        }
        assert!(result[10].is_nan());

        //////// asin
        legacy.asin(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(result[i], lhs[i].asin());
        }
        assert!(result[10].is_nan());

        avx2.asin(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(result[i], lhs[i].asin());
        }
        assert!(result[10].is_nan());
        
        cuda.asin(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(result[i], lhs[i].asin());
        }
        assert!(result[10].is_nan());

        //////// atan
        lhs[0] = -100.0f64;   lhs[1] = -10.0f64;   lhs[2] = -0.9f64;   lhs[3] = -0.5f64;
        lhs[4] = -0.1f64;   lhs[5] = 0.0f64;   lhs[6] = 0.1f64;    lhs[7] = 0.5f64;
        lhs[8] = 0.9f64;    lhs[9] = 1.0f64;    lhs[10] = 10.001f64;
        lhs[10] = 100.0f64;
        legacy.atan(&mut result, lhs);
        for i in 0..11{
            assert_eq_f64!(result[i], lhs[i].atan());
        }

        avx2.atan(&mut result, lhs);
        for i in 0..11{
            assert_eq_f64!(result[i], lhs[i].atan());
        }
        
        cuda.atan(&mut result, lhs);
        for i in 0..11{
            assert_eq_f64!(result[i], lhs[i].atan());
        }

        //////// cosh
        lhs[0] = -50f64;   lhs[1] = -10f64;    lhs[2] = -5f64; lhs[3] = -1f64;
        lhs[4] = -0.5f64;   lhs[5] = -0.1f64;   lhs[6] = -0.01f64;  lhs[7] = 0f64;
        lhs[8] = 0.01f64;   lhs[9] = 0.1f64;    lhs[10] = 0.5f64;   lhs[11] = 1f64;
        lhs[12] = 5f64;     lhs[13] = 10f64;    lhs[14] = 50f64;
        legacy.cosh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f64!(result[i], lhs[i].cosh());
        }
        avx2.cosh(&mut result, lhs);
        for i in 0..11{
            assert_eq_f64!(result[i], lhs[i].cosh());
        }
        cuda.cosh(&mut result, lhs);
        for i in 0..11{
            //// cuda float 연산시 오차가 좀 크다.
            assert_eq_f64_percent!(result[i], lhs[i].cosh());
        }

        //////// sinh
        legacy.sinh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f64!(result[i], lhs[i].sinh());
        }
        avx2.sinh(&mut result, lhs);
        for i in 0..11{
            assert_eq_f64!(result[i], lhs[i].sinh());
        }
        cuda.sinh(&mut result, lhs);
        for i in 0..11{
            //// 연산시 절대오차가 좀 크다.
            assert_eq_f64_percent!(result[i], lhs[i].sinh());
        }

        //////// tanh
        legacy.tanh(&mut result, lhs);
        for i in 1..15{
            assert_eq_f64!(result[i], lhs[i].tanh());
        }
        avx2.tanh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f64!(result[i], lhs[i].tanh());
        }
        cuda.tanh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f64!(result[i], lhs[i].tanh());
        }

        //////// acosh
        lhs[0] = -10f64;   lhs[1] = -5f64;    lhs[2] = -1f64; lhs[3] = 0f64;
        lhs[4] = 1f64;   lhs[5] = 5f64;   lhs[6] = 10f64;   lhs[7] = 100f64;
        legacy.acosh(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_nan());    assert!(result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(result[i], lhs[i].acosh());
        }
        avx2.acosh(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_nan());    assert!(result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(result[i], lhs[i].acosh());
        }
        cuda.acosh(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_nan());    assert!(result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(result[i], lhs[i].acosh());
        }

        //////// asinh
        legacy.asinh(&mut result, lhs);
        for i in 0..8{
            assert_eq_f64!(result[i], lhs[i].asinh());
        }
        avx2.asinh(&mut result, lhs);
        for i in 0..8{
            assert_eq_f64_percent!(result[i], lhs[i].asinh());
        }
        cuda.asinh(&mut result, lhs);
        for i in 0..8{
            assert_eq_f64_percent!(result[i], lhs[i].asinh());
        }

        //////// atanh
        lhs[0] = -2f64;   lhs[1] = -1f64;    lhs[2] = -0.99f64; lhs[3] = -0.9f64;
        lhs[4] = -0.5f64;   lhs[5] = 0f64;   lhs[6] = 0.5f64;   lhs[7] = 0.9f64;
        lhs[8] = 0.99f64;   lhs[9] = 1f64;   lhs[10] = 2f64;  

        legacy.atanh(&mut result, lhs);
        assert!(result[0].is_nan());   assert!(result[1].is_infinite());
        assert!(result[9].is_infinite());   assert!(result[10].is_nan());
        for i in 2..9{
            assert_eq_f64!(result[i], lhs[i].atanh());
        }
        avx2.atanh(&mut result, lhs);
        assert!(result[0].is_nan());   assert!(result[1].is_infinite());
        assert!(result[9].is_infinite());   assert!(result[10].is_nan());
        for i in 2..9{
            assert_eq_f64_percent!(result[i], lhs[i].atanh());
        }
        cuda.atanh(&mut result, lhs);
        assert!(result[0].is_nan());   assert!(result[1].is_infinite());
        assert!(result[9].is_infinite());   assert!(result[10].is_nan());
        for i in 2..9{
            assert_eq_f64_percent!(result[i], lhs[i].atanh());
        }

        println!("==== test_0002_01_trigonometric end ====");
    }

}
