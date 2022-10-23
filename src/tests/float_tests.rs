macro_rules! assert_eq_f32 {
    ($x:expr, $y:expr) => {
        if !($x.is_infinite() && $y.is_infinite() && $x.signum() == $y.signum()){
            let epsilonx:f32 = f32::EPSILON;
            if !($x - $y <= epsilonx && $y - $x <= epsilonx) { panic!(); }
        }
    }
}

macro_rules! assert_eq_f32_percent{
    ($x:expr, $y:expr) => {
        let diff = ($x - $y).abs();
        if diff > f32::EPSILON{
            let percent = diff / $x;
            let base_per = 0.000001f32;
            if !(percent <= base_per && percent <= base_per) { panic!(); }
        }
    }
}


macro_rules! assert_eq_f32_array256 {
    ($arr:expr, $y0:expr, $y1:expr, $y2:expr, $y100:expr, $y255:expr) => {
        let epsilonx:f32 = f32::EPSILON;

        if !($arr[0] - $y0 <= epsilonx && $arr[0] - $y0 <= epsilonx) { println!("index 0 error");panic!(); }
        if !($arr[1] - $y1 <= epsilonx && $arr[1] - $y1 <= epsilonx) { println!("index 1 error");panic!(); }
        if !($arr[2] - $y2 <= epsilonx && $arr[2] - $y2 <= epsilonx) { println!("index 2 error");panic!(); }
        if !($arr[100] - $y100 <= epsilonx && $arr[100] - $y100 <= epsilonx) { println!("index 100 error");panic!(); }
        if !($arr[255] - $y255 <= epsilonx && $arr[255] - $y255 <= epsilonx) { println!("index 255 error");panic!(); }
    }
}


macro_rules! assert_eq_f32_array256_2 {
    ($arr:expr, $y0:expr, $y1:expr, $y2:expr, $y100:expr, $y255:expr) => {
        let epsilonx:f32 = f32::EPSILON * 1.001f32;

        if !($arr[0] - $y0 < epsilonx && $arr[0] - $y0 < epsilonx) { println!("index 0 error");panic!(); }
        if !($arr[1] - $y1 < epsilonx && $arr[1] - $y1 < epsilonx) { println!("index 1 error");panic!(); }
        if !($arr[2] - $y2 < epsilonx && $arr[2] - $y2 < epsilonx) { println!("index 2 error");panic!(); }
        if !($arr[100] - $y100 < epsilonx && $arr[100] - $y100 < epsilonx) { println!("index 100 error");panic!(); }
        if !($arr[255] - $y255 < epsilonx && $arr[255] - $y255 < epsilonx) { println!("index 255 error");panic!(); }
    }
}

#[cfg(test)]
mod float_tests {
    use arr4com::arr4com::Arr4Com;
    use arr4com::arr4com::OpTarget;
    const BLOCK_SIZE: usize = 256;

    pub fn assert_eq_f32_percent(x:f32, y:f32){
        let diff = (x - y).abs();
        if diff > f32::EPSILON{
            let percent = diff / x;
            let base_per = 0.000001f32;
            if percent > base_per  { println!("{}", percent); panic!(); }
        }
    }

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
        assert_eq_f32_array256!(&result, 0f32, 3f32, 6f32, 300f32, 765f32);
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

        //////// cos 
        legacy.cos(&mut result, lhs);
        assert_eq_f32_array256!(result, 1f32, 0.5403023058681398f32, -0.4161468365471424f32,
            0.8623188722876839f32, -0.8623036078310824f32);
        avx2.cos(&mut result, lhs);
        assert_eq_f32_array256!(result, 1f32, 0.5403023058681398f32, -0.4161468365471424f32,
            0.8623188722876839f32, -0.8623036078310824f32);
        cuda.cos(&mut result, lhs);
        assert_eq_f32_array256!(result, 1f32, 0.5403023058681398f32, -0.4161468365471424f32,
            0.8623188722876839f32, -0.8623036078310824f32);

        //////// sin
        legacy.sin(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 0.8414709848078965f32, 0.9092974268256817f32,
            -0.5063656411097588f32, -0.5063916349244909f32);
        avx2.sin(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 0.8414709848078965f32, 0.9092974268256817f32,
            -0.5063656411097588f32, -0.5063916349244909f32);
        cuda.sin(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 0.8414709848078965f32, 0.9092974268256817f32,
            -0.5063656411097588f32, -0.5063916349244909f32);
        
        //////// tan
        legacy.tan(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 1.5574077246549023f32, -2.185039863261519f32,
            -0.5872139151569291f32, 0.5872544546093196f32);
        avx2.tan(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 1.5574077246549023f32, -2.185039863261519f32,
            -0.5872139151569291f32, 0.5872544546093196f32);
        cuda.tan(&mut result, lhs);
        assert_eq_f32_array256!(result, 0f32, 1.5574077246549023f32, -2.185039863261519f32,
            -0.5872139151569291f32, 0.5872544546093196f32);
        
        //////// acos
        lhs[0] = -1.001f32;   lhs[1] = -1.0f32;   lhs[2] = -0.9f32;   lhs[3] = -0.5f32;
        lhs[4] = -0.1f32;   lhs[5] = 0.0f32;   lhs[6] = 0.1f32;    lhs[7] = 0.5f32;
        lhs[8] = 0.9f32;    lhs[9] = 1.0f32;    lhs[10] = 1.001f32;
        legacy.acos(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f32!(result[i], lhs[i].acos());
        }
        assert!(result[10].is_nan());

        avx2.acos(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f32!(result[i], lhs[i].acos());
        }
        assert!(result[10].is_nan());
        
        cuda.acos(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f32!(result[i], lhs[i].acos());
        }
        assert!(result[10].is_nan());

        //////// asin
        legacy.asin(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f32!(result[i], lhs[i].asin());
        }
        assert!(result[10].is_nan());

        avx2.asin(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f32!(result[i], lhs[i].asin());
        }
        assert!(result[10].is_nan());
        
        cuda.asin(&mut result, lhs);
        assert!(result[0].is_nan());
        for i in 1..9{
            assert_eq_f32!(result[i], lhs[i].asin());
        }
        assert!(result[10].is_nan());

        //////// atan
        lhs[0] = -100.0f32;   lhs[1] = -10.0f32;   lhs[2] = -0.9f32;   lhs[3] = -0.5f32;
        lhs[4] = -0.1f32;   lhs[5] = 0.0f32;   lhs[6] = 0.1f32;    lhs[7] = 0.5f32;
        lhs[8] = 0.9f32;    lhs[9] = 1.0f32;    lhs[10] = 10.001f32;
        lhs[10] = 100.0f32;
        legacy.atan(&mut result, lhs);
        for i in 0..11{
            assert_eq_f32!(result[i], lhs[i].atan());
        }

        avx2.atan(&mut result, lhs);
        for i in 0..11{
            assert_eq_f32!(result[i], lhs[i].atan());
        }
        
        cuda.atan(&mut result, lhs);
        for i in 0..11{
            assert_eq_f32!(result[i], lhs[i].atan());
        }

        //////// cosh
        lhs[0] = -50f32;   lhs[1] = -10f32;    lhs[2] = -5f32; lhs[3] = -1f32;
        lhs[4] = -0.5f32;   lhs[5] = -0.1f32;   lhs[6] = -0.01f32;  lhs[7] = 0f32;
        lhs[8] = 0.01f32;   lhs[9] = 0.1f32;    lhs[10] = 0.5f32;   lhs[11] = 1f32;
        lhs[12] = 5f32;     lhs[13] = 10f32;    lhs[14] = 50f32;
        legacy.cosh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f32!(result[i], lhs[i].cosh());
        }
        avx2.cosh(&mut result, lhs);
        for i in 0..11{
            assert_eq_f32!(result[i], lhs[i].cosh());
        }
        cuda.cosh(&mut result, lhs);
        for i in 0..11{
            //// cuda float 연산시 오차가 좀 크다.
            assert_eq_f32_percent!(result[i], lhs[i].cosh());
        }

        //////// sinh
        legacy.sinh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f32!(result[i], lhs[i].sinh());
        }
        avx2.sinh(&mut result, lhs);
        for i in 0..11{
            assert_eq_f32!(result[i], lhs[i].sinh());
        }
        cuda.sinh(&mut result, lhs);
        for i in 0..11{
            //// 연산시 절대오차가 좀 크다.
            assert_eq_f32_percent!(result[i], lhs[i].sinh());
        }

        //////// tanh
        legacy.tanh(&mut result, lhs);
        for i in 1..15{
            assert_eq_f32!(result[i], lhs[i].tanh());
        }
        avx2.tanh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f32!(result[i], lhs[i].tanh());
        }
        cuda.tanh(&mut result, lhs);
        for i in 0..15{
            assert_eq_f32!(result[i], lhs[i].tanh());
        }

        //////// acosh
        lhs[0] = -10f32;   lhs[1] = -5f32;    lhs[2] = -1f32; lhs[3] = 0f32;
        lhs[4] = 1f32;   lhs[5] = 5f32;   lhs[6] = 10f32;   lhs[7] = 100f32;
        legacy.acosh(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_nan());    assert!(result[3].is_nan());
        for i in 4..8{
            assert_eq_f32!(result[i], lhs[i].acosh());
        }
        avx2.acosh(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_nan());    assert!(result[3].is_nan());
        for i in 4..8{
            assert_eq_f32!(result[i], lhs[i].acosh());
        }
        cuda.acosh(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_nan());    assert!(result[3].is_nan());
        for i in 4..8{
            assert_eq_f32!(result[i], lhs[i].acosh());
        }

        //////// asinh
        legacy.asinh(&mut result, lhs);
        for i in 0..8{
            assert_eq_f32!(result[i], lhs[i].asinh());
        }
        avx2.asinh(&mut result, lhs);
        for i in 0..8{
            assert_eq_f32_percent!(result[i], lhs[i].asinh());
        }
        cuda.asinh(&mut result, lhs);
        for i in 0..8{
            assert_eq_f32_percent!(result[i], lhs[i].asinh());
        }

        //////// atanh
        lhs[0] = -2f32;   lhs[1] = -1f32;    lhs[2] = -0.99f32; lhs[3] = -0.9f32;
        lhs[4] = -0.5f32;   lhs[5] = 0f32;   lhs[6] = 0.5f32;   lhs[7] = 0.9f32;
        lhs[8] = 0.99f32;   lhs[9] = 1f32;   lhs[10] = 2f32;  

        legacy.atanh(&mut result, lhs);
        assert!(result[0].is_nan());   assert!(result[1].is_infinite());
        assert!(result[9].is_infinite());   assert!(result[10].is_nan());
        for i in 2..9{
            assert_eq_f32!(result[i], lhs[i].atanh());
        }
        avx2.atanh(&mut result, lhs);
        assert!(result[0].is_nan());   assert!(result[1].is_infinite());
        assert!(result[9].is_infinite());   assert!(result[10].is_nan());
        for i in 2..9{
            assert_eq_f32_percent!(result[i], lhs[i].atanh());
        }
        cuda.atanh(&mut result, lhs);
        assert!(result[0].is_nan());   assert!(result[1].is_infinite());
        assert!(result[9].is_infinite());   assert!(result[10].is_nan());
        for i in 2..9{
            assert_eq_f32_percent!(result[i], lhs[i].atanh());
        }

        println!("==== test_0002_01_trigonometric end ====");
    }

    #[test]
    fn test_0003_01_math() {
        println!("==== test_0003_01_math start ====");
        let legacy:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::LEGACY);
        let avx2:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::AVX2);
        let cuda:Arr4Com<f32, BLOCK_SIZE> = Arr4Com::new(OpTarget::CUDA);

        let mut result = [0f32;BLOCK_SIZE];
        let mut lhs = [0f32;BLOCK_SIZE];
        let mut rhs = [0f32;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            lhs[i] = -2.0f32 +  i as f32;
            rhs[i] = i as f32;
        }

        //////// ln
        legacy.ln(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].ln());
        }
        avx2.ln(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].ln());
        }
        cuda.ln(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].ln());
        }

        //////// ln_1p    ln(x + 1)
        legacy.ln_1p(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_infinite());
        for i in 2..20{
            assert_eq_f32!(result[i], lhs[i].ln_1p());
        }
        avx2.ln_1p(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_infinite());
        for i in 2..20{
            assert_eq_f32!(result[i], lhs[i].ln_1p());
        }
        cuda.ln_1p(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_infinite());
        for i in 2..20{
            assert_eq_f32!(result[i], lhs[i].ln_1p());
        }

        //////// log10
        legacy.log10(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].log10());
        }
        avx2.log10(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].log10());
        }
        cuda.log10(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].log10());
        }

        //////// log2
        legacy.log2(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].log2());
        }
        avx2.log2(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32_percent!(result[i], lhs[i].log2());
        }
        cuda.log2(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        assert!(result[2].is_infinite());
        for i in 3..20{
            assert_eq_f32!(result[i], lhs[i].log2());
        }

        //////// exp
        legacy.exp(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp());
        }
        avx2.exp(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32_percent!(result[i], lhs[i].exp());
        }
        cuda.exp(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp());
        }

        //////// exp2
        legacy.exp2(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp2());
        }
        avx2.exp2(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp2());
        }
        cuda.exp2(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp2());
        }

        //////// exp_m1
        legacy.exp_m1(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp_m1());
        }
        avx2.exp_m1(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32!(result[i], lhs[i].exp_m1());
        }
        cuda.exp_m1(&mut result, lhs);
        for i in 1..10{
            assert_eq_f32_percent!(result[i], lhs[i].exp_m1());
        }

        for i in 0..BLOCK_SIZE{
            lhs[i] = -2.0f32 +  i as f32;
            rhs[i] = i as f32;
        }
        //////// sqrt
        legacy.sqrt(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        for i in 2..10{
            assert_eq_f32!(result[i], lhs[i].sqrt());
        }
        avx2.sqrt(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        for i in 2..10{
            assert_eq_f32!(result[i], lhs[i].sqrt());
        }
        cuda.sqrt(&mut result, lhs);
        assert!(result[0].is_nan());    assert!(result[1].is_nan());
        for i in 2..10{
            assert_eq_f32!(result[i], lhs[i].sqrt());
        }
        //////// cbrt
        legacy.cbrt(&mut result, lhs);
        for i in 0..10{
            assert_eq_f32!(result[i], lhs[i].cbrt());
        }
        avx2.cbrt(&mut result, lhs);
        for i in 0..10{
            assert_eq_f32!(result[i], lhs[i].cbrt());
        }
        cuda.cbrt(&mut result, lhs);
        for i in 0..10{
            assert_eq_f32!(result[i], lhs[i].cbrt());
        }

        //////// pow
        for i in 0..BLOCK_SIZE{
            lhs[i] = -4.0f32 +  i as f32;
            rhs[i] = -8.0f32 + i as f32;
        }
        legacy.powf(&mut result, lhs, rhs);
        for i in 0..12{
            assert_eq_f32!(result[i], lhs[i].powf(rhs[i]));
        }
        avx2.powf(&mut result, lhs, rhs);
        for i in 0..12{
            assert_eq_f32!(result[i], lhs[i].powf(rhs[i]));
        }
        cuda.powf(&mut result, lhs, rhs);
        for i in 0..12{
            assert_eq_f32!(result[i], lhs[i].powf(rhs[i]));
        }

        for i in 0..BLOCK_SIZE{
            lhs[i] = -4.0f32 +  i as f32;
            rhs[i] = -8.0f32 + i as f32;
        }
        legacy.hypot(&mut result, lhs, rhs);
        for i in 0..12{
            assert_eq_f32!(result[i], lhs[i].hypot(rhs[i]));
        }
        avx2.hypot(&mut result, lhs, rhs);
        for i in 0..12{
            assert_eq_f32!(result[i], lhs[i].hypot(rhs[i]));
        }
        cuda.hypot(&mut result, lhs, rhs);
        for i in 0..12{
            assert_eq_f32!(result[i], lhs[i].hypot(rhs[i]));
        }

        println!("==== test_0003_01_math end ====");
    }
}
