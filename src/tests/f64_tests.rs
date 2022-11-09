macro_rules! assert_eq_f64 {
    ($x:expr, $y:expr) => {
        if !($x.is_infinite() && $y.is_infinite() && $x.signum() == $y.signum()){
            let epsilonx:f64 = f64::EPSILON;
            if !($x - $y <= epsilonx && $y - $x <= epsilonx) { 
                println!("x = {}, y = {}", $x, $y);
                panic!(); }
        }
    }
}

macro_rules! assert_eq_f64_percent{
    ($x:expr, $y:expr) => {
        let diff = ($x - $y).abs();
        if diff > f64::EPSILON{
            let percent = diff / $x;
            let base_per = 0.0000001f64;
            if !(percent <= base_per && percent <= base_per) { 
                println!("x = {}, y = {}", $x, $y);
                panic!(); 
            }
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
        let opencl:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::OPENCL);

        let mut legacy_result = [0f64;BLOCK_SIZE];
        let mut avx2_result = [0f64;BLOCK_SIZE];
        let mut cuda_result = [0f64;BLOCK_SIZE];
        let mut opencl_result = [0f64;BLOCK_SIZE];
        let mut opr1 = [0f64;BLOCK_SIZE];
        let mut opr2 = [0f64;BLOCK_SIZE];
        let mut opr3 = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            opr1[i] = (i as f64) * 2f64;
            opr2[i] = i as f64;
            opr3[i] = (i as f64) * 0.2f64;
        }
        legacy.add(&mut legacy_result, opr1, opr2);
        assert_eq_f64_array256!(&legacy_result, 0f64, 3f64, 6f64, 300f64, 765f64);
        avx2.add(&mut avx2_result, opr1, opr2);
        assert_eq_f64_array256!(avx2_result, 0f64, 3f64, 6f64, 300f64, 765f64);
        cuda.add(&mut cuda_result, opr1, opr2);
        assert_eq_f64_array256!(cuda_result, 0f64, 3f64, 6f64, 300f64, 765f64);
        opencl.add(&mut opencl_result, opr1, opr2);
        assert_eq_f64_array256!(opencl_result, 0f64, 3f64, 6f64, 300f64, 765f64);

        legacy.sub(&mut legacy_result, opr1, opr2);
        assert_eq_f64_array256!(legacy_result, 0f64, 1f64, 2f64, 100f64, 255f64);
        avx2.sub(&mut avx2_result, opr1, opr2);
        assert_eq_f64_array256!(avx2_result, 0f64, 1f64, 2f64, 100f64, 255f64);
        cuda.sub(&mut cuda_result, opr1, opr2);
        assert_eq_f64_array256!(cuda_result, 0f64, 1f64, 2f64, 100f64, 255f64);
        opencl.sub(&mut opencl_result, opr1, opr2);
        assert_eq_f64_array256!(opencl_result, 0f64, 1f64, 2f64, 100f64, 255f64);

        legacy.mul(&mut legacy_result, opr1, opr2);
        assert_eq_f64_array256!(legacy_result, 0f64, 2f64, 8f64, 20000f64, 130050f64);
        avx2.mul(&mut avx2_result, opr1, opr2);
        assert_eq_f64_array256!(avx2_result, 0f64, 2f64, 8f64, 20000f64, 130050f64);
        cuda.mul(&mut cuda_result, opr1, opr2);
        assert_eq_f64_array256!(cuda_result, 0f64, 2f64, 8f64, 20000f64, 130050f64);
        opencl.mul(&mut opencl_result, opr1, opr2);
        assert_eq_f64_array256!(opencl_result, 0f64, 2f64, 8f64, 20000f64, 130050f64);

        legacy.div(&mut legacy_result, opr1, opr2);
        assert_eq_f64!(legacy_result[1], 2f64);  assert_eq_f64!(legacy_result[2], 2f64);
        assert_eq_f64!(legacy_result[100], 2f64);  assert_eq_f64!(legacy_result[255], 2f64);
        avx2.div(&mut avx2_result, opr1, opr2);
        assert_eq_f64!(avx2_result[1], 2f64);  assert_eq_f64!(avx2_result[2], 2f64);
        assert_eq_f64!(avx2_result[100], 2f64);  assert_eq_f64!(avx2_result[255], 2f64);
        cuda.div(&mut cuda_result, opr1, opr2);
        assert_eq_f64!(cuda_result[1], 2f64);  assert_eq_f64!(cuda_result[2], 2f64);
        assert_eq_f64!(cuda_result[100], 2f64);  assert_eq_f64!(cuda_result[255], 2f64);
        opencl.div(&mut opencl_result, opr1, opr2);
        assert_eq_f64!(opencl_result[1], 2f64);  assert_eq_f64!(opencl_result[2], 2f64);
        assert_eq_f64!(opencl_result[100], 2f64);  assert_eq_f64!(opencl_result[255], 2f64);

        legacy.mul_add(&mut legacy_result, opr1, opr2, opr3);
        assert_eq_f64!(legacy_result[1], 2.2f64);  assert_eq_f64!(legacy_result[2], 8.4f64);
        assert_eq_f64!(legacy_result[100], 20020f64);  assert_eq_f64!(legacy_result[255], 130101f64);
        avx2.mul_add(&mut avx2_result, opr1, opr2, opr3);
        assert_eq_f64!(avx2_result[1], 2.2f64);  assert_eq_f64!(avx2_result[2], 8.4f64);
        assert_eq_f64!(avx2_result[100], 20020f64);  assert_eq_f64!(avx2_result[255], 130101f64);
        cuda.mul_add(&mut cuda_result, opr1, opr2, opr3);
        assert_eq_f64!(cuda_result[1], 2.2f64);  assert_eq_f64!(cuda_result[2], 8.4f64);
        assert_eq_f64!(cuda_result[100], 20020f64);  assert_eq_f64!(cuda_result[255], 130101f64);
        opencl.mul_add(&mut opencl_result, opr1, opr2, opr3);
        assert_eq_f64!(opencl_result[1], 2.2f64);  assert_eq_f64!(opencl_result[2], 8.4f64);
        assert_eq_f64!(opencl_result[100], 20020f64);  assert_eq_f64!(opencl_result[255], 130101f64);

        println!("==== test_0001_01_arithmetic32 end ====");
    }

    #[test]
    fn test_0001_02_round() {
        println!("==== test_0001_02_round start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);
        let opencl:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::OPENCL);

        let mut legacy_result = [0f64;BLOCK_SIZE];
        let mut avx2_result = [0f64;BLOCK_SIZE];
        let mut cuda_result = [0f64;BLOCK_SIZE];
        let mut opencl_result = [0f64;BLOCK_SIZE];
        let mut opr1 = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            opr1[i] = (i as f64) * 2f64;
        }
        opr1[0] = -1.0099f64;           opr1[1] = -1.5001f64;
        opr1[2] = -1.5000f64;           opr1[3] = -1.4999f64;
        opr1[4] = -1.0001f64;           opr1[5] = -0.0008f64;
        opr1[6] = -0.00001f64;          opr1[7] = -0.0f64;
        opr1[8] = -0.00001f64;          opr1[9] = 0.00001f64;
        opr1[10] = 0.49999f64;          opr1[11] = 0.5000f64;
        opr1[12] = 0.50001f64;          opr1[13] = 0.99999f64;
        opr1[14] = 1f64;                opr1[15] = 1.00001f64;

        
        //////// ceil 
        legacy.ceil(&mut legacy_result, opr1);
        for i in 1..25{
            assert_eq_f64!(legacy_result[i], opr1[i].ceil());
        }
        avx2.ceil(&mut avx2_result, opr1);
        for i in 1..25{
            assert_eq_f64!(avx2_result[i], opr1[i].ceil());
        }
        cuda.ceil(&mut cuda_result, opr1);
        for i in 1..25{
            assert_eq_f64!(cuda_result[i], opr1[i].ceil());
        }
        opencl.ceil(&mut opencl_result, opr1);
        for i in 1..25{
            assert_eq_f64!(opencl_result[i], opr1[i].ceil());
        }

        //////// floor 
        legacy.floor(&mut legacy_result, opr1);
        for i in 1..25{
            assert_eq_f64!(legacy_result[i], opr1[i].floor());
        }
        avx2.floor(&mut avx2_result, opr1);
        for i in 1..25{
            assert_eq_f64!(avx2_result[i], opr1[i].floor());
        }
        cuda.floor(&mut cuda_result, opr1);
        for i in 1..25{
            assert_eq_f64!(cuda_result[i], opr1[i].floor());
        }
        opencl.floor(&mut opencl_result, opr1);
        for i in 1..25{
            assert_eq_f64!(opencl_result[i], opr1[i].floor());
        }

        //////// round 
        legacy.round(&mut legacy_result, opr1);
        for i in 1..25{
            assert_eq_f64!(legacy_result[i], opr1[i].round());
        }
        avx2.round(&mut avx2_result, opr1);
        for i in 1..25{
            assert_eq_f64!(avx2_result[i], opr1[i].round());
        }
        cuda.round(&mut cuda_result, opr1);
        for i in 1..25{
            assert_eq_f64!(cuda_result[i], opr1[i].round());
        }
        opencl.round(&mut opencl_result, opr1);
        for i in 1..25{
            assert_eq_f64!(opencl_result[i], opr1[i].round());
        }

        //////// trunc 
        legacy.trunc(&mut legacy_result, opr1);
        for i in 1..25{
            assert_eq_f64!(legacy_result[i], opr1[i].trunc());
        }
        avx2.trunc(&mut avx2_result, opr1);
        for i in 1..25{
            assert_eq_f64!(avx2_result[i], opr1[i].trunc());
        }
        cuda.trunc(&mut cuda_result, opr1);
        for i in 1..25{
            assert_eq_f64!(cuda_result[i], opr1[i].trunc());
        }
        opencl.trunc(&mut opencl_result, opr1);
        for i in 1..25{
            assert_eq_f64!(opencl_result[i], opr1[i].trunc());
        }

        println!("==== test_0001_02_round end ====");
    }

    #[test]
    fn test_0001_03_cmp() {
        println!("==== test_0001_03_cmp start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);
        let opencl:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::OPENCL);

        let mut result_legacy = [0f64;BLOCK_SIZE];
        let mut result_avx2 = [0f64;BLOCK_SIZE];
        let mut result_cuda = [0f64;BLOCK_SIZE];
        let mut result_opencl = [0f64;BLOCK_SIZE];
        let mut opr1 = [0f64;BLOCK_SIZE];
        let mut opr2 = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            opr1[i] = i as f64;
            opr2[i] = i as f64;
        }
        
        opr1[0] = -100.0f64;   opr1[1] = -10.0f64;   opr1[2] = -0.9f64;   opr1[3] = -0.5f64;
        opr1[4] = -0.1f64;   opr1[5] = 0.0f64;   opr1[6] = 0.0f64;    opr1[7] = -0.1f64;
        opr1[8] = 0.9f64;    opr1[9] = 1.0f64;    opr1[10] = 10.001f64;
        opr1[10] = 100.0f64;

        opr2[0] = -100.0f64;   opr2[1] = -10.0f64;   opr2[2] = -0.98f64;   opr2[3] = -0.49f64;
        opr2[4] = -0.11f64;   opr2[5] = 0.0f64;   opr2[6] = 0.1f64;    opr2[7] = 0.0f64;
        opr2[8] = 0.9f64;    opr2[9] = -1.0f64;    opr2[10] = 10.002f64;
        opr2[10] = 99.999f64;

        legacy.gtf(&mut result_legacy, opr1, opr2);
        avx2.gtf(&mut result_avx2, opr1, opr2);
        cuda.gtf(&mut result_cuda, opr1, opr2);
        opencl.gtf(&mut result_opencl, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_avx2[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_cuda[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_opencl[i]);
        }

        legacy.gtef(&mut result_legacy, opr1, opr2);
        avx2.gtef(&mut result_avx2, opr1, opr2);
        cuda.gtef(&mut result_cuda, opr1, opr2);
        opencl.gtef(&mut result_opencl, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_avx2[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_cuda[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_opencl[i]);
        }

        legacy.ltf(&mut result_legacy, opr1, opr2);
        avx2.ltf(&mut result_avx2, opr1, opr2);
        cuda.ltf(&mut result_cuda, opr1, opr2);
        opencl.ltf(&mut result_opencl, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_avx2[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_cuda[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_opencl[i]);
        }

        legacy.ltef(&mut result_legacy, opr1, opr2);
        avx2.ltef(&mut result_avx2, opr1, opr2);
        cuda.ltef(&mut result_cuda, opr1, opr2);
        opencl.ltef(&mut result_opencl, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_avx2[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_cuda[i]);
        }
        for i in 0..20{
            assert_eq_f64!(result_legacy[i], result_opencl[i]);
        }
        println!("==== test_0001_03_cmp end ====");
    }

    #[test]
    fn test_0002_01_trigonometric() {
        println!("==== test_0002_01_trigonometric start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);
        let opencl:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::OPENCL);

        let mut legacy_result = [0f64;BLOCK_SIZE];
        let mut avx2_result = [0f64;BLOCK_SIZE];
        let mut cuda_result = [0f64;BLOCK_SIZE];
        let mut opencl_result = [0f64;BLOCK_SIZE];
        let mut opr1 = [0f64;BLOCK_SIZE];
        let mut opr2 = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            opr1[i] = i as f64;
            opr2[i] = i as f64;
        }

        //////// cos 
        legacy.cos(&mut legacy_result, opr1);
        assert_eq_f64_array256!(legacy_result, 1f64, 0.5403023058681398f64, -0.4161468365471424f64,
            0.8623188722876839f64, -0.8623036078310824f64);
        avx2.cos(&mut avx2_result, opr1);
        assert_eq_f64_array256!(avx2_result, 1f64, 0.5403023058681398f64, -0.4161468365471424f64,
            0.8623188722876839f64, -0.8623036078310824f64);
        cuda.cos(&mut cuda_result, opr1);
        for i in 0..6{
            //// 연산시 절대오차가 좀 크다.
            assert_eq_f64_percent!(cuda_result[i], opr1[i].cos());
        }
        opencl.cos(&mut opencl_result, opr1);
        assert_eq_f64_array256!(opencl_result, 1f64, 0.5403023058681398f64, -0.4161468365471424f64,
            0.8623188722876839f64, -0.8623036078310824f64);

        //////// sin
        legacy.sin(&mut legacy_result, opr1);
        assert_eq_f64_array256!(legacy_result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        avx2.sin(&mut avx2_result, opr1);
        assert_eq_f64_array256!(avx2_result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        cuda.sin(&mut cuda_result, opr1);
        assert_eq_f64_array256!(cuda_result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        opencl.sin(&mut opencl_result, opr1);
         assert_eq_f64_array256!(opencl_result, 0f64, 0.8414709848078965f64, 0.9092974268256817f64,
            -0.5063656411097588f64, -0.5063916349244909f64);
        
        //////// tan
        legacy.tan(&mut legacy_result, opr1);
        assert_eq_f64_array256!(legacy_result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
        avx2.tan(&mut avx2_result, opr1);
        assert_eq_f64_array256!(avx2_result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
        cuda.tan(&mut cuda_result, opr1);
        assert_eq_f64_array256!(cuda_result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
            opencl.tan(&mut opencl_result, opr1);
        assert_eq_f64_array256!(opencl_result, 0f64, 1.5574077246549023f64, -2.185039863261519f64,
            -0.5872139151569291f64, 0.5872544546093196f64);
        
        //////// acos
        opr1[0] = -1.001f64;   opr1[1] = -1.0f64;   opr1[2] = -0.9f64;   opr1[3] = -0.5f64;
        opr1[4] = -0.1f64;   opr1[5] = 0.0f64;   opr1[6] = 0.1f64;    opr1[7] = 0.5f64;
        opr1[8] = 0.9f64;    opr1[9] = 1.0f64;    opr1[10] = 1.001f64;
        legacy.acos(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(legacy_result[i], opr1[i].acos());
        }
        assert!(legacy_result[10].is_nan());

        avx2.acos(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(avx2_result[i], opr1[i].acos());
        }
        assert!(avx2_result[10].is_nan());
        
        cuda.acos(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(cuda_result[i], opr1[i].acos());
        }
        assert!(cuda_result[10].is_nan());
        opencl.acos(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(opencl_result[i], opr1[i].acos());
        }
        assert!(opencl_result[10].is_nan());

        //////// asin
        legacy.asin(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(legacy_result[i], opr1[i].asin());
        }
        assert!(legacy_result[10].is_nan());

        avx2.asin(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(avx2_result[i], opr1[i].asin());
        }
        assert!(avx2_result[10].is_nan());
        
        cuda.asin(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(cuda_result[i], opr1[i].asin());
        }
        assert!(cuda_result[10].is_nan());

        opencl.asin(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());
        for i in 1..9{
            assert_eq_f64!(opencl_result[i], opr1[i].asin());
        }
        assert!(opencl_result[10].is_nan());

        //////// atan
        opr1[0] = -100.0f64;   opr1[1] = -10.0f64;   opr1[2] = -0.9f64;   opr1[3] = -0.5f64;
        opr1[4] = -0.1f64;   opr1[5] = 0.0f64;   opr1[6] = 0.1f64;    opr1[7] = 0.5f64;
        opr1[8] = 0.9f64;    opr1[9] = 1.0f64;    opr1[10] = 10.001f64;
        opr1[10] = 100.0f64;
        legacy.atan(&mut legacy_result, opr1);
        for i in 0..11{
            assert_eq_f64!(legacy_result[i], opr1[i].atan());
        }

        avx2.atan(&mut avx2_result, opr1);
        for i in 0..11{
            assert_eq_f64!(avx2_result[i], opr1[i].atan());
        }
        
        cuda.atan(&mut cuda_result, opr1);
        for i in 0..11{
            assert_eq_f64!(cuda_result[i], opr1[i].atan());
        }

        opencl.atan(&mut opencl_result, opr1);
        for i in 0..11{
            assert_eq_f64!(opencl_result[i], opr1[i].atan());
        }

        //////// atan2
        opr1[0] = -100.0f64;   opr1[1] = -10.0f64;   opr1[2] = -0.9f64;   opr1[3] = -0.5f64;
        opr1[4] = -0.1f64;   opr1[5] = 0.0f64;   opr1[6] = 0.1f64;    opr1[7] = 0.5f64;
        opr1[8] = 0.9f64;    opr1[9] = 1.0f64;    opr1[10] = 10.001f64;
        opr1[10] = 100.0f64;

        opr2[0] = 50.0f64;   opr2[1] = 25.0f64;   opr2[2] = -10f64;   opr2[3] = 1f64;
        opr2[4] = 0.01f64;   opr2[5] = 0.01f64;   opr2[6] = -0.01f64;    opr2[7] = 0.2f64;
        opr2[8] = -1f64;    opr2[9] = 10f64;    opr2[10] = 10.001f64;
        opr2[10] = 50f64;
        legacy.atan2(&mut legacy_result, opr1, opr2);
        for i in 0..11{
            assert_eq_f64!(legacy_result[i], opr1[i].atan2(opr2[i]));
        }

        avx2.atan2(&mut avx2_result, opr1, opr2);
        for i in 0..11{
            assert_eq_f64!(avx2_result[i], opr1[i].atan2(opr2[i]));
        }
        
        cuda.atan2(&mut cuda_result, opr1, opr2);
        for i in 0..11{
            assert_eq_f64_percent!(cuda_result[i], opr1[i].atan2(opr2[i]));
        }

        opencl.atan2(&mut opencl_result, opr1, opr2);
        for i in 0..11{
            assert_eq_f64_percent!(opencl_result[i], opr1[i].atan2(opr2[i]));
        }

        //////// cosh
        opr1[0] = -50f64;   opr1[1] = -10f64;    opr1[2] = -5f64; opr1[3] = -1f64;
        opr1[4] = -0.5f64;   opr1[5] = -0.1f64;   opr1[6] = -0.01f64;  opr1[7] = 0f64;
        opr1[8] = 0.01f64;   opr1[9] = 0.1f64;    opr1[10] = 0.5f64;   opr1[11] = 1f64;
        opr1[12] = 5f64;     opr1[13] = 10f64;    opr1[14] = 50f64;
        legacy.cosh(&mut legacy_result, opr1);
        for i in 0..15{
            assert_eq_f64!(legacy_result[i], opr1[i].cosh());
        }
        avx2.cosh(&mut avx2_result, opr1);
        for i in 0..11{
            assert_eq_f64!(avx2_result[i], opr1[i].cosh());
        }
        cuda.cosh(&mut cuda_result, opr1);
        for i in 0..11{
            //// cuda float 연산시 오차가 좀 크다.
            assert_eq_f64_percent!(cuda_result[i], opr1[i].cosh());
        }
        opencl.cosh(&mut opencl_result, opr1);
        for i in 0..11{
            assert_eq_f64!(opencl_result[i], opr1[i].cosh());
        }

        //////// sinh
        legacy.sinh(&mut legacy_result, opr1);
        for i in 0..15{
            assert_eq_f64!(legacy_result[i], opr1[i].sinh());
        }
        avx2.sinh(&mut avx2_result, opr1);
        for i in 0..11{
            assert_eq_f64!(avx2_result[i], opr1[i].sinh());
        }
        cuda.sinh(&mut cuda_result, opr1);
        for i in 0..11{
            //// 연산시 절대오차가 좀 크다.
            assert_eq_f64_percent!(cuda_result[i], opr1[i].sinh());
        }
        opencl.sinh(&mut opencl_result, opr1);
        for i in 0..11{
            assert_eq_f64!(opencl_result[i], opr1[i].sinh());
        }

        //////// tanh
        legacy.tanh(&mut legacy_result, opr1);
        for i in 1..15{
            assert_eq_f64!(legacy_result[i], opr1[i].tanh());
        }
        avx2.tanh(&mut avx2_result, opr1);
        for i in 0..15{
            assert_eq_f64!(avx2_result[i], opr1[i].tanh());
        }
        cuda.tanh(&mut cuda_result, opr1);
        for i in 0..15{
            assert_eq_f64!(cuda_result[i], opr1[i].tanh());
        }
        opencl.tanh(&mut opencl_result, opr1);
        for i in 0..15{
            assert_eq_f64!(opencl_result[i], opr1[i].tanh());
        }

        //////// acosh
        opr1[0] = -10f64;   opr1[1] = -5f64;    opr1[2] = -1f64; opr1[3] = 0f64;
        opr1[4] = 1f64;   opr1[5] = 5f64;   opr1[6] = 10f64;   opr1[7] = 100f64;
        legacy.acosh(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());    assert!(legacy_result[1].is_nan());
        assert!(legacy_result[2].is_nan());    assert!(legacy_result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(legacy_result[i], opr1[i].acosh());
        }
        avx2.acosh(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());    assert!(avx2_result[1].is_nan());
        assert!(avx2_result[2].is_nan());    assert!(avx2_result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(avx2_result[i], opr1[i].acosh());
        }
        cuda.acosh(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());    assert!(cuda_result[1].is_nan());
        assert!(cuda_result[2].is_nan());    assert!(cuda_result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(cuda_result[i], opr1[i].acosh());
        }
        opencl.acosh(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());    assert!(opencl_result[1].is_nan());
        assert!(opencl_result[2].is_nan());    assert!(opencl_result[3].is_nan());
        for i in 4..8{
            assert_eq_f64!(opencl_result[i], opr1[i].acosh());
        }

        //////// asinh
        legacy.asinh(&mut legacy_result, opr1);
        for i in 0..8{
            assert_eq_f64!(legacy_result[i], opr1[i].asinh());
        }
        avx2.asinh(&mut avx2_result, opr1);
        for i in 0..8{
            assert_eq_f64_percent!(avx2_result[i], opr1[i].asinh());
        }
        cuda.asinh(&mut cuda_result, opr1);
        for i in 0..8{
            assert_eq_f64_percent!(cuda_result[i], opr1[i].asinh());
        }
        opencl.asinh(&mut opencl_result, opr1);
        for i in 0..8{
            assert_eq_f64_percent!(opencl_result[i], opr1[i].asinh());
        }

        //////// atanh
        opr1[0] = -2f64;   opr1[1] = -1f64;    opr1[2] = -0.99f64; opr1[3] = -0.9f64;
        opr1[4] = -0.5f64;   opr1[5] = 0f64;   opr1[6] = 0.5f64;   opr1[7] = 0.9f64;
        opr1[8] = 0.99f64;   opr1[9] = 1f64;   opr1[10] = 2f64;  

        legacy.atanh(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());   assert!(legacy_result[1].is_infinite());
        assert!(legacy_result[9].is_infinite());   assert!(legacy_result[10].is_nan());
        for i in 2..9{
            assert_eq_f64!(legacy_result[i], opr1[i].atanh());
        }
        avx2.atanh(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());   assert!(avx2_result[1].is_infinite());
        assert!(avx2_result[9].is_infinite());   assert!(avx2_result[10].is_nan());
        for i in 2..9{
            assert_eq_f64_percent!(avx2_result[i], opr1[i].atanh());
        }
        cuda.atanh(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());   assert!(cuda_result[1].is_infinite());
        assert!(cuda_result[9].is_infinite());   assert!(cuda_result[10].is_nan());
        for i in 2..9{
            assert_eq_f64_percent!(cuda_result[i], opr1[i].atanh());
        }
        opencl.atanh(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());   assert!(opencl_result[1].is_infinite());
        assert!(opencl_result[9].is_infinite());   assert!(opencl_result[10].is_nan());
        for i in 2..9{
            assert_eq_f64_percent!(opencl_result[i], opr1[i].atanh());
        }

        println!("==== test_0002_01_trigonometric end ====");
    }

    #[test]
    fn test_0003_01_math() {
        println!("==== test_0003_01_math start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);
        let opencl:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::OPENCL);

        let mut legacy_result = [0f64;BLOCK_SIZE];
        let mut avx2_result = [0f64;BLOCK_SIZE];
        let mut cuda_result = [0f64;BLOCK_SIZE];
        let mut opencl_result = [0f64;BLOCK_SIZE];
        let mut opr1 = [0f64;BLOCK_SIZE];
        let mut opr2 = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            opr1[i] = -2.0f64 +  i as f64;
            opr2[i] = i as f64;
        }

        //////// ln
        legacy.ln(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());    assert!(legacy_result[1].is_nan());
        assert!(legacy_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(legacy_result[i], opr1[i].ln());
        }
        avx2.ln(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());    assert!(avx2_result[1].is_nan());
        assert!(avx2_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(avx2_result[i], opr1[i].ln());
        }
        cuda.ln(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());    assert!(cuda_result[1].is_nan());
        assert!(cuda_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(cuda_result[i], opr1[i].ln());
        }
        opencl.ln(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());    assert!(opencl_result[1].is_nan());
        assert!(opencl_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(opencl_result[i], opr1[i].ln());
        }

        //////// ln_1p    ln(x + 1)
        legacy.ln_1p(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());    assert!(legacy_result[1].is_infinite());
        for i in 2..20{
            assert_eq_f64!(legacy_result[i], opr1[i].ln_1p());
        }
        avx2.ln_1p(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());    assert!(avx2_result[1].is_infinite());
        for i in 2..20{
            assert_eq_f64!(avx2_result[i], opr1[i].ln_1p());
        }
        cuda.ln_1p(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());    assert!(cuda_result[1].is_infinite());
        for i in 2..20{
            assert_eq_f64!(cuda_result[i], opr1[i].ln_1p());
        }
        opencl.ln_1p(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());    assert!(opencl_result[1].is_infinite());
        for i in 2..20{
            assert_eq_f64!(opencl_result[i], opr1[i].ln_1p());
        }

        //////// log10
        legacy.log10(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());    assert!(legacy_result[1].is_nan());
        assert!(legacy_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(legacy_result[i], opr1[i].log10());
        }
        avx2.log10(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());    assert!(avx2_result[1].is_nan());
        assert!(avx2_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(avx2_result[i], opr1[i].log10());
        }
        cuda.log10(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());    assert!(cuda_result[1].is_nan());
        assert!(cuda_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(cuda_result[i], opr1[i].log10());
        }
        opencl.log10(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());    assert!(opencl_result[1].is_nan());
        assert!(opencl_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(opencl_result[i], opr1[i].log10());
        }

        //////// log2
        legacy.log2(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());    assert!(legacy_result[1].is_nan());
        assert!(legacy_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64!(legacy_result[i], opr1[i].log2());
        }
        avx2.log2(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());    assert!(avx2_result[1].is_nan());
        assert!(avx2_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64_percent!(avx2_result[i], opr1[i].log2());
        }
        cuda.log2(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());    assert!(cuda_result[1].is_nan());
        assert!(cuda_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64_percent!(cuda_result[i], opr1[i].log2());
        }
        opencl.log2(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());    assert!(opencl_result[1].is_nan());
        assert!(opencl_result[2].is_infinite());
        for i in 3..20{
            assert_eq_f64_percent!(opencl_result[i], opr1[i].log2());
        }

        //////// exp
        legacy.exp(&mut legacy_result, opr1);
        for i in 1..10{
            assert_eq_f64!(legacy_result[i], opr1[i].exp());
        }
        avx2.exp(&mut avx2_result, opr1);
        for i in 1..10{
            assert_eq_f64_percent!(avx2_result[i], opr1[i].exp());
        }
        cuda.exp(&mut cuda_result, opr1);
        for i in 1..10{
            assert_eq_f64!(cuda_result[i], opr1[i].exp());
        }
        opencl.exp(&mut opencl_result, opr1);
        for i in 1..10{
            assert_eq_f64!(opencl_result[i], opr1[i].exp());
        }

        //////// exp2
        legacy.exp2(&mut legacy_result, opr1);
        for i in 1..10{
            assert_eq_f64!(legacy_result[i], opr1[i].exp2());
        }
        avx2.exp2(&mut avx2_result, opr1);
        for i in 1..10{
            assert_eq_f64!(avx2_result[i], opr1[i].exp2());
        }
        cuda.exp2(&mut cuda_result, opr1);
        for i in 1..10{
            assert_eq_f64!(cuda_result[i], opr1[i].exp2());
        }
        opencl.exp2(&mut opencl_result, opr1);
        for i in 1..10{
            assert_eq_f64!(opencl_result[i], opr1[i].exp2());
        }

        //////// exp_m1
        legacy.exp_m1(&mut legacy_result, opr1);
        for i in 1..10{
            assert_eq_f64!(legacy_result[i], opr1[i].exp_m1());
        }
        avx2.exp_m1(&mut avx2_result, opr1);
        for i in 1..10{
            assert_eq_f64!(avx2_result[i], opr1[i].exp_m1());
        }
        cuda.exp_m1(&mut cuda_result, opr1);
        for i in 1..10{
            assert_eq_f64_percent!(cuda_result[i], opr1[i].exp_m1());
        }
        opencl.exp_m1(&mut opencl_result, opr1);
        for i in 1..10{
            assert_eq_f64_percent!(opencl_result[i], opr1[i].exp_m1());
        }

        for i in 0..BLOCK_SIZE{
            opr1[i] = -2.0f64 +  i as f64;
            opr2[i] = i as f64;
        }
        //////// sqrt
        legacy.sqrt(&mut legacy_result, opr1);
        assert!(legacy_result[0].is_nan());    assert!(legacy_result[1].is_nan());
        for i in 2..10{
            assert_eq_f64!(legacy_result[i], opr1[i].sqrt());
        }
        avx2.sqrt(&mut avx2_result, opr1);
        assert!(avx2_result[0].is_nan());    assert!(avx2_result[1].is_nan());
        for i in 2..10{
            assert_eq_f64!(avx2_result[i], opr1[i].sqrt());
        }
        cuda.sqrt(&mut cuda_result, opr1);
        assert!(cuda_result[0].is_nan());    assert!(cuda_result[1].is_nan());
        for i in 2..10{
            assert_eq_f64!(cuda_result[i], opr1[i].sqrt());
        }
        opencl.sqrt(&mut opencl_result, opr1);
        assert!(opencl_result[0].is_nan());    assert!(opencl_result[1].is_nan());
        for i in 2..10{
            assert_eq_f64!(opencl_result[i], opr1[i].sqrt());
        }

        //////// cbrt
        legacy.cbrt(&mut legacy_result, opr1);
        for i in 0..10{
            assert_eq_f64!(legacy_result[i], opr1[i].cbrt());
        }
        avx2.cbrt(&mut avx2_result, opr1);
        for i in 0..10{
            assert_eq_f64!(avx2_result[i], opr1[i].cbrt());
        }
        cuda.cbrt(&mut cuda_result, opr1);
        for i in 0..10{
            assert_eq_f64!(cuda_result[i], opr1[i].cbrt());
        }
        opencl.cbrt(&mut opencl_result, opr1);
        for i in 0..10{
            assert_eq_f64!(opencl_result[i], opr1[i].cbrt());
        }

        //////// pow
        for i in 0..BLOCK_SIZE{
            opr1[i] = -4.0f64 +  i as f64;
            opr2[i] = -8.0f64 + i as f64;
        }
        legacy.powf(&mut legacy_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64!(legacy_result[i], opr1[i].powf(opr2[i]));
        }
        avx2.powf(&mut avx2_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64!(avx2_result[i], opr1[i].powf(opr2[i]));
        }
        cuda.powf(&mut cuda_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64!(cuda_result[i], opr1[i].powf(opr2[i]));
        }
        opencl.powf(&mut opencl_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64!(opencl_result[i], opr1[i].powf(opr2[i]));
        }

        for i in 0..BLOCK_SIZE{
            opr1[i] = -4.0f64 +  i as f64;
            opr2[i] = -8.0f64 + i as f64;
        }
        legacy.hypot(&mut legacy_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64!(legacy_result[i], opr1[i].hypot(opr2[i]));
        }
        avx2.hypot(&mut avx2_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64!(avx2_result[i], opr1[i].hypot(opr2[i]));
        }
        cuda.hypot(&mut cuda_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64_percent!(cuda_result[i], opr1[i].hypot(opr2[i]));
        }
        opencl.hypot(&mut opencl_result, opr1, opr2);
        for i in 0..12{
            assert_eq_f64_percent!(opencl_result[i], opr1[i].hypot(opr2[i]));
        }

        println!("==== test_0003_01_math end ====");
    }

    #[test]
    fn test_0004_01_math() {
        println!("==== test_0004_01_math start ====");
        let legacy:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::LEGACY);
        let avx2:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::AVX2);
        let cuda:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::CUDA);
        let opencl:Arr4Com<f64, BLOCK_SIZE> = Arr4Com::newf64(OpTarget::OPENCL);

        let mut legacy_result = [0f64;BLOCK_SIZE];
        let mut avx2_result = [0f64;BLOCK_SIZE];
        let mut cuda_result = [0f64;BLOCK_SIZE];
        let mut opencl_result = [0f64;BLOCK_SIZE];
        let mut opr1 = [0f64;BLOCK_SIZE];
        let mut opr2 = [0f64;BLOCK_SIZE];
        for i in 0..BLOCK_SIZE{
            opr1[i] = -2.0f64 +  i as f64;
            opr2[i] = i as f64;
        }
        opr1[0] = -1.0099f64;           opr1[1] = -1.5001f64;
        opr1[2] = -1.5000f64;           opr1[3] = -1.4999f64;
        opr1[4] = -1.0001f64;           opr1[5] = -0.0008f64;
        opr1[6] = -0.00001f64;          opr1[7] = -0.0f64;
        opr1[8] = -0.00001f64;          opr1[9] = 0.00001f64;
        opr1[10] = 0.49999f64;          opr1[11] = 0.5000f64;
        opr1[12] = 0.50001f64;          opr1[13] = 0.99999f64;
        opr1[14] = 1f64;                opr1[15] = 1.00001f64;
        opr1[16] = 10.00001f64;

        legacy.abs(&mut legacy_result, opr1);
        for i in 0..20{
            assert_eq_f64!(legacy_result[i], opr1[i].abs());
        }
        avx2.abs(&mut avx2_result, opr1);
        for i in 0..20{
            assert_eq_f64!(avx2_result[i], opr1[i].abs());
        }
        cuda.abs(&mut cuda_result, opr1);
        for i in 0..20{
            assert_eq_f64!(cuda_result[i], opr1[i].abs());
        }
        opencl.abs(&mut opencl_result, opr1);
        for i in 0..20{
            assert_eq_f64!(opencl_result[i], opr1[i].abs());
        }

        opr2[0] = -1.0098f64;           opr2[1] = -1.5001f64;
        opr2[2] = -1.5001f64;           opr2[3] = -1.5f64;
        opr2[4] = 1.0000f64;           opr2[5] = -0.00089f64;
        opr2[6] = -0.000001f64;          opr2[7] = -0.0f64;
        opr2[8] = 0.00001f64;          opr2[9] = 0.00001f64;
        opr2[10] = 0.49999f64;          opr2[11] = 0.5001f64;
        opr2[12] = 0.50000f64;          opr2[13] = 0.99999f64;
        opr2[14] = 10000f64;                opr2[15] = -1.00001f64;
        opr2[16] = -1000.00001f64;


        legacy.max(&mut legacy_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(legacy_result[i], opr1[i].max(opr2[i]));
        }
        avx2.max(&mut avx2_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(avx2_result[i], opr1[i].max(opr2[i]));
        }
        cuda.max(&mut cuda_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(cuda_result[i], opr1[i].max(opr2[i]));
        }
        opencl.max(&mut opencl_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(opencl_result[i], opr1[i].max(opr2[i]));
        }

        legacy.min(&mut legacy_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(legacy_result[i], opr1[i].min(opr2[i]));
        }
        avx2.min(&mut avx2_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(avx2_result[i], opr1[i].min(opr2[i]));
        }
        cuda.min(&mut cuda_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(cuda_result[i], opr1[i].min(opr2[i]));
        }
        opencl.min(&mut opencl_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(opencl_result[i], opr1[i].min(opr2[i]));
        }

        legacy.copysign(&mut legacy_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(legacy_result[i], opr1[i].copysign(opr2[i]));
        }
        avx2.copysign(&mut avx2_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(avx2_result[i], opr1[i].copysign(opr2[i]));
        }
        cuda.copysign(&mut cuda_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(cuda_result[i], opr1[i].copysign(opr2[i]));
        }
        opencl.copysign(&mut opencl_result, opr1, opr2);
        for i in 0..20{
            assert_eq_f64!(opencl_result[i], opr1[i].copysign(opr2[i]));
        }

        println!("==== test_0004_01_math end ====");
    }
    
}
