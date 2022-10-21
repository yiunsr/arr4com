
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

#[cfg(test)]
mod float_tests {
    

    use arr4com::arr4com::Arr4Com;
    use arr4com::arr4com::OpTarget;
    const BLOCK_SIZE: usize = 256;

    #[test]
    fn test_0001_01_checkshort() {
        println!("==== test_0001_01_checkshort start ====");
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
        assert_eq_f32!( result[0], 0.0f32);
        assert_eq_f32!( result[1], 3.0f32);
        assert_eq_f32!( result[255], 765.0f32);

        avx2.add(&mut result, lhs, rhs);
        assert_eq_f32!( result[0], 0.0f32);
        assert_eq_f32!( result[1], 3.0f32);
        assert_eq_f32!( result[2], 6.0f32);
        assert_eq_f32!( result[255], 765.0f32);

        cuda.add(&mut result, lhs, rhs);
        assert_eq_f32!( result[0], 0.0f32);
        assert_eq_f32!( result[1], 3.0f32);
        assert_eq_f32!( result[2], 6.0f32);
        assert_eq_f32!( result[255], 765.0f32);

        println!("==== test_0001_01_checkshort end ====");
    }

    #[test]
    fn test_0001_02_checkshort() {
        println!("==== test_0001_02_checkshort start ====");
        assert_eq!("a1", "a1");
        assert_eq!("a2", "a2");
        println!("==== test_0001_02_checkshort end ====");
    }
}