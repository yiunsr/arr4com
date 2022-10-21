#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]

pub const M_PI:f64 = 3.141592653589793238462643383279502884;
pub const M_1_PI:f64 = 0.318309886183790671537767526745028724f64;
pub const M_2_PI:f64 = 0.636619772367581343075535053490057448f64;

pub const SLEEF_FP_ILOGB0:i32 = 0x80000000u32 as i32;
pub const SLEEF_FP_ILOGBNAN:i32 = 2147483647;

pub const SLEEF_FLT_MIN:f32 = 1.175494350822287508e-38;
pub const SLEEF_DBL_MIN:f64 = 2.225073858507201383e-308;
pub const SLEEF_INT_MAX:i32 = 2147483647;
pub const SLEEF_DBL_DENORM_MIN:f64 = 4.9406564584124654e-324;
pub const SLEEF_FLT_DENORM_MIN:f32 = 1.40129846e-45f32;

pub const PI_A:f64 = 3.1415926218032836914;
pub const PI_B:f64 = 3.1786509424591713469e-08;
pub const PI_C:f64 = 1.2246467864107188502e-16;
pub const PI_D:f64 = 1.2736634327021899816e-24;
pub const TRIGRANGEMAX:f64 = 1e+14;


pub const PI_A2:f64 = 3.141592653589793116;
pub const PI_B2:f64 = 1.2246467991473532072e-16;
pub const TRIGRANGEMAX2:i32 = 15;
pub const M_2_PI_H:f64 = 0.63661977236758138243;
pub const M_2_PI_L:f64 = -3.9357353350364971764e-17;
pub const SQRT_DBL_MAX:f64 =1.3407807929942596355e+154;
pub const TRIGRANGEMAX3:f64 = 1e+9;

pub const M_4_PI:f64 = 1.273239544735162542821171882678754627704620361328125;
pub const L2U:f64 = 0.69314718055966295651160180568695068359375;
pub const L2L:f64 = 0.28235290563031577122588448175013436025525412068e-12;
pub const R_LN2:f64 = 1.442695040888963407359924681001892137426645954152985934135449406931;

pub const L10U:f64 = 0.30102999566383914498; // log 2 / log 10
pub const L10L:f64 = 1.4205023227266099418e-13;
pub const LOG10_2:f64 = 3.3219280948873623478703194294893901758648313930;

pub const L10Uf:f32 =  0.3010253906f32;
pub const L10Lf:f32 = 4.605038981e-06f32;

pub const PI_Af:f32 = 3.140625f32;
pub const PI_Bf:f32 = 0.0009670257568359375f32;
pub const PI_Cf:f32 = 6.2771141529083251953e-07f32;
pub const PI_Df:f32 = 1.2154201256553420762e-10f32;
pub const TRIGRANGEMAXf:i32 = 39000;

pub const PI_A2f:f32 = 3.1414794921875f32;
pub const PI_B2f:f32 = 0.00011315941810607910156f32;
pub const PI_C2f:f32 = 1.9841872589410058936e-09f32;
pub const TRIGRANGEMAX2f:f32 = 125.0f32;

pub const TRIGRANGEMAX4f:f32 = 8e+6f32;

pub const SQRT_FLT_MAX:f64 = 18446743523953729536.0;

pub const L2Uf:f32 = 0.693145751953125f32;
pub const L2Lf:f32 = 1.428606765330187045e-06f32;

pub const R_LN2f:f32 = 1.442695040888963407359924681001892137426645954152985934135449406931f32;
pub const M_PIf:f32 = M_PI as f32;

pub const SLEEF_INFINITY:f64 = 1e+300 * 1e+300;
pub const SLEEF_NAN:f64 = SLEEF_INFINITY - SLEEF_INFINITY;
pub const SLEEF_INFINITYf:f32 = SLEEF_INFINITY as f32;

pub const SLEEF_NANf:f32 = SLEEF_NAN as f32;
