#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_macros,
    unused_imports
)]

use crate::arr4com::sleef::helperavx2::*;
use crate::arr4com::sleef::simddp::{MLA, C2V};


// macro_rules! POLY2{
//     ($x:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x, C2V($c1), C2V($c0))
//     };
// }
// pub(crate) use POLY2;

// macro_rules! POLY3{
//     ($x:stmt, $x2:stmt,  $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x2, C2V($c2), MLA($x, C2V($c1), C2V($c0)))
//     };
// }
// pub(crate) use POLY3;

// macro_rules! POLY4{
//     ($x:stmt, $x2:stmt,  $c3:stmt, $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x2, MLA($x, C2V($c3), C2V($c2)), MLA($x, C2V($c1), C2V($c0)))
//     };
// }
// pub(crate) use POLY4;

// macro_rules! POLY5{
//     ($x:stmt, $x2:stmt, $x4:stmt,  $c4:stmt,  $c3:stmt, $c2:stmt, 
//         $c1:stmt, $c0:stmt) => { 
//         MLA($x4, C2V($c4), POLY4!($x, $x2, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY5;

// macro_rules! POLY6{
//     ($x:stmt, $x2:stmt, $x4:stmt,  $c5:stmt, $c4:stmt,  $c3:stmt, $c2:stmt,
//         $c1:stmt, $c0:stmt) => {
//         MLA($x4, POLY2!($x, $c5, $c4), POLY4!($x, $x2, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY6;

// macro_rules! POLY7{
//     ($x:stmt, $x2:stmt, $x4:stmt, $c6:stmt, $c5:stmt, $c4:stmt,  $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x4, POLY3!($x, $x2, $c6, $c5, $c4), POLY4!($x, $x2, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY7;

// macro_rules! POLY8{
//     ($x:stmt, $x2:stmt, $x4:stmt, $c7:stmt, $c6:stmt, $c5:stmt, $c4:stmt,  $c3:stmt,
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x4, POLY4!($x, $x2, $c7, $c6, $c5, $c4), POLY4!($x, $x2, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY8;

// macro_rules! POLY9{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt, 
//         $c4:stmt, $c3:stmt, $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, C2V($c8), POLY8!($x, $x2,$ x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY9;

// macro_rules! POLY10{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt, 
//         $c4:stmt, $c3:stmt, $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, POLY2!($x, $c9, $c8), POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY10;

// macro_rules! POLY11{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $ca:stmt, $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt, 
//         $c4:stmt, $c3:stmt, $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, POLY3!($x, $x2, $ca, $c9, $c8), 
//             POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))
//     };
// }
// pub(crate) use POLY11;

// macro_rules! POLY12{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $cb:stmt, $ca:stmt, $c9:stmt, $c8:stmt, $c7:stmt, 
//         $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, POLY4!($x, $x2, $cb, $ca, $c9, $c8), 
//             POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY12;

// macro_rules! POLY13{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, POLY5!($x, $x2, $x4,$ cc, $cb, $ca, $c9, $c8), 
//             POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY13;

// macro_rules! POLY14{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $cd:stmt, $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, POLY6!($x, $x2, $x4, $cd, $cc, $cb, $ca, $c9, $c8), 
//             POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY14;

// macro_rules! POLY15{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $ce:stmt, $cd:stmt, $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//         MLA($x8, POLY7!($x, $x2, $x4, $ce, $cd, $cc, $cb, $ca, $c9, $c8), 
//             POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY15;

// macro_rules! POLY16{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $cf:stmt, $ce:stmt, $cd:stmt, $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//             MLA($x8, POLY8!($x, $x2, $x4, $cf, $ce, $cd, $cc, $cb, $ca, $c9, $c8), 
//                 POLY8!($x, $x2, $x4, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY16;

// macro_rules! POLY17{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $x16:stmt, $d0:stmt, $cf:stmt, $ce:stmt, $cd:stmt, 
//         $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//             MLA($x16, C2V($d0), POLY16!($x, $x2, $x4, $x8, $cf, $ce, $cd, $cc, $cb, 
//                     $ca, $c9, $c8, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY17;

// macro_rules! POLY18{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $x16:stmt, 
//         $d1:stmt, $d0:stmt, $cf:stmt, $ce:stmt, $cd:stmt, 
//         $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//             MLA($x16, POLY2($x, $d1, $d0), POLY16!($x, $x2, $x4, $x8, $cf, $ce, $cd, $cc, $cb, 
//                     $ca, $c9, $c8, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY18;

// macro_rules! POLY19{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $x16:stmt, 
//         $d2:stmt, $d1:stmt, $d0:stmt, $cf:stmt, $ce:stmt, $cd:stmt, 
//         $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//             MLA($x16, POLY3($x, $x2, $d2, $d1, $d0), POLY16!($x, $x2, $x4, $x8, $cf, $ce, $cd, $cc, $cb, 
//                     $ca, $c9, $c8, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY19;

// macro_rules! POLY20{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $x16:stmt, 
//         $d3:stmt, $d2:stmt, $d1:stmt, $d0:stmt, $cf:stmt, $ce:stmt, $cd:stmt, 
//         $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//             MLA($x16, POLY4($x, $x2, $d3, $d2, $d1, $d0), 
//                 POLY16!($x, $x2, $x4, $x8, $cf, $ce, $cd, $cc, $cb, 
//                     $ca, $c9, $c8, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY20;

// macro_rules! POLY21{
//     ($x:stmt, $x2:stmt, $x4:stmt, $x8:stmt, $x16:stmt, 
//         $d4:stmt, $d3:stmt, $d2:stmt, $d1:stmt, $d0:stmt, $cf:stmt, $ce:stmt, $cd:stmt, 
//         $cc:stmt, $cb:stmt, $ca:stmt,
//         $c9:stmt, $c8:stmt, $c7:stmt, $c6:stmt, $c5:stmt,  $c4:stmt, $c3:stmt, 
//         $c2:stmt, $c1:stmt, $c0:stmt) => { 
//             MLA($x16, POLY5($x, $x2, $x4, $d4, $d3, $d2, $d1, $d0), 
//                 POLY16!($x, $x2, $x4, $x8, $cf, $ce, $cd, $cc, $cb, 
//                     $ca, $c9, $c8, $c7, $c6, $c5, $c4, $c3, $c2, $c1, $c0))

//     };
// }
// pub(crate) use POLY21;


pub fn POLY2(x:vdouble, c1:f64, c0:f64)->vdouble{
    MLA(x, C2V(c1), C2V(c0))
}
pub fn POLY3(x:vdouble, x2:vdouble, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x2, C2V(c2), MLA(x, C2V(c1), C2V(c0)))
}
pub fn POLY4(x:vdouble, x2:vdouble, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x2, MLA(x, C2V(c3), C2V(c2)), MLA(x, C2V(c1), C2V(c0)))
}
pub fn POLY5(x:vdouble, x2:vdouble, x4:vdouble, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x4, C2V(c4), POLY4(x, x2, c3, c2, c1, c0))
}
pub fn POLY6(x:vdouble, x2:vdouble, x4:vdouble, c5:f64, c4:f64, c3:f64, 
        c2:f64, c1:f64, c0:f64)->vdouble{
        MLA(x4, POLY2(x, c5, c4), POLY4(x, x2, c3, c2, c1, c0))
}
pub fn POLY7(x:vdouble, x2:vdouble, x4:vdouble, c6:f64, c5:f64, c4:f64,
        c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x4, POLY3(x, x2, c6, c5, c4), POLY4(x, x2, c3, c2, c1, c0))
    }
pub fn POLY8(x:vdouble, x2:vdouble, x4:vdouble, c7:f64, c6:f64, c5:f64,
        c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x4, POLY4(x, x2, c7, c6, c5, c4), POLY4(x, x2, c3, c2, c1, c0))
}
pub fn POLY9(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, c8:f64, c7:f64,
        c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x8, C2V(c8), POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY10(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, c9:f64, c8:f64,
        c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x8, POLY2(x, c9, c8), POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY11(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, ca:f64, c9:f64, 
        c8:f64, c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
  MLA(x8, POLY3(x, x2, ca, c9, c8), 
    POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY12(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, cb:f64, ca:f64, 
        c9:f64, c8:f64, c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x8, POLY4(x, x2, cb, ca, c9, c8), 
        POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY13(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, cc:f64, cb:f64, ca:f64, 
        c9:f64, c8:f64, c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x8, POLY5(x, x2, x4, cc, cb, ca, c9, c8), 
        POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY14(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, cd:f64, cc:f64, cb:f64, 
        ca:f64, c9:f64, c8:f64, c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64,
         c1:f64, c0:f64)->vdouble{
    MLA(x8, POLY6(x, x2, x4, cd, cc, cb, ca, c9, c8), 
        POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY15(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, ce:f64, cd:f64,
        cc:f64, cb:f64, ca:f64, c9:f64, c8:f64, c7:f64, c6:f64, c5:f64, 
        c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x8, POLY7(x, x2, x4, ce, cd, cc, cb, ca, c9, c8), 
    POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY16(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, cf:f64, ce:f64, cd:f64,
        cc:f64, cb:f64, ca:f64, c9:f64, c8:f64, c7:f64, c6:f64, c5:f64, 
        c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x8, POLY8(x, x2, x4, cf, ce, cd, cc, cb, ca, c9, c8),
    POLY8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY17(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, x16:vdouble,
        d0:f64, cf:f64, ce:f64, cd:f64, cc:f64, cb:f64, ca:f64, c9:f64, 
        c8:f64, c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x16, C2V(d0), 
    POLY16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY18(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, x16:vdouble, 
        d1:f64, d0:f64, cf:f64, ce:f64, cd:f64, cc:f64, cb:f64, ca:f64, c9:f64, 
        c8:f64, c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x16, POLY2(x, d1, d0), 
    POLY16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY19(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, x16:vdouble, d2:f64, d1:f64, 
        d0:f64, cf:f64, ce:f64, cd:f64, cc:f64, cb:f64, ca:f64, c9:f64, c8:f64, c7:f64, c6:f64, 
        c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x16, POLY3(x, x2, d2, d1, d0), 
    POLY16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}
pub fn POLY20(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, x16:vdouble, d3:f64, d2:f64, d1:f64, 
        d0:f64, cf:f64, ce:f64, cd:f64, cc:f64, cb:f64, ca:f64, c9:f64, c8:f64, c7:f64, 
        c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x16, POLY4(x, x2, d3, d2, d1, d0), 
    POLY16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}
    
pub fn POLY21(x:vdouble, x2:vdouble, x4:vdouble, x8:vdouble, x16:vdouble, d4:f64, d3:f64, d2:f64,
        d1:f64, d0:f64, cf:f64, ce:f64, cd:f64, cc:f64, cb:f64, ca:f64, c9:f64, c8:f64, 
        c7:f64, c6:f64, c5:f64, c4:f64, c3:f64, c2:f64, c1:f64, c0:f64)->vdouble{
    MLA(x16, POLY5(x, x2, x4, d4, d3, d2, d1, d0), 
    POLY16(x, x2, x4, x8, cf, ce, cd, cc, cb, ca, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0))
}
