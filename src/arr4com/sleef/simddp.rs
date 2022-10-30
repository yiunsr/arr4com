#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_parens
)]

use crate::arr4com::sleef::dd::*;
use crate::arr4com::sleef::helperavx2::*;
use crate::arr4com::sleef::commonfuncs::*;
use crate::arr4com::sleef::misc::*;
use crate::arr4com::sleef::rempitab::Sleef_rempitabdp;
use crate::arr4com::sleef::estrin::*;

// macro_rules! MLA{
//     ($x:stmt, $y:stmt, $z:stmt) => { 
//         unsafe{
//             vmla_vf_vf_vf_vf($x, $y, $z)
//         }
//     };
// }
// pub(crate) use MLA;


// macro_rules! C2V{
//     ($c:stmt) => { 
//         unsafe{
//             vcast_vf_f($c)
//         }
//     };
// }
// pub(crate) use C2V;

pub fn MLA(x:vdouble, y:vdouble, z:vdouble)->vdouble{
    vmla_vd_vd_vd_vd((x), (y), (z))
}
pub fn C2V(c:f64)->vdouble{
    vcast_vd_d(c)
}


// return d0 < d1 ? x : y
fn vsel_vi_vd_vd_vi_vi(d0:vdouble, d1:vdouble, x:vint, y:vint)->vint { vsel_vi_vo_vi_vi(vcast_vo32_vo64(vlt_vo_vd_vd(d0, d1)), x, y) } 

// return d0 < 0 ? x : 0
fn vsel_vi_vd_vi(d:vdouble, x:vint)->vint{ return vand_vi_vo_vi(vcast_vo32_vo64(vsignbit_vo_vd(d)), x); }

pub fn xldexp(x:vdouble, q:vint)->vdouble{ vldexp_vd_vd_vi(x, q) }

pub fn xilogb(d:vdouble)->vint{
    let mut e = vcast_vd_vi(vilogbk_vi_vd(vabs_vd_vd(d)));
    e = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), vcast_vd_d(SLEEF_FP_ILOGB0 as f64), e);
    e = vsel_vd_vo_vd_vd(visnan_vo_vd(d), vcast_vd_d(SLEEF_FP_ILOGBNAN as f64), e);
    e = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vcast_vd_d(SLEEF_INT_MAX as f64), e);
    vrint_vi_vd(e)
}

fn rempi(arg_a:vdouble)->ddi_t{
    let mut ex = vilogb2k_vi_vd(arg_a);
    ex = vsub_vi_vi_vi(ex, vcast_vi_i(55));
    let mut q = vand_vi_vo_vi(vgt_vo_vi_vi(ex, vcast_vi_i(700-55)), vcast_vi_i(-64));
    let a = vldexp3_vd_vd_vi(arg_a, q);
    ex = vandnot_vi_vi_vi(vsra_vi_vi_i::<31>(ex), ex);
    ex = vsll_vi_vi_i::<2>(ex);
    let mut x = ddmul_vd2_vd_vd(a, vgather_vd_p_vi(&Sleef_rempitabdp[0], ex));
    let di = rempisub(vd2getx_vd_vd2(x));
    q = digeti_vi_di(&di);
    x = vd2setx_vd2_vd2_vd(x, digetd_vd_di(&di));
    x = ddnormalize_vd2_vd2(x);
    let mut y = ddmul_vd2_vd_vd(a, vgather_vd_p_vi(&Sleef_rempitabdp[1], ex));
    x = ddadd2_vd2_vd2_vd2(x, y);
    let di = rempisub(vd2getx_vd_vd2(x));
    q = vadd_vi_vi_vi(q, digeti_vi_di(&di));
    x = vd2setx_vd2_vd2_vd(x, digetd_vd_di(&di));
    x = ddnormalize_vd2_vd2(x);
    y = vcast_vd2_vd_vd(vgather_vd_p_vi(&Sleef_rempitabdp[2], ex), vgather_vd_p_vi(&Sleef_rempitabdp[3], ex));
    y = ddmul_vd2_vd2_vd(y, a);
    x = ddadd2_vd2_vd2_vd2(x, y);
    x = ddnormalize_vd2_vd2(x);
    x = ddmul_vd2_vd2_vd2(x, vcast_vd2_d_d(3.141592653589793116*2.0, 1.2246467991473532072e-16*2.0));
    let o = vlt_vo_vd_vd(vabs_vd_vd(a), vcast_vd_d(0.7));
    x = vd2setx_vd2_vd2_vd(x, vsel_vd_vo_vd_vd(o, a, vd2getx_vd_vd2(x)));
    x = vd2sety_vd2_vd2_vd(x, vreinterpret_vd_vm(vandnot_vm_vo64_vm(o, vreinterpret_vm_vd(vd2gety_vd_vd2(x)))));
    ddisetddi_ddi_vd2_vi!(x, q)
}

pub fn xsin_u1(d:vdouble)->vdouble{
    let mut u;
    let mut s;  let  t; let  x;
    let mut ql;
      
    if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2 as f64))){
        let dql = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)));
        ql = vrint_vi_vd(dql);
        u = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A2), d);
        s = ddadd_vd2_vd_vd (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2)));
    }else if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX))) {
        let mut dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 24) as f64)));
        dqh = vmul_vd_vd_vd(dqh, vcast_vd_d( (1 << 24) as f64));
        let dql = vrint_vd_vd(vmlapn_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), dqh));
        ql = vrint_vi_vd(dql);
    
        u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A), d);
        s = ddadd_vd2_vd_vd  (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C)));
        s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D)));
    }else{
        let mut ddi = rempi(d);
        ql = vand_vi_vi_vi(ddigeti_vi_ddi(&ddi), vcast_vi_i(3));
        ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(vd2getx_vd_vd2(ddigetdd_vd2_ddi(&ddi)), vcast_vd_d(0.0))), vcast_vi_i(2), vcast_vi_i(1)));
        ql = vsra_vi_vi_i::<2>(ql);
        let o = veq_vo_vi_vi(vand_vi_vi_vi(ddigeti_vi_ddi(&ddi), vcast_vi_i(1)), vcast_vi_i(1));
        let mut x = vcast_vd2_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(-3.141592653589793116 * 0.5), vd2getx_vd_vd2(ddigetdd_vd2_ddi(&ddi))), 
                     vmulsign_vd_vd_vd(vcast_vd_d(-1.2246467991473532072e-16 * 0.5), vd2getx_vd_vd2(ddigetdd_vd2_ddi(&ddi))));
        x = ddadd2_vd2_vd2_vd2(ddigetdd_vd2_ddi(&ddi), x);
        let aa = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(o), x, ddigetdd_vd2_ddi(&ddi));
        let ddi = ddisetdd_ddi_ddi_vd2!(ddi, aa);
        s = ddnormalize_vd2_vd2(ddigetdd_vd2_ddi(&ddi));
        s = vd2setx_vd2_vd2_vd(s, vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visinf_vo_vd(d), visnan_vo_vd(d)), vreinterpret_vm_vd(vd2getx_vd_vd2(s)))));
    }
      
    t = s;
    s = ddsqu_vd2_vd2(s);

    let s2 = vmul_vd_vd_vd(vd2getx_vd_vd2(s), vd2getx_vd_vd2(s));
    let s4 = vmul_vd_vd_vd(s2, s2);
    u = POLY6(
        vd2getx_vd_vd2(s), s2, s4,
        2.72052416138529567917983e-15,
        -7.6429259411395447190023e-13,
        1.60589370117277896211623e-10,
        -2.5052106814843123359368e-08,
        2.75573192104428224777379e-06,
        -0.000198412698412046454654947);
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(s), vcast_vd_d(0.00833333333333318056201922));

    x = ddadd_vd2_vd_vd2(vcast_vd_d(1.0), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(-0.166666666666666657414808), vmul_vd_vd_vd(u, vd2getx_vd_vd2(s))), s));
    u = ddmul_vd_vd2_vd2(t, x);
    
    u = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1))),
                                vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(u)));
    u = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), d, u);
    
    u
}


pub fn xcos_u1(d:vdouble)->vdouble{
      let mut u;
      let mut s; let t; let x;
      let mut ql;
      
    if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2 as f64))) {
        let mut dql = vrint_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), vcast_vd_d(-0.5)));
        dql = vmla_vd_vd_vd_vd(vcast_vd_d(2.0), dql, vcast_vd_d(1.0));
        ql = vrint_vi_vd(dql);
        s = ddadd2_vd2_vd_vd(d, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A2*0.5)));
        s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2*0.5)));
    }else if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX))) {
        let mut dqh = vtruncate_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 23) as f64), vcast_vd_d(-M_1_PI / (1 << 24) as f64 )));
        ql = vrint_vi_vd(vadd_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
                        vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-(1 << 23) as f64), vcast_vd_d(-0.5))));
        dqh = vmul_vd_vd_vd(dqh, vcast_vd_d( (1 << 24) as f64));
        ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));
        let dql = vcast_vd_vi(ql);
    
        u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
        s = ddadd2_vd2_vd_vd(u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5)));
        s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D*0.5)));
    }else{
        let mut ddi = rempi(d);
        ql = vand_vi_vi_vi(ddigeti_vi_ddi(&ddi), vcast_vi_i(3));
        ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vsel_vi_vo_vi_vi(vcast_vo32_vo64(vgt_vo_vd_vd(vd2getx_vd_vd2(ddigetdd_vd2_ddi(&ddi)), vcast_vd_d(0.0))), vcast_vi_i(8), vcast_vi_i(7)));
        ql = vsra_vi_vi_i::<1>(ql);
        let o = veq_vo_vi_vi(vand_vi_vi_vi(ddigeti_vi_ddi(&ddi), vcast_vi_i(1)), vcast_vi_i(0));
        let y = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(vd2getx_vd_vd2(ddigetdd_vd2_ddi(&ddi)), vcast_vd_d(0.0)), vcast_vd_d(0.0), vcast_vd_d(-1.0));
        let mut x = vcast_vd2_vd_vd(vmulsign_vd_vd_vd(vcast_vd_d(-3.141592653589793116 * 0.5), y), 
                     vmulsign_vd_vd_vd(vcast_vd_d(-1.2246467991473532072e-16 * 0.5), y));
        x = ddadd2_vd2_vd2_vd2(ddigetdd_vd2_ddi(&ddi), x);
        let aa = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(o), x, ddigetdd_vd2_ddi(&ddi));
        ddi = ddisetdd_ddi_ddi_vd2!(ddi, aa);
        s = ddnormalize_vd2_vd2(ddigetdd_vd2_ddi(&ddi));
        s = vd2setx_vd2_vd2_vd(s, vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visinf_vo_vd(d), visnan_vo_vd(d)), vreinterpret_vm_vd(vd2getx_vd_vd2(s)))));
    }
    
    t = s;
    s = ddsqu_vd2_vd2(s);

    let s2 = vmul_vd_vd_vd(vd2getx_vd_vd2(s), vd2getx_vd_vd2(s));
    let s4 = vmul_vd_vd_vd(s2, s2);
    u = POLY6(vd2getx_vd_vd2(s), s2, s4,
        2.72052416138529567917983e-15,
        -7.6429259411395447190023e-13,
        1.60589370117277896211623e-10,
        -2.5052106814843123359368e-08,
        2.75573192104428224777379e-06,
        -0.000198412698412046454654947);
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(s), vcast_vd_d(0.00833333333333318056201922));

    x = ddadd_vd2_vd_vd2(vcast_vd_d(1.0), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(-0.166666666666666657414808), vmul_vd_vd_vd(u, vd2getx_vd_vd2(s))), s));
    u = ddmul_vd_vd2_vd2(t, x);
    
    u = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(0))), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(u)));
    
    u
}

pub fn xtan_u1(d:vdouble)->vdouble{
    let u;
    let mut s;  let t; let mut x; let y;
    let o;
    let ql;
      
    if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2 as f64))){
        let dql = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2.0 * M_1_PI)));
        ql = vrint_vi_vd(dql);
        u = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A2*0.5), d);
        s = ddadd_vd2_vd_vd (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2*0.5)));
    }else if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX))){
        let mut dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2.0*M_1_PI / (1 << 24) as f64)));
        dqh = vmul_vd_vd_vd(dqh, vcast_vd_d( (1 << 24) as f64));
        s = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd(vcast_vd2_d_d(M_2_PI_H, M_2_PI_L), d),
                  vsub_vd_vd_vd(vsel_vd_vo_vd_vd(vlt_vo_vd_vd(d, vcast_vd_d(0.0)),
                                 vcast_vd_d(-0.5), vcast_vd_d(0.5)), dqh));
        let  dql = vtruncate_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(s), vd2gety_vd_vd2(s)));
        ql = vrint_vi_vd(dql);
    
        u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
        s = ddadd_vd2_vd_vd(u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5            )));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5            )));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5)));
        s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5            )));
        s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D*0.5)));
    }else {
        let ddi = rempi(d);
        ql = ddigeti_vi_ddi(&ddi);
        s = ddigetdd_vd2_ddi(&ddi);
        let o = vor_vo_vo_vo(visinf_vo_vd(d), visnan_vo_vd(d));
        s = vd2setx_vd2_vd2_vd(s, vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(vd2getx_vd_vd2(s)))));
        s = vd2sety_vd2_vd2_vd(s, vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(vd2gety_vd_vd2(s)))));
    }
    
    t = ddscale_vd2_vd2_vd(s, vcast_vd_d(0.5));
    s = ddsqu_vd2_vd2(t);
    
    let s2 = vmul_vd_vd_vd(vd2getx_vd_vd2(s), vd2getx_vd_vd2(s));
    let s4 = vmul_vd_vd_vd(s2, s2);
    let mut u = POLY8(vd2getx_vd_vd2(s), s2, s4,
            0.3245098826639276316e-3,
            0.5619219738114323735e-3,
            0.1460781502402784494e-2,
            0.3591611540792499519e-2,
            0.8863268409563113126e-2,
            0.2186948728185535498e-1,
            0.5396825399517272970e-1,
            0.1333333333330500581e+0);
    
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(s), vcast_vd_d(0.3333333333333343695e+0));
    x = ddadd_vd2_vd2_vd2(t, ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(s, t), u));

    y = ddadd_vd2_vd_vd2(vcast_vd_d(-1.0), ddsqu_vd2_vd2(x));
    x = ddscale_vd2_vd2_vd(x, vcast_vd_d(-2.0));

    o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1)));

    x = dddiv_vd2_vd2_vd2(vsel_vd2_vo_vd2_vd2(o, ddneg_vd2_vd2(y), x),
            vsel_vd2_vo_vd2_vd2(o, x, y));

    u = vadd_vd_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(x));

    u = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), d, u);
    
    u
}

pub fn atan2k_u1(y:vdouble2, x:vdouble2)->vdouble2{
    let mut q = vsel_vi_vd_vi(vd2getx_vd_vd2(x), vcast_vi_i(-2));
    let mut p = vlt_vo_vd_vd(vd2getx_vd_vd2(x), vcast_vd_d(0.0));
    let b = vand_vm_vo64_vm(p, vreinterpret_vm_vd(vcast_vd_d(-0.0)));
    let mut x = vd2setx_vd2_vd2_vd(x, vreinterpret_vd_vm(vxor_vm_vm_vm(b, vreinterpret_vm_vd(vd2getx_vd_vd2(x)))));
    x = vd2sety_vd2_vd2_vd(x, vreinterpret_vd_vm(vxor_vm_vm_vm(b, vreinterpret_vm_vd(vd2gety_vd_vd2(x)))));
  
    q = vsel_vi_vd_vd_vi_vi(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y), vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
    p = vlt_vo_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    let mut s = vsel_vd2_vo_vd2_vd2(p, ddneg_vd2_vd2(x), y);
    let mut t = vsel_vd2_vo_vd2_vd2(p, y, x);
  
    s = dddiv_vd2_vd2_vd2(s, t);
    t = ddsqu_vd2_vd2(s);
    t = ddnormalize_vd2_vd2(t);
  
    let t2 = vmul_vd_vd_vd(vd2getx_vd_vd2(t), vd2getx_vd_vd2(t));
    let t4 = vmul_vd_vd_vd(t2, t2);
    let t8 = vmul_vd_vd_vd(t4, t4);
    let mut u = POLY16(vd2getx_vd_vd2(t), t2, t4, t8,
           1.06298484191448746607415e-05,
           -0.000125620649967286867384336,
           0.00070557664296393412389774,
           -0.00251865614498713360352999,
           0.00646262899036991172313504,
           -0.0128281333663399031014274,
           0.0208024799924145797902497,
           -0.0289002344784740315686289,
           0.0359785005035104590853656,
           -0.041848579703592507506027,
           0.0470843011653283988193763,
           -0.0524914210588448421068719,
           0.0587946590969581003860434,
           -0.0666620884778795497194182,
           0.0769225330296203768654095,
           -0.0909090442773387574781907);
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(t), vcast_vd_d(0.111111108376896236538123));
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(t), vcast_vd_d(-0.142857142756268568062339));
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(t), vcast_vd_d(0.199999999997977351284817));
    u = vmla_vd_vd_vd_vd(u, vd2getx_vd_vd2(t), vcast_vd_d(-0.333333333333317605173818));
  
    t = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(s, t), u));
    
    t = ddadd_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(1.570796326794896557998982, 6.12323399573676603586882e-17), vcast_vd_vi(q)), t);
  
    t
}

fn visinf2_vd_vd_vd(d:vdouble, m:vdouble)->vdouble{
    vreinterpret_vd_vm(vand_vm_vo64_vm(visinf_vo_vd(d), vor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(m))))
}

pub fn xatan2_u1(mut y:vdouble, mut x:vdouble)->vdouble{
    let o = vlt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(5.5626846462680083984e-309)); // nexttoward((1.0 / DBL_MAX), 1)
    x = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(x, vcast_vd_d((1u64 << 53) as f64 )), x);
    y = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(y, vcast_vd_d((1u64 << 53) as f64)), y);
  
    let d = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(y), vcast_vd_d(0.0)), vcast_vd2_vd_vd(x, vcast_vd_d(0.0)));
    let mut r = vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d));
  
    r = vmulsign_vd_vd_vd(r, x);
    r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0.0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2.0), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2.0), x))), r);
    r = vsel_vd_vo_vd_vd(visinf_vo_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2.0), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4.0), x))), r);
    r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(0.0)), vreinterpret_vd_vm(vand_vm_vo64_vm(vsignbit_vo_vd(x), vreinterpret_vm_vd(vcast_vd_d(M_PI)))), r);
  
    r = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(vmulsign_vd_vd_vd(r, y))));
    r
}

pub fn xasin_u1(d:vdouble)->vdouble{
    let o = vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.5));
    let x2 = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, d), vmul_vd_vd_vd(vsub_vd_vd_vd(vcast_vd_d(1.0), vabs_vd_vd(d)), vcast_vd_d(0.5)));
    
    let mut x = vsel_vd2_vo_vd2_vd2(o, vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.0)), ddsqrt_vd2_vd(x2));
    x = vsel_vd2_vo_vd2_vd2(veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1.0)), vcast_vd2_d_d(0.0, 0.0), x);
  
    let x4 = vmul_vd_vd_vd(x2, x2);
    let x8 = vmul_vd_vd_vd(x4, x4);
    let x16 = vmul_vd_vd_vd(x8, x8);
    let mut u = POLY12(x2, x4, x8, x16,
           0.3161587650653934628e-1,
           -0.1581918243329996643e-1,
           0.1929045477267910674e-1,
           0.6606077476277170610e-2,
           0.1215360525577377331e-1,
           0.1388715184501609218e-1,
           0.1735956991223614604e-1,
           0.2237176181932048341e-1,
           0.3038195928038132237e-1,
           0.4464285681377102438e-1,
           0.7500000000378581611e-1,
           0.1666666666666497543e+0);
  
    u = vmul_vd_vd_vd(u, vmul_vd_vd_vd(x2, vd2getx_vd_vd2(x)));
  
    let y = ddsub_vd2_vd2_vd(ddsub_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116/4.0, 1.2246467991473532072e-16/4.0), x), u);
    
    let r = vsel_vd_vo_vd_vd(o, vadd_vd_vd_vd(u, vd2getx_vd_vd2(x)),
                     vmul_vd_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(y), vd2gety_vd_vd2(y)), vcast_vd_d(2.0)));
    vmulsign_vd_vd_vd(r, d)
}

pub fn xacos_u1(d:vdouble)->vdouble{
    let o = vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.5));
    let x2 = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, d), vmul_vd_vd_vd(vsub_vd_vd_vd(vcast_vd_d(1.0), vabs_vd_vd(d)), vcast_vd_d(0.5)));
    let mut x = vsel_vd2_vo_vd2_vd2(o, vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.0)), ddsqrt_vd2_vd(x2));
    x = vsel_vd2_vo_vd2_vd2(veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1.0)), vcast_vd2_d_d(0.0, 0.0), x);

  
    let x4 = vmul_vd_vd_vd(x2, x2);
    let x8 = vmul_vd_vd_vd(x4, x4);
    let x16 = vmul_vd_vd_vd(x8, x8);
    let mut u = POLY12(x2, x4, x8, x16,
           0.3161587650653934628e-1,
           -0.1581918243329996643e-1,
           0.1929045477267910674e-1,
           0.6606077476277170610e-2,
           0.1215360525577377331e-1,
           0.1388715184501609218e-1,
           0.1735956991223614604e-1,
           0.2237176181932048341e-1,
           0.3038195928038132237e-1,
           0.4464285681377102438e-1,
           0.7500000000378581611e-1,
           0.1666666666666497543e+0);
  
    u = vmul_vd_vd_vd(u, vmul_vd_vd_vd(x2, vd2getx_vd_vd2(x)));
  
    let mut y = ddsub_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116/2.0, 1.2246467991473532072e-16/2.0),
                   ddadd_vd2_vd_vd(vmulsign_vd_vd_vd(vd2getx_vd_vd2(x), d), vmulsign_vd_vd_vd(u, d)));
    x = ddadd_vd2_vd2_vd(x, u);
    
    y = vsel_vd2_vo_vd2_vd2(o, y, ddscale_vd2_vd2_vd(x, vcast_vd_d(2.0)));
    
    y = vsel_vd2_vo_vd2_vd2(vandnot_vo_vo_vo(o, vlt_vo_vd_vd(d, vcast_vd_d(0.0))),
                ddsub_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116, 1.2246467991473532072e-16), y), y);
  
    vadd_vd_vd_vd(vd2getx_vd_vd2(y), vd2gety_vd_vd2(y))
}
  
pub fn xatan_u1(d:vdouble)->vdouble{
    let d2 = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.0)), vcast_vd2_d_d(1.0, 0.0));
    let mut r = vadd_vd_vd_vd(vd2getx_vd_vd2(d2), vd2gety_vd_vd2(d2));
    r = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vcast_vd_d(1.570796326794896557998982), r);
    vmulsign_vd_vd_vd(r, d)
}

pub fn xexp(d:vdouble)->vdouble{
    let mut u = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(R_LN2)));
    let q = vrint_vi_vd(u);
  
    let mut s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L2U), d);
    s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L2L), s);
  
    let s2 = vmul_vd_vd_vd(s, s);
    let s4 = vmul_vd_vd_vd(s2, s2);
    let s8 = vmul_vd_vd_vd(s4, s4);
    u = POLY10(s, s2, s4, s8,
           2.08860621107283687536341e-09,
           2.51112930892876518610661e-08,
           2.75573911234900471893338e-07,
           2.75572362911928827629423e-06,
           2.4801587159235472998791e-05,
           0.000198412698960509205564975,
           0.00138888888889774492207962,
           0.00833333333331652721664984,
           0.0416666666666665047591422,
           0.166666666666666851703837);
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.5000000000000000000e+0));
    u = vadd_vd_vd_vd(vcast_vd_d(1.0), vmla_vd_vd_vd_vd(vmul_vd_vd_vd(s, s), u, s));    
    u = vldexp2_vd_vd_vi(u, q);
  
    u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d, vcast_vd_d(709.78271114955742909217217426)), vcast_vd_d(SLEEF_INFINITY), u);
    u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d, vcast_vd_d(-1000.0)), vreinterpret_vm_vd(u)));
  
    u
}

pub fn expm1k(d:vdouble)->vdouble{
    let u = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(R_LN2)));
    let q = vrint_vi_vd(u);
  
    let mut s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L2U), d);
    s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L2L), s);
  
    let s2 = vmul_vd_vd_vd(s, s);
    let s4 = vmul_vd_vd_vd(s2, s2);
    let s8 = vmul_vd_vd_vd(s4, s4);
    let mut u = POLY10(s, s2, s4, s8,
           2.08860621107283687536341e-09,
           2.51112930892876518610661e-08,
           2.75573911234900471893338e-07,
           2.75572362911928827629423e-06,
           2.4801587159235472998791e-05,
           0.000198412698960509205564975,
           0.00138888888889774492207962,
           0.00833333333331652721664984,
           0.0416666666666665047591422,
           0.166666666666666851703837);
  
    u = vadd_vd_vd_vd(vmla_vd_vd_vd_vd(s2, vcast_vd_d(0.5), vmul_vd_vd_vd(vmul_vd_vd_vd(s2, s), u)), s);
    
    u = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(q, vcast_vi_i(0))), u,
                 vsub_vd_vd_vd(vldexp2_vd_vd_vi(vadd_vd_vd_vd(u, vcast_vd_d(1.0)), q), vcast_vd_d(1.0)));
  
    u
}

pub fn logk(d:vdouble)->vdouble2{
    let o = vlt_vo_vd_vd(d, vcast_vd_d(SLEEF_DBL_MIN));
    let d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((1u64 << 32) as f64 * (1u64 << 32) as f64)), d);
    let mut e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
    let m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
    e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);
  
  
    let mut x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1.0), m), ddadd2_vd2_vd_vd(vcast_vd_d(1.0), m));
    let x2 = ddsqu_vd2_vd2(x);
  
    let x4 = vmul_vd_vd_vd(vd2getx_vd_vd2(x2), vd2getx_vd_vd2(x2));
    let x8 = vmul_vd_vd_vd(x4, x4);
    let x16 = vmul_vd_vd_vd(x8, x8);
    let t = POLY9(vd2getx_vd_vd2(x2), x4, x8, x16,
          0.116255524079935043668677,
          0.103239680901072952701192,
          0.117754809412463995466069,
          0.13332981086846273921509,
          0.153846227114512262845736,
          0.181818180850050775676507,
          0.222222222230083560345903,
          0.285714285714249172087875,
          0.400000000000000077715612);
  
    let c = vcast_vd2_d_d(0.666666666666666629659233, 3.80554962542412056336616e-17);
    let mut s = ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e));
  
    s = ddadd_vd2_vd2_vd2(s, ddscale_vd2_vd2_vd(x, vcast_vd_d(2.0)));
    x = ddmul_vd2_vd2_vd2(x2, x);
    s = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd2(x, c));
    x = ddmul_vd2_vd2_vd2(x2, x);
    s = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd(x, t));
  
    s
}


pub fn xlog_u1(mut d:vdouble)->vdouble{

    let o = vlt_vo_vd_vd(d, vcast_vd_d(SLEEF_DBL_MIN));
    d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((1u64 << 32) as f64 * (1u64 << 32) as f64 )), d);
    let mut e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
    let m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
    e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);


    let x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1.0), m), ddadd2_vd2_vd_vd(vcast_vd_d(1.0), m));
    let x2 = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));

    let x4 = vmul_vd_vd_vd(x2, x2);
    let x8 = vmul_vd_vd_vd(x4, x4);
    let t = POLY7(x2, x4, x8,
	    0.1532076988502701353e+0,
	    0.1525629051003428716e+0,
	    0.1818605932937785996e+0,
	    0.2222214519839380009e+0,
	    0.2857142932794299317e+0,
	    0.3999999999635251990e+0,
	    0.6666666666667333541e+0);

    let s = ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e));

    let mut s = ddadd_vd2_vd2_vd2(s, ddscale_vd2_vd2_vd(x, vcast_vd_d(2.0)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vmul_vd_vd_vd(x2, vd2getx_vd_vd2(x)), t));

  let mut r = vadd_vd_vd_vd(vd2getx_vd_vd2(s), vd2gety_vd_vd2(s));

    r = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(SLEEF_INFINITY), r);
    r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(d, vcast_vd_d(0.0)), visnan_vo_vd(d)), vcast_vd_d(SLEEF_NAN), r);
    r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), vcast_vd_d(-SLEEF_INFINITY), r);

  
    r
}

pub fn expk(d:vdouble2)->vdouble{
    let u = vmul_vd_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)), vcast_vd_d(R_LN2));
    let dq = vrint_vd_vd(u);
    let q = vrint_vi_vd(dq);
  
    let mut s = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(dq, vcast_vd_d(-L2U)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dq, vcast_vd_d(-L2L)));
  
    s = ddnormalize_vd2_vd2(s);
  
    let s2 = vmul_vd_vd_vd(vd2getx_vd_vd2(s), vd2getx_vd_vd2(s));
    let s4 = vmul_vd_vd_vd(s2, s2);
    let s8 = vmul_vd_vd_vd(s4, s4);
    let mut u = POLY10(vd2getx_vd_vd2(s), s2, s4, s8,
           2.51069683420950419527139e-08,
           2.76286166770270649116855e-07,
           2.75572496725023574143864e-06,
           2.48014973989819794114153e-05,
           0.000198412698809069797676111,
           0.0013888888939977128960529,
           0.00833333333332371417601081,
           0.0416666666665409524128449,
           0.166666666666666740681535,
           0.500000000000000999200722);
  
    let mut t = ddadd_vd2_vd_vd2(vcast_vd_d(1.0), s);
    t = ddadd_vd2_vd2_vd2(t, ddmul_vd2_vd2_vd(ddsqu_vd2_vd2(s), u));
  
    u = vadd_vd_vd_vd(vd2getx_vd_vd2(t), vd2gety_vd_vd2(t));
    u = vldexp2_vd_vd_vi(u, q);
  
    u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(-1000.0)), vreinterpret_vm_vd(u)));
    
    u
}

pub fn xpow(x:vdouble, y:vdouble)->vdouble{
    let yisint = visint_vo_vd(y);
    let yisodd = vand_vo_vo_vo(visodd_vo_vd(y), yisint);
    
    let d = ddmul_vd2_vd2_vd(logk(vabs_vd_vd(x)), y);
    let mut result = expk(d);
    result = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(709.78271114955742909217217426)), vcast_vd_d(SLEEF_INFINITY), result);
    
    result = vmul_vd_vd_vd(result,
                 vsel_vd_vo_vd_vd(vgt_vo_vd_vd(x, vcast_vd_d(0.0)),
                          vcast_vd_d(1.0),
                          vsel_vd_vo_vd_vd(yisint, vsel_vd_vo_vd_vd(yisodd, vcast_vd_d(-1.0), vcast_vd_d(1.0)), vcast_vd_d(SLEEF_NAN))));
    
    let efx = vmulsign_vd_vd_vd(vsub_vd_vd_vd(vabs_vd_vd(x), vcast_vd_d(1.0)), y);

    result = vsel_vd_vo_vd_vd(visinf_vo_vd(y),
                vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(efx, vcast_vd_d(0.0)),
                                    vreinterpret_vm_vd(vsel_vd_vo_vd_vd(veq_vo_vd_vd(efx, vcast_vd_d(0.0)),
                                                        vcast_vd_d(1.0),
                                                        vcast_vd_d(SLEEF_INFINITY))))),
                    result);
    
    result = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0.0))),
                vmulsign_vd_vd_vd(vsel_vd_vo_vd_vd(vxor_vo_vo_vo(vsignbit_vo_vd(y), veq_vo_vd_vd(x, vcast_vd_d(0.0))),
                                    vcast_vd_d(0.0), vcast_vd_d(SLEEF_INFINITY)),
                            vsel_vd_vo_vd_vd(yisodd, x, vcast_vd_d(1.0))), result);
    
    result = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(result)));
    
    result = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(y, vcast_vd_d(0.0)), veq_vo_vd_vd(x, vcast_vd_d(1.0))), vcast_vd_d(1.0), result);
    
    result
    
}

pub fn expk2(d:vdouble2)->vdouble2{
    let u = vmul_vd_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)), vcast_vd_d(R_LN2));
    let dq = vrint_vd_vd(u);
    let q = vrint_vi_vd(dq);
  
    let mut s = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(dq, vcast_vd_d(-L2U)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dq, vcast_vd_d(-L2L)));
  
    let s2 = ddsqu_vd2_vd2(s);
    let s4 = ddsqu_vd2_vd2(s2);
    let s8 = vmul_vd_vd_vd(vd2getx_vd_vd2(s4), vd2getx_vd_vd2(s4));
    let u = POLY10(vd2getx_vd_vd2(s), vd2getx_vd_vd2(s2), vd2getx_vd_vd2(s4), s8,
           0.1602472219709932072e-9,
           0.2092255183563157007e-8,
           0.2505230023782644465e-7,
           0.2755724800902135303e-6,
           0.2755731892386044373e-5,
           0.2480158735605815065e-4,
           0.1984126984148071858e-3,
           0.1388888888886763255e-2,
           0.8333333333333347095e-2,
           0.4166666666666669905e-1);
  
    let mut t = ddadd_vd2_vd_vd2(vcast_vd_d(0.5), ddmul_vd2_vd2_vd(s, vcast_vd_d(0.1666666666666666574e+0)));
    t = ddadd_vd2_vd_vd2(vcast_vd_d(1.0), ddmul_vd2_vd2_vd2(t, s));
    t = ddadd_vd2_vd_vd2(vcast_vd_d(1.0), ddmul_vd2_vd2_vd2(t, s));
    t = ddadd_vd2_vd2_vd2(t, ddmul_vd2_vd2_vd(s4, u));
  
    t = vd2setx_vd2_vd2_vd(t, vldexp2_vd_vd_vi(vd2getx_vd_vd2(t), q));
    t = vd2sety_vd2_vd2_vd(t, vldexp2_vd_vd_vi(vd2gety_vd_vd2(t), q));
  
    t = vd2setx_vd2_vd2_vd(t, vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(-1000.0)), vreinterpret_vm_vd(vd2getx_vd_vd2(t)))));
    t = vd2sety_vd2_vd2_vd(t, vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(-1000.0)), vreinterpret_vm_vd(vd2gety_vd_vd2(t)))));
  
    t
}



pub fn xsinh(x:vdouble)->vdouble{
    let y = vabs_vd_vd(x);
    let d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0.0)));
    let d = ddsub_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
    let mut y = vmul_vd_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)), vcast_vd_d(0.5));

    y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(710.0)), visnan_vo_vd(y)), vcast_vd_d(SLEEF_INFINITY), y);
    y = vmulsign_vd_vd_vd(y, x);
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

    y
}

pub fn xcosh(x:vdouble)->vdouble{
    let mut y = vabs_vd_vd(x);
    let mut d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0.0)));
    d = ddadd_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
    y = vmul_vd_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)), vcast_vd_d(0.5));

    y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(710.0)), visnan_vo_vd(y)), vcast_vd_d(SLEEF_INFINITY), y);
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

    y
}

pub fn xtanh(x:vdouble)->vdouble{
    let mut y = vabs_vd_vd(x);
    let mut d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0.0)));
    let e = ddrec_vd2_vd2(d);
    d = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd2(d, ddneg_vd2_vd2(e)), ddadd2_vd2_vd2_vd2(d, e));
    y = vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d));

    y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(18.714973875)), visnan_vo_vd(y)), vcast_vd_d(1.0), y);
    y = vmulsign_vd_vd_vd(y, x);
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

    y
}

pub fn logk2(d:vdouble2)->vdouble2{
    let e = vilogbk_vi_vd(vmul_vd_vd_vd(vd2getx_vd_vd2(d), vcast_vd_d(1.0/0.75)));
  
    let m = vd2setxy_vd2_vd_vd(vldexp2_vd_vd_vi(vd2getx_vd_vd2(d), vneg_vi_vi(e)), 
               vldexp2_vd_vd_vi(vd2gety_vd_vd2(d), vneg_vi_vi(e)));
  
    let x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(m, vcast_vd_d(-1.0)), ddadd2_vd2_vd2_vd(m, vcast_vd_d(1.0)));
    let x2 = ddsqu_vd2_vd2(x);
  
    let x4 = vmul_vd_vd_vd(vd2getx_vd_vd2(x2), vd2getx_vd_vd2(x2));
    let x8 = vmul_vd_vd_vd(x4, x4);
    let mut t = POLY7(vd2getx_vd_vd2(x2), x4, x8,
          0.13860436390467167910856,
          0.131699838841615374240845,
          0.153914168346271945653214,
          0.181816523941564611721589,
          0.22222224632662035403996,
          0.285714285511134091777308,
          0.400000000000914013309483);
    t = vmla_vd_vd_vd_vd(t, vd2getx_vd_vd2(x2), vcast_vd_d(0.666666666666664853302393));
  
    let mut s = ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e));
    s = ddadd_vd2_vd2_vd2(s, ddscale_vd2_vd2_vd(x, vcast_vd_d(2.0)));
    s = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(x2, x), t));
  
    s
}

pub fn xasinh(x:vdouble)->vdouble {
    let y = vabs_vd_vd(x);
    let o = vgt_vo_vd_vd(y, vcast_vd_d(1.0));
    
    let mut d = vsel_vd2_vo_vd2_vd2(o, ddrec_vd2_vd(x), vcast_vd2_vd_vd(y, vcast_vd_d(0.0)));
    d = ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddsqu_vd2_vd2(d), vcast_vd_d(1.0)));
    d = vsel_vd2_vo_vd2_vd2(o, ddmul_vd2_vd2_vd(d, y), d);
  
    d = logk2(ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd(d, x)));
    let mut y = vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d));
    
    y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(SQRT_DBL_MAX)),
                      visnan_vo_vd(y)),
                 vmulsign_vd_vd_vd(vcast_vd_d(SLEEF_INFINITY), x), y);
  
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));
    y = vsel_vd_vo_vd_vd(visnegzero_vo_vd(x), vcast_vd_d(-0.0), y);
    
    y
}
  
pub fn xacosh(x:vdouble)->vdouble{
    let d = logk2(ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddsqrt_vd2_vd2(ddadd2_vd2_vd_vd(x, vcast_vd_d(1.0))), ddsqrt_vd2_vd2(ddadd2_vd2_vd_vd(x, vcast_vd_d(-1.0)))), x));
    let mut y = vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d));
  
    y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(SQRT_DBL_MAX)),
                      visnan_vo_vd(y)),
                 vcast_vd_d(SLEEF_INFINITY), y);
    y = vreinterpret_vd_vm(vandnot_vm_vo64_vm(veq_vo_vd_vd(x, vcast_vd_d(1.0)), vreinterpret_vm_vd(y)));
  
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(vlt_vo_vd_vd(x, vcast_vd_d(1.0)), vreinterpret_vm_vd(y)));
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));
    
    y
}
  
pub fn xatanh(x:vdouble)->vdouble{
    let mut y = vabs_vd_vd(x);
    let d = logk2(dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(1.0), y), ddadd2_vd2_vd_vd(vcast_vd_d(1.0), vneg_vd_vd(y))));
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(vgt_vo_vd_vd(y, vcast_vd_d(1.0)), vreinterpret_vm_vd(vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(1.0)), vcast_vd_d(SLEEF_INFINITY), vmul_vd_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)), vcast_vd_d(0.5))))));
  
    y = vmulsign_vd_vd_vd(y, x);
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visinf_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(y)));
    y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));
  
    y
}


pub fn xcbrt_u1(d:vdouble)->vdouble{
    let q2 = vcast_vd2_d_d(1.0, 0.0);
  
    let e = vadd_vi_vi_vi(vilogbk_vi_vd(vabs_vd_vd(d)), vcast_vi_i(1));
    let mut d = vldexp2_vd_vd_vi(d, vneg_vi_vi(e));
  
    let t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144.0));
    let qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
    let re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3.0))));
  
    let mut q2 = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(1))), vcast_vd2_d_d(1.2599210498948731907, -2.5899333753005069177e-17), q2);
    q2 = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(2))), vcast_vd2_d_d(1.5874010519681995834, -1.0869008194197822986e-16), q2);
  
    q2 = vd2setxy_vd2_vd_vd(vmulsign_vd_vd_vd(vd2getx_vd_vd2(q2), d), vmulsign_vd_vd_vd(vd2gety_vd_vd2(q2), d));
    d = vabs_vd_vd(d);
  
    let mut x = vcast_vd_d(-0.640245898480692909870982);
    x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.96155103020039511818595));
    x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-5.73353060922947843636166));
    x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(6.03990368989458747961407));
    x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-3.85841935510444988821632));
    x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.2307275302496609725722));
  
    let mut y = vmul_vd_vd_vd(x, x); y = vmul_vd_vd_vd(y, y); x = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vmlapn_vd_vd_vd_vd(d, y, x), vcast_vd_d(1.0 / 3.0)));
  
    let mut z = x;
  
    let mut u = ddmul_vd2_vd_vd(x, x);
    u = ddmul_vd2_vd2_vd2(u, u);
    u = ddmul_vd2_vd2_vd(u, d);
    u = ddadd2_vd2_vd2_vd(u, vneg_vd_vd(x));
    y = vadd_vd_vd_vd(vd2getx_vd_vd2(u), vd2gety_vd_vd2(u));
  
    y = vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(-2.0 / 3.0), y), z);
    let mut v = ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd(z, z), y);
    v = ddmul_vd2_vd2_vd(v, d);
    v = ddmul_vd2_vd2_vd2(v, q2);
    z = vldexp2_vd_vd_vi(vadd_vd_vd_vd(vd2getx_vd_vd2(v), vd2gety_vd_vd2(v)), vsub_vi_vi_vi(qu, vcast_vi_i(2048)));
  
    z = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vmulsign_vd_vd_vd(vcast_vd_d(SLEEF_INFINITY), vd2getx_vd_vd2(q2)), z);
    z = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), vreinterpret_vd_vm(vsignbit_vm_vd(vd2getx_vd_vd2(q2))), z);

    
    z
}

pub fn xexp2(d:vdouble)->vdouble{
    let u = vrint_vd_vd(d);
    let q = vrint_vi_vd(u);
  
    let s = vsub_vd_vd_vd(d, u);
  
    let s2 = vmul_vd_vd_vd(s, s);
    let s4 = vmul_vd_vd_vd(s2, s2);
    let s8 = vmul_vd_vd_vd(s4, s4);
    let mut u = POLY10(s, s2, s4, s8,
           0.4434359082926529454e-9,
           0.7073164598085707425e-8,
           0.1017819260921760451e-6,
           0.1321543872511327615e-5,
           0.1525273353517584730e-4,
           0.1540353045101147808e-3,
           0.1333355814670499073e-2,
           0.9618129107597600536e-2,
           0.5550410866482046596e-1,
           0.2402265069591012214e+0);
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.6931471805599452862e+0));
    
    u = vd2getx_vd_vd2(ddnormalize_vd2_vd2(ddadd_vd2_vd_vd2(vcast_vd_d(1.0), ddmul_vd2_vd_vd(u, s))));
    
    u = vldexp2_vd_vd_vi(u, q);
  
    u = vsel_vd_vo_vd_vd(vge_vo_vd_vd(d, vcast_vd_d(1024.0)), vcast_vd_d(SLEEF_INFINITY), u);
    u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d, vcast_vd_d(-2000.0)), vreinterpret_vm_vd(u)));
  
    u
}


pub fn xexp10(d:vdouble)->vdouble{
    let mut u = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(LOG10_2)));
    let q = vrint_vi_vd(u);
  
    let mut s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L10U), d);
    s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L10L), s);
  
    u = vcast_vd_d(0.2411463498334267652e-3);
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.1157488415217187375e-2));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.5013975546789733659e-2));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.1959762320720533080e-1));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.6808936399446784138e-1));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.2069958494722676234e+0));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.5393829292058536229e+0));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.1171255148908541655e+1));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.2034678592293432953e+1));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.2650949055239205876e+1));
    u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.2302585092994045901e+1));
    
    u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(1.0));
    
    u = vldexp2_vd_vd_vi(u, q);
  
    u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d, vcast_vd_d(308.25471555991671)), vcast_vd_d(SLEEF_INFINITY), u);
    u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d, vcast_vd_d(-350.0)), vreinterpret_vm_vd(u)));
  
    u
}

pub fn xexpm1(a:vdouble)->vdouble{
    let d = ddadd2_vd2_vd2_vd(expk2(vcast_vd2_vd_vd(a, vcast_vd_d(0.0))), vcast_vd_d(-1.0));
    let mut x = vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d));
    x = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(709.782712893383996732223)), vcast_vd_d(SLEEF_INFINITY), x);
    x = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(a, vcast_vd_d(-36.736800569677101399113302437)), vcast_vd_d(-1.0), x);
    x = vsel_vd_vo_vd_vd(visnegzero_vo_vd(a), vcast_vd_d(-0.0), x);
    x
}

pub fn xlog10(mut d:vdouble)->vdouble{
    let o = vlt_vo_vd_vd(d, vcast_vd_d(SLEEF_DBL_MIN));
    d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((1u64 << 32) as f64 * (1u64 << 32) as f64 )), d);
    let mut e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
    let m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
    e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);

  
    let x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1.0), m), ddadd2_vd2_vd_vd(vcast_vd_d(1.0), m));
    let x2 = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));
  
    let x4 = vmul_vd_vd_vd(x2, x2);
    let x8 = vmul_vd_vd_vd(x4, x4);
    let t = POLY7(x2, x4, x8,
          0.6653725819576758460e-1,
          0.6625722782820833712e-1,
          0.7898105214313944078e-1,
          0.9650955035715275132e-1,
          0.1240841409721444993e+0,
          0.1737177927454605086e+0,
          0.2895296546021972617e+0);
    
    let mut s = ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.30102999566398119802, -2.803728127785170339e-18), vcast_vd_vi(e));

  
    s = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd2(x, vcast_vd2_d_d(0.86858896380650363334, 1.1430059694096389311e-17)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vmul_vd_vd_vd(x2, vd2getx_vd_vd2(x)), t));
  
    let mut r = vadd_vd_vd_vd(vd2getx_vd_vd2(s), vd2gety_vd_vd2(s));
  
    r = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(SLEEF_INFINITY), r);
    r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(d, vcast_vd_d(0.0)), visnan_vo_vd(d)), vcast_vd_d(SLEEF_NAN), r);
    r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), vcast_vd_d(-SLEEF_INFINITY), r);
    
    r
}


pub fn xlog2(mut d:vdouble)->vdouble{
    let o = vlt_vo_vd_vd(d, vcast_vd_d(SLEEF_DBL_MIN));
    d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((1u64 << 32) as f64 * (1u64 << 32) as f64)), d);
    let mut e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
    let m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
    e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);
  
  
    let x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1.0), m), ddadd2_vd2_vd_vd(vcast_vd_d(1.0), m));
    let x2 = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));
  
    let x4 = vmul_vd_vd_vd(x2, x2);
    let x8 = vmul_vd_vd_vd(x4, x4);
    let t = POLY7(x2, x4, x8,
          0.2211941750456081490e+0,
          0.2200768693152277689e+0,
          0.2623708057488514656e+0,
          0.3205977477944495502e+0,
          0.4121985945485324709e+0,
          0.5770780162997058982e+0,
          0.96179669392608091449);
    
    let mut s = ddadd2_vd2_vd_vd2(vcast_vd_vi(e),
                   ddmul_vd2_vd2_vd2(x, vcast_vd2_d_d(2.885390081777926774, 6.0561604995516736434e-18)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(vmul_vd_vd_vd(x2, vd2getx_vd_vd2(x)), t));
  
    let mut r = vadd_vd_vd_vd(vd2getx_vd_vd2(s), vd2gety_vd_vd2(s));
  
    r = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(SLEEF_INFINITY), r);
    r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(d, vcast_vd_d(0.0)), visnan_vo_vd(d)), vcast_vd_d(SLEEF_NAN), r);
    r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), vcast_vd_d(-SLEEF_INFINITY), r);
  
    r
}


pub fn xlog1p(d:vdouble)->vdouble{
    let dp1 = vadd_vd_vd_vd(d, vcast_vd_d(1.0));
  
    let o = vlt_vo_vd_vd(dp1, vcast_vd_d(SLEEF_DBL_MIN));
    let dp1 = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(dp1, vcast_vd_d((1u64 << 32) as f64 * (1u64 << 32) as f64)), dp1);
    let mut e = vilogb2k_vi_vd(vmul_vd_vd_vd(dp1, vcast_vd_d(1.0/0.75)));
    let t = vldexp3_vd_vd_vi(vcast_vd_d(1.0), vneg_vi_vi(e));
    let m = vmla_vd_vd_vd_vd(d, t, vsub_vd_vd_vd(t, vcast_vd_d(1.0)));
    e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);
    let s = ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e));
  
    let x = dddiv_vd2_vd2_vd2(vcast_vd2_vd_vd(m, vcast_vd_d(0.0)), ddadd_vd2_vd_vd(vcast_vd_d(2.0), m));
    let x2 = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));
  
    let x4 = vmul_vd_vd_vd(x2, x2);
    let x8 = vmul_vd_vd_vd(x4, x4);
    let t = POLY7(x2, x4, x8,
          0.1532076988502701353e+0,
          0.1525629051003428716e+0,
          0.1818605932937785996e+0,
          0.2222214519839380009e+0,
          0.2857142932794299317e+0,
          0.3999999999635251990e+0,
          0.6666666666667333541e+0);
    
    let mut s = ddadd_vd2_vd2_vd2(s, ddscale_vd2_vd2_vd(x, vcast_vd_d(2.0)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vmul_vd_vd_vd(x2, vd2getx_vd_vd2(x)), t));
  
    let mut r = vadd_vd_vd_vd(vd2getx_vd_vd2(s), vd2gety_vd_vd2(s));
    
    r = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d, vcast_vd_d(1e+307)), vcast_vd_d(SLEEF_INFINITY), r);
    r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(d, vcast_vd_d(-1.0)), visnan_vo_vd(d)), vcast_vd_d(SLEEF_NAN), r);
    r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(-1.0)), vcast_vd_d(-SLEEF_INFINITY), r);
    r = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), r);
    
    r
}


pub fn xfdim(x:vdouble, y:vdouble)->vdouble{
    let ret = vsub_vd_vd_vd(x, y);
    vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(ret, vcast_vd_d(0.0)), veq_vo_vd_vd(x, y)), vcast_vd_d(0.0), ret)
}

pub fn xround(x:vdouble)->vdouble{
    vround2_vd_vd(x)
}

pub fn xnextafter(mut x:vdouble, y:vdouble)->vdouble{
    x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0.0)), vmulsign_vd_vd_vd(vcast_vd_d(0.0), y), x);
    let mut xi2 = vreinterpret_vm_vd(x);
    let c = vxor_vo_vo_vo(vsignbit_vo_vd(x), vge_vo_vd_vd(y, x));
  
    xi2 = vsel_vm_vo64_vm_vm(c, vneg64_vm_vm(vxor_vm_vm_vm(xi2, vcast_vm_i_i((1u32 << 31) as i32, 0))), xi2);
  
    xi2 = vsel_vm_vo64_vm_vm(vneq_vo_vd_vd(x, y), vsub64_vm_vm_vm(xi2, vcast_vm_i_i(0, 1)), xi2);
  
    xi2 = vsel_vm_vo64_vm_vm(c, vneg64_vm_vm(vxor_vm_vm_vm(xi2, vcast_vm_i_i((1u32 << 31) as i32, 0))), xi2);
  
    let mut ret = vreinterpret_vd_vm(xi2);
  
    ret = vsel_vd_vo_vd_vd(vand_vo_vo_vo(veq_vo_vd_vd(ret, vcast_vd_d(0.0)), vneq_vo_vd_vd(x, vcast_vd_d(0.0))), 
               vmulsign_vd_vd_vd(vcast_vd_d(0.0), x), ret);
  
    ret = vsel_vd_vo_vd_vd(vand_vo_vo_vo(veq_vo_vd_vd(x, vcast_vd_d(0.0)), veq_vo_vd_vd(y, vcast_vd_d(0.0))), y, ret);
  
    ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vcast_vd_d(SLEEF_NAN), ret);
    
    ret
}

pub fn xsqrt(d:vdouble)->vdouble{
    vsqrt_vd_vd(d)
}

pub fn xhypot_u05(mut x:vdouble, mut y:vdouble)->vdouble{
    x = vabs_vd_vd(x);
    y = vabs_vd_vd(y);
    let min = vmin_vd_vd_vd(x, y);
    let mut n = min;
    let max = vmax_vd_vd_vd(x, y);
    let mut d = max;
  
    let o = vlt_vo_vd_vd(max, vcast_vd_d(SLEEF_DBL_MIN));
    n = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(n, vcast_vd_d( (1u64 << 54) as f64 )), n);
    d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d( (1u64 << 54) as f64)), d);
  
    let mut t = dddiv_vd2_vd2_vd2(vcast_vd2_vd_vd(n, vcast_vd_d(0.0)), vcast_vd2_vd_vd(d, vcast_vd_d(0.0)));
    t = ddmul_vd2_vd2_vd(ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddsqu_vd2_vd2(t), vcast_vd_d(1.0))), max);
    let mut ret = vadd_vd_vd_vd(vd2getx_vd_vd2(t), vd2gety_vd_vd2(t));
    ret = vsel_vd_vo_vd_vd(visnan_vo_vd(ret), vcast_vd_d(SLEEF_INFINITY), ret);
    ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(min, vcast_vd_d(0.0)), max, ret);
    ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vcast_vd_d(SLEEF_NAN), ret);
    ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(x, vcast_vd_d(SLEEF_INFINITY)), veq_vo_vd_vd(y, vcast_vd_d(SLEEF_INFINITY))), vcast_vd_d(SLEEF_INFINITY), ret);
  
    ret
}

pub fn vptrunc_vd_vd(x:vdouble)->vdouble{ // round to integer toward 0, positive argument only
    vtruncate_vd_vd(x)
}

pub fn xfmod(x:vdouble, y:vdouble)->vdouble{
    let mut n = vabs_vd_vd(x);
    let mut d = vabs_vd_vd(y);
    let mut s = vcast_vd_d(1.0);
    let o = vlt_vo_vd_vd(d, vcast_vd_d(SLEEF_DBL_MIN));
    n = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(n, vcast_vd_d( (1u64 << 54) as f64 )), n);
    d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d( (1u64 << 54) as f64 )), d);
    s  = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(s , vcast_vd_d(1.0 / (1u64 << 54) as f64 )), s);
    let mut r = vcast_vd2_vd_vd(n, vcast_vd_d(0.0));
    let rd = vtoward0_vd_vd(vrec_vd_vd(d));
  
    for _ in 0..21{     // // ceil(log2(DBL_MAX) / 52)
        let mut q = vptrunc_vd_vd(vmul_vd_vd_vd(vtoward0_vd_vd(vd2getx_vd_vd2(r)), rd));
        q = vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(q), vcast_vm_u64(0xfffffffffffffffeu64)));
        q = vsel_vd_vo_vd_vd(vand_vo_vo_vo(vgt_vo_vd_vd(vmul_vd_vd_vd(vcast_vd_d(3.0), d), vd2getx_vd_vd2(r)),
                         vge_vo_vd_vd(vd2getx_vd_vd2(r), d)),
               vcast_vd_d(2.0), q);
        q = vsel_vd_vo_vd_vd(vand_vo_vo_vo(vgt_vo_vd_vd(vadd_vd_vd_vd(d, d), vd2getx_vd_vd2(r)),
                         vge_vo_vd_vd(vd2getx_vd_vd2(r), d)),
               vcast_vd_d(1.0), q);
        r = ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd2(r, ddmul_vd2_vd_vd(q, vneg_vd_vd(d))));
        if 0 != vtestallones_i_vo64(vlt_vo_vd_vd(vd2getx_vd_vd2(r), d)){
            break;
        }
    }
    
    let mut ret = vmul_vd_vd_vd(vd2getx_vd_vd2(r), s);
    ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(r), vd2gety_vd_vd2(r)), d), vcast_vd_d(0.0), ret);
  
    ret = vmulsign_vd_vd_vd(ret, x);
  
    ret = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(n, d), x, ret);
    ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.0)), vcast_vd_d(SLEEF_NAN), ret);
  
    ret
}

pub fn vrintk2_vd_vd(d:vdouble)->vdouble{
    vrint_vd_vd(d)
}

pub fn ddmla_vd2_vd_vd2_vd2(x:vdouble, y:vdouble2, z:vdouble2)->vdouble2{
    ddadd_vd2_vd2_vd2(z, ddmul_vd2_vd2_vd(y, x))
}
  
pub fn poly2dd_b(x:vdouble, c1:vdouble2, c0:vdouble2)->vdouble2{
    ddmla_vd2_vd_vd2_vd2(x, c1, c0)
}

pub fn poly2dd(x:vdouble, c1:vdouble, c0:vdouble2)->vdouble2{
    ddmla_vd2_vd_vd2_vd2(x, vcast_vd2_vd_vd(c1, vcast_vd_d(0.0)), c0)
}

pub fn poly4dd(x:vdouble, c3:vdouble, c2:vdouble2, c1:vdouble2, c0:vdouble2)->vdouble2{
    ddmla_vd2_vd_vd2_vd2(vmul_vd_vd_vd(x, x), poly2dd(x, c3, c2), poly2dd_b(x, c1, c0))
}

