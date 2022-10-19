use core::{arch::x86_64::*};
use crate::arr4com::sleef::df::*;
use crate::arr4com::sleef::helperavx2::*;
use crate::arr4com::sleef::misc::*;
use crate::arr4com::sleef::rempitab::Sleef_rempitabsp;

pub struct vquad{
    pub x: vmask,
    y: vmask,
}

pub  fn MLA(x:vfloat, y:vfloat, z:vfloat)->vfloat{
    vmla_vf_vf_vf_vf(x, y, z)
}
pub fn C2V(c:f32)->vfloat{
    vcast_vf_f(c)
}

pub fn visnegzero_vo_vf(d:vfloat)->vopmask {
    veq_vo_vi2_vi2(vreinterpret_vi2_vf(d), vreinterpret_vi2_vf(vcast_vf_f(-0.0f32)))
}
  
pub fn vnot_vo32_vo32(x:vopmask)->vopmask {
    vxor_vo_vo_vo(x, veq_vo_vi2_vi2(vcast_vi2_i(0), vcast_vi2_i(0)))
}
  
pub fn vsignbit_vm_vf(f:vfloat)->vmask{
    vand_vm_vm_vm(vreinterpret_vm_vf(f), vreinterpret_vm_vf(vcast_vf_f(-0.0f32)))
}
  
pub fn vmulsign_vf_vf_vf(x:vfloat, y:vfloat)->vfloat{
    vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(x), vsignbit_vm_vf(y)))
}

pub fn vcopysign_vf_vf_vf(x:vfloat, y:vfloat)->vfloat{
    vreinterpret_vf_vm(vxor_vm_vm_vm(vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f32)), vreinterpret_vm_vf(x)), 
                        vand_vm_vm_vm   (vreinterpret_vm_vf(vcast_vf_f(-0.0f32)), vreinterpret_vm_vf(y))))
}

pub fn vsign_vf_vf(f:vfloat)->vfloat{
    vreinterpret_vf_vm(vor_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(1.0f32)), vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f32)), vreinterpret_vm_vf(f))))
}

pub fn vsignbit_vo_vf(d:vfloat)->vopmask{
    veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0x80000000u32 as i32)), vcast_vi2_i(0x80000000u32 as i32))
}

pub fn vsel_vi2_vf_vf_vi2_vi2(f0:vfloat, f1:vfloat, x:vint2, y:vint2)->vint2{
    vsel_vi2_vo_vi2_vi2(vlt_vo_vf_vf(f0, f1), x, y)
}

pub fn vsel_vi2_vf_vi2(d:vfloat, x:vint2)->vint2{
    vand_vi2_vo_vi2(vsignbit_vo_vf(d), x)
}

pub fn visint_vo_vf(y:vfloat)->vopmask { veq_vo_vf_vf(vtruncate_vf_vf(y), y) }

pub fn visnumber_vo_vf(x:vfloat)->vopmask{ vnot_vo32_vo32(vor_vo_vo_vo(visinf_vo_vf(x), visnan_vo_vf(x))) }

pub fn vilogbk_vi2_vf(d:vfloat)->vint2{
    let o = vlt_vo_vf_vf(d, vcast_vf_f(5.421010862427522E-20f32));
    let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(vcast_vf_f(1.8446744073709552E19f32), d), d);
    let mut q = vand_vi2_vi2_vi2(vsrl_vi2_vi2_i::<23>(vreinterpret_vi2_vf(d)), vcast_vi2_i(0xff));
    vsub_vi2_vi2_vi2(q, vsel_vi2_vo_vi2_vi2(o, vcast_vi2_i(64 + 0x7f), vcast_vi2_i(0x7f)))
}

pub fn vilogb2k_vi2_vf(d:vfloat)->vint2{
    let mut q = vreinterpret_vi2_vf(d);
    q = vsrl_vi2_vi2_i::<23>(q);
    q = vand_vi2_vi2_vi2(q, vcast_vi2_i(0xff));
    vsub_vi2_vi2_vi2(q, vcast_vi2_i(0x7f))
}

pub fn xilogbf(d:vfloat)->vint2{
    let mut e = vilogbk_vi2_vf(vabs_vf_vf(d));
    e = vsel_vi2_vo_vi2_vi2(veq_vo_vf_vf(d, vcast_vf_f(0.0f32)), vcast_vi2_i(SLEEF_FP_ILOGB0), e);
    e = vsel_vi2_vo_vi2_vi2(visnan_vo_vf(d), vcast_vi2_i(SLEEF_FP_ILOGBNAN), e);
    vsel_vi2_vo_vi2_vi2(visinf_vo_vf(d), vcast_vi2_i(SLEEF_INT_MAX), e)
}
  
pub fn vpow2i_vf_vi2(q:vint2)->vfloat{
    vreinterpret_vf_vi2(vsll_vi2_vi2_i::<23>(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f))))
}
  
pub fn vldexp_vf_vf_vi2(x:vfloat, q:vint2)->vfloat{
    let mut m = vsra_vi2_vi2_i::<31>(q);
    m = vsll_vi2_vi2_i::<4>(vsub_vi2_vi2_vi2(vsra_vi2_vi2_i::<6>(vadd_vi2_vi2_vi2(m, q)), m));
    let q = vsub_vi2_vi2_vi2(q, vsll_vi2_vi2_i::<2>(m));
    m = vadd_vi2_vi2_vi2(m, vcast_vi2_i(0x7f));
    m = vand_vi2_vi2_vi2(vgt_vi2_vi2_vi2(m, vcast_vi2_i(0)), m);
    let n = vgt_vi2_vi2_vi2(m, vcast_vi2_i(0xff));
    m = vor_vi2_vi2_vi2(vandnot_vi2_vi2_vi2(n, m), vand_vi2_vi2_vi2(n, vcast_vi2_i(0xff)));
    let u = vreinterpret_vf_vi2(vsll_vi2_vi2_i::<23>(m));
    let x = vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(x, u), u), u), u);
    let u = vreinterpret_vf_vi2(vsll_vi2_vi2_i::<23>(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f))));
    vmul_vf_vf_vf(x, u)
}
  
pub fn vldexp2_vf_vf_vi2(d:vfloat, e:vint2)->vfloat{
    vmul_vf_vf_vf(vmul_vf_vf_vf(d, vpow2i_vf_vi2(vsra_vi2_vi2_i::<1>(e))), vpow2i_vf_vi2(vsub_vi2_vi2_vi2(e, vsra_vi2_vi2_i::<1>(e))))
}
  
pub fn vldexp3_vf_vf_vi2(d:vfloat, q:vint2)->vfloat{
    vreinterpret_vf_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vsll_vi2_vi2_i::<23>(q)))
}

pub fn xldexpf(x:vfloat, q:vint2)->vfloat{
    vldexp_vf_vf_vi2(x, q)
}

#[derive(Debug, Copy, Clone)]
struct fi_t{
    pub d:vfloat,
    pub i:vint2,
}

fn figetd_vf_di(d:fi_t)->vfloat { d.d }
fn figeti_vi2_di(d:fi_t)->vint2{ d.i }
fn fisetdi_fi_vf_vi2(d:vfloat, i:vint2)->fi_t{
    fi_t{d, i }
}

#[derive(Debug, Copy, Clone)]
struct dfi_t{
    pub df:vfloat2,
    pub i:vint2,
}

fn dfigetdf_vf2_dfi(d:dfi_t)->vfloat2{ d.df }
fn dfigeti_vi2_dfi(d:dfi_t)->vint2 { d.i }
fn dfisetdfi_dfi_vf2_vi2(v:vfloat2, i:vint2)->dfi_t{
    dfi_t{ df:v, i }
}
fn dfisetdf_dfi_dfi_vf2(dfi:dfi_t, v:vfloat2)->dfi_t{
    dfi_t{
        df:v,
        i:dfi.i,
    }
}

fn vorsign_vf_vf_vf(x:vfloat, y:vfloat)->vfloat{
    vreinterpret_vf_vm(vor_vm_vm_vm(vreinterpret_vm_vf(x), vsignbit_vm_vf(y)))
}
  
fn rempisubf(x:vfloat)->fi_t{
    let y = vrint_vf_vf(vmul_vf_vf_vf(x, vcast_vf_f(4f32)));
    let vi = vtruncate_vi2_vf(vsub_vf_vf_vf(y, vmul_vf_vf_vf(vrint_vf_vf(x), vcast_vf_f(4f32))));
    fisetdi_fi_vf_vi2(vsub_vf_vf_vf(x, vmul_vf_vf_vf(y, vcast_vf_f(0.25f32))), vi)
}
fn rempif(a:vfloat)->dfi_t{
    let mut x:vfloat2; let mut y:vfloat2;
    let mut ex:vint2 = vilogb2k_vi2_vf(a);
    ex = vsub_vi2_vi2_vi2(ex, vcast_vi2_i(25));
    let mut q:vint2 = vand_vi2_vo_vi2(vgt_vo_vi2_vi2(ex, vcast_vi2_i(90-25)), vcast_vi2_i(-64));
    let a = vldexp3_vf_vf_vi2(a, q);
    ex = vandnot_vi2_vi2_vi2(vsra_vi2_vi2_i::<31>(ex), ex);
    ex = vsll_vi2_vi2_i::<2>(ex);
    let mut x = dfmul_vf2_vf_vf(a, vgather_vf_p_vi2(Sleef_rempitabsp.as_ptr(), ex));
    let mut di:fi_t = rempisubf(vf2getx_vf_vf2(x));
    q = figeti_vi2_di(di);
    x = vf2setx_vf2_vf2_vf(x, figetd_vf_di(di));
    x = dfnormalize_vf2_vf2(x);
    y = dfmul_vf2_vf_vf(a, vgather_vf_p_vi2(&Sleef_rempitabsp[1], ex));
    x = dfadd2_vf2_vf2_vf2(x, y);
    di = rempisubf(vf2getx_vf_vf2(x));
    q = vadd_vi2_vi2_vi2(q, figeti_vi2_di(di));
    x = vf2setx_vf2_vf2_vf(x, figetd_vf_di(di));
    x = dfnormalize_vf2_vf2(x);
    y = vcast_vf2_vf_vf(vgather_vf_p_vi2(&Sleef_rempitabsp[2], ex), vgather_vf_p_vi2(&Sleef_rempitabsp[3], ex));
    y = dfmul_vf2_vf2_vf(y, a);
    x = dfadd2_vf2_vf2_vf2(x, y);
    x = dfnormalize_vf2_vf2(x);
    x = dfmul_vf2_vf2_vf2(x, vcast_vf2_f_f(3.1415927410125732422f32*2f32, -8.7422776573475857731e-08f32*2f32));
    x = vsel_vf2_vo_vf2_vf2(vlt_vo_vf_vf(vabs_vf_vf(a), vcast_vf_f(0.7f32)), vcast_vf2_vf_vf(a, vcast_vf_f(0f32)), x);
    dfisetdfi_dfi_vf2_vi2(x, q)
}

pub fn xsinf_u1(d:vfloat)->vfloat{
    let mut q:vint2;
    let mut u:vfloat; let mut v:vfloat;
    let mut s:vfloat2; let mut t:vfloat2; let mut x:vfloat2;

    if vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f))) != 0 {
        u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(M_1_PI as f32)));
        q = vrint_vi2_vf(u);
        v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2f), d);
        s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2f)));
        s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2f)));
    }
    else {
        let mut dfi = rempif(d);
        q = vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(3));
        q = vadd_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, q), vsel_vi2_vo_vi2_vi2(vgt_vo_vf_vf(vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)), vcast_vf_f(0f32)), vcast_vi2_i(2), vcast_vi2_i(1)));
        q = vsra_vi2_vi2_i::<2>(q);
        let o:vopmask = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(1)), vcast_vi2_i(1));
        let mut x:vfloat2 = vcast_vf2_vf_vf(vmulsign_vf_vf_vf(vcast_vf_f(3.1415927410125732422f32*-0.5), vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi))), 
                    vmulsign_vf_vf_vf(vcast_vf_f(-8.7422776573475857731e-08f32*-0.5), vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi))));
        x = dfadd2_vf2_vf2_vf2(dfigetdf_vf2_dfi(dfi), x);
        dfi = dfisetdf_dfi_dfi_vf2(dfi, vsel_vf2_vo_vf2_vf2(o, x, dfigetdf_vf2_dfi(dfi)));
        s = dfnormalize_vf2_vf2(dfigetdf_vf2_dfi(dfi));
        s.x = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d)), vreinterpret_vm_vf(s.x)));
    }

    t = s;
    s = dfsqu_vf2_vf2(s);

    u = vcast_vf_f(2.6083159809786593541503e-06f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(-0.0001981069071916863322258f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.00833307858556509017944336f32));

    x = dfadd_vf2_vf_vf2(vcast_vf_f(1f32), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(-0.166666597127914428710938f32), vmul_vf_vf_vf(u, vf2getx_vf_vf2(s))), s));

    u = dfmul_vf_vf2_vf2(t, x);

    u = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), vreinterpret_vm_vf(vcast_vf_f(-0.0f32))), vreinterpret_vm_vf(u)));

    vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), d, u)
}

pub fn xcosf_u1(d:vfloat)->vfloat{
    let mut q:vint2;
    let mut u:vfloat;
    let mut s:vfloat2;
    let mut t:vfloat2;
    let mut x:vfloat2;
    
    if vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f))) != 0 {
        let dq = vmla_vf_vf_vf_vf(vrint_vf_vf(vmla_vf_vf_vf_vf(d, vcast_vf_f(M_1_PI as f32), vcast_vf_f(-0.5f32))),
                     vcast_vf_f(2f32), vcast_vf_f(1f32));
        q = vrint_vi2_vf(dq);
        s = dfadd2_vf2_vf_vf (d, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_A2f*0.5f32)));
        s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_B2f*0.5f32)));
        s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_C2f*0.5f32)));
      } else {
        let mut dfi:dfi_t = rempif(d);
        q = vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(3));
        q = vadd_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, q), vsel_vi2_vo_vi2_vi2(vgt_vo_vf_vf(vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)), vcast_vf_f(0f32)), vcast_vi2_i(8), vcast_vi2_i(7)));
        q = vsra_vi2_vi2_i::<1>(q);
        let o:vopmask = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(dfigeti_vi2_dfi(dfi), vcast_vi2_i(1)), vcast_vi2_i(0));
        let y:vfloat = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(vf2getx_vf_vf2(dfigetdf_vf2_dfi(dfi)), vcast_vf_f(0f32)), vcast_vf_f(0f32), vcast_vf_f(-1f32));
        let mut x:vfloat2 = vcast_vf2_vf_vf(vmulsign_vf_vf_vf(vcast_vf_f(3.1415927410125732422f32*-0.5), y),
                    vmulsign_vf_vf_vf(vcast_vf_f(-8.7422776573475857731e-08f32*-0.5), y));
        x = dfadd2_vf2_vf2_vf2(dfigetdf_vf2_dfi(dfi), x);
        dfi = dfisetdf_dfi_dfi_vf2(dfi, vsel_vf2_vo_vf2_vf2(o, x, dfigetdf_vf2_dfi(dfi)));
        s = dfnormalize_vf2_vf2(dfigetdf_vf2_dfi(dfi));
        s.x = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d)), vreinterpret_vm_vf(s.x)));
    }
    
    t = s;
    s = dfsqu_vf2_vf2(s);
    
    u = vcast_vf_f(2.6083159809786593541503e-06f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(-0.0001981069071916863322258f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.00833307858556509017944336f32));

    x = dfadd_vf2_vf_vf2(vcast_vf_f(1f32), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(-0.166666597127914428710938f32), vmul_vf_vf_vf(u, vf2getx_vf_vf2(s))), s));

    u = dfmul_vf_vf2_vf2(t, x);
    u = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0)), vreinterpret_vm_vf(vcast_vf_f(-0.0f32))), vreinterpret_vm_vf(u)));
    return u;
}

pub fn XMODFF(x:vfloat)->vfloat2{
    let mut fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
    fr = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f((1i64  << 23) as f32)), vcast_vf_f(0f32), fr);
    vf2setxy_vf2_vf_vf(vcopysign_vf_vf_vf(fr, x), vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x))
}

pub fn xtanf_u1(d:vfloat)->vfloat{
    let mut q:vint2;
    let mut u:vfloat ; let mut v:vfloat;
    let mut s:vfloat2; let mut t:vfloat2; let mut x:vfloat2;
    let mut o:vopmask;

    if vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f))) != 0{
        u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f( (2f64 * M_1_PI) as f32 )));
        q = vrint_vi2_vf(u);
        v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2f*0.5f32), d);
        s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2f*0.5f32)));
        s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2f*0.5f32)));
    }else{
        let dfi = rempif(d);
        q = dfigeti_vi2_dfi(dfi);
        s = dfigetdf_vf2_dfi(dfi);
        o = vor_vo_vo_vo(visinf_vo_vf(d), visnan_vo_vf(d));
        s = vf2setx_vf2_vf2_vf(s, vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(vf2getx_vf_vf2(s)))));
        s = vf2sety_vf2_vf2_vf(s, vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(vf2gety_vf_vf2(s)))));
    }
    
    o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1));
    let n = vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0)));
    s.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(s.x), n));
    s.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(s.y), n));

    t = s;
    s = dfsqu_vf2_vf2(s);
    s = dfnormalize_vf2_vf2(s);

    u = vcast_vf_f(0.00446636462584137916564941f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(-8.3920182078145444393158e-05f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.0109639242291450500488281f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.0212360303848981857299805f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.0540687143802642822265625f32));

    x = dfadd_vf2_vf_vf(vcast_vf_f(0.133325666189193725585938f32), vmul_vf_vf_vf(u, vf2getx_vf_vf2(s)));
    x = dfadd_vf2_vf_vf2(vcast_vf_f(1f32), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf2(vcast_vf_f(0.33333361148834228515625f32), dfmul_vf2_vf2_vf2(s, x)), s));
    x = dfmul_vf2_vf2_vf2(t, x);
    x = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf2(x), x);

    u = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(x));
    u = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), d, u);
    u
    
}

pub fn atan2kf_u1(y:vfloat2, x:vfloat2)->vfloat2{
    let mut q = vsel_vi2_vf_vf_vi2_vi2(vf2getx_vf_vf2(x), vcast_vf_f(0f32), vcast_vi2_i(-2), vcast_vi2_i(0));
    let mut p = vlt_vo_vf_vf(vf2getx_vf_vf2(x), vcast_vf_f(0f32));
    let mut r = vand_vm_vo32_vm(p, vreinterpret_vm_vf(vcast_vf_f(-0.0f32)));
    let mut x = vf2setx_vf2_vf2_vf(x, vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2getx_vf_vf2(x)), r)));
    x = vf2sety_vf2_vf2_vf(x, vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2gety_vf_vf2(x)), r)));
  
    q = vsel_vi2_vf_vf_vi2_vi2(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y), vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
    p = vlt_vo_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let mut s = vsel_vf2_vo_vf2_vf2(p, dfneg_vf2_vf2(x), y);
    let mut t = vsel_vf2_vo_vf2_vf2(p, y, x);
  
    s = dfdiv_vf2_vf2_vf2(s, t);
    t = dfsqu_vf2_vf2(s);
    t = dfnormalize_vf2_vf2(t);
  
    let mut u = vcast_vf_f(-0.00176397908944636583328247f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(0.0107900900766253471374512f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(-0.0309564601629972457885742f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(0.0577365085482597351074219f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(-0.0838950723409652709960938f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(0.109463557600975036621094f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(-0.142626821994781494140625f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(t), vcast_vf_f(0.199983194470405578613281f32));
  
    t = dfmul_vf2_vf2_vf2(t, dfadd_vf2_vf_vf(vcast_vf_f(-0.333332866430282592773438f32), vmul_vf_vf_vf(u, vf2getx_vf_vf2(t))));
    t = dfmul_vf2_vf2_vf2(s, dfadd_vf2_vf_vf2(vcast_vf_f(1f32), t));
    dfadd_vf2_vf2_vf2(dfmul_vf2_vf2_vf(vcast_vf2_f_f(1.5707963705062866211f32, -4.3711388286737928865e-08f32), vcast_vf_vi2(q)), t)
}

fn visinf2_vf_vf_vf(d:vfloat, m:vfloat)->vfloat{
    vreinterpret_vf_vm(vand_vm_vo32_vm(visinf_vo_vf(d), vor_vm_vm_vm(vsignbit_vm_vf(d), vreinterpret_vm_vf(m))))
}

pub fn xatan2f_u1(y:vfloat, x:vfloat)->vfloat{
    let o = vlt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(2.9387372783541830947e-39f32)); // nexttowardf((1.0 / FLT_MAX), 1)
    let x = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(x, vcast_vf_f( (1 << 24) as f32)), x);
    let y = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(y, vcast_vf_f( (1 << 24) as f32)), y);
    
    let d = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(y), vcast_vf_f(0f32)), vcast_vf2_vf_vf(x, vcast_vf_f(0 as f32)));
    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));
  
    r = vmulsign_vf_vf_vf(r, x);
    r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0f32))), vsub_vf_vf_vf(vcast_vf_f((M_PI as f32)/2f32), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f((M_PI as f32)/2f32), x))), r);
    r = vsel_vf_vo_vf_vf(visinf_vo_vf(y), vsub_vf_vf_vf(vcast_vf_f((M_PI as f32)/2f32), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f((M_PI as f32)/4f32), x))), r);
    r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(0.0f32)), vreinterpret_vf_vm(vand_vm_vo32_vm(vsignbit_vo_vf(x), vreinterpret_vm_vf(vcast_vf_f(M_PI as f32)))), r);
  
    r = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vreinterpret_vm_vf(vmulsign_vf_vf_vf(r, y))));
    r
  }
  
pub fn xasinf_u1(d:vfloat)->vfloat{
    let o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f32));
    let x2 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, d), vmul_vf_vf_vf(vsub_vf_vf_vf(vcast_vf_f(1f32), vabs_vf_vf(d)), vcast_vf_f(0.5f32)));
    let mut x = vsel_vf2_vo_vf2_vf2(o, vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0f32)), dfsqrt_vf2_vf(x2));
    x = vsel_vf2_vo_vf2_vf2(veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f32)), vcast_vf2_f_f(0f32, 0f32), x);
  
    let mut u = vcast_vf_f(0.4197454825e-1f32);
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.2424046025e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.4547423869e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.7495029271e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.1666677296e+0f32));
    u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)));
  
    let y = dfsub_vf2_vf2_vf(dfsub_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f32/4f32,-8.7422776573475857731e-08f32/4f32), x), u);
    
    let r = vsel_vf_vo_vf_vf(o, vadd_vf_vf_vf(u, vf2getx_vf_vf2(x)),
                     vmul_vf_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(y), vf2gety_vf_vf2(y)), vcast_vf_f(2f32)));
    vmulsign_vf_vf_vf(r, d)
  }
  
pub fn xacosf_u1(d:vfloat)->vfloat{
    let o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f32));
    let x2 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, d), vmul_vf_vf_vf(vsub_vf_vf_vf(vcast_vf_f(1f32), vabs_vf_vf(d)), vcast_vf_f(0.5f32)));
    let mut x = vsel_vf2_vo_vf2_vf2(o, vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0f32)), dfsqrt_vf2_vf(x2));
    x = vsel_vf2_vo_vf2_vf2(veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f32)), vcast_vf2_f_f(0f32, 0f32), x);
  
    let u = vcast_vf_f(0.4197454825e-1f32);
    let mut u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.2424046025e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.4547423869e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.7495029271e-1f32));
    u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(0.1666677296e+0f32));
    u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)));
  
    let mut y = dfsub_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f32/2f32, -8.7422776573475857731e-08f32/2f32),
                   dfadd_vf2_vf_vf(vmulsign_vf_vf_vf(vf2getx_vf_vf2(x), d), vmulsign_vf_vf_vf(u, d)));
    x = dfadd_vf2_vf2_vf(x, u);
  
    y = vsel_vf2_vo_vf2_vf2(o, y, dfscale_vf2_vf2_vf(x, vcast_vf_f(2f32)));
    
    y = vsel_vf2_vo_vf2_vf2(vandnot_vo_vo_vo(o, vlt_vo_vf_vf(d, vcast_vf_f(0f32))),
                dfsub_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f32, -8.7422776573475857731e-08f32), y), y);
  
    vadd_vf_vf_vf(vf2getx_vf_vf2(y), vf2gety_vf_vf2(y))
}
  
pub fn xatanf_u1(d:vfloat)->vfloat{
    let d2 = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0f32)), vcast_vf2_f_f(1f32, 0f32));
    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(d2), vf2gety_vf_vf2(d2));
    r = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vcast_vf_f(1.570796326794896557998982), r);
    vmulsign_vf_vf_vf(r, d)
}

pub fn xexpf(d:vfloat)->vfloat{
    let q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(R_LN2f)));

    let mut s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf), d);
    s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf), s);
  
    let mut u = vcast_vf_f(0.000198527617612853646278381f32);
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00139304355252534151077271f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833336077630519866943359f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416664853692054748535156f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166666671633720397949219f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5f32));
  
    u = vadd_vf_vf_vf(vcast_vf_f(1.0f32), vmla_vf_vf_vf_vf(vmul_vf_vf_vf(s, s), u, s));
  
    u = vldexp2_vf_vf_vi2(u, q);
  
    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d, vcast_vf_f(-104f32)), vreinterpret_vm_vf(u)));
    u = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(vcast_vf_f(100f32), d), vcast_vf_f(SLEEF_INFINITYf), u);
  
    u
}

pub fn xcbrtf_u1(d:vfloat)->vfloat{
    let q2 = vcast_vf2_f_f(1f32, 0f32);
  
    let e = vadd_vi2_vi2_vi2(vilogbk_vi2_vf(vabs_vf_vf(d)), vcast_vi2_i(1));
    let mut d = vldexp2_vf_vf_vi2(d, vneg_vi2_vi2(e));
  
    let t = vadd_vf_vf_vf(vcast_vf_vi2(e), vcast_vf_f(6144f32));
    let qu = vtruncate_vi2_vf(vmul_vf_vf_vf(t, vcast_vf_f(1.0/3.0)));
    let re = vtruncate_vi2_vf(vsub_vf_vf_vf(t, vmul_vf_vf_vf(vcast_vf_vi2(qu), vcast_vf_f(3f32))));
  
    let mut q2 = vsel_vf2_vo_vf2_vf2(veq_vo_vi2_vi2(re, vcast_vi2_i(1)), vcast_vf2_f_f(1.2599210739135742188f32, -2.4018701694217270415e-08f32), q2);
    q2 = vsel_vf2_vo_vf2_vf2(veq_vo_vi2_vi2(re, vcast_vi2_i(2)), vcast_vf2_f_f(1.5874010324478149414f32,  1.9520385308169352356e-08f32), q2);
  
    q2 = vf2setx_vf2_vf2_vf(q2, vmulsign_vf_vf_vf(vf2getx_vf_vf2(q2), d));
    q2 = vf2sety_vf2_vf2_vf(q2, vmulsign_vf_vf_vf(vf2gety_vf_vf2(q2), d));
    d = vabs_vf_vf(d);
  
    let mut x = vcast_vf_f(-0.601564466953277587890625f32);
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.8208892345428466796875f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-5.532182216644287109375f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(5.898262500762939453125f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-3.8095417022705078125f32));
    x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.2241256237030029296875f32));
  
    let mut y = vmul_vf_vf_vf(x, x); y = vmul_vf_vf_vf(y, y); x = vsub_vf_vf_vf(x, vmul_vf_vf_vf(vmlanp_vf_vf_vf_vf(d, y, x), vcast_vf_f(-1.0f32 / 3.0f32)));
  
    let z = x;
  
    let mut u = dfmul_vf2_vf_vf(x, x);
    u = dfmul_vf2_vf2_vf2(u, u);
    u = dfmul_vf2_vf2_vf(u, d);
    u = dfadd2_vf2_vf2_vf(u, vneg_vf_vf(x));
    y = vadd_vf_vf_vf(vf2getx_vf_vf2(u), vf2gety_vf_vf2(u));
  
    y = vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(-2.0f32 / 3.0f32), y), z);
    let mut v = dfadd2_vf2_vf2_vf(dfmul_vf2_vf_vf(z, z), y);
    v = dfmul_vf2_vf2_vf(v, d);
    v = dfmul_vf2_vf2_vf2(v, q2);
    let mut z = vldexp2_vf_vf_vi2(vadd_vf_vf_vf(vf2getx_vf_vf2(v), vf2gety_vf_vf2(v)), vsub_vi2_vi2_vi2(qu, vcast_vi2_i(2048)));
  
    z = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vmulsign_vf_vf_vf(vcast_vf_f(SLEEF_INFINITYf), vf2getx_vf_vf2(q2)), z);
    z = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0f32)), vreinterpret_vf_vm(vsignbit_vm_vf(vf2getx_vf_vf2(q2))), z);
  
    z
}

fn logkf(d:vfloat)->vfloat2{
    let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
    let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f((( 1i64 << 32) as f32) * ((1i64 << 32) as f32))), d);
    let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32/0.75f32)));
    let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
    e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
  
    let x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(-1f32), m), dfadd2_vf2_vf_vf(vcast_vf_f(1f32), m));
    let x2 = dfsqu_vf2_vf2(x);
  
    let mut t = vcast_vf_f(0.240320354700088500976562f32);
    t = vmla_vf_vf_vf_vf(t, vf2getx_vf_vf2(x2), vcast_vf_f(0.285112679004669189453125f32));
    t = vmla_vf_vf_vf_vf(t, vf2getx_vf_vf2(x2), vcast_vf_f(0.400007992982864379882812f32));
    let c = vcast_vf2_f_f(0.66666662693023681640625f32, 3.69183861259614332084311e-09f32);

    let mut s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-09f32), vcast_vf_vi2(e));
  
    s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2f32)));
    s = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(x2, x),
                           dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(x2, t), c)));
    s
}

fn logk3f(d:vfloat)->vfloat{
    let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
    let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(((1i64 << 32) as f32) * ((1i64 << 32) as f32))), d);
    let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32/0.75f32)));
    let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
    e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
  
  
    let x = vdiv_vf_vf_vf(vsub_vf_vf_vf(m, vcast_vf_f(1.0f32)), vadd_vf_vf_vf(vcast_vf_f(1.0f32), m));
    let x2 = vmul_vf_vf_vf(x, x);
  
    let mut t = vcast_vf_f(0.2392828464508056640625f32);
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.28518211841583251953125f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.400005877017974853515625f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.666666686534881591796875f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(2.0f32));
  
    vmla_vf_vf_vf_vf(x, t, vmul_vf_vf_vf(vcast_vf_f(0.693147180559945286226764f32), vcast_vf_vi2(e)))
}

pub fn xlogf_u1(d:vfloat)->vfloat{
    
    let o = vlt_vo_vf_vf(d, vcast_vf_f(SLEEF_FLT_MIN));
    let d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(((1i64 << 32) as f32) * ((1i64 << 32) as f32))), d);
    let mut e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f32/0.75f32)));
    let m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
    e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
    let s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f32, -1.904654323148236017e-09f32), vcast_vf_vi2(e));

  
    let x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(-1f32), m), dfadd2_vf2_vf_vf(vcast_vf_f(1f32), m));
    let x2 = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));
  
    let mut t = vcast_vf_f(0.3027294874e+0f32);
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.3996108174e+0f32));
    t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.6666694880e+0f32));
    
    let mut s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2f32)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, vf2getx_vf_vf2(x)), t));
  
    let mut r = vadd_vf_vf_vf(vf2getx_vf_vf2(s), vf2gety_vf_vf2(s));
  
    r = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(SLEEF_INFINITYf), r);
    r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0f32)), visnan_vo_vf(d)), vcast_vf_f(SLEEF_NANf), r);
    r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0f32)), vcast_vf_f(-SLEEF_INFINITYf), r);
    r
}


fn expkf(d:vfloat2)->vfloat{
    let mut u = vmul_vf_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)), vcast_vf_f(R_LN2f));
    let  q = vrint_vi2_vf(u);
  
    let mut s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf)));
  
    s = dfnormalize_vf2_vf2(s);
  
    u = vcast_vf_f(0.00136324646882712841033936f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.00836596917361021041870117f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.0416710823774337768554688f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.166665524244308471679688f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.499999850988388061523438f32));
  
    let mut t = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfsqu_vf2_vf2(s), u));
  
    t = dfadd_vf2_vf_vf2(vcast_vf_f(1f32), t);
    u = vadd_vf_vf_vf(vf2getx_vf_vf2(t), vf2gety_vf_vf2(t));
    u = vldexp_vf_vf_vi2(u, q);
  
    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(vf2getx_vf_vf2(d), vcast_vf_f(-104f32)), vreinterpret_vm_vf(u)));
    u
}

fn expk3f(d:vfloat)->vfloat{
    let q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(R_LN2f)));
  
    let mut s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf), d);
    s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf), s);
  
    let mut u = vcast_vf_f(0.000198527617612853646278381f32);
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00139304355252534151077271f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833336077630519866943359f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416664853692054748535156f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166666671633720397949219f32));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5f32));
  
    u = vmla_vf_vf_vf_vf(vmul_vf_vf_vf(s, s), u, vadd_vf_vf_vf(s, vcast_vf_f(1.0f32)));
    u = vldexp2_vf_vf_vi2(u, q);
  
    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d, vcast_vf_f(-104f32)), vreinterpret_vm_vf(u)));
    
    u
}


pub fn xpowf(x:vfloat, y:vfloat)->vfloat{
  let yisint = vor_vo_vo_vo(veq_vo_vf_vf(vtruncate_vf_vf(y), y), vgt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f( (1 << 24) as f32)));
  let yisodd = vand_vo_vo_vo(vand_vo_vo_vo(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vtruncate_vi2_vf(y), vcast_vi2_i(1)), vcast_vi2_i(1)), yisint),
				 vlt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f( (1 << 24) as f32)));

  let mut  result = expkf(dfmul_vf2_vf2_vf(logkf(vabs_vf_vf(x)), y));

  result = vsel_vf_vo_vf_vf(visnan_vo_vf(result), vcast_vf_f(SLEEF_INFINITYf), result);
  
  result = vmul_vf_vf_vf(result,
			 vsel_vf_vo_vf_vf(vgt_vo_vf_vf(x, vcast_vf_f(0f32)),
					  vcast_vf_f(1f32),
					  vsel_vf_vo_vf_vf(yisint, vsel_vf_vo_vf_vf(yisodd, vcast_vf_f(-1.0f32), vcast_vf_f(1f32)), vcast_vf_f(SLEEF_NANf))));

  let efx = vmulsign_vf_vf_vf(vsub_vf_vf_vf(vabs_vf_vf(x), vcast_vf_f(1f32)), y);

  result = vsel_vf_vo_vf_vf(visinf_vo_vf(y),
			    vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(efx, vcast_vf_f(0.0f32)),
								  vreinterpret_vm_vf(vsel_vf_vo_vf_vf(veq_vo_vf_vf(efx, vcast_vf_f(0.0f32)),
												      vcast_vf_f(1.0f32),
												      vcast_vf_f(SLEEF_INFINITYf))))),
			    result);

  result = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0.0f32))),
			    vmulsign_vf_vf_vf(vsel_vf_vo_vf_vf(vxor_vo_vo_vo(vsignbit_vo_vf(y), veq_vo_vf_vf(x, vcast_vf_f(0.0f32))),
							       vcast_vf_f(0f32), vcast_vf_f(SLEEF_INFINITYf)),
					      vsel_vf_vo_vf_vf(yisodd, x, vcast_vf_f(1f32))), result);

  result = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vreinterpret_vm_vf(result)));

  result = vsel_vf_vo_vf_vf(vor_vo_vo_vo(veq_vo_vf_vf(y, vcast_vf_f(0f32)), veq_vo_vf_vf(x, vcast_vf_f(1f32))), vcast_vf_f(1f32), result);
  result
}

fn expk2f(d:vfloat2)->vfloat2{
    let mut u = vmul_vf_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)), vcast_vf_f(R_LN2f));
    let q = vrint_vi2_vf(u);

    let mut s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf)));
  
    u = vcast_vf_f(0.1980960224e-3f32);
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.1394256484e-2f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.8333456703e-2f32));
    u = vmla_vf_vf_vf_vf(u, vf2getx_vf_vf2(s), vcast_vf_f(0.4166637361e-1f32));
  
    let mut t = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf(s, u), vcast_vf_f(0.166666659414234244790680580464e+0f32));
    t = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf2(s, t), vcast_vf_f(0.5f32));
    t = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf2(dfsqu_vf2_vf2(s), t));
  
    t = dfadd_vf2_vf_vf2(vcast_vf_f(1f32), t);
  
    t = vf2setx_vf2_vf2_vf(t, vldexp2_vf_vf_vi2(vf2getx_vf_vf2(t), q));
    t = vf2sety_vf2_vf2_vf(t, vldexp2_vf_vf_vi2(vf2gety_vf_vf2(t), q));
  
    t = vf2setx_vf2_vf2_vf(t, vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(vf2getx_vf_vf2(d), vcast_vf_f(-104f32)), vreinterpret_vm_vf(vf2getx_vf_vf2(t)))));
    t = vf2sety_vf2_vf2_vf(t, vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(vf2getx_vf_vf2(d), vcast_vf_f(-104f32)), vreinterpret_vm_vf(vf2gety_vf_vf2(t)))));
  
    t
}

pub fn xsinhf(x:vfloat)->vfloat{
    let mut y = vabs_vf_vf(x);
    let mut d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0f32)));
    d = dfsub_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
    y = vmul_vf_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)), vcast_vf_f(0.5f32));
  
    y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89f32)),
                      visnan_vo_vf(y)), vcast_vf_f(SLEEF_INFINITYf), y);
    y = vmulsign_vf_vf_vf(y, x);
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));
  
    y
}
  
pub fn xcoshf(x:vfloat)->vfloat{
    let mut y = vabs_vf_vf(x);
    let mut d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0f32)));
    d = dfadd_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
    y = vmul_vf_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)), vcast_vf_f(0.5f32));
  
    y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89f32)),
                      visnan_vo_vf(y)), vcast_vf_f(SLEEF_INFINITYf), y);
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));
    y
}
  
pub fn xtanhf(x:vfloat)->vfloat{
    let mut y = vabs_vf_vf(x);
    let mut d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0f32)));
    let e = dfrec_vf2_vf2(d);
    d = dfdiv_vf2_vf2_vf2(dfadd_vf2_vf2_vf2(d, dfneg_vf2_vf2(e)), dfadd_vf2_vf2_vf2(d, e));
    y = vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d));
  
    y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(8.664339742f32)),
                      visnan_vo_vf(y)), vcast_vf_f(1.0f32), y);
    y = vmulsign_vf_vf_vf(y, x);
    y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));
  
    y
}