#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unused_macros,
    unused_imports
)]

use crate::arr4com::sleef::helperavx2::*;

#[derive(Debug, Copy, Clone)]
pub struct vdouble3{
    pub x:vdouble,
    pub y:vdouble,
    pub z:vdouble,
}

pub fn vd3getx_vd_vd3(v:vdouble3)->vdouble { v.x }
pub fn vd3gety_vd_vd3(v:vdouble3)->vdouble { v.y }
pub fn vd3getz_vd_vd3(v:vdouble3)->vdouble{ v.z }
macro_rules! vd3setxyz_vd3_vd_vd_vd{
    ($x:ident, $y:ident, $z:ident) => {
        vdouble3{x:$x, y:$y, z:$z }
    };
}
pub(crate) use vd3setxyz_vd3_vd_vd_vd;
macro_rules! vd3setx_vd3_vd3_vd{
    ($v:ident, $d:ident) => {
        v.x = d; 
        v
    };
}
pub(crate) use vd3setx_vd3_vd3_vd;
macro_rules! vd3sety_vd3_vd3_vd{
    ($v:ident, $d:ident) => {
        v.y = d; 
        v
    };
}
pub(crate) use vd3sety_vd3_vd3_vd;

macro_rules! vd3setz_vd3_vd3_vd{
    ($v:ident, $d:ident) => {
        v.z = d; 
        v
    };
}
pub(crate) use vd3setz_vd3_vd3_vd;

pub struct dd2{
    pub a:vdouble2,
    pub b:vdouble2,
}

macro_rules! dd2setab_dd2_vd2_vd2{
    ($a:ident, $b:ident) => {
        dd2{a:$a, b:$b}
    };
}
pub(crate) use dd2setab_dd2_vd2_vd2;

pub fn dd2geta_vd2_dd2(d:dd2)->vdouble2{ d.a }
pub fn dd2getb_vd2_dd2(d:dd2)->vdouble2 { d.b }


pub struct tdx{
    pub e:vmask,
    pub d3:vdouble3,
}
  
pub fn tdxgete_vm_tdx(t:tdx)->vmask{ t.e }
pub fn tdxgetd3_vd3_tdx(t:tdx)->vdouble3{ t.d3 }
pub fn tdxgetd3x_vd_tdx(t:tdx)->vdouble{ t.d3.x }
pub fn tdxgetd3y_vd_tdx(t:tdx)->vdouble{ t.d3.y }
pub fn tdxgetd3z_vd_tdx(t:tdx)->vdouble{ t.d3.z }

macro_rules! tdxsete_tdx_tdx_vm{
    ($t:ident, $e:ident) => {   $t.e = e;    $t};
}
pub(crate) use tdxsete_tdx_tdx_vm;
macro_rules! tdxsetd3_tdx_tdx_vd3{
    ($t:ident, $d3:ident) => {  $t.d3 = d3;  $t};
}
pub(crate) use tdxsetd3_tdx_tdx_vd3;

macro_rules! tdxsetx_tdx_tdx_vd{
    ($t:ident, $x:ident) => {  $t.d3.x = x;  $t};
}
pub(crate) use tdxsetx_tdx_tdx_vd;
macro_rules! tdxsety_tdx_tdx_vd{
    ($t:ident, $y:ident) => {  $t.d3.y = y;  $t};
}
pub(crate) use tdxsety_tdx_tdx_vd;
macro_rules! tdxsetz_tdx_tdx_vd{
    ($t:ident, $z:ident) => {  $t.d3.z = z;  $t};
}
pub(crate) use tdxsetz_tdx_tdx_vd;
macro_rules! tdxsetxyz_tdx_tdx_vd_vd_vd{
    ($t:ident, $x:ident, $y:ident, $z:ident) => {  
        $t.d3 = vdouble3{x:$x, y:$y, z:$z};
        $t
    };
}
pub(crate) use tdxsetxyz_tdx_tdx_vd_vd_vd;
macro_rules! tdxseted3_tdx_vm_vd3{
    ($e:ident, $d3:ident) => {  tdx{ e:$e, d3:$d3 } };
}
pub(crate) use tdxseted3_tdx_vm_vd3;
macro_rules! tdxsetexyz_tdx_vm_vd_vd_vd{
    ($e:ident, $x:ident, $y:ident, $z:ident) => {
        tdx{ e:$e, d3:vdouble3{x:$x, y:$y, z:$z} } 
    };
}
pub(crate) use tdxsetexyz_tdx_vm_vd_vd_vd;
fn vqgetx_vm_vq(v:vquad)->vmask{ v.x }
fn vqgety_vm_vq(v:vquad)->vmask{ v.y }
fn vqsetxy_vq_vm_vm(x:vmask, y:vmask)->vquad{ vquad{ x, y } }
macro_rules! vqsetx_vq_vq_vm{
    ($v:ident, $x:ident) => {$v.x = $x;  $v};
}
pub(crate) use vqsetx_vq_vq_vm;
macro_rules! vqsety_vq_vq_vm{
    ($v:ident, $y:ident) => {$v.y = $y;  $v};
}
pub(crate) use vqsety_vq_vq_vm;

#[derive(Debug, Copy, Clone)]
pub struct di_t{
    pub d:vdouble,
    pub i:vint,
} 
  
pub fn digetd_vd_di(d:&di_t)->vdouble{ d.d }
pub fn digeti_vi_di(d:&di_t)->vint{ d.i }
macro_rules! disetdi_di_vd_vi{
    ($d:ident, $i:ident) => {  di_t{d:$d, i:$i}  };
}
pub(crate) use disetdi_di_vd_vi;

#[derive(Debug, Copy, Clone)]
pub struct ddi_t{
    pub dd: vdouble2,
    pub i:vint
}
  
pub fn ddigetdd_vd2_ddi(d:&ddi_t)->vdouble2{ d.dd }
pub fn ddigeti_vi_ddi(d:&ddi_t)->vint{ d.i }
macro_rules! ddisetddi_ddi_vd2_vi{
    ($v:ident, $i:ident) => {  ddi_t{dd:$v, i:$i}  };
}
pub(crate) use ddisetddi_ddi_vd2_vi;
macro_rules! ddisetdd_ddi_ddi_vd2{
    ($ddi:ident, $v:ident) => {  
        {$ddi.dd = $v; $ddi  }
    };
}
pub(crate) use ddisetdd_ddi_ddi_vd2;

#[derive(Debug, Copy, Clone)]
pub struct tdi_t{
    pub td:vdouble3,
    pub i:vint,
}

pub fn tdigettd_vd3_tdi(d:&tdi_t)->vdouble3{ d.td }
pub fn tdigetx_vd_tdi(d:&tdi_t)->vdouble{ d.td.x }
fn tdigeti_vi_tdi(d:tdi_t)->vint{ d.i }
macro_rules! tdisettdi_tdi_vd3_vi{
    ($v:ident, $i:ident) => {  tdi_t{v:$v, i:$i}  };
}
pub(crate) use tdisettdi_tdi_vd3_vi;

pub fn visnegzero_vo_vd(d:vdouble)->vopmask{
    veq64_vo_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)))
}
  
pub fn visnumber_vo_vd(x:vdouble)->vopmask{
    vandnot_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, x))
}

pub fn visnonfinite_vo_vd( x:vdouble)->vopmask{
    veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(x), vcast_vm_i64(0x7ff0000000000000i64)), vcast_vm_i64(0x7ff0000000000000i64))
}
  
pub fn vsignbit_vm_vd(d:vdouble)->vmask{
    return vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)))
}
  
pub fn vsignbit_vo_vd(d:vdouble)->vopmask{
    veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(vcast_vd_d(-0.0)))
}
  
pub fn vclearlsb_vd_vd_i(d:vdouble, n:i32)->vdouble{
    vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_u64((!(0u64)) << n)))
}
  
pub fn vtoward0_vd_vd(x:vdouble)->vdouble{ // returns nextafter(x, 0)
    let t = vreinterpret_vd_vm(vadd64_vm_vm_vm(vreinterpret_vm_vd(x), vcast_vm_i64(-1)));
    vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0.0)), vcast_vd_d(0.0), t)
}
  
pub fn vmulsign_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{
    vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)))
}
  
pub fn vsign_vd_vd(d:vdouble)->vdouble{
    vmulsign_vd_vd_vd(vcast_vd_d(1.0), d)
}
  
pub fn vorsign_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{
    vreinterpret_vd_vm(vor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)))
}
  
pub fn vcopysign_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{
    vreinterpret_vd_vm(vxor_vm_vm_vm(vandnot_vm_vm_vm(vreinterpret_vm_vd(vcast_vd_d(-0.0)), vreinterpret_vm_vd(x)), 
                        vand_vm_vm_vm   (vreinterpret_vm_vd(vcast_vd_d(-0.0)), vreinterpret_vm_vd(y))))
}

pub fn vtruncate2_vd_vd(x:vdouble)->vdouble{
    vtruncate_vd_vd(x)
}

pub fn vfloor2_vd_vd(x:vdouble)->vdouble{
    let mut fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d( (1i64 << 31) as f64 ), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1i64 << 31) as f64 ))))));
    fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
    fr = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(fr, vcast_vd_d(0f64)), vadd_vd_vd_vd(fr, vcast_vd_d(1.0)), fr);
    vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), vge_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d( (1i64 << 52) as f64 ))), x, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x))
}
  
pub fn vceil2_vd_vd(x:vdouble)->vdouble{
    let mut fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d( (1i64 << 31) as f64 ), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1i64 << 31) as f64))))));
    fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
    fr = vsel_vd_vo_vd_vd(vle_vo_vd_vd(fr, vcast_vd_d(0f64)), fr, vsub_vd_vd_vd(fr, vcast_vd_d(1.0)));
    vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), vge_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d( (1i64 << 52) as f64))), x, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x))
}
  
pub fn vround2_vd_vd(d:vdouble)->vdouble{
    let mut x = vadd_vd_vd_vd(d, vcast_vd_d(0.5));
    let mut fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d( (1i64 << 31) as f64 ), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / ( (1i64 << 31) as f64 )))))));
    fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
    x = vsel_vd_vo_vd_vd(vand_vo_vo_vo(vle_vo_vd_vd(x, vcast_vd_d(0f64)), veq_vo_vd_vd(fr, vcast_vd_d(0f64))), vsub_vd_vd_vd(x, vcast_vd_d(1.0)), x);
    fr = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(fr, vcast_vd_d(0f64)), vadd_vd_vd_vd(fr, vcast_vd_d(1.0)), fr);
    x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.49999999999999994449)), vcast_vd_d(0f64), x);
    vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(d), vge_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d( (1i64 << 52) as f64 ))), d, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), d))
}

pub fn vrint2_vd_vd(d:vdouble)->vdouble{
    vrint_vd_vd(d)
}

pub fn visint_vo_vd(d:vdouble)->vopmask{
    veq_vo_vd_vd(vrint2_vd_vd(d), d)
}
    
pub fn visodd_vo_vd(d:vdouble)->vopmask{
    let x = vmul_vd_vd_vd(d, vcast_vd_d(0.5));
    vneq_vo_vd_vd(vrint2_vd_vd(x), x)
}

pub fn vilogbk_vi_vd(arg_d:vdouble)->vint{
    let o = vlt_vo_vd_vd(arg_d, vcast_vd_d(4.9090934652977266E-91));
    let d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(vcast_vd_d(2.037035976334486E90), arg_d), arg_d);
    let mut q = vcastu_vi_vm(vreinterpret_vm_vd(d));
    q = vand_vi_vi_vi(q, vcast_vi_i((((1u32 << 12) - 1) << 20) as i32));
    q = vsrl_vi_vi_i::<20>(q);
    q = vsub_vi_vi_vi(q, vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vcast_vi_i(300 + 0x3ff), vcast_vi_i(0x3ff)));
    q
}

pub fn vilogb2k_vi_vd(d:vdouble)->vint{
    let mut q = vcastu_vi_vm(vreinterpret_vm_vd(d));
    q = vsrl_vi_vi_i::<20>(q);
    q = vand_vi_vi_vi(q, vcast_vi_i(0x7ff));
    q = vsub_vi_vi_vi(q, vcast_vi_i(0x3ff));
    q
}

pub fn vilogb2k_vm_vd(d:vdouble)->vmask{
    let mut m = vreinterpret_vm_vd(d);
    const c:i32 = 20 + 32;
    m = vsrl64_vm_vm_i::<52>(m);
    m = vand_vm_vm_vm(m, vcast_vm_i64(0x7ff));
    m = vsub64_vm_vm_vm(m, vcast_vm_i64(0x3ff));
    m
}
  
pub fn vilogb3k_vm_vd(d:vdouble)->vmask{
    let mut m = vreinterpret_vm_vd(d);
    const c:i32 = 20 + 32;
    m = vsrl64_vm_vm_i::<c>(m);
    m = vand_vm_vm_vm(m, vcast_vm_i64(0x7ff));
    m
}


pub fn vpow2i_vd_vi(mut q:vint)->vdouble{
    q = vadd_vi_vi_vi(vcast_vi_i(0x3ff), q);
    let r = vcastu_vm_vi(vsll_vi_vi_i::<20>(q));
    vreinterpret_vd_vm(r)
}


pub fn vldexp_vd_vd_vi(x:vdouble, mut q:vint)->vdouble{
    let mut m = vsra_vi_vi_i::<31>(q);
    m = vsll_vi_vi_i::<7>(vsub_vi_vi_vi(vsra_vi_vi_i::<9>(vadd_vi_vi_vi(m, q)), m));
    q = vsub_vi_vi_vi(q, vsll_vi_vi_i::<2>(m));
    m = vadd_vi_vi_vi(vcast_vi_i(0x3ff), m);
    m = vandnot_vi_vo_vi(vgt_vo_vi_vi(vcast_vi_i(0), m), m);
    m = vsel_vi_vo_vi_vi(vgt_vo_vi_vi(m, vcast_vi_i(0x7ff)), vcast_vi_i(0x7ff), m);
    let r = vcastu_vm_vi(vsll_vi_vi_i::<20>(m));
    let y = vreinterpret_vd_vm(r);
    vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, y), y), y), y), vpow2i_vd_vi(q))
}

pub fn vldexp2_vd_vd_vi(d:vdouble, e:vint)->vdouble{
    vmul_vd_vd_vd(vmul_vd_vd_vd(d, vpow2i_vd_vi(vsra_vi_vi_i::<1>(e))), vpow2i_vd_vi(vsub_vi_vi_vi(e, vsra_vi_vi_i::<1>(e))))
}

pub fn vldexp3_vd_vd_vi(d:vdouble, q:vint)->vdouble{
    vreinterpret_vd_vm(vadd64_vm_vm_vm(vreinterpret_vm_vd(d), vcastu_vm_vi(vsll_vi_vi_i::<20>(q))))
}

pub fn rempisub(x:vdouble)->di_t{
    let y = vrint_vd_vd(vmul_vd_vd_vd(x, vcast_vd_d(4.0)));
    let vi = vtruncate_vi_vd(vsub_vd_vd_vd(y, vmul_vd_vd_vd(vrint_vd_vd(x), vcast_vd_d(4.0))));
    let t = vsub_vd_vd_vd(x, vmul_vd_vd_vd(y, vcast_vd_d(0.25)));
    disetdi_di_vd_vi!(t, vi)
}

