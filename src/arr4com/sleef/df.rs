use crate::arr4com::sleef::helperavx2::*;

pub fn vf2getx_vf_vf2(v:vfloat2)->vfloat { v.x }
pub fn vf2gety_vf_vf2(v:vfloat2)->vfloat { v.y }
pub fn vf2setxy_vf2_vf_vf(x:vfloat, y:vfloat)->vfloat2  { vfloat2{x, y} }
pub fn vf2setx_vf2_vf2_vf(v: vfloat2, d:vfloat)->vfloat2 { vfloat2{x:d, y:v.y}}
pub fn  vf2sety_vf2_vf2_vf(v: vfloat2, d:vfloat)->vfloat2 { vfloat2{x:v.x, y:d}}

pub fn vupper_vf_vf(d:vfloat)->vfloat {
    vreinterpret_vf_vi2(vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0xfffff000u32 as i32)))
}
  
pub fn vcast_vf2_vf_vf(h:vfloat, l:vfloat)->vfloat2 {
    vf2setxy_vf2_vf_vf(h, l)
}
  
pub fn vcast_vf2_f_f(h:f32, l:f32)->vfloat2 {
    vf2setxy_vf2_vf_vf(vcast_vf_f(h), vcast_vf_f(l))
}
  
pub fn vcast_vf2_d(d:f64)->vfloat2 {
    let f = d as f32;
    vf2setxy_vf2_vf_vf(vcast_vf_f(f), vcast_vf_f( ( d - f as f64) as f32 ))
}
  
pub fn vsel_vf2_vo_vf2_vf2(m:vopmask, x:vfloat2, y:vfloat2)->vfloat2 {
    vf2setxy_vf2_vf_vf(vsel_vf_vo_vf_vf(m, vf2getx_vf_vf2(x), vf2getx_vf_vf2(y)), vsel_vf_vo_vf_vf(m, vf2gety_vf_vf2(x), vf2gety_vf_vf2(y)))
}
  
pub fn vsel_vf2_vo_f_f_f_f(o:vopmask, x1:f32, y1:f32, x0:f32, y0:f32)->vfloat2 {
    vf2setxy_vf2_vf_vf(vsel_vf_vo_f_f(o, x1, x0), vsel_vf_vo_f_f(o, y1, y0))
}
  
pub fn vsel_vf2_vo_vo_d_d_d(o0:vopmask, o1:vopmask, d0:f64, d1:f64, d2:f64)->vfloat2 {
    vsel_vf2_vo_vf2_vf2(o0, vcast_vf2_d(d0), vsel_vf2_vo_vf2_vf2(o1, vcast_vf2_d(d1), vcast_vf2_d(d2)))
}
  
pub fn vsel_vf2_vo_vo_vo_d_d_d_d(o0:vopmask, o1:vopmask, o2:vopmask, d0:f64, d1:f64, d2:f64, d3:f64)->vfloat2 {
    vsel_vf2_vo_vf2_vf2(o0, vcast_vf2_d(d0), vsel_vf2_vo_vf2_vf2(o1, vcast_vf2_d(d1), vsel_vf2_vo_vf2_vf2(o2, vcast_vf2_d(d2), vcast_vf2_d(d3))))
}
  
pub fn vabs_vf2_vf2(x:vfloat2)->vfloat2 {
    vcast_vf2_vf_vf(vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(vf2getx_vf_vf2(x))), vreinterpret_vm_vf(vf2getx_vf_vf2(x)))),
               vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(vf2getx_vf_vf2(x))), vreinterpret_vm_vf(vf2gety_vf_vf2(x)))))
}
  
pub fn vadd_vf_3vf(v0:vfloat, v1:vfloat, v2:vfloat)->vfloat {
    vadd_vf_vf_vf(vadd_vf_vf_vf(v0, v1), v2)
}
  
pub fn vadd_vf_4vf(v0:vfloat, v1:vfloat, v2:vfloat, v3:vfloat)->vfloat {
    vadd_vf_3vf(vadd_vf_vf_vf(v0, v1), v2, v3)
}
  
pub fn vadd_vf_5vf(v0:vfloat, v1:vfloat, v2:vfloat, v3:vfloat, v4:vfloat)->vfloat {
    vadd_vf_4vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4)
}
  
pub fn vadd_vf_6vf(v0:vfloat, v1:vfloat, v2:vfloat, v3:vfloat, v4:vfloat, v5:vfloat)->vfloat{
    vadd_vf_5vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4, v5)
}
  
pub fn vadd_vf_7vf(v0:vfloat, v1:vfloat, v2:vfloat, v3:vfloat, v4:vfloat, v5:vfloat, v6:vfloat)->vfloat {
    vadd_vf_6vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4, v5, v6)
}
  
pub fn vsub_vf_3vf(v0:vfloat, v1:vfloat, v2:vfloat)->vfloat {
    vsub_vf_vf_vf(vsub_vf_vf_vf(v0, v1), v2)
}
  
pub fn vsub_vf_4vf(v0:vfloat, v1:vfloat, v2:vfloat, v3:vfloat)->vfloat {
    vsub_vf_3vf(vsub_vf_vf_vf(v0, v1), v2, v3)
}
  
pub fn vsub_vf_5vf(v0:vfloat, v1:vfloat, v2:vfloat, v3:vfloat, v4:vfloat)->vfloat {
    vsub_vf_4vf(vsub_vf_vf_vf(v0, v1), v2, v3, v4)
}

pub fn dfneg_vf2_vf2(x:vfloat2)->vfloat2 {
    vcast_vf2_vf_vf(vneg_vf_vf(vf2getx_vf_vf2(x)), vneg_vf_vf(vf2gety_vf_vf2(x)))
}

pub fn dfabs_vf2_vf2(x:vfloat2)->vfloat2 {
    vcast_vf2_vf_vf(vabs_vf_vf(vf2getx_vf_vf2(x)),
               vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vf2gety_vf_vf2(x)), vand_vm_vm_vm(vreinterpret_vm_vf(vf2getx_vf_vf2(x)), vreinterpret_vm_vf(vcast_vf_f(-0.0f32))))))
}
  
pub fn dfnormalize_vf2_vf2(t:vfloat2)->vfloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(t), vf2gety_vf_vf2(t));
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(vf2getx_vf_vf2(t), s), vf2gety_vf_vf2(t)))
}
  
pub fn dfscale_vf2_vf2_vf(d:vfloat2, s:vfloat)->vfloat2 {
    vf2setxy_vf2_vf_vf(vmul_vf_vf_vf(vf2getx_vf_vf2(d), s), vmul_vf_vf_vf(vf2gety_vf_vf2(d), s))
}
  
pub fn dfadd_vf2_vf_vf(x:vfloat, y:vfloat)->vfloat2 {
    let s = vadd_vf_vf_vf(x, y);
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(x, s), y))
}
  
pub fn dfadd2_vf2_vf_vf(x:vfloat, y:vfloat)->vfloat2 {
    let s = vadd_vf_vf_vf(x, y);
    let v = vsub_vf_vf_vf(s, x);
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(x, vsub_vf_vf_vf(s, v)), vsub_vf_vf_vf(y, v)))
}
  
pub fn dfadd2_vf2_vf_vf2(x:vfloat, y:vfloat2)->vfloat2 {
    let s = vadd_vf_vf_vf(x, vf2getx_vf_vf2(y));
    let v = vsub_vf_vf_vf(s, x);
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vadd_vf_vf_vf(vsub_vf_vf_vf(x, vsub_vf_vf_vf(s, v)), vsub_vf_vf_vf(vf2getx_vf_vf2(y), v)), vf2gety_vf_vf2(y)))
}
  
pub fn dfadd_vf2_vf2_vf(x:vfloat2, y:vfloat)->vfloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), y);
    return vf2setxy_vf2_vf_vf(s, vadd_vf_3vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), s), y, vf2gety_vf_vf2(x)));
}
  
pub fn dfsub_vf2_vf2_vf(x:vfloat2, y:vfloat)->vfloat2{
    let s = vsub_vf_vf_vf(vf2getx_vf_vf2(x), y);
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(vsub_vf_vf_vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), s), y), vf2gety_vf_vf2(x)))
}
  
pub fn dfadd2_vf2_vf2_vf(x:vfloat2, y:vfloat)->vfloat2{
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), y);
    let v = vsub_vf_vf_vf(s, vf2getx_vf_vf2(x));
    let t = vadd_vf_vf_vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), vsub_vf_vf_vf(s, v)), vsub_vf_vf_vf(y, v));
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(t, vf2gety_vf_vf2(x)))
}
  
pub fn dfadd_vf2_vf_vf2(x:vfloat, y:vfloat2)->vfloat2 {
    let s = vadd_vf_vf_vf(x, vf2getx_vf_vf2(y));
    vf2setxy_vf2_vf_vf(s, vadd_vf_3vf(vsub_vf_vf_vf(x, s), vf2getx_vf_vf2(y), vf2gety_vf_vf2(y)))
}
  
pub fn dfadd_vf2_vf2_vf2(x:vfloat2, y:vfloat2)->vfloat2 {
    // |x| >= |y|
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    vf2setxy_vf2_vf_vf(s, vadd_vf_4vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), s), vf2getx_vf_vf2(y), vf2gety_vf_vf2(x), vf2gety_vf_vf2(y)))
}
  
pub fn dfadd2_vf2_vf2_vf2(x:vfloat2, y:vfloat2)->vfloat2 {
    let s = vadd_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let v = vsub_vf_vf_vf(s, vf2getx_vf_vf2(x));
    let t = vadd_vf_vf_vf(vsub_vf_vf_vf(vf2getx_vf_vf2(x), vsub_vf_vf_vf(s, v)), vsub_vf_vf_vf(vf2getx_vf_vf2(y), v));
    vf2setxy_vf2_vf_vf(s, vadd_vf_vf_vf(t, vadd_vf_vf_vf(vf2gety_vf_vf2(x), vf2gety_vf_vf2(y))))
}
  
pub fn dfsub_vf2_vf_vf(x:vfloat, y:vfloat)->vfloat2{
    // |x| >= |y|
  
    let s = vsub_vf_vf_vf(x, y);
    vf2setxy_vf2_vf_vf(s, vsub_vf_vf_vf(vsub_vf_vf_vf(x, s), y))
}

pub fn dfsub_vf2_vf2_vf2(x:vfloat2, y:vfloat2)->vfloat2 {
    // |x| >= |y|
  
    let s = vsub_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let t = vsub_vf_vf_vf(vf2getx_vf_vf2(x), s);
    let mut t = vsub_vf_vf_vf(t, vf2getx_vf_vf2(y));
    t = vadd_vf_vf_vf(t, vf2gety_vf_vf2(x));
    vf2setxy_vf2_vf_vf(s, vsub_vf_vf_vf(t, vf2gety_vf_vf2(y)))
}

pub fn dfdiv_vf2_vf2_vf2(n:vfloat2, d:vfloat2)->vfloat2{
    let t = vrec_vf_vf(vf2getx_vf_vf2(d));
    let dh  = vupper_vf_vf(vf2getx_vf_vf2(d));
    let dl  = vsub_vf_vf_vf(vf2getx_vf_vf2(d),  dh);
    let th  = vupper_vf_vf(t  );
    let tl  = vsub_vf_vf_vf(t  ,  th);
    let nhh = vupper_vf_vf(vf2getx_vf_vf2(n));
    let nhl = vsub_vf_vf_vf(vf2getx_vf_vf2(n), nhh);
  
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(n), t);
  
    let mut u;
    let mut w;
    w = vcast_vf_f(-1f32);
    w = vmla_vf_vf_vf_vf(dh, th, w);
    w = vmla_vf_vf_vf_vf(dh, tl, w);
    w = vmla_vf_vf_vf_vf(dl, th, w);
    w = vmla_vf_vf_vf_vf(dl, tl, w);
    w = vneg_vf_vf(w);
  
    u = vmla_vf_vf_vf_vf(nhh, th, vneg_vf_vf(s));
    u = vmla_vf_vf_vf_vf(nhh, tl, u);
    u = vmla_vf_vf_vf_vf(nhl, th, u);
    u = vmla_vf_vf_vf_vf(nhl, tl, u);
    u = vmla_vf_vf_vf_vf(s, w, u);
  
    vf2setxy_vf2_vf_vf(s, vmla_vf_vf_vf_vf(t, vsub_vf_vf_vf(vf2gety_vf_vf2(n), vmul_vf_vf_vf(s, vf2gety_vf_vf2(d))), u))
  }
  
pub fn dfmul_vf2_vf_vf(x:vfloat, y:vfloat)->vfloat2{
    let xh = vupper_vf_vf(x);
    let xl = vsub_vf_vf_vf(x, xh);
    let yh = vupper_vf_vf(y);
    let yl = vsub_vf_vf_vf(y, yh);
  
    let s = vmul_vf_vf_vf(x, y);
    let mut t;
  
    t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(xl, yh, t);
    t = vmla_vf_vf_vf_vf(xh, yl, t);
    t = vmla_vf_vf_vf_vf(xl, yl, t);
  
    vf2setxy_vf2_vf_vf(s, t)
}
  
pub fn dfmul_vf2_vf2_vf(x:vfloat2, y:vfloat)->vfloat2 {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
    let yh = vupper_vf_vf(y  );
    let yl = vsub_vf_vf_vf(y  , yh);
  
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), y);
    let mut t;
  
    t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(xl, yh, t);
    t = vmla_vf_vf_vf_vf(xh, yl, t);
    t = vmla_vf_vf_vf_vf(xl, yl, t);
    t = vmla_vf_vf_vf_vf(vf2gety_vf_vf2(x), y, t);
  
    vf2setxy_vf2_vf_vf(s, t)
}
  
pub fn dfmul_vf2_vf2_vf2(x:vfloat2, y:vfloat2)->vfloat2{
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
    let yh = vupper_vf_vf(vf2getx_vf_vf2(y));
    let yl = vsub_vf_vf_vf(vf2getx_vf_vf2(y), yh);
  
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(y));
    let mut t;
  
    t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(xl, yh, t);
    t = vmla_vf_vf_vf_vf(xh, yl, t);
    t = vmla_vf_vf_vf_vf(xl, yl, t);
    t = vmla_vf_vf_vf_vf(vf2getx_vf_vf2(x), vf2gety_vf_vf2(y), t);
    t = vmla_vf_vf_vf_vf(vf2gety_vf_vf2(x), vf2getx_vf_vf2(y), t);
  
    vf2setxy_vf2_vf_vf(s, t)
}
  
pub fn dfmul_vf_vf2_vf2(x:vfloat2, y:vfloat2)->vfloat {
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
    let yh = vupper_vf_vf(vf2getx_vf_vf2(y));
    let yl = vsub_vf_vf_vf(vf2getx_vf_vf2(y), yh);
  
    return vadd_vf_6vf(vmul_vf_vf_vf(vf2gety_vf_vf2(x), yh), vmul_vf_vf_vf(xh, vf2gety_vf_vf2(y)), vmul_vf_vf_vf(xl, yl), vmul_vf_vf_vf(xh, yl), vmul_vf_vf_vf(xl, yh), vmul_vf_vf_vf(xh, yh));
}
  
pub fn dfsqu_vf2_vf2(x:vfloat2)->vfloat2{
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
  
    let s = vmul_vf_vf_vf(vf2getx_vf_vf2(x), vf2getx_vf_vf2(x));
    let mut t;
  
    t = vmla_vf_vf_vf_vf(xh, xh, vneg_vf_vf(s));
    t = vmla_vf_vf_vf_vf(vadd_vf_vf_vf(xh, xh), xl, t);
    t = vmla_vf_vf_vf_vf(xl, xl, t);
    t = vmla_vf_vf_vf_vf(vf2getx_vf_vf2(x), vadd_vf_vf_vf(vf2gety_vf_vf2(x), vf2gety_vf_vf2(x)), t);
  
    vf2setxy_vf2_vf_vf(s, t)
}
  
pub fn dfsqu_vf_vf2(x:vfloat2)->vfloat{
    let xh = vupper_vf_vf(vf2getx_vf_vf2(x));
    let xl = vsub_vf_vf_vf(vf2getx_vf_vf2(x), xh);
  
    vadd_vf_5vf(vmul_vf_vf_vf(xh, vf2gety_vf_vf2(x)), vmul_vf_vf_vf(xh, vf2gety_vf_vf2(x)), vmul_vf_vf_vf(xl, xl), vadd_vf_vf_vf(vmul_vf_vf_vf(xh, xl), vmul_vf_vf_vf(xh, xl)), vmul_vf_vf_vf(xh, xh))
}
  
pub fn dfrec_vf2_vf(d:vfloat)->vfloat2 {
    let t = vrec_vf_vf(d);
    let dh = vupper_vf_vf(d);
    let dl = vsub_vf_vf_vf(d, dh);
    let th = vupper_vf_vf(t);
    let tl = vsub_vf_vf_vf(t, th);
  
    let mut u = vcast_vf_f(-1f32);
    u = vmla_vf_vf_vf_vf(dh, th, u);
    u = vmla_vf_vf_vf_vf(dh, tl, u);
    u = vmla_vf_vf_vf_vf(dl, th, u);
    u = vmla_vf_vf_vf_vf(dl, tl, u);
  
    vf2setxy_vf2_vf_vf(t, vmul_vf_vf_vf(vneg_vf_vf(t), u))
}
  
pub fn dfrec_vf2_vf2(d:vfloat2)->vfloat2 {
    let t = vrec_vf_vf(vf2getx_vf_vf2(d));
    let dh = vupper_vf_vf(vf2getx_vf_vf2(d));
    let dl = vsub_vf_vf_vf(vf2getx_vf_vf2(d), dh);
    let th = vupper_vf_vf(t  );
    let tl = vsub_vf_vf_vf(t  , th);
  
    let mut u = vcast_vf_f(-1f32);
    u = vmla_vf_vf_vf_vf(dh, th, u);
    u = vmla_vf_vf_vf_vf(dh, tl, u);
    u = vmla_vf_vf_vf_vf(dl, th, u);
    u = vmla_vf_vf_vf_vf(dl, tl, u);
    u = vmla_vf_vf_vf_vf(vf2gety_vf_vf2(d), t, u);
  
    vf2setxy_vf2_vf_vf(t, vmul_vf_vf_vf(vneg_vf_vf(t), u))
}


pub fn dfsqrt_vf2_vf2(d:vfloat2)->vfloat2 {

    let t = vsqrt_vf_vf(vadd_vf_vf_vf(vf2getx_vf_vf2(d), vf2gety_vf_vf2(d)));
    dfscale_vf2_vf2_vf(dfmul_vf2_vf2_vf2(dfadd2_vf2_vf2_vf2(d, dfmul_vf2_vf_vf(t, t)), dfrec_vf2_vf(t)), vcast_vf_f(0.5))
}
    
pub fn dfsqrt_vf2_vf(d:vfloat)->vfloat2 {
    let t = vsqrt_vf_vf(d);
    dfscale_vf2_vf2_vf(dfmul_vf2_vf2_vf2(dfadd2_vf2_vf_vf2(d, dfmul_vf2_vf_vf(t, t)), dfrec_vf2_vf(t)), vcast_vf_f(0.5f32))
}
    