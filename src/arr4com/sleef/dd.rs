use crate::arr4com::sleef::helperavx2::*;

pub fn  vd2getx_vd_vd2(v:vdouble2)->vdouble { v.x }
pub fn  vd2gety_vd_vd2(v:vdouble2)->vdouble { v.y }
pub fn vd2setxy_vd2_vd_vd(x:vdouble, y:vdouble)->vdouble2  { vdouble2{x, y} }
pub fn vd2setx_vd2_vd2_vd(v:vdouble2, d:vdouble)->vdouble2 { vdouble2{x:d, y:v.y} }
pub fn vd2sety_vd2_vd2_vd(v:vdouble2, d:vdouble)->vdouble2 { vdouble2{x:v.x, y:d}}


pub fn dd(h:f64, l:f64)->double2 {
    double2{ x:h, y:l }
}

pub fn vupper_vd_vd(d:vdouble)->vdouble {
    vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_i_i(0xffffffffu32 as i32, 0xf8000000u32 as i32)))
}
  
pub fn vcast_vd2_vd_vd(h:vdouble, l:vdouble)->vdouble2{
    vd2setxy_vd2_vd_vd(h, l)
}
  
pub fn vcast_vd2_d_d(h:f64, l:f64)->vdouble2 {
    vd2setxy_vd2_vd_vd(vcast_vd_d(h), vcast_vd_d(l))
}
  
pub fn vcast_vd2_d2(dd:double2)->vdouble2 {
    vd2setxy_vd2_vd_vd(vcast_vd_d(dd.x), vcast_vd_d(dd.y))
}
  
pub fn vsel_vd2_vo_vd2_vd2(m:vopmask, x:vdouble2, y:vdouble2)->vdouble2 {
    vd2setxy_vd2_vd_vd(vsel_vd_vo_vd_vd(m, vd2getx_vd_vd2(x), vd2getx_vd_vd2(y)),
                  vsel_vd_vo_vd_vd(m, vd2gety_vd_vd2(x), vd2gety_vd_vd2(y)))
}
  
pub fn vsel_vd2_vo_d_d_d_d(o:vopmask, x1:f64, y1:f64, x0:f64, y0:f64)->vdouble2 {
    vd2setxy_vd2_vd_vd(vsel_vd_vo_d_d(o, x1, x0),
                  vsel_vd_vo_d_d(o, y1, y0))
}
  
pub fn vadd_vd_3vd(v0:vdouble, v1:vdouble, v2:vdouble)->vdouble {
    vadd_vd_vd_vd(vadd_vd_vd_vd(v0, v1), v2)
}
  
pub fn vadd_vd_4vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble)->vdouble {
    vadd_vd_3vd(vadd_vd_vd_vd(v0, v1), v2, v3)
}
  
pub fn vadd_vd_5vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble, v4:vdouble)->vdouble {
    vadd_vd_4vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4)
}
  
pub fn vadd_vd_6vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble, v4:vdouble, v5:vdouble)->vdouble {
    vadd_vd_5vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4, v5)
}
  
pub fn vadd_vd_7vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble, v4:vdouble, v5:vdouble, v6:vdouble)->vdouble{
    vadd_vd_6vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4, v5, v6)
}
  
pub fn vsub_vd_3vd(v0:vdouble, v1:vdouble, v2:vdouble)->vdouble{
    vsub_vd_vd_vd(vsub_vd_vd_vd(v0, v1), v2)
}
  
pub fn vsub_vd_4vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble)->vdouble {
    vsub_vd_3vd(vsub_vd_vd_vd(v0, v1), v2, v3)
}
  
pub fn vsub_vd_5vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble, v4:vdouble)->vdouble{
    vsub_vd_4vd(vsub_vd_vd_vd(v0, v1), v2, v3, v4)
}
  
pub fn vsub_vd_6vd(v0:vdouble, v1:vdouble, v2:vdouble, v3:vdouble, v4:vdouble, v5:vdouble)->vdouble{
    vsub_vd_5vd(vsub_vd_vd_vd(v0, v1), v2, v3, v4, v5)
}

pub fn ddneg_vd2_vd2(x:vdouble2)->vdouble2{
    vcast_vd2_vd_vd(vneg_vd_vd(vd2getx_vd_vd2(x)), vneg_vd_vd(vd2gety_vd_vd2(x)))
}
  
pub fn ddabs_vd2_vd2(x:vdouble2)->vdouble2 {
    vcast_vd2_vd_vd(vabs_vd_vd(vd2getx_vd_vd2(x)),
               vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(vd2gety_vd_vd2(x)),
                                vand_vm_vm_vm(vreinterpret_vm_vd(vd2getx_vd_vd2(x)),
                                      vreinterpret_vm_vd(vcast_vd_d(-0.0))))))
}

pub fn ddnormalize_vd2_vd2(t:vdouble2)->vdouble2 {
    let s = vadd_vd_vd_vd(vd2getx_vd_vd2(t), vd2gety_vd_vd2(t));
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(vsub_vd_vd_vd(vd2getx_vd_vd2(t), s), vd2gety_vd_vd2(t)))
}

pub fn ddscale_vd2_vd2_vd(d:vdouble2, s:vdouble)->vdouble2 {
    vd2setxy_vd2_vd_vd(vmul_vd_vd_vd(vd2getx_vd_vd2(d), s), vmul_vd_vd_vd(vd2gety_vd_vd2(d), s))
}

pub fn ddscale_vd2_vd2_d(d:vdouble2, s:f64)->vdouble2{ ddscale_vd2_vd2_vd(d, vcast_vd_d(s)) }
  
pub fn ddadd_vd2_vd_vd(x:vdouble, y:vdouble)->vdouble2{
    let s = vadd_vd_vd_vd(x, y);
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(vsub_vd_vd_vd(x, s), y))
}
  
pub fn ddadd2_vd2_vd_vd(x:vdouble, y:vdouble)->vdouble2 {
    let s = vadd_vd_vd_vd(x, y);
    let v = vsub_vd_vd_vd(s, x);
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(s, v)), vsub_vd_vd_vd(y, v)))
}
  
pub fn ddadd_vd2_vd2_vd(x:vdouble2, y:vdouble)->vdouble2 {
    let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), y);
    vd2setxy_vd2_vd_vd(s, vadd_vd_3vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), s), y, vd2gety_vd_vd2(x)))
}
  
pub fn ddsub_vd2_vd2_vd(x:vdouble2, y:vdouble)->vdouble2 {
    let s = vsub_vd_vd_vd(vd2getx_vd_vd2(x), y);
    return vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(vsub_vd_vd_vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), s), y), vd2gety_vd_vd2(x)));
}
  
pub fn ddadd2_vd2_vd2_vd(x:vdouble2, y:vdouble)->vdouble2 {
    let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), y);
    let v = vsub_vd_vd_vd(s, vd2getx_vd_vd2(x));
    let w = vadd_vd_vd_vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), vsub_vd_vd_vd(s, v)), vsub_vd_vd_vd(y, v));
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(w, vd2gety_vd_vd2(x)))
}
  
pub fn ddadd_vd2_vd_vd2(x:vdouble, y:vdouble2)->vdouble2{
    let s = vadd_vd_vd_vd(x, vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(s, vadd_vd_3vd(vsub_vd_vd_vd(x, s), vd2getx_vd_vd2(y), vd2gety_vd_vd2(y)))
}
  
pub fn ddadd2_vd2_vd_vd2(x:vdouble, y:vdouble2)->vdouble2{
    let s = vadd_vd_vd_vd(x, vd2getx_vd_vd2(y));
    let v = vsub_vd_vd_vd(s, x);
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(s, v)),
                                 vsub_vd_vd_vd(vd2getx_vd_vd2(y), v)), vd2gety_vd_vd2(y)))
}
  
pub fn ddadd_vd2_vd2_vd2(x:vdouble2, y:vdouble2)->vdouble2{
    // |x| >= |y|
  
    let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(s, vadd_vd_4vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), s), vd2getx_vd_vd2(y), vd2gety_vd_vd2(x), vd2gety_vd_vd2(y)))
}
  
pub fn ddadd2_vd2_vd2_vd2(x:vdouble2, y:vdouble2)->vdouble2{
    let s = vadd_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    let v = vsub_vd_vd_vd(s, vd2getx_vd_vd2(x));
    let t = vadd_vd_vd_vd(vsub_vd_vd_vd(vd2getx_vd_vd2(x), vsub_vd_vd_vd(s, v)), vsub_vd_vd_vd(vd2getx_vd_vd2(y), v));
    vd2setxy_vd2_vd_vd(s, vadd_vd_vd_vd(t, vadd_vd_vd_vd(vd2gety_vd_vd2(x), vd2gety_vd_vd2(y))))
  }
  
pub fn ddsub_vd2_vd_vd(x:vdouble, y:vdouble)->vdouble2{
    // |x| >= |y|
  
    let s = vsub_vd_vd_vd(x, y);
    vd2setxy_vd2_vd_vd(s, vsub_vd_vd_vd(vsub_vd_vd_vd(x, s), y))
}
  
pub fn ddsub_vd2_vd2_vd2(x:vdouble2, y:vdouble2)->vdouble2 {
    // |x| >= |y|
  
    let s = vsub_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    let mut t = vsub_vd_vd_vd(vd2getx_vd_vd2(x), s);
    t = vsub_vd_vd_vd(t, vd2getx_vd_vd2(y));
    t = vadd_vd_vd_vd(t, vd2gety_vd_vd2(x));
    vd2setxy_vd2_vd_vd(s, vsub_vd_vd_vd(t, vd2gety_vd_vd2(y)))
}

pub fn dddiv_vd2_vd2_vd2(n:vdouble2, d:vdouble2)->vdouble2 {
    let t = vrec_vd_vd(vd2getx_vd_vd2(d));
    let dh  = vupper_vd_vd(vd2getx_vd_vd2(d));
    let dl  = vsub_vd_vd_vd(vd2getx_vd_vd2(d),  dh);
    let th  = vupper_vd_vd(t  );
    let tl  = vsub_vd_vd_vd(t  ,  th);
    let nhh = vupper_vd_vd(vd2getx_vd_vd2(n));
    let nhl = vsub_vd_vd_vd(vd2getx_vd_vd2(n), nhh);
  
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(n), t);
  
    let u = vadd_vd_5vd(vsub_vd_vd_vd(vmul_vd_vd_vd(nhh, th), s), vmul_vd_vd_vd(nhh, tl), vmul_vd_vd_vd(nhl, th), vmul_vd_vd_vd(nhl, tl),
              vmul_vd_vd_vd(s, vsub_vd_5vd(vcast_vd_d(1f64), vmul_vd_vd_vd(dh, th), vmul_vd_vd_vd(dh, tl), vmul_vd_vd_vd(dl, th), vmul_vd_vd_vd(dl, tl))));
  
    vd2setxy_vd2_vd_vd(s, vmla_vd_vd_vd_vd(t, vsub_vd_vd_vd(vd2gety_vd_vd2(n), vmul_vd_vd_vd(s, vd2gety_vd_vd2(d))), u))
  }
  
pub fn ddmul_vd2_vd_vd(x:vdouble, y:vdouble)->vdouble2 {
    let xh = vupper_vd_vd(x);
    let xl = vsub_vd_vd_vd(x, xh);
    let yh = vupper_vd_vd(y);
    let yl = vsub_vd_vd_vd(y, yh);
  
    let s = vmul_vd_vd_vd(x, y);
    vd2setxy_vd2_vd_vd(s, vadd_vd_5vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(s), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl)))
}
  
pub fn ddmul_vd2_vd2_vd(x:vdouble2, y:vdouble)->vdouble2 {
    let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);
    let yh = vupper_vd_vd(y  );
    let yl = vsub_vd_vd_vd(y  , yh);
  
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), y);
    vd2setxy_vd2_vd_vd(s, vadd_vd_6vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(s), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl), vmul_vd_vd_vd(vd2gety_vd_vd2(x), y)))
}

pub fn ddmul_vd2_vd2_vd2(x:vdouble2, y:vdouble2)->vdouble2{
    let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);
    let yh = vupper_vd_vd(vd2getx_vd_vd2(y));
    let yl = vsub_vd_vd_vd(vd2getx_vd_vd2(y), yh);
  
    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(y));
    vd2setxy_vd2_vd_vd(s, vadd_vd_7vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(s), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl), vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2gety_vd_vd2(y)), vmul_vd_vd_vd(vd2gety_vd_vd2(x), vd2getx_vd_vd2(y))))
  }
  
pub fn ddmul_vd_vd2_vd2(x:vdouble2, y:vdouble2)->vdouble {
    let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);
    let yh = vupper_vd_vd(vd2getx_vd_vd2(y));
    let yl = vsub_vd_vd_vd(vd2getx_vd_vd2(y), yh);
  
    vadd_vd_6vd(vmul_vd_vd_vd(vd2gety_vd_vd2(x), yh), vmul_vd_vd_vd(xh, vd2gety_vd_vd2(y)), vmul_vd_vd_vd(xl, yl), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yh))
}

pub fn ddsqu_vd2_vd2(x:vdouble2)->vdouble2{
    let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);

    let s = vmul_vd_vd_vd(vd2getx_vd_vd2(x), vd2getx_vd_vd2(x));
    vd2setxy_vd2_vd_vd(s, vadd_vd_5vd(vmul_vd_vd_vd(xh, xh), vneg_vd_vd(s), vmul_vd_vd_vd(vadd_vd_vd_vd(xh, xh), xl), vmul_vd_vd_vd(xl, xl), vmul_vd_vd_vd(vd2getx_vd_vd2(x), vadd_vd_vd_vd(vd2gety_vd_vd2(x), vd2gety_vd_vd2(x)))))
}
  
pub fn ddsqu_vd_vd2(x:vdouble2)->vdouble{
    let xh = vupper_vd_vd(vd2getx_vd_vd2(x));
    let xl = vsub_vd_vd_vd(vd2getx_vd_vd2(x), xh);

    vadd_vd_5vd(vmul_vd_vd_vd(xh, vd2gety_vd_vd2(x)), vmul_vd_vd_vd(xh, vd2gety_vd_vd2(x)), vmul_vd_vd_vd(xl, xl), vadd_vd_vd_vd(vmul_vd_vd_vd(xh, xl), vmul_vd_vd_vd(xh, xl)), vmul_vd_vd_vd(xh, xh))
}

pub fn ddrec_vd2_vd(d:vdouble)->vdouble2{
    let t = vrec_vd_vd(d);
    let dh = vupper_vd_vd(d);
    let dl = vsub_vd_vd_vd(d, dh);
    let th = vupper_vd_vd(t);
    let tl = vsub_vd_vd_vd(t, th);

    vd2setxy_vd2_vd_vd(t, vmul_vd_vd_vd(t, vsub_vd_5vd(vcast_vd_d(1f64), vmul_vd_vd_vd(dh, th), vmul_vd_vd_vd(dh, tl), vmul_vd_vd_vd(dl, th), vmul_vd_vd_vd(dl, tl))))
}

pub fn ddrec_vd2_vd2(d:vdouble2)->vdouble2{
    let t = vrec_vd_vd(vd2getx_vd_vd2(d));
    let dh = vupper_vd_vd(vd2getx_vd_vd2(d));
    let dl = vsub_vd_vd_vd(vd2getx_vd_vd2(d), dh);
    let th = vupper_vd_vd(t  );
    let tl = vsub_vd_vd_vd(t  , th);

    vd2setxy_vd2_vd_vd(t, vmul_vd_vd_vd(t, vsub_vd_6vd(vcast_vd_d(1f64), vmul_vd_vd_vd(dh, th), vmul_vd_vd_vd(dh, tl), vmul_vd_vd_vd(dl, th), vmul_vd_vd_vd(dl, tl), vmul_vd_vd_vd(vd2gety_vd_vd2(d), t))))
}

pub fn ddsqrt_vd2_vd2(d:vdouble2)->vdouble2{
    let t = vsqrt_vd_vd(vadd_vd_vd_vd(vd2getx_vd_vd2(d), vd2gety_vd_vd2(d)));
    ddscale_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd2(d, ddmul_vd2_vd_vd(t, t)), ddrec_vd2_vd(t)), vcast_vd_d(0.5))
}
  
pub fn ddsqrt_vd2_vd(d:vdouble)->vdouble2{
    let t = vsqrt_vd_vd(d);
    ddscale_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd_vd2(d, ddmul_vd2_vd_vd(t, t)), ddrec_vd2_vd(t)), vcast_vd_d(0.5))
}

pub fn ddmla_vd2_vd2_vd2_vd2(x:vdouble2, y:vdouble2, z:vdouble2)->vdouble2{
    ddadd_vd2_vd2_vd2(z, ddmul_vd2_vd2_vd2(x, y))
}
