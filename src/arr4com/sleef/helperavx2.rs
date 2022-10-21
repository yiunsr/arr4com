#![allow(
    improper_ctypes,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]

use core::{arch::x86_64::*};
use crate::arr4com::sleef::misc::*;
pub type vmask = __m256i;
pub type vopmask = __m256i;

pub type vdouble = __m256d;
pub type vint = __m128i;


pub type vfloat = __m256;
pub type vint2 = __m256i;

pub type vint64 = __m256i;
pub type vuint64 = __m256i;

#[derive(Debug, Copy, Clone)]
pub struct vquad{
    pub x: vmask,
    pub y: vmask,
}

#[derive(Debug, Copy, Clone)]
pub struct vfloat2{
    pub x:vfloat,
    pub y:vfloat,
}

#[derive(Debug, Copy, Clone)]
pub struct vdouble2{
    pub x: vdouble,
    pub y: vdouble,
}

#[derive(Debug, Copy, Clone)]
pub struct double2{
    pub x:f64,
    pub y:f64,
}

#[derive(Debug, Copy, Clone)]
pub struct dfi_t{
    pub df:vfloat2,
    pub i:vint2,
}


pub fn vtestallones_i_vo32(g:vopmask)->i32{
    unsafe {
        _mm_test_all_ones(_mm_and_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1)))
    }
}

pub fn vtestallones_i_vo64(g:vopmask)->i32{
    unsafe{
        _mm_test_all_ones(_mm_and_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1)))
    }
}

pub fn vcast_vd_d(d:f64)->vdouble{
    unsafe{_mm256_set1_pd(d)}
}
pub fn vreinterpret_vm_vd(vd:vdouble)->vmask{
    unsafe{_mm256_castpd_si256(vd)}
}
pub fn vreinterpret_vd_vm(vm:vmask)->vdouble{ 
    unsafe{_mm256_castsi256_pd(vm)}
}

pub fn vand_vm_vm_vm(x:vmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vandnot_vm_vm_vm(x:vmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vor_vm_vm_vm(x:vmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vxor_vm_vm_vm(x:vmask, y:vmask)->vmask{
    unsafe{return vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}

pub fn vand_vo_vo_vo(x:vopmask, y:vopmask)->vopmask{
    unsafe{vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vandnot_vo_vo_vo(x:vopmask, y:vopmask)->vopmask{
    unsafe{vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vor_vo_vo_vo(x:vopmask, y:vopmask)->vopmask{
    unsafe{vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vxor_vo_vo_vo(x:vopmask, y:vopmask)->vopmask{
    unsafe{vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}

pub fn vand_vm_vo64_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vandnot_vm_vo64_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vor_vm_vo64_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vxor_vm_vo64_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}

pub fn vand_vm_vo32_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vandnot_vm_vo32_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vor_vm_vo32_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}
pub fn vxor_vm_vo32_vm(x:vopmask, y:vmask)->vmask{
    unsafe{vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y)))}
}

pub fn vcast_vo32_vo64(o:vopmask)->vopmask{
    unsafe{_mm256_permutevar8x32_epi32(o, _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0))}
}
  
pub fn vcast_vo64_vo32(o:vopmask)->vopmask{
    unsafe{ _mm256_permutevar8x32_epi32(o, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0))}
}


pub fn vrint_vi_vd(vd:vdouble)->vint{ unsafe{_mm256_cvtpd_epi32(vd)}}
pub fn vtruncate_vi_vd(vd:vdouble)->vint{ unsafe{_mm256_cvttpd_epi32(vd)}}
pub fn vrint_vd_vd(vd:vdouble)->vdouble{ unsafe{_mm256_round_pd(vd, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)} }
pub fn vrint_vf_vf(vd:vfloat)->vfloat{ unsafe{_mm256_round_ps(vd, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)}}
pub fn vtruncate_vd_vd(vd:vdouble)->vdouble { unsafe{_mm256_round_pd(vd, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)}}
pub fn vtruncate_vf_vf(vf:vfloat)->vfloat{ unsafe{_mm256_round_ps(vf, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)}}
pub fn vcast_vd_vi(vi:vint)->vdouble{ unsafe{_mm256_cvtepi32_pd(vi)} }
pub fn vcast_vi_i(i: i32)->vint{ unsafe{_mm_set1_epi32(i)} }

pub fn vcastu_vm_vi(vi:vint)->vmask{
    unsafe{ _mm256_slli_epi64(_mm256_cvtepi32_epi64(vi), 32) }
}

pub fn vcastu_vi_vm(vi:vmask)->vint{
    unsafe{ _mm_or_si128(_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm256_castsi256_si128(vi)), _mm_set1_ps(0f32), 0x0d)),
  		      _mm_castps_si128(_mm_shuffle_ps(_mm_set1_ps(0f32), _mm_castsi128_ps(_mm256_extractf128_si256(vi, 1)), 0xd0))) }
}

pub fn vcast_vm_i_i(i0:i32, i1:i32)->vmask{
    unsafe{ _mm256_set_epi32(i0, i1, i0, i1, i0, i1, i0, i1)}
}

pub fn vcast_vm_i64(i:i64)->vmask{ unsafe{_mm256_set1_epi64x(i)} }
pub fn vcast_vm_u64(i:u64)->vmask{ unsafe{_mm256_set1_epi64x(i as i64)} }

pub fn veq64_vo_vm_vm(x:vmask, y:vmask)->vopmask { unsafe{_mm256_cmpeq_epi64(x, y)} }
pub fn vadd64_vm_vm_vm(x:vmask, y:vmask)->vmask { unsafe{_mm256_add_epi64(x, y)} }

pub fn vadd_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{ unsafe{_mm256_add_pd(x, y)} }
pub fn vsub_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{ unsafe{_mm256_sub_pd(x, y)} }
pub fn vmul_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{ unsafe{_mm256_mul_pd(x, y)} }
pub fn vdiv_vd_vd_vd(x:vdouble, y:vdouble)->vdouble{ unsafe{_mm256_div_pd(x, y)} }
pub fn vrec_vd_vd(x:vdouble)->vdouble { unsafe{_mm256_div_pd(_mm256_set1_pd(1f64), x)} }
pub fn vsqrt_vd_vd(x:vdouble)->vdouble { unsafe{ _mm256_sqrt_pd(x)} }
pub fn vabs_vd_vd(d:vdouble)->vdouble { unsafe{_mm256_andnot_pd(_mm256_set1_pd(-0.0), d)} }
pub fn vneg_vd_vd(d:vdouble)->vdouble { unsafe{_mm256_xor_pd(_mm256_set1_pd(-0.0), d)} }
pub fn vmla_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{_mm256_fmadd_pd(x, y, z)} }
pub fn vmlapn_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{_mm256_fmsub_pd(x, y, z)} }
pub fn vmlanp_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{_mm256_fnmadd_pd(x, y, z)} }
pub fn vmax_vd_vd_vd(x:vdouble, y:vdouble)->vdouble { unsafe{ _mm256_max_pd(x, y)} }
pub fn vmin_vd_vd_vd(x:vdouble, y:vdouble)->vdouble { unsafe{ _mm256_min_pd(x, y)} }

pub fn vfma_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{ _mm256_fmadd_pd(x, y, z)} }
pub fn vfmapp_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{ _mm256_fmadd_pd(x, y, z)} }
pub fn vfmapn_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{ _mm256_fmsub_pd(x, y, z)} }
pub fn vfmanp_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{ _mm256_fnmadd_pd(x, y, z)} }
pub fn vfmann_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { unsafe{ _mm256_fnmsub_pd(x, y, z)} }

pub fn veq_vo_vd_vd(x:vdouble, y:vdouble)->vopmask { unsafe{ vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_EQ_OQ))} }
pub fn vneq_vo_vd_vd(x:vdouble, y:vdouble)->vopmask { unsafe{ vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_NEQ_UQ))} }
pub fn vlt_vo_vd_vd(x:vdouble, y:vdouble)->vopmask { unsafe{ vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_LT_OQ))} }
pub fn vle_vo_vd_vd(x:vdouble, y:vdouble)->vopmask { unsafe{ vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_LE_OQ))} }
pub fn vgt_vo_vd_vd(x:vdouble, y:vdouble)->vopmask { unsafe{ vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_GT_OQ))} }
pub fn vge_vo_vd_vd(x:vdouble, y:vdouble)->vopmask { unsafe{ vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_GE_OQ))} }

pub fn vadd_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_add_epi32(x, y)} }
pub fn vsub_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_sub_epi32(x, y)} }
pub fn vneg_vi_vi(e:vint)->vint {  vsub_vi_vi_vi(vcast_vi_i(0), e) }

pub fn vand_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_and_si128(x, y)} }
pub fn vandnot_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_andnot_si128(x, y)} }
pub fn vor_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_or_si128(x, y)} }
pub fn vxor_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_xor_si128(x, y)} }

pub fn vandnot_vi_vo_vi(m:vopmask, y:vint)->vint { unsafe{ _mm_andnot_si128(_mm256_castsi256_si128(m), y)} }
pub fn vand_vi_vo_vi(m:vopmask, y:vint)->vint { unsafe{ _mm_and_si128(_mm256_castsi256_si128(m), y)} }

pub fn vsll_vi_vi_i<const c: i32>(x:vint)->vint { unsafe{ _mm_slli_epi32(x, c)} }
pub fn vsrl_vi_vi_i<const c: i32>(x:vint)->vint { unsafe{ _mm_srli_epi32(x, c)} }
pub fn vsra_vi_vi_i<const c: i32>(x:vint)->vint { unsafe{ _mm_srai_epi32(x, c)} }

pub fn veq_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_cmpeq_epi32(x, y)} }
pub fn vgt_vi_vi_vi(x:vint, y:vint)->vint { unsafe{ _mm_cmpgt_epi32(x, y)} }

pub fn veq_vo_vi_vi(x:vint, y:vint)->vopmask { unsafe{ _mm256_castsi128_si256(_mm_cmpeq_epi32(x, y))} }
pub fn vgt_vo_vi_vi(x:vint, y:vint)->vopmask { unsafe{ _mm256_castsi128_si256(_mm_cmpgt_epi32(x, y))} }

pub fn vsel_vi_vo_vi_vi(m:vopmask, x:vint, y:vint)->vint { unsafe{ _mm_blendv_epi8(y, x, _mm256_castsi256_si128(m))} }

pub fn vsel_vd_vo_vd_vd(o:vopmask, x:vdouble, y:vdouble)->vdouble { unsafe{ _mm256_blendv_pd(y, x, _mm256_castsi256_pd(o))} }
pub fn vsel_vd_vo_d_d(o:vopmask, v1:f64, v0:f64)->vdouble { unsafe{ _mm256_permutevar_pd(_mm256_set_pd(v1, v0, v1, v0), o)} }

pub fn vsel_vd_vo_vo_vo_d_d_d_d(o0:vopmask, o1:vopmask, o2:vopmask, d0:f64, d1:f64, d2:f64, d3:f64)->vdouble {
    unsafe{
        let v:__m256i = _mm256_castpd_si256(vsel_vd_vo_vd_vd(o0, _mm256_castsi256_pd(_mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0)),
                                vsel_vd_vo_vd_vd(o1, _mm256_castsi256_pd(_mm256_set_epi32(3, 2, 3, 2, 3, 2, 3, 2)),
                                        vsel_vd_vo_vd_vd(o2, _mm256_castsi256_pd(_mm256_set_epi32(5, 4, 5, 4, 5, 4, 5, 4)),
                                                _mm256_castsi256_pd(_mm256_set_epi32(7, 6, 7, 6, 7, 6, 7, 6))))));
        _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_mm256_set_pd(d3, d2, d1, d0)), v))
    }
}
  
pub fn vsel_vd_vo_vo_d_d_d(o0:vopmask, o1:vopmask, d0:f64, d1:f64, d2:f64)->vdouble {
    vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o1, d0, d1, d2, d2)
}
  
pub fn visinf_vo_vd(d:vdouble)->vopmask{
    unsafe{
        vreinterpret_vm_vd(_mm256_cmp_pd(vabs_vd_vd(d), _mm256_set1_pd(SLEEF_INFINITY), _CMP_EQ_OQ))
    }
}
  
pub fn vispinf_vo_vd(d:vdouble)->vopmask {
    unsafe{
        vreinterpret_vm_vd(_mm256_cmp_pd(d, _mm256_set1_pd(SLEEF_INFINITY), _CMP_EQ_OQ))
    }
}
  
pub fn visminf_vo_vd(d:vdouble)->vopmask {
    unsafe{
        vreinterpret_vm_vd(_mm256_cmp_pd(d, _mm256_set1_pd(-SLEEF_INFINITY), _CMP_EQ_OQ))
    }
}
  
pub fn visnan_vo_vd(d:vdouble)->vopmask{
    unsafe{
        vreinterpret_vm_vd(_mm256_cmp_pd(d, d, _CMP_NEQ_UQ))
    }
}

pub fn vload_vd_p(ptr: *const f64)->vdouble{ unsafe{ _mm256_load_pd(ptr)} }
pub fn vloadu_vd_p(ptr:  *const f64)->vdouble{ unsafe{ _mm256_loadu_pd(ptr)} }

pub fn vstore_v_p_vd(ptr: *mut f64, v:vdouble) { unsafe{ _mm256_store_pd(ptr, v)} }
pub fn vstoreu_v_p_vd(ptr: *mut f64, v:vdouble) { unsafe{_mm256_storeu_pd(ptr, v)} }

pub fn vgather_vd_p_vi(ptr: *const f64, vi:vint)->vdouble { unsafe{_mm256_i32gather_pd(ptr, vi, 8)} }

pub fn vcast_vi2_vm(vm:vmask)->vint2 { vm as vint2}
pub fn vcast_vm_vi2(vi:vint2)->vmask { vi as vint2}

pub fn vrint_vi2_vf(vf:vfloat)->vint2 { unsafe{vcast_vi2_vm(_mm256_cvtps_epi32(vf))} }
pub fn vtruncate_vi2_vf(vf:vfloat)->vint2 { unsafe{vcast_vi2_vm(_mm256_cvttps_epi32(vf))} }
pub fn vcast_vf_vi2(vi:vint2)->vfloat { unsafe{ _mm256_cvtepi32_ps(vcast_vm_vi2(vi))} }
pub fn vcast_vf_f(f:f32)->vfloat { unsafe{ _mm256_set1_ps(f)} }
pub fn vcast_vi2_i(i:i32)->vint2 { unsafe{ _mm256_set1_epi32(i)} }
pub fn vreinterpret_vm_vf(vf:vfloat)->vmask { unsafe{ _mm256_castps_si256(vf)} }
pub fn vreinterpret_vf_vm(vm:vmask)->vfloat { unsafe{ _mm256_castsi256_ps(vm)} }

pub fn vreinterpret_vf_vi2(vi:vint2)->vfloat {  vreinterpret_vf_vm(vcast_vm_vi2(vi)) }
pub fn vreinterpret_vi2_vf(vf:vfloat)->vint2 { vcast_vi2_vm(vreinterpret_vm_vf(vf)) }

pub fn vadd_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{_mm256_add_ps(x, y)} }
pub fn vsub_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{ _mm256_sub_ps(x, y)} }
pub fn vmul_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{ _mm256_mul_ps(x, y)} }
pub fn vdiv_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{ _mm256_div_ps(x, y)} }
pub fn vrec_vf_vf(x:vfloat)->vfloat { vdiv_vf_vf_vf(vcast_vf_f(1.0f32), x) }
pub fn vsqrt_vf_vf(x:vfloat)->vfloat { unsafe{ _mm256_sqrt_ps(x)} }
pub fn vabs_vf_vf(f:vfloat)->vfloat { vreinterpret_vf_vm(vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f32)), vreinterpret_vm_vf(f))) }
pub fn vneg_vf_vf(d:vfloat)->vfloat { vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f32)), vreinterpret_vm_vf(d)))}
pub fn vmla_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat )->vfloat { unsafe{ _mm256_fmadd_ps(x, y, z)} }
pub fn vmlapn_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat )->vfloat { unsafe{ _mm256_fmsub_ps(x, y, z)} }
pub fn vmlanp_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat )->vfloat { unsafe{ _mm256_fnmadd_ps(x, y, z)} }
pub fn vmax_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{ _mm256_max_ps(x, y)} }
pub fn vmin_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{ _mm256_min_ps(x, y)} }

pub fn vfma_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat)->vfloat { unsafe{_mm256_fmadd_ps(x, y, z)} }
pub fn vfmapp_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat)->vfloat { unsafe{ _mm256_fmadd_ps(x, y, z)} }
pub fn vfmapn_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat)->vfloat { unsafe{_mm256_fmsub_ps(x, y, z)} }
pub fn vfmanp_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat)->vfloat { unsafe{_mm256_fnmadd_ps(x, y, z)} }
pub fn  vfmann_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat)->vfloat { unsafe{_mm256_fnmsub_ps(x, y, z)} }

pub fn veq_vo_vf_vf(x:vfloat, y:vfloat)->vopmask { unsafe{vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_EQ_OQ))} }
pub fn vneq_vo_vf_vf(x:vfloat, y:vfloat)->vopmask { unsafe{ vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_NEQ_UQ))} }
pub fn vlt_vo_vf_vf(x:vfloat, y:vfloat)->vopmask { unsafe{ vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LT_OQ))} }
pub fn vle_vo_vf_vf(x:vfloat, y:vfloat)->vopmask { unsafe{ vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LE_OQ))} }
pub fn vgt_vo_vf_vf(x:vfloat, y:vfloat)->vopmask { unsafe{ vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_GT_OQ))} }
pub fn vge_vo_vf_vf(x:vfloat, y:vfloat)->vopmask { unsafe{ vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_GE_OQ))} }

pub fn vadd_vi2_vi2_vi2(x:vint2, y:vint2)->vint2 { unsafe{ _mm256_add_epi32(x, y)} }
pub fn vsub_vi2_vi2_vi2(x:vint2, y:vint2)->vint2 { unsafe{ _mm256_sub_epi32(x, y)} }
pub fn vneg_vi2_vi2(e:vint2)->vint2 { vsub_vi2_vi2_vi2(vcast_vi2_i(0), e) }

pub fn vand_vi2_vi2_vi2(x:vint2, y:vint2 )->vint2 { unsafe{ _mm256_and_si256(x, y)} }
pub fn vandnot_vi2_vi2_vi2(x:vint2, y:vint2 )->vint2 { unsafe{ _mm256_andnot_si256(x, y)} }
pub fn vor_vi2_vi2_vi2(x:vint2, y:vint2 )->vint2 { unsafe{ _mm256_or_si256(x, y)} }
pub fn vxor_vi2_vi2_vi2(x:vint2, y:vint2 )->vint2 { unsafe{ _mm256_xor_si256(x, y)} }

pub fn vand_vi2_vo_vi2(x:vopmask, y:vint2)->vint2 { vand_vi2_vi2_vi2(vcast_vi2_vm(x), y) }
pub fn vandnot_vi2_vo_vi2(x:vopmask, y:vint2)->vint2 { vandnot_vi2_vi2_vi2(vcast_vi2_vm(x), y) }

pub fn vsll_vi2_vi2_i<const c: i32>(x:vint2)->vint2 { unsafe{ _mm256_slli_epi32(x, c)} }
pub fn vsrl_vi2_vi2_i<const c: i32>(x:vint2)->vint2 { unsafe{ _mm256_srli_epi32(x, c)} }
pub fn vsra_vi2_vi2_i<const c: i32>(x:vint2)->vint2 { unsafe{ _mm256_srai_epi32(x, c)} }

pub fn veq_vo_vi2_vi2(x:vint2, y:vint2)->vopmask { unsafe{ _mm256_cmpeq_epi32(x, y)} }
pub fn vgt_vo_vi2_vi2(x:vint2, y:vint2)->vopmask { unsafe{ _mm256_cmpgt_epi32(x, y)} }
pub fn veq_vi2_vi2_vi2(x:vint2, y:vint2)->vint2 { unsafe{ _mm256_cmpeq_epi32(x, y)} }
pub fn vgt_vi2_vi2_vi2(x:vint2, y:vint2)->vint2 { unsafe{ _mm256_cmpgt_epi32(x, y)} }


pub fn vsel_vi2_vo_vi2_vi2(m:vopmask, x:vint2, y:vint2)->vint2 {
    unsafe{_mm256_blendv_epi8(y, x, m)}
}

pub fn vsel_vf_vo_vf_vf(o:vopmask, x:vfloat, y:vfloat)->vfloat { unsafe{_mm256_blendv_ps(y, x, _mm256_castsi256_ps(o))} }

pub fn vsel_vf_vo_f_f(o:vopmask, v1:f32, v0:f32)->vfloat {
    vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0))
}

pub fn vsel_vf_vo_vo_f_f_f(o0:vopmask, o1:vopmask, d0:f32, d1:f32, d2:f32)->vfloat {
    vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2))
}

pub fn vsel_vf_vo_vo_vo_f_f_f_f(o0:vopmask, o1:vopmask, o2:vopmask, d0:f32, d1:f32, d2:f32, d3:f32)->vfloat {
    vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3)))
}

pub fn visinf_vo_vf(d:vfloat)->vopmask { veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(SLEEF_INFINITYf)) }
pub fn vispinf_vo_vf(d:vfloat)->vopmask { veq_vo_vf_vf(d, vcast_vf_f(SLEEF_INFINITYf)) }
pub fn visminf_vo_vf(d:vfloat)->vopmask { veq_vo_vf_vf(d, vcast_vf_f(-SLEEF_INFINITYf)) }
pub fn visnan_vo_vf(d:vfloat)->vopmask { vneq_vo_vf_vf(d, d)}

pub fn vload_vf_p(ptr: *const f32)->vfloat { unsafe{_mm256_load_ps(ptr)} }
pub fn vloadu_vf_p(ptr: *const f32)->vfloat { unsafe{_mm256_loadu_ps(ptr)} }

pub fn vstore_v_p_vf(ptr: *mut f32, v:vfloat) { unsafe{_mm256_store_ps(ptr, v)} }
pub fn vstoreu_v_p_vf(ptr: *mut f32, v:vfloat) { unsafe{_mm256_storeu_ps(ptr, v)} }

pub fn vgather_vf_p_vi2(ptr:*const f32, vi2:vint2)->vfloat { unsafe{ _mm256_i32gather_ps(ptr, vi2, 4)} }

const PNMASK:vdouble = unsafe {
    std::mem::transmute::<[f64;4], vdouble>([0.0, -0.0, 0.0, -0.0])
};
const NPMASK:vdouble = unsafe {
    std::mem::transmute::<[f64;4], vdouble>([ -0.0, 0.0, -0.0, 0.0])
};

const PNMASKf:vfloat = unsafe {
    std::mem::transmute::<[f32;8], vfloat>([0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32])
};
const NPMASKf:vfloat = unsafe {
    std::mem::transmute::<[f32;8], vfloat>([-0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32, -0.0f32, 0.0f32 ])
};

pub fn vposneg_vd_vd(d:vdouble)->vdouble { vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(PNMASK))) }
pub fn vnegpos_vd_vd(d:vdouble)->vdouble { vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(NPMASK))) }
pub fn vposneg_vf_vf(d:vfloat)->vfloat { vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(PNMASKf))) }
pub fn vnegpos_vf_vf(d:vfloat)->vfloat { vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(NPMASKf))) }

pub fn vsubadd_vd_vd_vd(x:vdouble, y:vdouble)->vdouble { unsafe{ _mm256_addsub_pd(x, y)} }
pub fn vsubadd_vf_vf_vf(x:vfloat, y:vfloat)->vfloat { unsafe{_mm256_addsub_ps(x, y)} }

pub fn vmlsubadd_vd_vd_vd_vd(x:vdouble, y:vdouble, z:vdouble)->vdouble { vmla_vd_vd_vd_vd(x, y, vnegpos_vd_vd(z)) }
pub fn vmlsubadd_vf_vf_vf_vf(x:vfloat, y:vfloat, z:vfloat)->vfloat { vmla_vf_vf_vf_vf(x, y, vnegpos_vf_vf(z)) }

pub fn  vrev21_vd_vd(d0:vdouble)->vdouble { unsafe{ _mm256_shuffle_pd(d0, d0, (0 << 3) | (1 << 2) | (0 << 1) | (1 << 0))} }
pub fn vreva2_vd_vd(d0:vdouble)->vdouble { 
    let d0 = unsafe{_mm256_permute2f128_pd(d0, d0, 1)};
    unsafe{ _mm256_shuffle_pd(d0, d0, (1 << 3) | (0 << 2) | (1 << 1) | (0 << 0))}
}

pub fn vstream_v_p_vd(ptr:*mut f64, v:vdouble) { unsafe{_mm256_stream_pd(ptr, v)} }
// pub fn vscatter2_v_p_i_i_vd(ptr:*mut f64, offset: i32, step:i32, v:vdouble) {
//     unsafe{
//         _mm_store_pd(ptr.as_ptr() + (offset as usize + step * 0)*2, _mm256_extractf128_pd(v, 0));
//         _mm_store_pd(ptr[(offset + step * 1)*2], _mm256_extractf128_pd(v, 1));
//     }
// }

// pub fn vsscatter2_v_p_i_i_vd(ptr:*mut f64, offset:i32, step:i32, v:vdouble) {
//     unsafe{
//         _mm_stream_pd(&ptr[(offset + step * 0)*2], _mm256_extractf128_pd(v, 0));
//         _mm_stream_pd(&ptr[(offset + step * 1)*2], _mm256_extractf128_pd(v, 1));
//     }
// }

pub fn vrev21_vf_vf(d0:vfloat)->vfloat { unsafe{_mm256_shuffle_ps(d0, d0, (2 << 6) | (3 << 4) | (0 << 2) | (1 << 0))} }
const vreva2_vf_vf_mask:i32 = (1 << 6) | (0 << 4) | (3 << 2) | (2 << 0);
pub fn vreva2_vf_vf(d0:vfloat)->vfloat {
    let d0 = unsafe{_mm256_permute2f128_ps(d0, d0, 1)};
    unsafe{_mm256_shuffle_ps(d0, d0, vreva2_vf_vf_mask)}
}

// pub fn vstream_v_p_vf(ptr:*mut f32, v:vfloat) { _mm256_stream_ps(ptr, v); }

pub fn vtestallzeros_i_vo64(g:vopmask)->i32 {
    let ret = unsafe{_mm_movemask_epi8(_mm_or_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1))) == 0};
    ret as i32
}

pub fn vsel_vm_vo64_vm_vm(o:vopmask, x:vmask, y:vmask)->vmask { unsafe{ _mm256_blendv_epi8(y, x, o)} }

pub fn vsub64_vm_vm_vm(x:vmask, y:vmask)->vmask { unsafe{ _mm256_sub_epi64(x, y)} }
pub fn vneg64_vm_vm(x:vmask)->vmask { unsafe{ _mm256_sub_epi64(vcast_vm_i_i(0, 0), x)} }
pub fn vgt64_vo_vm_vm(x:vmask, y:vmask)->vopmask { unsafe{ _mm256_cmpgt_epi64(x, y)} } // signed compare

