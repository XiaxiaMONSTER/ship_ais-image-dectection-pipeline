"""
Rockchip RGA IM2D ctypes 封装 — 基于 librga 2.2.0 头文件

C 接口 (IM_C_API):
  wrapbuffer_virtualaddr_t(va, w, h, ws, hs, fmt) → rga_buffer_t
  imcopy_t(src, dst, sync)                         → IM_STATUS
  imcvtcolor_t(src, dst, sfmt, dfmt, mode, sync)    → IM_STATUS
"""

import ctypes
import numpy as np

_lib = ctypes.cdll.LoadLibrary("librga.so.2")

# RK_FORMAT_* 值来自 rga.h — 所有格式值左移 8 位
FORMAT_RGBA_8888 = 0x0 << 8
FORMAT_RGBX_8888 = 0x1 << 8
FORMAT_RGB_888   = 0x2 << 8
FORMAT_BGRA_8888 = 0x3 << 8
FORMAT_BGRX_8888 = 0x16 << 8
FORMAT_BGR_888   = 0x7 << 8
FORMAT_NV12      = 0xa << 8   # RK_FORMAT_YCbCr_420_SP


class im_colorkey_range(ctypes.Structure):
    _fields_ = [("max", ctypes.c_int), ("min", ctypes.c_int)]

class im_nn_t(ctypes.Structure):
    _fields_ = [
        ("scale_r", ctypes.c_int), ("scale_g", ctypes.c_int),
        ("scale_b", ctypes.c_int), ("offset_r", ctypes.c_int),
        ("offset_g", ctypes.c_int), ("offset_b", ctypes.c_int),
    ]

class rga_buffer_t(ctypes.Structure):
    _fields_ = [
        ("vir_addr",         ctypes.c_void_p),     #   0
        ("phy_addr",         ctypes.c_void_p),     #   8
        ("fd",               ctypes.c_int),        #  16
        ("width",            ctypes.c_int),        #  20
        ("height",           ctypes.c_int),        #  24
        ("wstride",          ctypes.c_int),        #  28
        ("hstride",          ctypes.c_int),        #  32
        ("format",           ctypes.c_int),        #  36
        ("color_space_mode", ctypes.c_int),        #  40
        ("global_alpha",     ctypes.c_int),        #  44
        ("rd_mode",          ctypes.c_int),        #  48
        ("color",            ctypes.c_int),        #  52
        ("colorkey_range",   im_colorkey_range),   #  56
        ("nn",               im_nn_t),             #  64
        ("rop_code",         ctypes.c_int),        #  88
        ("handle",           ctypes.c_int),        #  92
    ]


# —— C API (IM_C_API = extern "C") ——
_wrap = _lib.wrapbuffer_virtualaddr_t
_wrap.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_wrap.restype = rga_buffer_t

_imcopy_t = _lib.imcopy_t
_imcopy_t.argtypes = [rga_buffer_t, rga_buffer_t, ctypes.c_int]
_imcopy_t.restype = ctypes.c_int

_imresize_t = _lib.imresize_t
_imresize_t.argtypes = [
    rga_buffer_t, rga_buffer_t,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int,
]
_imresize_t.restype = ctypes.c_int

_imcvtcolor_t = _lib.imcvtcolor_t
_imcvtcolor_t.argtypes = [
    rga_buffer_t, rga_buffer_t, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_imcvtcolor_t.restype = ctypes.c_int


def _buf(arr, fmt, w=None, h=None):
    """wrapbuffer_virtualaddr_t — wstride/hstride 均为像素步幅"""
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    if w is None:
        h, w = arr.shape[:2]
    ptr = arr.ctypes.data_as(ctypes.c_void_p)
    return _wrap(ptr, w, h, w, h, fmt)


def rga_resize(src, dst_w, dst_h):
    src_h, src_w = src.shape[:2]
    dst = np.empty((dst_h, dst_w, 3), dtype=np.uint8)
    fx = dst_w / src_w
    fy = dst_h / src_h
    ret = _imresize_t(_buf(src, FORMAT_BGR_888),
                      _buf(dst, FORMAT_BGR_888),
                      fx, fy, 1, 1)  # INTER_LINEAR, sync
    if ret <= 0:
        raise RuntimeError(f"rga resize status={ret}")
    return dst


def rga_bgr_to_nv12(src, dst_w=None, dst_h=None):
    src_h, src_w = src.shape[:2]
    if dst_w is None: dst_w = src_w
    if dst_h is None: dst_h = src_h

    if src_w != dst_w or src_h != dst_h:
        resized = rga_resize(src, dst_w, dst_h)
    else:
        resized = src

    nv12_size = dst_w * dst_h * 3 // 2
    dst = np.empty(nv12_size, dtype=np.uint8)

    ret = _imcvtcolor_t(_buf(resized, FORMAT_BGR_888),
                         _buf(dst, FORMAT_NV12, w=dst_w, h=dst_h),
                         FORMAT_BGR_888, FORMAT_NV12, 0, 1)
    if ret <= 0:
        raise RuntimeError(f"rga cvtcolor status={ret}")

    return dst, dst_w, dst_h


# ======================== 自检 ========================
def _self_test():
    def say(msg):
        print(msg, flush=True)

    say(f"[rga_utils] sizeof={ctypes.sizeof(rga_buffer_t)} (expect 96)")

    test = np.zeros((64, 64, 3), dtype=np.uint8)

    say("[rga_utils] Phase1 — imcopy_t 同尺寸 copy ...")
    try:
        same = np.empty_like(test)
        ret = _imcopy_t(_buf(test, FORMAT_BGR_888),
                        _buf(same, FORMAT_BGR_888), 1)
        if ret <= 0:
            raise RuntimeError(f"status={ret}")
        say("[rga_utils] Phase1 OK")
    except Exception as e:
        say(f"[rga_utils] Phase1 FAIL: {e}")
        return

    say("[rga_utils] Phase2 — imresize_t 64→32 ...")
    try:
        out = rga_resize(test, 32, 32)
        assert out.shape == (32, 32, 3)
        say("[rga_utils] Phase2 OK")
    except Exception as e:
        say(f"[rga_utils] Phase2 FAIL: {e}")
        return

    say("[rga_utils] Phase3 — imcvtcolor_t BGR→NV12 ...")
    try:
        nv12, w, h = rga_bgr_to_nv12(test, 32, 32)
        assert nv12.nbytes == 32 * 32 * 3 // 2
        say("[rga_utils] Phase3 OK")
    except Exception as e:
        say(f"[rga_utils] Phase3 FAIL: {e}")

_self_test()
