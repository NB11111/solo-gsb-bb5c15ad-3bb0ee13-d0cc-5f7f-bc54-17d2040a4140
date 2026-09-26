"""scanlines：水平扫描线填充区间单元测试。"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.scanlines import MAX_SCANLINES, _line_intervals, compute_scanlines  # noqa: E402
from app.svggeo import Contour, analyze_contours  # noqa: E402


def _scan(d, spacing, fill_rule="nonzero"):
    contours, _warnings = analyze_contours(d, fill_rule)
    return compute_scanlines(contours, spacing), contours


def test_single_outer_square():
    r, _ = _scan("M0,0h10v10h-10Z", 2.0)
    assert r["bounds"] == {"miny": 0.0, "maxy": 10.0}
    assert r["scanline_count"] == 5
    assert [line["y"] for line in r["lines"]] == [0.0, 2.0, 4.0, 6.0, 8.0]
    for line in r["lines"]:
        assert line["intervals"] == [[0.0, 10.0]]
    assert r["skipped_contours"] == []


def test_max_y_exclusive_exact_integer_height():
    # 高度恰好等于 spacing 整数倍：最大 y 不取线
    r, _ = _scan("M0,0h4v4h-4Z", 1.0)
    assert r["scanline_count"] == 4
    assert [line["y"] for line in r["lines"]] == [0.0, 1.0, 2.0, 3.0]


def test_spacing_larger_than_height_still_emits_min_y_line():
    r, _ = _scan("M0,0h4v4h-4Z", 5.0)
    assert r["scanline_count"] == 1
    assert r["lines"][0]["y"] == 0.0
    assert r["lines"][0]["intervals"] == [[0.0, 4.0]]


def test_outer_with_hole_alternates():
    # 外方 0..10，内孔 2..8
    r, _ = _scan("M0,0h10v10h-10Z M2,2h6v6h-6Z", 2.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[0.0] == [[0.0, 10.0]]
    assert by_y[2.0] == [[0.0, 2.0], [8.0, 10.0]]
    assert by_y[4.0] == [[0.0, 2.0], [8.0, 10.0]]
    assert by_y[8.0] == [[0.0, 10.0]]
    # 区间升序、互不重叠
    for line in r["lines"]:
        ivs = line["intervals"]
        assert ivs == sorted(ivs)
        assert all(b[0] >= a[1] for a, b in zip(ivs, ivs[1:]))


def test_hole_with_island_three_levels():
    # 外方 0..10、孔 2..8、岛 6..8
    r, _ = _scan("M0,0h10v10h-10Z M2,2h6v6h-6Z M6,6h2v2h-2Z", 2.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    # y=6/8 切到岛（y=6 为岛底边，半开规则不计；y=8 是岛的最大 y）
    assert by_y[6.0] == [[0.0, 2.0], [6.0, 10.0]]
    assert by_y[8.0] == [[0.0, 10.0]]
    assert by_y[2.0] == [[0.0, 2.0], [8.0, 10.0]]


def test_vertex_scanline_no_double_count():
    # 扫描线精确落在横/纵边连接的顶点上：半开边规则不重复计数
    d = "M0,0 L10,0 L10,5 L5,5 L5,10 L0,10 Z"  # L 形
    r, _ = _scan(d, 5.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[0.0] == [[0.0, 10.0]]   # 底边
    assert by_y[5.0] == [[0.0, 5.0]]    # 内凹顶点（10,5)-(5,5) 水平边不计数）


def test_half_open_rule_bottom_tangent_is_empty():
    # 扫描线恰为整体最小 y（圆底切点）：交点成对规则下无填充区间
    r, _ = _scan("M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z", 0.5)
    first = r["lines"][0]
    assert first["y"] == -1.0
    assert first["intervals"] == []
    # 其余线为单一对称区间，宽度关于 y=0 对称且向中部递增
    mids = [line for line in r["lines"] if line["intervals"]]
    widths = [line["intervals"][0][1] - line["intervals"][0][0]
              for line in mids]
    assert widths == pytest.approx(widths[::-1], abs=1e-9)
    half = widths[:len(widths) // 2 + 1]
    assert half == sorted(half)
    assert max(widths) <= 2.0 + 1e-9


def test_two_disjoint_outers_produce_sorted_intervals():
    r, _ = _scan("M0,0h2v10h-2Z M8,0h2v10h-2Z", 5.0)
    assert r["bounds"] == {"miny": 0.0, "maxy": 10.0}
    for line in r["lines"]:
        assert line["intervals"] == [[0.0, 2.0], [8.0, 10.0]]


def test_far_apart_unit_squares_keep_local_intervals():
    # 回归：单个 1×1 方形 spacing=0.5 得 [0,1]；再加入 x=1e9 处同样大小的
    # 方形后，旧实现按全局包围盒宽度（约 1e9）取 1e-9 比例容差（约 1.0），
    # 把每个方形自身相距 1 的左右交点并为同点而抵消，两条线全空。
    single, _ = _scan("M0,0h1v1h-1Z", 0.5)
    assert single["scanline_count"] == 2
    assert [line["y"] for line in single["lines"]] == [0.0, 0.5]
    for line in single["lines"]:
        assert line["intervals"] == [[0.0, 1.0]]

    d = "M0,0h1v1h-1Z M1000000000,0h1v1h-1Z"
    r, _ = _scan(d, 0.5)
    assert r["bounds"] == {"miny": 0.0, "maxy": 1.0}
    assert r["scanline_count"] == 2
    for line in r["lines"]:
        # 两个独立填充区间按 x 升序返回；远处轮廓不影响局部容差
        assert line["intervals"] == [[0.0, 1.0],
                                     [1000000000.0, 1000000001.0]]


def test_far_apart_unit_square_interval_wider_than_local_tolerance():
    # 大坐标处宽度远大于局部相对容差（1e-12·1e9=1e-3）的窄矩形必须保留
    r, _ = _scan("M1000000000,0h0.01v2h-0.01Z", 1.0)
    assert r["scanline_count"] == 2
    for line in r["lines"]:
        assert line["intervals"] == [[1000000000.0, 1000000000.01]]


def test_coincident_outer_and_hole_endpoints_cancel():
    # 孔左边与外环左边共边（x=0）：共点的 +1/-1 端点在同一组抵消
    r, _ = _scan("M0,0h10v10h-10Z M0,2h6v6h-6Z", 2.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[0.0] == [[0.0, 10.0]]
    assert by_y[2.0] == [[6.0, 10.0]]
    assert by_y[6.0] == [[6.0, 10.0]]
    assert by_y[8.0] == [[0.0, 10.0]]


def test_coincident_hole_and_island_endpoints_cancel():
    # 孔中岛左边与孔左边共边（x=2）：外环 0..10、孔 2..8、岛 2..4/y2..8
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z M2,2h2v6h-2Z"
    r, _ = _scan(d, 2.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[2.0] == [[0.0, 4.0], [8.0, 10.0]]
    assert by_y[6.0] == [[0.0, 4.0], [8.0, 10.0]]
    assert by_y[8.0] == [[0.0, 10.0]]


def test_coincident_endpoints_cancel_at_far_coordinates():
    # 整体平移到 x=1e9 后共点抵消仍然成立（局部容差随交点量级缩放，
    # 但不会吞掉宽 4 的真实区间）
    d = ("M1000000000,0h10v10h-10Z "
         "M1000000000,2h6v6h-6Z "
         "M1000000002,2h2v6h-2Z")
    r, _ = _scan(d, 2.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[2.0] == [[1000000002.0, 1000000004.0],
                         [1000000006.0, 1000000010.0]]
    assert by_y[8.0] == [[1000000000.0, 1000000010.0]]


def test_sub_tolerance_zero_width_interval_not_emitted():
    # 梯形在 y=0（底边）的两腰交点仅相距 5e-13（< 原点绝对容差 1e-12），
    # 视为数值零宽度不输出；同轮廓其余扫描线宽度真实，正常返回
    r, contours = _scan("M0,0 L0,2 L10,2 L0.0000000000005,0 Z", 1.0)
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[0.0] == []
    assert by_y[1.0] == [[0.0, 5.0]]


def _make_contour(points, depth=0, original_index=0):
    """构造非退化 Contour 供 _line_intervals 白盒测试。"""
    return Contour(
        original_index=original_index,
        subpath={"segments": [], "closed": True},
        points=points,
        area=1.0,
        bounds=(min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points)),
        closed=True,
        degenerate=False,
        role="outer" if depth % 2 == 0 else "hole",
        depth=depth,
    )


def test_zero_width_filter_uses_local_scale():
    # 宽 5e-13（< 1e-12 绝对下限）：丢弃；宽 0.5：保留
    thin_origin = _make_contour(
        [(0.0, -1.0), (0.0, 1.0), (5e-13, 1.0), (5e-13, -1.0)])
    assert _line_intervals([thin_origin], 0.0) == []
    wide_origin = _make_contour(
        [(0.0, -1.0), (0.0, 1.0), (0.5, 1.0), (0.5, -1.0)])
    assert _line_intervals([wide_origin], 0.0) == [(0.0, 0.5)]

    # 大坐标处：宽 5e-4（< 1e-12·1e9=1e-3 局部容差）视为零宽丢弃；
    # 宽 1（远方形）与宽 0.01（> 1e-3）保留
    thin_far = _make_contour(
        [(1e9, -1.0), (1e9, 1.0), (1e9 + 5e-4, 1.0), (1e9 + 5e-4, -1.0)])
    assert _line_intervals([thin_far], 0.0) == []
    unit_far = _make_contour(
        [(1e9, -1.0), (1e9, 1.0), (1e9 + 1.0, 1.0), (1e9 + 1.0, -1.0)])
    assert _line_intervals([unit_far], 0.0) == [(1e9, 1e9 + 1.0)]


def test_skips_open_and_degenerate_contours():
    d = "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10"
    r, _ = _scan(d, 5.0)
    assert r["scanline_count"] == 2
    for line in r["lines"]:
        assert line["intervals"] == [[0.0, 10.0]]
    reasons = {(s["original_index"], s["reason"])
               for s in r["skipped_contours"]}
    assert reasons == {(1, "degenerate"), (2, "open")}


def test_no_valid_contours_returns_empty_and_none_bounds():
    r, _ = _scan("M0,0 L10,0 L10,10", 1.0)
    assert r["bounds"] is None
    assert r["scanline_count"] == 0
    assert r["lines"] == []
    assert r["skipped_contours"][0]["reason"] == "open"

    r2, _ = _scan("", 1.0)
    assert r2["bounds"] is None and r2["lines"] == []


def test_curves_flattened_are_used():
    # 与 normalize/metrics 相同的弧细分：y=0 过圆心，半宽约为半径 1
    r, _ = _scan("M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z", 0.25)
    mid = next(line for line in r["lines"] if line["y"] == 0.0)
    x0, x1 = mid["intervals"][0]
    assert abs(x0 + 1.0) < 1e-6 and abs(x1 - 1.0) < 1e-6


def test_evenodd_same_hierarchy_result():
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z M6,6h2v2h-2Z"
    r_nonzero, _ = _scan(d, 2.0, "nonzero")
    r_evenodd, _ = _scan(d, 2.0, "evenodd")
    assert r_nonzero["lines"] == r_evenodd["lines"]


def test_too_many_scanlines_raises():
    # 高 10 的轮廓，spacing=0.005 -> 2000 条，超过上限
    with pytest.raises(ValueError, match=str(MAX_SCANLINES)):
        _scan("M0,0h1v10h-1Z", 0.005)


def test_exactly_at_cap_allowed():
    # 4 / 0.004 = 1000 条：恰好上限允许
    r, _ = _scan("M0,0h4v4h-4Z", 0.004)
    assert r["scanline_count"] == MAX_SCANLINES


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_spacing_raises(bad):
    with pytest.raises(ValueError):
        _scan("M0,0h4v4h-4Z", bad)
