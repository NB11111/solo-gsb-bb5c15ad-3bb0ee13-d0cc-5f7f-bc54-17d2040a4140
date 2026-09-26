"""scanlines：水平扫描线填充区间单元测试。"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.scanlines import MAX_SCANLINES, compute_scanlines, _line_intervals  # noqa: E402
from app.svggeo import Contour, analyze_contours, bbox  # noqa: E402


def _scan(d, spacing, fill_rule="nonzero"):
    contours, _warnings = analyze_contours(d, fill_rule)
    return compute_scanlines(contours, spacing), contours


def _forced_contour(idx, points, depth):
    """直接构造已分类 Contour（用于包围盒相同、无法走包含分类的构型）。"""
    return Contour(
        original_index=idx, subpath={}, points=points, area=0.0,
        bounds=bbox(points), closed=True, degenerate=False,
        role=("outer" if depth % 2 == 0 else "hole"), depth=depth)


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


def test_distant_small_squares_keep_local_intervals():
    # 回归：单个 1×1 方形 spacing=0.5 返回 [0,1]；再加入 x=1e9 处同尺寸
    # 方形后，两条扫描线都必须各自给出两个独立填充区间（按 x 升序）。
    # 旧实现按整体包围盒跨度缩放的全局比例容差（~1.0）会把每个方形的
    # 左右交点并为同点，导致 intervals 全部为空。
    d = "M0,0h1v1h-1Z M1000000000,0h1v1h-1Z"
    r, _ = _scan(d, 0.5)
    assert r["scanline_count"] == 2
    assert [line["y"] for line in r["lines"]] == [0.0, 0.5]
    assert r["bounds"] == {"miny": 0.0, "maxy": 1.0}
    for line in r["lines"]:
        assert line["intervals"] == [
            [0.0, 1.0],
            [1000000000.0, 1000000001.0],
        ]
        ivs = line["intervals"]
        assert ivs == sorted(ivs)
        assert all(b[1] <= a2[0] for b, a2 in zip(ivs, ivs[1:]))


def test_distant_small_squares_consistent_with_single_local_square():
    # 远处轮廓不应改变局部区间：加 / 不加 1e9 处的方形，原点 1×1 方形
    # 在两条线上的区间都保持 [0,1]（纯平移不变性的另一侧体现）
    r_single, _ = _scan("M0,0h1v1h-1Z", 0.5)
    r_pair, _ = _scan("M0,0h1v1h-1Z M1000000000,0h1v1h-1Z", 0.5)
    near = [[iv for iv in line["intervals"] if iv[1] <= 1.0]
            for line in r_pair["lines"]]
    assert near == [line["intervals"] for line in r_single["lines"]]


def test_shared_outer_hole_edge_endpoints_cancel_at_large_coordinate():
    # 外环 0..10 与孔 0..6 在 x=0 处共享整条竖边：共点端点必须抵消，
    # 每行只剩 [6,10]；整体平移到 1e9 后逐位一致（共点判定只用局部
    # 浮点不确定度，不再依赖整体包围盒容差）。
    d0 = "M0,0h10v10h-10Z M0,0h6v10h-6Z"
    d1 = ("M1000000000,0h10v10h-10Z "
          "M1000000000,0h6v10h-6Z")
    for d, ox in ((d0, 0.0), (d1, 1000000000.0)):
        r, contours = _scan(d, 5.0)
        assert [c.role for c in contours if c.role in ("outer", "hole")] \
            == ["outer", "hole"]
        for line in r["lines"]:
            assert line["intervals"] == [[6.0 + ox, 10.0 + ox]]


def test_shared_hole_island_edge_endpoints_cancel():
    # 外方 0..10、孔 2..8、岛 2..4：孔与岛在 x=2 共享竖边，共点端点
    # 必须抵消——岛把 [2,4] 填回后左侧净填充为 [0,4]（共点合并既不能
    # 漏掉抵消，也不能把 4 处的孔/岛边界错误吞掉）
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z M2,2h2v6h-2Z"
    r, contours = _scan(d, 2.0)
    roles = [(c.original_index, c.role)
             for c in contours if c.role in ("outer", "hole")]
    assert roles == [(0, "outer"), (1, "hole"), (2, "outer")]
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    # y=2/4/6：孔 [2,8] 被岛 [2,4] 填回 -> 净填充 [0,4] 与 [8,10]
    for yy in (2.0, 4.0, 6.0):
        assert by_y[yy] == [[0.0, 4.0], [8.0, 10.0]]
    assert by_y[0.0] == [[0.0, 10.0]]
    assert by_y[8.0] == [[0.0, 10.0]]


def test_coincident_diagonal_and_vertical_crossings_cancel_far_from_origin():
    # 外环右竖边与三角孔的斜边在 x=1e9+10 处共顶点：半开规则下斜边
    # 插值交点与竖边精确交点仅差若干 ulp，必须按共点抵消，不留零宽度
    # 残差区间，也不把外环左半边吞掉。
    d = ("M1000000000,0 L1000000010,0 L1000000010,10 L1000000000,10 Z "
         "M1000000005,5 L1000000010,0 L1000000010,10 Z")
    r, contours = _scan(d, 1.0)
    assert [c.role for c in contours if c.role in ("outer", "hole")] \
        == ["outer", "hole"]
    by_y = {line["y"]: line["intervals"] for line in r["lines"]}
    assert by_y[0.0] == [[1000000000.0, 1000000010.0]]
    assert by_y[5.0] == [[1000000000.0, 1000000005.0]]
    # 所有区间宽度均为正，且右端点附近没有零宽度伪区间
    for line in r["lines"]:
        for a, b in line["intervals"]:
            assert b > a


def test_zero_width_interval_on_shared_slanted_edge_not_emitted():
    # 外环三角 (0,0)-(20,0)-(10,10) 与被强制为孔的三角
    # (10,10)-(20,0)-(30,10) 在几何上共享斜边 (20,0)-(10,10)，
    # 但两环对该边的遍历方向/端点顺序不同，插值得到的交点 x 可能仅在
    # 若干 ulp 内相同：共点抵消后应为覆盖度 0，真正的零宽度区间不输出。
    outer = [(0.0, 0.0), (20.0, 0.0), (10.0, 10.0)]
    hole = [(10.0, 10.0), (20.0, 0.0), (30.0, 10.0)]
    contours = [_forced_contour(0, outer, 0), _forced_contour(1, hole, 1)]
    for y in (1.0, 2.5, 5.0, 7.5, 9.0):
        assert _line_intervals(contours, y, 1.0) == []


def test_distinct_nearly_coincident_crossings_keep_positive_interval():
    # 嵌套三角孔的右斜边 (30,10)-(10,30)：x=40-y，与外环右斜边
    # (40.5,0)-(0,40)：x=40.5-1.0125y 几何错开（半开规则下两斜边都
    # 参与求交）。共点附近 0.35/0.25/0.15 宽的真实嵌缝填充区间必须
    # 保留，不能被局部不确定度（~1e-13）吞掉。
    outer = [(0.0, 0.0), (40.5, 0.0), (0.0, 40.0)]
    hole = [(10.0, 10.0), (30.0, 10.0), (10.0, 30.0)]
    contours = [_forced_contour(0, outer, 0), _forced_contour(1, hole, 1)]
    assert _line_intervals(contours, 12.0, 1.0) == pytest.approx(
        [(0.0, 10.0), (28.0, 28.35)], abs=1e-9)
    assert _line_intervals(contours, 20.0, 1.0) == pytest.approx(
        [(0.0, 10.0), (20.0, 20.25)], abs=1e-9)
    assert _line_intervals(contours, 28.0, 1.0) == pytest.approx(
        [(0.0, 10.0), (12.0, 12.15)], abs=1e-9)


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
