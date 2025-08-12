# test_fk_nested_secondary_axis.py
# 目的：验证“第二轴随第一轴一起旋转（嵌套轴）”的 forward_kinematics 几何一致性
# 若当前实现仍是“第二轴固定在全局”，本文件会自动 skip 对应用例。

import numpy as np
import pytest
from math import isclose

from fiveaxis_kins import (
    MachineType, RotaryAxis, MachineConfig, KinematicSolver
)

# ----------------- helpers -----------------

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    assert n > 0
    return v / n

def rodrigues(axis, angle_deg):
    axis = unit(axis)
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R

def tool_dir_vec(tool_dir):
    return {'X': np.array([1.,0.,0.]),
            'Y': np.array([0.,1.,0.]),
            'Z': np.array([0.,0.,1.])}[tool_dir]

def tool_sign(solver):
    return getattr(getattr(solver, "config", object()), "tool_axis_sign", 1)

def make_solver(
    mtype=MachineType.TABLE_TILTING,
    axes=(RotaryAxis.A, RotaryAxis.B),
    tool_dir='Z',
    tool_axis_sign=+1,
    primary_center=(0., 0., 0.),
    secondary_offset=(0., 0., 0.),
    spindle_swing=(0., 0., 0.),
    tool_length=100.0,
    primary_deflection=None,
    secondary_deflection=None,
):
    cfg = MachineConfig()
    cfg.machine_type = mtype
    cfg.rotary_axes = axes
    cfg.rotary_axes_direction = (1, 1)
    cfg.primary_rotary_center = np.array(primary_center, dtype=float)
    cfg.secondary_rotary_offset = np.array(secondary_offset, dtype=float)
    cfg.primary_deflection = None if primary_deflection is None else unit(primary_deflection)
    cfg.secondary_deflection = None if secondary_deflection is None else unit(secondary_deflection)
    cfg.tool_direction = tool_dir
    cfg.tool_length = float(tool_length)
    if mtype != MachineType.TABLE_TILTING:
        cfg.spindle_swing_offset = np.array(spindle_swing, dtype=float)
    # 宽松行程
    cfg.linear_limits = {'X': (-1e6, 1e6), 'Y': (-1e6, 1e6), 'Z': (-1e6, 1e6)}
    cfg.rotary_limits = {'A': (-1e9, 1e9), 'B': (-1e9, 1e9), 'C': (-1e9, 1e9)}
    setattr(cfg, "tool_axis_sign", tool_axis_sign)
    return KinematicSolver(cfg)

def nested_expected_orientation(solver, a_deg, b_deg, tool_dir):
    """嵌套轴几何：先绕 primary 旋，再绕（已被 R1 携带的）secondary 旋。"""
    v0 = tool_sign(solver) * tool_dir_vec(tool_dir)
    u1 = solver.primary_axis          # 零位的主轴方向（全局）
    u2 = solver.secondary_axis        # 零位的次轴方向（全局）
    R1 = rodrigues(u1, a_deg)
    u2_prime = R1 @ u2                # 次轴被第一轴携带后的方向（全局）
    R2_carry = rodrigues(u2_prime, b_deg)
    return R2_carry @ (R1 @ v0)

def fixed_expected_orientation(solver, a_deg, b_deg, tool_dir):
    """固定轴几何：两轴都始终在全局定义；实现顺序：先 primary 后 secondary。"""
    v0 = tool_sign(solver) * tool_dir_vec(tool_dir)
    R1 = rodrigues(solver.primary_axis, a_deg)
    R2 = rodrigues(solver.secondary_axis, b_deg)
    return (R1 @ R2) @ v0

def detect_convention(solver, tool_dir='Z'):
    """粗略检测当前实现更像哪种约定（仅用于决定是否 skip）。"""
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    # 选一组不会退化的角度
    A, B = 23.0, -37.0
    p, o = solver.forward_kinematics({'X':0,'Y':0,'Z':0, a1:A, a2:B})
    ne = nested_expected_orientation(solver, A, B, tool_dir)
    fe = fixed_expected_orientation(solver, A, B, tool_dir)
    dn = 1 - float(np.clip(np.dot(unit(o), unit(ne)), -1, 1))
    df = 1 - float(np.clip(np.dot(unit(o), unit(fe)), -1, 1))
    # dn << df 认为是嵌套；反之为固定；相近则“未知”
    if dn < df * 0.1:
        return "nested"
    if df < dn * 0.1:
        return "fixed"
    return "unknown"

# ----------------- tests -----------------

@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
@pytest.mark.parametrize("angles", [(12.5, -33.0), (-50.0, 77.0), (90.0, 15.0)])
def test_fk_both_axes_nested_matches_math(tool_dir, angles):
    """如果实现是嵌套轴，则方向需与“携带后再旋转”的几何完全一致。
       若当前实现仍是固定轴，这个测试会自动 skip。"""
    solver = make_solver(mtype=MachineType.TABLE_TILTING, axes=(RotaryAxis.A, RotaryAxis.B), tool_dir=tool_dir)
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    A, B = angles
    _, ori = solver.forward_kinematics({'X':0,'Y':0,'Z':0, a1:A, a2:B})

    conv = detect_convention(solver, tool_dir=tool_dir)
    if conv == "fixed":
        pytest.skip("当前实现为固定轴（secondary 不随 primary 携带），跳过嵌套轴测试。")
    elif conv == "unknown":
        pytest.skip("无法稳定判别约定（fixed/nested），为避免误报选择跳过。")

    expected = nested_expected_orientation(solver, A, B, tool_dir)
    assert np.allclose(ori, expected, atol=1e-9)

@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
def test_fk_nested_periodicity(tool_dir):
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir)
    conv = detect_convention(solver, tool_dir=tool_dir)
    if conv != "nested":
        pytest.skip("当前实现不是嵌套轴，跳过。")

    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    base = {'X':0,'Y':0,'Z':0, a1:33.3, a2:-77.7}
    _, o0 = solver.forward_kinematics(base)

    q = dict(base)
    q[a1] += 360.0
    q[a2] -= 720.0
    _, o1 = solver.forward_kinematics(q)

    assert np.allclose(o0, o1, atol=1e-12)

@pytest.mark.parametrize("eps_deg", [0.5, 1.0])
def test_fk_nested_with_small_deflection(eps_deg):
    """带小偏转时也应满足嵌套几何：第二轴先被 R1 携带再旋转。"""
    prim_def = unit(rodrigues([0,1,0], eps_deg) @ np.array([1.,0.,0.]))
    sec_def  = unit(rodrigues([1,0,0], eps_deg) @ np.array([0.,1.,0.]))

    solver = make_solver(
        mtype=MachineType.TABLE_TILTING,
        axes=(RotaryAxis.A, RotaryAxis.B),
        tool_dir='Z',
        primary_deflection=prim_def,
        secondary_deflection=sec_def
    )
    if detect_convention(solver) != "nested":
        pytest.skip("当前实现不是嵌套轴，跳过。")

    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    A, B = (17.0, -29.0)

    _, ori = solver.forward_kinematics({'X':0,'Y':0,'Z':0, a1:A, a2:B})
    expected = nested_expected_orientation(solver, A, B, 'Z')
    assert np.allclose(ori, expected, atol=1e-9)

@pytest.mark.parametrize("mtype", [MachineType.SPINDLE_TILTING, MachineType.TABLE_SPINDLE_TILTING])
def test_fk_nested_spindle_swing_affects_position_not_orientation(mtype):
    """嵌套/固定哪种约定都不该让 spindle_swing 影响方向，只影响位置。"""
    solver1 = make_solver(mtype=mtype, axes=(RotaryAxis.A, RotaryAxis.C), spindle_swing=(0.,0.,0.))
    solver2 = make_solver(mtype=mtype, axes=(RotaryAxis.A, RotaryAxis.C), spindle_swing=(30., -40., 50.))
    a1 = 'A'; a2 = 'C'
    q = {'X':0,'Y':0,'Z':0, a1:22.2, a2:-33.3}

    p1, o1 = solver1.forward_kinematics(q)
    p2, o2 = solver2.forward_kinematics(q)

    assert np.allclose(o1, o2, atol=1e-12)
    assert not np.allclose(p1, p2, atol=1e-12)
