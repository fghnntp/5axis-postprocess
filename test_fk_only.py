# test_fk_only.py
# 只测试 forward_kinematics 的一致性与几何性质
# 使用前请把下面的导入改成你工程中的模块名
from math import isclose
import numpy as np
import pytest

from fiveaxis_kins import (
    MachineType, RotaryAxis, MachineConfig, KinematicSolver
)

# ---------- 小工具 ----------

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    assert n > 0
    return v / n

def rodrigues(axis, angle_deg):
    """返回 3x3 旋转矩阵：绕 axis 旋转 angle_deg（度）。"""
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

def get_tool_axis_sign(solver):
    # 兼容：如果你实现了 tool_axis_sign，则用之；否则按 +1 处理
    return getattr(getattr(solver, "config", object()), "tool_axis_sign", 1)

# ---------- 构造求解器 ----------

def make_solver(
    mtype=MachineType.TABLE_TILTING,
    axes=(RotaryAxis.A, RotaryAxis.B),
    tool_dir='Z',
    tool_axis_sign=+1,
    primary_center=(10., 20., 30.),
    secondary_offset=(1., 2., 3.),
    spindle_swing=(0., 0., 0.),
    tool_length=50.0,
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
    # 非双转台机型需要 spindle_swing_offset
    if mtype != MachineType.TABLE_TILTING:
        cfg.spindle_swing_offset = np.array(spindle_swing, dtype=float)
    # 行程限制放宽
    cfg.linear_limits = {'X': (-1e6, 1e6), 'Y': (-1e6, 1e6), 'Z': (-1e6, 1e6)}
    cfg.rotary_limits = {'A': (-1e9, 1e9), 'B': (-1e9, 1e9), 'C': (-1e9, 1e9)}
    # 可选：刀轴方向符号
    setattr(cfg, "tool_axis_sign", tool_axis_sign)

    return KinematicSolver(cfg)

# ---------- 测试 1：零位一致性 ----------

@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
@pytest.mark.parametrize("sign", [+1, -1])
def test_fk_zero_pose_orientation_matches_tool_dir(tool_dir, sign):
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir, tool_axis_sign=sign)
    pos, ori = solver.forward_kinematics({'X':0,'Y':0,'Z':0,
                                          solver.config.rotary_axes[0].value:0.0,
                                          solver.config.rotary_axes[1].value:0.0})
    expected = sign * tool_dir_vec(tool_dir)
    assert np.allclose(ori, expected, atol=1e-9)

def test_fk_zero_pose_position_matches_fixed_chain():
    solver = make_solver(
        mtype=MachineType.TABLE_TILTING,
        primary_center=(10., 20., 30.),
        secondary_offset=(1., 2., 3.),
        tool_length=80.0,
    )
    # 零姿态、线性零
    q = {'X':0,'Y':0,'Z':0,
         solver.config.rotary_axes[0].value:0.0,
         solver.config.rotary_axes[1].value:0.0}
    pos, ori = solver.forward_kinematics(q)

    # 期望：线性(0) + 固定链路平移
    fixed = (solver.T_base_to_primary @
             np.eye(4) @
             solver.T_primary_to_secondary @
             np.eye(4) @
             solver.spindle_swing_matrix @
             solver.T_secondary_to_tool)
    expected_pos = fixed[:3,3]
    assert np.allclose(pos, expected_pos, atol=1e-9)

# ---------- 测试 2：单轴旋转与 Rodrigues 一致 ----------

@pytest.mark.parametrize("angle_deg", [-40.0, -5.0, 0.0, 7.5, 33.3, 120.0])
@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
def test_fk_primary_axis_only_matches_rodrigues(angle_deg, tool_dir):
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir)
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value

    q = {'X':0,'Y':0,'Z':0, a1:angle_deg, a2:0.0}
    pos, ori = solver.forward_kinematics(q)

    sign = get_tool_axis_sign(solver)
    v0 = sign * tool_dir_vec(tool_dir)
    R1 = rodrigues(solver.primary_axis, angle_deg)
    expected = R1 @ v0
    assert np.allclose(ori, expected, atol=1e-9)

@pytest.mark.parametrize("angle_deg", [-40.0, -5.0, 0.0, 7.5, 33.3, 120.0])
@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
def test_fk_secondary_axis_only_matches_rodrigues(angle_deg, tool_dir):
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir)
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value

    q = {'X':0,'Y':0,'Z':0, a1:0.0, a2:angle_deg}
    pos, ori = solver.forward_kinematics(q)

    sign = get_tool_axis_sign(solver)
    v0 = sign * tool_dir_vec(tool_dir)
    R2 = rodrigues(solver.secondary_axis, angle_deg)
    expected = R2 @ v0
    assert np.allclose(ori, expected, atol=1e-9)

# ---------- 测试 3：两轴同时转的次序一致性（与实现相同的顺序） ----------

@pytest.mark.parametrize("angles", [
    (10.0, 20.0), (-30.0, 45.0), (0.0, 0.0), (120.0, -80.0)
])
@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
def test_fk_both_axes_order_matches_impl(angles, tool_dir):
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir)
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    A, B = angles

    q = {'X':0,'Y':0,'Z':0, a1:A, a2:B}
    pos, ori = solver.forward_kinematics(q)

    sign = get_tool_axis_sign(solver)
    v0 = sign * tool_dir_vec(tool_dir)
    # 与实现相同：先绕 primary，再绕 secondary（两者的轴向量取 solver 中定义的“零位全局轴向”）
    R1 = rodrigues(solver.primary_axis, A)
    R2 = rodrigues(solver.secondary_axis, B)
    expected = (R1 @ R2) @ v0
    assert np.allclose(ori, expected, atol=1e-9)

# ---------- 测试 4：角度周期性 ----------

@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_fk_periodicity_orientation(k):
    solver = make_solver()
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value

    base = {'X':0,'Y':0,'Z':0, a1:33.3, a2:-77.7}
    p0, o0 = solver.forward_kinematics(base)

    q = dict(base)
    q[a1] += 360.0 * k
    q[a2] -= 720.0 * k
    p1, o1 = solver.forward_kinematics(q)

    assert np.allclose(o0, o1, atol=1e-9)
    # 位置同样应一致（只有旋转，不影响固定链的平移结果）
    assert np.allclose(p0, p1, atol=1e-9)

# ---------- 测试 5：线性平移与姿态解耦 ----------

@pytest.mark.parametrize("dxyz", [
    (0., 0., 0.),
    (100., -50., 30.),
    (-1.5, 2.25, -3.75),
])
def test_fk_linear_translation_superposition(dxyz):
    solver = make_solver()
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    angles = {a1: 12.3, a2: -45.6}

    q0 = {'X':0,'Y':0,'Z':0, **angles}
    p0, o0 = solver.forward_kinematics(q0)

    q1 = {'X':dxyz[0], 'Y':dxyz[1], 'Z':dxyz[2], **angles}
    p1, o1 = solver.forward_kinematics(q1)

    # 姿态不随线性平移改变
    assert np.allclose(o0, o1, atol=1e-12)
    # 位置按平移叠加
    assert np.allclose(p0 + np.array(dxyz, dtype=float), p1, atol=1e-12)

# ---------- 测试 6：小偏转轴的健壮性（与 Rodrigues 一致） ----------

@pytest.mark.parametrize("which", ["primary", "secondary"])
def test_fk_with_small_axis_deflection(which):
    eps_deg = 0.8  # 轴小偏转角
    if which == "primary":
        # 让主轴略微偏离 X 轴（绕 Y 转 eps）
        prim_def = unit(rodrigues([0,1,0], eps_deg) @ np.array([1.,0.,0.]))
        sec_def = None
    else:
        prim_def = None
        # 让次轴略微偏离 Y 轴（绕 X 转 eps）
        sec_def = unit(rodrigues([1,0,0], eps_deg) @ np.array([0.,1.,0.]))

    solver = make_solver(
        mtype=MachineType.TABLE_TILTING,
        tool_dir='Z',
        primary_deflection=prim_def,
        secondary_deflection=sec_def
    )
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value

    angles = {a1: 17.0, a2: -29.0}
    pos, ori = solver.forward_kinematics({'X':0,'Y':0,'Z':0, **angles})

    sign = get_tool_axis_sign(solver)
    v0 = sign * tool_dir_vec('Z')
    R1 = rodrigues(solver.primary_axis, angles[a1])
    R2 = rodrigues(solver.secondary_axis, angles[a2])
    expected = (R1 @ R2) @ v0
    # 允许更松的容差（因为 deflection 与旋转组合的浮点误差稍大）
    assert np.allclose(ori, expected, atol=1e-9)

# ---------- 测试 7：摆头/混合机型：spindle_swing 只影响位置，不影响姿态 ----------

@pytest.mark.parametrize("mtype", [MachineType.SPINDLE_TILTING, MachineType.TABLE_SPINDLE_TILTING])
def test_fk_spindle_swing_affects_position_not_orientation(mtype):
    axes = (RotaryAxis.A, RotaryAxis.C)
    solver1 = make_solver(mtype=mtype, axes=axes, spindle_swing=(0., 0., 0.))
    solver2 = make_solver(mtype=mtype, axes=axes, spindle_swing=(30., -40., 50.))
    a1 = axes[0].value
    a2 = axes[1].value
    q = {'X':0,'Y':0,'Z':0, a1:22.2, a2:-33.3}

    p1, o1 = solver1.forward_kinematics(q)
    p2, o2 = solver2.forward_kinematics(q)

    # 姿态应一致（平移不改变方向）
    assert np.allclose(o1, o2, atol=1e-12)
    # 位置不同（由 spindle_swing 决定）
    assert not np.allclose(p1, p2, atol=1e-12)
