# test_kinematics_full_v2.py
# 适配你当前 fiveaxis_kins.py 的实现（含“第二轴随动”的方向模型，FK 中 orientation 调用
# _tool_axis_from_angles_direction_only，IK 无偏转时优先解析、有偏转/失败时走数值）。
#
# 关注点：
#  1) FK 基础几何正确（零位、一轴旋转、两轴随动、周期性、线性平移解耦）
#  2) 平移构件影响：spindle_swing 只影响位置不影响姿态；tool_length 与姿态方向一致（或相反）
#  3) IK 数值路径回代一致（强制给 deflection 以跳过解析），并验证连续性
#  4) 兼容三种机型/多个轴序/不同 tool_dir 与 tool_axis_sign
#
# 使用前把导入改成你的模块名（下面按你贴出的文件名）
from math import isclose
import numpy as np
import pytest

from fiveaxis_kins import (
    MachineType, RotaryAxis, MachineConfig, KinematicSolver, _wrap180
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
    return getattr(getattr(solver, "config", object()), "tool_axis_sign", 1)

def angles_close(a, b, tol=1e-5):
    """按 360° 周期比较两角接近程度。"""
    da = abs(_wrap180(a - b))
    return da < tol

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
    rot_sign=(1, 1),
):
    cfg = MachineConfig()
    cfg.machine_type = mtype
    cfg.rotary_axes = axes
    cfg.rotary_axes_direction = rot_sign
    cfg.primary_rotary_center = np.array(primary_center, dtype=float)
    cfg.secondary_rotary_offset = np.array(secondary_offset, dtype=float)
    cfg.primary_deflection = None if primary_deflection is None else unit(primary_deflection)
    cfg.secondary_deflection = None if secondary_deflection is None else unit(secondary_deflection)
    cfg.tool_direction = tool_dir
    cfg.tool_length = float(tool_length)
    # 非双转台机型需要 spindle_swing_offset
    if mtype != MachineType.TABLE_TILTING:
        cfg.spindle_swing_offset = np.array(spindle_swing, dtype=float)
    # 很宽的行程/角度限制
    cfg.linear_limits = {'X': (-1e6, 1e6), 'Y': (-1e6, 1e6), 'Z': (-1e6, 1e6)}
    cfg.rotary_limits = {'A': (-1e6, 1e6), 'B': (-1e6, 1e6), 'C': (-1e6, 1e6)}
    setattr(cfg, "tool_axis_sign", tool_axis_sign)
    return KinematicSolver(cfg)

# ========= FK 基本一致性 =========

@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
@pytest.mark.parametrize("sign", [+1, -1])
def test_fk_zero_pose_orientation_matches_tool_dir(tool_dir, sign):
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir, tool_axis_sign=sign)
    q = {'X':0,'Y':0,'Z':0,
         solver.config.rotary_axes[0].value:0.0,
         solver.config.rotary_axes[1].value:0.0}
    pos, ori = solver.forward_kinematics(q)
    expected = sign * tool_dir_vec(tool_dir)
    assert np.allclose(ori, expected, atol=1e-12)

def test_fk_zero_pose_position_matches_fixed_chain():
    solver = make_solver(
        mtype=MachineType.TABLE_TILTING,
        primary_center=(10., 20., 30.),
        secondary_offset=(1., 2., 3.),
        tool_length=80.0,
    )
    q = {'X':0,'Y':0,'Z':0,
         solver.config.rotary_axes[0].value:0.0,
         solver.config.rotary_axes[1].value:0.0}
    pos, ori = solver.forward_kinematics(q)

    fixed = (solver.T_base_to_primary @
             np.eye(4) @
             solver.T_primary_to_secondary @
             np.eye(4) @
             solver.spindle_swing_matrix @
             solver.T_secondary_to_tool)
    expected_pos = fixed[:3,3]
    assert np.allclose(pos, expected_pos, atol=1e-12)

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
    assert np.allclose(ori, expected, atol=1e-12)

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
    assert np.allclose(ori, expected, atol=1e-12)

@pytest.mark.parametrize("angles", [(10.0, 20.0), (-30.0, 45.0), (0.0, 0.0), (120.0, -80.0)])
@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
def test_fk_both_axes_with_following_axis_model(angles, tool_dir):
    """按实现的“第二轴随第一轴转”模型验证：expected = (R1 @ R2_eff) @ v0"""
    solver = make_solver(mtype=MachineType.TABLE_TILTING, tool_dir=tool_dir)
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    A, B = angles

    q = {'X':0,'Y':0,'Z':0, a1:A, a2:B}
    pos, ori = solver.forward_kinematics(q)

    sign = get_tool_axis_sign(solver)
    v0 = sign * tool_dir_vec(tool_dir)
    R1 = rodrigues(solver.primary_axis, A)
    n2_eff = R1 @ solver.secondary_axis
    R2_eff = rodrigues(n2_eff, B)
    expected = (R1 @ R2_eff) @ v0
    assert np.allclose(ori, expected, atol=1e-12)

@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_fk_periodicity_orientation_and_position(k):
    solver = make_solver()
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value

    base = {'X':0,'Y':0,'Z':0, a1:33.3, a2:-77.7}
    p0, o0 = solver.forward_kinematics(base)

    q = dict(base)
    q[a1] += 360.0 * k
    q[a2] -= 720.0 * k
    p1, o1 = solver.forward_kinematics(q)

    assert np.allclose(o0, o1, atol=1e-12)
    assert np.allclose(p0, p1, atol=1e-12)

@pytest.mark.parametrize("dxyz", [(0.,0.,0.), (100., -50., 30.), (-1.5, 2.25, -3.75)])
def test_fk_linear_translation_superposition(dxyz):
    solver = make_solver()
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    angles = {a1: 12.3, a2: -45.6}

    q0 = {'X':0,'Y':0,'Z':0, **angles}
    p0, o0 = solver.forward_kinematics(q0)

    q1 = {'X':dxyz[0], 'Y':dxyz[1], 'Z':dxyz[2], **angles}
    p1, o1 = solver.forward_kinematics(q1)

    assert np.allclose(o0, o1, atol=1e-12)
    assert np.allclose(p0 + np.array(dxyz, dtype=float), p1, atol=1e-12)

# ========= 平移构件的影响 =========

@pytest.mark.parametrize("mtype", [MachineType.SPINDLE_TILTING, MachineType.TABLE_SPINDLE_TILTING])
def test_spindle_swing_affects_position_not_orientation(mtype):
    axes = (RotaryAxis.A, RotaryAxis.C)
    solver1 = make_solver(mtype=mtype, axes=axes, spindle_swing=(0., 0., 0.))
    solver2 = make_solver(mtype=mtype, axes=axes, spindle_swing=(30., -40., 50.))
    a1 = axes[0].value; a2 = axes[1].value
    q = {'X':0,'Y':0,'Z':0, a1:22.2, a2:-33.3}

    p1, o1 = solver1.forward_kinematics(q)
    p2, o2 = solver2.forward_kinematics(q)

    assert np.allclose(o1, o2, atol=1e-12)
    assert not np.allclose(p1, p2, atol=1e-9)

@pytest.mark.parametrize("sign", [+1, -1])
@pytest.mark.parametrize("angles", [(0.,0.), (10.,-20.), (33.3, 40.)])
def test_tool_length_moves_along_tool_axis(sign, angles):
    """当 tool_dir='Z' 时，tool_length 增加 ΔL，应使位置沿当前刀轴方向移动：
       Δp ≈ -(R[:,2])*ΔL；而 orientation = sign*R[:,2]，故 Δp = -(1/sign)*orientation*ΔL。"""
    a1, a2 = RotaryAxis.A, RotaryAxis.B
    solverL = make_solver(axes=(a1,a2), tool_dir='Z', tool_axis_sign=sign, tool_length=20.0)
    solverH = make_solver(axes=(a1,a2), tool_dir='Z', tool_axis_sign=sign, tool_length=55.0)
    A, B = angles
    q = {'X':0,'Y':0,'Z':0, a1.value:A, a2.value:B}

    pL, oL = solverL.forward_kinematics(q)
    pH, oH = solverH.forward_kinematics(q)

    # 姿态不因刀长变化
    assert np.allclose(oL, oH, atol=1e-12)

    dL = 55.0 - 20.0
    expected_dp = -(1.0/sign) * oL * dL
    assert np.allclose(pH - pL, expected_dp, atol=1e-9)

# ========= IK（强制数值路径）回代一致与连续性 =========

@pytest.mark.parametrize("mtype,axes", [
    (MachineType.TABLE_TILTING, (RotaryAxis.A, RotaryAxis.B)),
    (MachineType.SPINDLE_TILTING, (RotaryAxis.A, RotaryAxis.C)),
    (MachineType.TABLE_SPINDLE_TILTING, (RotaryAxis.B, RotaryAxis.C)),
])
@pytest.mark.parametrize("tool_dir", ['X','Y','Z'])
def test_ik_numeric_roundtrip_fk(mtype, axes, tool_dir):
    # 通过给 deflection 一个“等于标准轴向”的向量，强制 IK 走数值解路径（避免解析分支差异）。
    std1 = {'A':[1,0,0],'B':[0,1,0],'C':[0,0,1]}[axes[0].value]
    std2 = {'A':[1,0,0],'B':[0,1,0],'C':[0,0,1]}[axes[1].value]
    # 对于非 TABLE_TILTING，给一个 swing（数值上非零）
    swing = (10., -20., 15.) if mtype != MachineType.TABLE_TILTING else (0.,0.,0.)

    solver = make_solver(
        mtype=mtype, axes=axes, tool_dir=tool_dir,
        primary_deflection=std1, secondary_deflection=std2,
        spindle_swing=swing, tool_length=40.0
    )

    # 随机选一组角与线性位置
    rng = np.random.RandomState(123)
    A = float(rng.uniform(-120, 120))
    B = float(rng.uniform(-120, 120))
    X, Y, Z = rng.uniform(-500, 500, size=3)

    q_gt = {'X':X,'Y':Y,'Z':Z, axes[0].value:A, axes[1].value:B}
    pos_gt, dir_gt = solver.forward_kinematics(q_gt)

    # IK
    ik = solver.inverse_kinematics(pos_gt, dir_gt, prev_angles=(0.0, 0.0))

    # FK 回代
    pos_fk, dir_fk = solver.forward_kinematics(ik)

    # 位置与方向应回到目标
    assert np.allclose(pos_fk, pos_gt, atol=1e-6)
    dot = float(np.clip(np.dot(dir_fk, dir_gt), -1.0, 1.0))
    assert dot > 0.9999

@pytest.mark.parametrize("delta", [1e-3, 2e-3, -1.5e-3])
def test_ik_numeric_continuity_small_perturbation(delta):
    # 强制走数值解
    solver = make_solver(
        mtype=MachineType.TABLE_TILTING, axes=(RotaryAxis.A, RotaryAxis.B),
        tool_dir='Z',
        primary_deflection=[1,0,0], secondary_deflection=[0,1,0],
        tool_length=30.0
    )
    pos = np.array([100., -200., 300.])
    d1 = unit([0.2, -0.7, 0.68])

    ik1 = solver.inverse_kinematics(pos, d1, prev_angles=(0.0, 0.0))
    a1 = ik1['A']; b1 = ik1['B']

    # 轻微扰动目标方向
    d2 = unit(d1 + np.array([delta, -2*delta, delta]))
    ik2 = solver.inverse_kinematics(pos, d2, prev_angles=(a1, b1))
    a2 = ik2['A']; b2 = ik2['B']

    # 角度变化不应突跳（由于连续性正则）
    da = abs(_wrap180(a2 - a1))
    db = abs(_wrap180(b2 - b1))
    assert da + db < 5.0  # 一般远小于 1°，留出裕度

# ========= 实现内部一致性（方向） =========

@pytest.mark.parametrize("angles", [(11.0, -33.0), (0.0, 0.0), (120.0, 80.0)])
@pytest.mark.parametrize("tool_dir,sign", [('Z',+1), ('Z',-1), ('X',+1), ('Y',+1)])
def test_fk_orientation_matches_internal_tool_axis_function(angles, tool_dir, sign):
    solver = make_solver(tool_dir=tool_dir, tool_axis_sign=sign)
    a1 = solver.config.rotary_axes[0].value
    a2 = solver.config.rotary_axes[1].value
    A,B = angles
    q = {'X':0,'Y':0,'Z':0, a1:A, a2:B}

    pos, ori = solver.forward_kinematics(q)
    k = solver._tool_axis_from_angles_direction_only(A, B)
    assert np.allclose(ori, k, atol=1e-12)
