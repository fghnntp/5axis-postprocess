import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, List
from scipy.optimize import minimize
import warnings

class MachineType(Enum):
    TABLE_TILTING = 1  # 两旋转轴在工作台 (如AC双转台)
    SPINDLE_TILTING = 2  # 两旋转轴在主轴 (如AB双摆头)
    TABLE_SPINDLE_TILTING = 3  # 各一旋转轴 (如A转台+C摆头)


class RotaryAxis(Enum):
    A = 'A'  # 绕X轴旋转
    B = 'B'  # 绕Y轴旋转
    C = 'C'  # 绕Z轴旋转

def _build_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    使用 Rodrigues 公式构建三维旋转矩阵。
    参数:
        axis: 旋转轴（单位向量）
        angle: 旋转角度（度）
    返回:
        3x3 旋转矩阵
    """
    angle_rad = np.radians(angle)
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    # 构建旋转轴的叉乘矩阵 K
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Rodrigues 旋转公式：R = I + sinθ*K + (1-cosθ)*K^2
    R = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)
    return R

def rotation_error(theta1: float,
                   theta2: float,
                   n1: np.ndarray,
                   n2: np.ndarray,
                   tool_direction: np.ndarray,
                   machine) -> float:
    """
    给定两个旋转角，计算当前刀具方向与目标刀具方向之间的角度误差（弧度）

    参数:
        theta1: 第一轴角度（deg）
        theta2: 第二轴角度（deg）
        n1: 第一轴单位向量
        n2: 第二轴单位向量
        tool_direction: 期望刀具方向单位向量
        machine: 当前 KinematicSolver 实例

    返回:
        实际刀具方向与目标方向的夹角（弧度）
    """
    # 构建两个旋转矩阵（仅 3x3）
    R1 = _build_rotation_matrix(n1, theta1)
    R2 = _build_rotation_matrix(n2, theta2)

    # 构建两个 4x4 齐次旋转矩阵
    R1_hom = np.eye(4)
    R1_hom[:3, :3] = R1

    R2_hom = np.eye(4)
    R2_hom[:3, :3] = R2

    # 构建完整变换链（用于求解刀具方向）
    T = (
        machine.T_base_to_primary @
        R1_hom @
        machine.T_primary_to_secondary @
        R2_hom @
        machine.spindle_swing_matrix @
        machine.T_secondary_to_tool
    )

    # 提取刀具方向（根据 config 中设定的刀具方向轴）
    col_index = {'X': 0, 'Y': 1, 'Z': 2}[machine.config.tool_direction]
    K_actual = T[:3, col_index]
    K_actual = K_actual / np.linalg.norm(K_actual)

    # 计算方向夹角（弧度）
    dot = np.clip(np.dot(K_actual, tool_direction), -1.0, 1.0)
    angle_error = np.arccos(dot)
    return angle_error


def solve_rotary_angles_numeric(tool_direction: np.ndarray,
                                machine,
                                initial_guess=(0.0, 0.0)) -> tuple:
    """
    使用数值优化方式求解逆运动学中的两个旋转角度（适用于偏心旋转轴）。
    若优化失败，自动回退为简化近似解。

    参数:
        tool_direction: 目标刀具方向（单位向量）
        machine: KinematicSolver 实例
        initial_guess: 初始角度猜测 (θ1, θ2)，单位为度

    返回:
        (θ1, θ2): 优化求解的角度解（单位为度）
    """

    # ---------- 1. 归一化方向 ----------
    tool_direction = np.asarray(tool_direction, dtype=float)
    norm = np.linalg.norm(tool_direction)
    if norm < 1e-6:
        print("[警告] 输入的刀具方向向量太小，可能为零向量")
        return 0.0, 0.0
    tool_direction /= norm

    # ---------- 2. 提取旋转轴信息 ----------
    n1 = machine.primary_axis
    n2 = machine.secondary_axis

    # ---------- 3. 构造损失函数 ----------
    def loss_fn(angles):
        theta1, theta2 = angles
        return rotation_error(theta1, theta2, n1, n2, tool_direction, machine)

    # ---------- 4. 设定角度范围 ----------
    bounds = [
        machine.config.rotary_limits[machine.config.rotary_axes[0].value],
        machine.config.rotary_limits[machine.config.rotary_axes[1].value],
    ]

    # ---------- 5. 进行优化 ----------
    try:
        result = minimize(
            loss_fn,
            x0=np.array(initial_guess, dtype=float),
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-9}
        )

        if result.success:
            # ---------- 6. 正常返回 ----------
            def normalize(angle):
                return (angle + 180) % 360 - 180

            theta1 = normalize(result.x[0])
            theta2 = normalize(result.x[1])
            return theta1, theta2

        else:
            raise RuntimeError(result.message)

    except Exception as e:
        # ---------- 7. 失败时回退处理 ----------
        print("\n[❌ 数值求解失败，自动回退简化解]")
        print(f"失败信息: {str(e)}")
        print(f"→ 工具方向: {tool_direction}")
        print(f"→ 初始角度猜测: {initial_guess}")
        print(f"→ 回退方式: 固定第二轴为初始值，粗略估算第一轴")

        # 用固定初始的第二轴角度，快速估算第一轴角度
        theta2 = initial_guess[1]

        def quick_theta1_estimate():
            min_error = float("inf")
            best_theta1 = initial_guess[0]
            for t in np.linspace(-180, 180, 73):  # 每隔 5°
                err = rotation_error(t, theta2, n1, n2, tool_direction, machine)
                if err < min_error:
                    min_error = err
                    best_theta1 = t
            return best_theta1

        theta1 = quick_theta1_estimate()

        return theta1, theta2


class MachineConfig:
    """
    五轴机床运动学配置类 - 基于笛卡尔坐标系
    完整定义机床几何结构和运动学参数
    """

    def __init__(self):
        # 机床基本构型
        self.machine_type: MachineType = None

        # 线性轴配置
        se
        linear_axes: Tuple[str, str, str] = ('X', 'Y', 'Z')  # 线性轴名称

        # 旋转轴配置
        self.rotary_axes: Tuple[RotaryAxis, RotaryAxis] = None  # 两旋转轴类型(A/B/C)
        self.rotary_axes_direction: Tuple[int, int] = None  # 旋转方向(1或-1)

        # 主旋转轴参数
        self.primary_rotary_center: np.ndarray = None  # 主旋转轴心坐标(机械坐标系下XYZ)
        self.primary_deflection: np.ndarray = None  # 主旋转轴偏角单位向量(相对于法线)

        # 次旋转轴参数
        self.secondary_rotary_offset: np.ndarray = None  # 次旋转轴相对于主旋转轴心的偏移
        self.secondary_deflection: np.ndarray = None  # 次旋转轴偏角单位向量

        # 刀具参数
        self.tool_direction: str = 'Z'  # 默认刀具方向(可能为XYZ,通常为Z轴)
        self.spindle_swing_offset: np.ndarray = None  # 主轴中心到第二旋转中心的偏移, 通常只有摆头有这个偏移量
        self.tool_length: float = 0.0  # 刀具长度(刀尖到旋转中心的距离)

        # 机床限制参数
        self.linear_limits: dict = {}  # 线性轴行程限制 {'X': (min, max), ...}
        self.rotary_limits: dict = {}  # 旋转轴角度限制 {'A': (min, max), ...}

    def validate_config(self):
        """验证配置是否完整有效"""
        required_params = [
            'machine_type', 'rotary_axes', 'primary_rotary_center',
            'secondary_rotary_offset'
        ]
        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(f"缺少必要参数: {param}")

    def get_rotary_axis_sequence(self) -> str:
        """获取旋转轴序列字符串(如'AC'、'BA'等)"""
        return ''.join([axis.value for axis in self.rotary_axes])

    def get_tool_direction_vector(self) -> np.ndarray:
        """获取刀具方向单位向量"""
        if self.tool_direction == 'X':
            return np.array([1, 0, 0])
        elif self.tool_direction == 'Y':
            return np.array([0, 1, 0])
        else:  # 默认为Z
            return np.array([0, 0, 1])

    def get_rotary_center(self, axis: str) -> np.ndarray:
        """获取指定旋转轴的中心坐标"""
        if axis == self.rotary_axes[0].value:
            return self.primary_rotary_center
        elif axis == self.rotary_axes[1].value:
            return self.primary_rotary_center + self.secondary_rotary_offset
        else:
            raise ValueError(f"无效的旋转轴: {axis}")

_TOOL_DIRECTION_MAP = {'X': 0, 'Y': 1, 'Z': 2}
class KinematicSolver:
    def __init__(self, config: MachineConfig):
        self.config = config
        self._validate_config()
        self._build_transformation_matrices()

    def _validate_config(self):
        """验证配置完整性"""
        self.config.validate_config()
        if not self.config.rotary_axes or len(self.config.rotary_axes) != 2:
            raise ValueError("必须指定两个旋转轴")
        if self.config.primary_rotary_center is None:
            raise ValueError("必须指定主旋转中心")
        if self.config.secondary_rotary_offset is None:
            raise ValueError("必须指定次旋转中心偏移")
        if self.config.machine_type != MachineType.TABLE_TILTING and \
                self.config.spindle_swing_offset is None:
            raise ValueError("非双转台需要定义第二旋转中心到主轴的偏移")
        if self.config.tool_length < 0:
            raise ValueError("刀具长度必须大于等于0")

    def _build_transformation_matrices(self):
        """根据机床配置构建变换矩阵（改进版）

        现在正确处理旋转轴方向和偏转矢量，使用Rodrigues旋转公式构建旋转矩阵
        """
        # 1. 构建平移变换矩阵
        # 从机床基座到主旋转轴的变换
        self.T_base_to_primary = self._build_translation_matrix(self.config.primary_rotary_center)

        # 从主旋转轴到次旋转轴的变换
        offset = self.config.secondary_rotary_offset
        self.T_primary_to_secondary = self._build_translation_matrix(offset)

        # 从次旋转轴到刀具的变换
        tool_offset = np.array([0, 0, -self.config.tool_length])
        self.T_secondary_to_tool = self._build_translation_matrix(tool_offset)

        # 2. 确定旋转轴方向向量（考虑deflection和旋转方向）
        # 获取标准旋转轴向量
        primary_std_axis = self._get_standard_axis_vector(self.config.rotary_axes[0])
        secondary_std_axis = self._get_standard_axis_vector(self.config.rotary_axes[1])

        # 应用偏转矢量（如果有）
        self.primary_axis = self.config.primary_deflection if self.config.primary_deflection is not None \
            else primary_std_axis
        self.secondary_axis = self.config.secondary_deflection if self.config.secondary_deflection is not None \
            else secondary_std_axis

        # 标准化方向向量
        self.primary_axis = self.primary_axis / np.linalg.norm(self.primary_axis)
        self.secondary_axis = self.secondary_axis / np.linalg.norm(self.secondary_axis)

        # 3. 确定旋转方向（考虑旋转轴方向设置）
        self.primary_rotation_sign = self.config.rotary_axes_direction[0] if self.config.rotary_axes_direction else 1
        self.secondary_rotation_sign = self.config.rotary_axes_direction[1] if self.config.rotary_axes_direction else 1

        # 4.摆头偏移矩阵应在需要时构建
        if self.config.spindle_swing_offset is not None:
            self.spindle_swing_matrix = self._build_translation_matrix(
                self.config.spindle_swing_offset
            )
        else:
            self.spindle_swing_matrix = np.eye(4)

    def _get_standard_axis_vector(self, axis: RotaryAxis) -> np.ndarray:
        """获取标准旋转轴向量

        根据旋转轴类型返回标准轴向量，不考虑偏转和方向
        """
        if axis == RotaryAxis.A:
            return np.array([1, 0, 0])  # X轴
        elif axis == RotaryAxis.B:
            return np.array([0, 1, 0])  # Y轴
        elif axis == RotaryAxis.C:
            return np.array([0, 0, 1])  # Z轴
        else:
            raise ValueError(f"未知旋转轴类型: {axis}")

    def _build_translation_matrix(self, translation: np.ndarray) -> np.ndarray:
        """构建平移变换矩阵"""
        return np.array([
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1]
        ])

    def _build_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """使用Rodrigues旋转公式构建旋转矩阵

        参数:
            axis: 旋转轴单位向量
            angle: 旋转角度（度）

        返回:
            4x4齐次旋转矩阵
        """
        angle_rad = np.radians(angle)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        # Rodrigues旋转公式
        # R = I + sin(θ)*K + (1-cos(θ))*K^2
        # 其中K是旋转轴的叉积矩阵

        # 构建叉积矩阵
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # 计算旋转矩阵
        R = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)

        # 转换为齐次坐标
        return np.vstack([
            np.hstack([R, np.zeros((3, 1))]),
            np.array([0, 0, 0, 1])
        ])

    def forward_kinematics(self, joint_values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算五轴机床的正向运动学（刀具位置和方向）

        返回:
            position: (3,) 刀具尖端在工件坐标系中的位置
            orientation: (3,) 刀具的方向单位向量（已归一化）
        """
        # 1. 获取线性轴位置（程序坐标/工件坐标系下的X,Y,Z）
        linear_pos = np.array([
            joint_values.get('X', 0.0),
            joint_values.get('Y', 0.0),
            joint_values.get('Z', 0.0)
        ], dtype=float)

        # 2. 获取旋转轴名与角度（并乘以设置的旋转方向符号）
        first_axis = self.config.rotary_axes[0].value
        second_axis = self.config.rotary_axes[1].value

        first_angle = float(joint_values.get(first_axis, 0.0)) * float(self.primary_rotation_sign)
        second_angle = float(joint_values.get(second_axis, 0.0)) * float(self.secondary_rotation_sign)

        # 3. 构建变换链（按注释顺序执行）
        # 注意：矩阵乘法顺序很重要。这里的语义是：
        #    先做 "程序坐标系 -> 平移(linear_pos)"，再做 "到主旋转轴中心的平移(T_base_to_primary)"。
        # 也就是 T_machine_to_first_center = Trans(linear_pos) * T_base_to_primary
        T_machine_to_first_center = self._build_translation_matrix(linear_pos) @ self.T_base_to_primary

        # 第一轴旋转（关于 primary_axis，角度 first_angle）
        R_first_axis = self._build_rotation_matrix(self.primary_axis, first_angle)

        # 从第一轴中心到第二轴中心的偏移（固定）
        T_first_to_second_center = self.T_primary_to_secondary

        # 第二轴旋转（关于 secondary_axis，角度 second_angle）
        R_second_axis = self._build_rotation_matrix(self.secondary_axis, second_angle)

        # 第二轴到刀具（包含摆头偏移与刀具长度偏移）
        T_second_to_tool = self.spindle_swing_matrix @ self.T_secondary_to_tool

        # 组合完整的齐次变换：机床程序坐标 -> 刀具尖端
        T_total = (T_machine_to_first_center @
                   R_first_axis @
                   T_first_to_second_center @
                   R_second_axis @
                   T_second_to_tool)

        # 4. 提取刀具位置（齐次矩阵的平移分量）
        position = T_total[:3, 3].astype(float)

        # 5. 提取刀具方向向量：取变换矩阵的第 tool_dir_index 列（0->X,1->Y,2->Z）
        tool_dir_index = _TOOL_DIRECTION_MAP.get(self.config.tool_direction, 2)
        orientation = T_total[:3, tool_dir_index].astype(float)

        # 6. 归一化方向并返回（并在异常情况下提供诊断）
        norm = np.linalg.norm(orientation)
        if norm < 1e-9:
            raise ValueError("Computed tool orientation has near-zero length; check axes/deflection settings.")
        orientation = orientation / norm

        return position, orientation

    def _select_optimal_angle(self, candidates: List[float], prev_angle: float, axis: str) -> float:
        """
        选择最优角度解 (论文第3节最短路径原则)
        """
        valid_angles = [a for a in candidates
                        if self.config.rotary_limits[axis][0] <= a <= self.config.rotary_limits[axis][1]]

        if not valid_angles:
            raise ValueError(f"No valid solution for {axis} axis within limits")

        return min(valid_angles, key=lambda x: abs(x - prev_angle))

    def _generate_periodic_candidates(self, base_angle: float, axis: str) -> List[float]:
        """
        生成周期候选解 (处理360°周期性问题)
        """
        return [base_angle + k * 360 for k in [-1, 0, 1]
                if self.config.rotary_limits[axis][0] <= base_angle + k * 360 <= self.config.rotary_limits[axis][1]]

    def _solve_spindle_tilting_angles(self, tool_dir_local: np.ndarray,
                                      prev_angles: Tuple[float, float]) -> Tuple[float, float]:
        """
        主轴摆动型机床角度解算（完整实现论文 Table 1）
        """
        Kx, Ky, Kz = tool_dir_local
        axis1 = self.config.rotary_axes[0].value
        axis2 = self.config.rotary_axes[1].value
        td = self.config.tool_direction  # 'X', 'Y', or 'Z'

        # 工具函数
        def sel(base, axis_name, prev):
            return self._select_optimal_angle(
                self._generate_periodic_candidates(base, axis_name),
                prev, axis_name
            )

        # AB 型
        if axis1 == 'A' and axis2 == 'B':
            if td == 'X':
                fy_cand = [np.degrees(np.arccos(Kx)), -np.degrees(np.arccos(Kx))]
                fy = self._select_optimal_angle(fy_cand, prev_angles[1], 'B')
                fx = sel(np.degrees(np.arctan2(Ky, -Kz)), 'A', prev_angles[0])
                return fx, fy
            elif td == 'Z':
                fy_cand = [np.degrees(np.arcsin(Kx)),
                           np.copysign(180 - abs(np.degrees(np.arcsin(Kx))), Kx)]
                fy = self._select_optimal_angle(fy_cand, prev_angles[1], 'B')
                fx = sel(np.degrees(np.arctan2(-Ky, Kz)), 'A', prev_angles[0])
                return fx, fy

        # AC 型
        elif axis1 == 'A' and axis2 == 'C':
            if td == 'X':
                fz_cand = [np.degrees(np.arccos(Kx)), -np.degrees(np.arccos(Kx))]
                fz = self._select_optimal_angle(fz_cand, prev_angles[1], 'C')
                fx = sel(np.degrees(np.arctan2(Kz, Ky)), 'A', prev_angles[0])
                return fx, fz
            elif td == 'Y':
                fz_cand = [np.degrees(np.arccos(-Kx)), -np.degrees(np.arccos(-Kx))]
                fz = self._select_optimal_angle(fz_cand, prev_angles[1], 'C')
                fx = sel(np.degrees(np.arctan2(Kz, Ky)), 'A', prev_angles[0])
                return fx, fz

        # BA 型
        elif axis1 == 'B' and axis2 == 'A':
            if td == 'Y':
                fx_cand = [np.degrees(np.arccos(Ky)), -np.degrees(np.arccos(Ky))]
                fx = self._select_optimal_angle(fx_cand, prev_angles[1], 'A')
                fy = sel(np.degrees(np.arctan2(Kx, -Kz)), 'B', prev_angles[0])
                return fy, fx
            elif td == 'Z':
                fx_cand = [np.degrees(np.arccos(-Ky)), -np.degrees(np.arccos(-Ky))]
                fx = self._select_optimal_angle(fx_cand, prev_angles[1], 'A')
                fy = sel(np.degrees(np.arctan2(Kx, Kz)), 'B', prev_angles[0])
                return fy, fx

        # BC 型
        elif axis1 == 'B' and axis2 == 'C':
            if td == 'X':
                fz_cand = [np.degrees(np.arccos(Ky)), -np.degrees(np.arccos(Ky))]
                fz = self._select_optimal_angle(fz_cand, prev_angles[1], 'C')
                fy = sel(np.degrees(np.arctan2(-Kz, Kx)), 'B', prev_angles[0])
                return fy, fz
            elif td == 'Y':
                fz_cand = [np.degrees(np.arccos(Ky)), -np.degrees(np.arccos(Ky))]
                fz = self._select_optimal_angle(fz_cand, prev_angles[1], 'C')
                fy = sel(np.degrees(np.arctan2(Kz, -Kx)), 'B', prev_angles[0])
                return fy, fz

        # CA 型
        elif axis1 == 'C' and axis2 == 'A':
            if td == 'Y':
                fx_cand = [np.degrees(np.arccos(Kz)), -np.degrees(np.arccos(Kz))]
                fx = self._select_optimal_angle(fx_cand, prev_angles[1], 'A')
                fz = sel(np.degrees(np.arctan2(-Kx, Ky)), 'C', prev_angles[0])
                return fz, fx
            elif td == 'Z':
                fx_cand = [np.degrees(np.arccos(Kz)), -np.degrees(np.arccos(Kz))]
                fx = self._select_optimal_angle(fx_cand, prev_angles[1], 'A')
                fz = sel(np.degrees(np.arctan2(Kx, Ky)), 'C', prev_angles[0])
                return fz, fx

        # CB 型
        elif axis1 == 'C' and axis2 == 'B':
            if td == 'X':
                fy_cand = [np.degrees(np.arccos(-Kz)), -np.degrees(np.arccos(-Kz))]
                fy = self._select_optimal_angle(fy_cand, prev_angles[1], 'B')
                fz = sel(np.degrees(np.arctan2(Ky, Kx)), 'C', prev_angles[0])
                return fz, fy
            elif td == 'Z':
                fy_cand = [np.degrees(np.arccos(Kz)), -np.degrees(np.arccos(Kz))]
                fy = self._select_optimal_angle(fy_cand, prev_angles[1], 'B')
                fz = sel(np.degrees(np.arctan2(Ky, Kx)), 'C', prev_angles[0])
                return fz, fy

        raise ValueError(f"Unsupported spindle-tilting config {axis1}{axis2} with tool dir {td}")


    def _solve_table_tilting_angles(self, tool_dir_local: np.ndarray,
                                    prev_angles: Tuple[float, float]) -> Tuple[float, float]:
        """
        双转台机床角度解算（Table-tilting）
        返回: (angle_first_axis, angle_second_axis) 单位: 度
        说明: 公式与论文给出的 spindle-tilting 形式类似，但针对两个转轴都在工件侧的情况，
        某些三角函数的被求量符号会有不同（paper 中已给出对应形式）。
        """
        Kx, Ky, Kz = tool_dir_local
        axis1 = self.config.rotary_axes[0].value  # e.g. 'A'
        axis2 = self.config.rotary_axes[1].value  # e.g. 'B'
        td = self.config.tool_direction  # 'X','Y','Z'

        # Helper: 选择最优周期解（与 prev_angles 连续性一致）
        def sel(base, axis_name, prev):
            return self._select_optimal_angle(
                self._generate_periodic_candidates(base, axis_name),
                prev, axis_name
            )

        # 以下分支基于论文给出的 Table 型解式（但针对 table-tilting 的符号/位置做了对应调整）
        # 返回顺序为 (angle_for_axis1, angle_for_axis2)

        # AB 型
        if axis1 == 'A' and axis2 == 'B':
            if td == 'X':
                # 对应 table 型：phi_B = arccos(-Kx) 或 -arccos(-Kx) 等；phi_A = arctan2(Ky, Kz)
                B1 = np.degrees(np.arccos(-Kx))
                B2 = -B1
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                A_base = np.degrees(np.arctan2(Ky, Kz))
                A = sel(A_base, 'A', prev_angles[0])
                return A, B
            elif td == 'Z':
                # phi_B = arcsin(-Kx), phi_A = arctan2(Ky, -Kz)
                B1 = np.degrees(np.arcsin(-Kx))
                B2 = np.copysign(180 - abs(B1), B1)
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                A_base = np.degrees(np.arctan2(Ky, -Kz))
                A = sel(A_base, 'A', prev_angles[0])
                return A, B
            elif td == 'Y':
                # AB 型刀具朝 Y 不可行（参考论文，若可行需按具体机型扩展）
                raise ValueError("AB table-tilting with tool_direction='Y' is typically infeasible")

        # AC 型
        elif axis1 == 'A' and axis2 == 'C':
            if td == 'X':
                # phi_C = arccos(-Kx), phi_A = arctan2(Kz, -Ky)
                C1 = np.degrees(np.arccos(-Kx))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                A_base = np.degrees(np.arctan2(Kz, -Ky))
                A = sel(A_base, 'A', prev_angles[0])
                return A, C
            elif td == 'Y':
                # phi_C = arccos(Kx), phi_A = arctan2(Kz, -Ky)
                C1 = np.degrees(np.arccos(Kx))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                A_base = np.degrees(np.arctan2(Kz, -Ky))
                A = sel(A_base, 'A', prev_angles[0])
                return A, C
            elif td == 'Z':
                raise ValueError("AC table-tilting with tool_direction='Z' is typically infeasible")

        # BA 型
        elif axis1 == 'B' and axis2 == 'A':
            if td == 'Y':
                # phi_A = arccos(-Ky), phi_B = arctan2(Kx, Kz)
                A1 = np.degrees(np.arccos(-Ky))
                A2 = -A1
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                B_base = np.degrees(np.arctan2(Kx, Kz))
                B = sel(B_base, 'B', prev_angles[0])
                return B, A
            elif td == 'Z':
                # phi_A = arcsin(-Ky), phi_B = arctan2(-Kx, Kz)
                A1 = np.degrees(np.arcsin(-Ky))
                A2 = np.copysign(180 - abs(A1), A1)
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                B_base = np.degrees(np.arctan2(-Kx, Kz))
                B = sel(B_base, 'B', prev_angles[0])
                return B, A
            elif td == 'X':
                raise ValueError("BA table-tilting with tool_direction='X' is typically infeasible")

        # BC 型
        elif axis1 == 'B' and axis2 == 'C':
            if td == 'X':
                # phi_C = arccos(-Ky), phi_B = arctan2(Kz, Kx)
                C1 = np.degrees(np.arccos(-Ky))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                B_base = np.degrees(np.arctan2(Kz, Kx))
                B = sel(B_base, 'B', prev_angles[0])
                return B, C
            elif td == 'Y':
                # phi_C = arccos(-Ky), phi_B = arctan2(-Kz, -Kx)
                C1 = np.degrees(np.arccos(-Ky))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                B_base = np.degrees(np.arctan2(-Kz, -Kx))
                B = sel(B_base, 'B', prev_angles[0])
                return B, C
            elif td == 'Z':
                raise ValueError("BC table-tilting with tool_direction='Z' is typically infeasible")

        # CA 型
        elif axis1 == 'C' and axis2 == 'A':
            if td == 'Y':
                # phi_A = arccos(-Kz), phi_C = arctan2(Kx, -Ky)
                A1 = np.degrees(np.arccos(-Kz))
                A2 = -A1
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                C_base = np.degrees(np.arctan2(Kx, -Ky))
                C = sel(C_base, 'C', prev_angles[0])
                return C, A
            elif td == 'Z':
                # phi_A = arcsin(-Kz), phi_C = arctan2(-Kx, -Ky)
                A1 = np.degrees(np.arcsin(-Kz))
                A2 = np.copysign(180 - abs(A1), A1)
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                C_base = np.degrees(np.arctan2(-Kx, -Ky))
                C = sel(C_base, 'C', prev_angles[0])
                return C, A
            elif td == 'X':
                raise ValueError("CA table-tilting with tool_direction='X' is typically infeasible")

        # CB 型
        elif axis1 == 'C' and axis2 == 'B':
            if td == 'X':
                # phi_B = arccos(Kz), phi_C = arctan2(-Ky, -Kx)
                B1 = np.degrees(np.arccos(Kz))
                B2 = -B1
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                C_base = np.degrees(np.arctan2(-Ky, -Kx))
                C = sel(C_base, 'C', prev_angles[0])
                return C, B
            elif td == 'Z':
                # phi_B = arccos(-Kz), phi_C = arctan2(-Ky, Kx)
                B1 = np.degrees(np.arccos(-Kz))
                B2 = -B1
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                C_base = np.degrees(np.arctan2(-Ky, Kx))
                C = sel(C_base, 'C', prev_angles[0])
                return C, B
            elif td == 'Y':
                raise ValueError("CB table-tilting with tool_direction='Y' is typically infeasible")

        # 如果没有匹配：说明该组合在 table-tilting 下不可行或需要特殊处理
        raise ValueError(f"Unsupported TABLE_TILTING config {axis1}{axis2} with tool dir {td}")


    def _solve_hybrid_angles(self, tool_dir_local: np.ndarray,
                             prev_angles: Tuple[float, float]) -> Tuple[float, float]:
        """
        混合型机床逆解（TABLE_SPINDLE_TILTING）
        一轴在工作台，一轴在主轴（或反过来）——实现所有轴组合与刀具朝向。
        返回: (angle_for_first_axis, angle_for_second_axis)
        说明: 采用分步法（先解靠近刀具/工件侧的轴，再解另一轴），并使用论文中给出的角公式形式。
        """
        Kx, Ky, Kz = tool_dir_local
        axis1 = self.config.rotary_axes[0].value
        axis2 = self.config.rotary_axes[1].value
        td = self.config.tool_direction

        def sel(base, axis_name, prev):
            return self._select_optimal_angle(
                self._generate_periodic_candidates(base, axis_name),
                prev, axis_name
            )

        # 我们区分两种情况：第一轴在 table（workpiece），第二轴在 spindle（spindle）
        # 还是第一轴在 spindle，第二轴在 table。你在 config.rotary_axes 的顺序决定了 axis1/axis2 的语义。
        # 下面按 axis1=table, axis2=spindle 的习惯进行实现；若你的配置是反过来的，请交换 axis 的含义或在调用处保证顺序。

        # CASE: axis1 (first) = table-side, axis2 (second) = spindle-side
        # 需要先解 table 角（去掉 spindle 对方向的影响），再解 spindle 角。
        # 下面实现针对常见组合的解析式（覆盖 AB/AC/BA/BC/CA/CB 与 X/Y/Z）
        if axis1 == 'A' and axis2 == 'B':
            # 假设 A 在 table、B 在 spindle
            if td == 'X':
                # phi_B (spindle) = arccos(Kx) 之类（参考 spindle 型）
                B1 = np.degrees(np.arccos(Kx))
                B2 = -B1
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                # phi_A (table) 从旋转后关系推得（近似式，按论文混合型对应项）
                A_base = np.degrees(np.arctan2(Ky, -Kz))
                A = sel(A_base, 'A', prev_angles[0])
                return A, B
            elif td == 'Z':
                B1 = np.degrees(np.arcsin(Kx))
                B2 = np.copysign(180 - abs(B1), B1)
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                A_base = np.degrees(np.arctan2(-Ky, Kz))
                A = sel(A_base, 'A', prev_angles[0])
                return A, B
            elif td == 'Y':
                # 某些混合型下 Y 方向可能不可行（视具体轴序）
                raise ValueError("Hybrid AB with tool_direction='Y' may be infeasible for this axis ordering.")

        if axis1 == 'A' and axis2 == 'C':
            if td == 'X':
                C1 = np.degrees(np.arccos(Kx))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                A_base = np.degrees(np.arctan2(Kz, Ky))
                A = sel(A_base, 'A', prev_angles[0])
                return A, C
            elif td == 'Y':
                C1 = np.degrees(np.arccos(-Kx))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                A_base = np.degrees(np.arctan2(Kz, Ky))
                A = sel(A_base, 'A', prev_angles[0])
                return A, C
            elif td == 'Z':
                raise ValueError("Hybrid AC with tool_direction='Z' may be infeasible.")

        # BA 型 (first=table B, second=spindle A)
        if axis1 == 'B' and axis2 == 'A':
            if td == 'Y':
                A1 = np.degrees(np.arccos(Ky))
                A2 = -A1
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                B_base = np.degrees(np.arctan2(Kx, -Kz))
                B = sel(B_base, 'B', prev_angles[0])
                return B, A
            elif td == 'Z':
                A1 = np.degrees(np.arccos(-Ky))
                A2 = -A1
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                B_base = np.degrees(np.arctan2(Kx, Kz))
                B = sel(B_base, 'B', prev_angles[0])
                return B, A

        # BC 型 (B table, C spindle)
        if axis1 == 'B' and axis2 == 'C':
            if td == 'X':
                C1 = np.degrees(np.arccos(Ky))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                B_base = np.degrees(np.arctan2(-Kz, Kx))
                B = sel(B_base, 'B', prev_angles[0])
                return B, C
            elif td == 'Y':
                C1 = np.degrees(np.arccos(Ky))
                C2 = -C1
                C = self._select_optimal_angle([C1, C2], prev_angles[1], 'C')
                B_base = np.degrees(np.arctan2(Kz, -Kx))
                B = sel(B_base, 'B', prev_angles[0])
                return B, C

        # CA 型 (C table, A spindle)
        if axis1 == 'C' and axis2 == 'A':
            if td == 'Y':
                A1 = np.degrees(np.arccos(Kz))
                A2 = -A1
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                C_base = np.degrees(np.arctan2(-Kx, Ky))
                C = sel(C_base, 'C', prev_angles[0])
                return C, A
            elif td == 'Z':
                A1 = np.degrees(np.arccos(Kz))
                A2 = -A1
                A = self._select_optimal_angle([A1, A2], prev_angles[1], 'A')
                C_base = np.degrees(np.arctan2(Kx, Ky))
                C = sel(C_base, 'C', prev_angles[0])
                return C, A

        # CB 型 (C table, B spindle)
        if axis1 == 'C' and axis2 == 'B':
            if td == 'X':
                B1 = np.degrees(np.arccos(-Kz))
                B2 = -B1
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                C_base = np.degrees(np.arctan2(Ky, Kx))
                C = sel(C_base, 'C', prev_angles[0])
                return C, B
            elif td == 'Z':
                B1 = np.degrees(np.arccos(Kz))
                B2 = -B1
                B = self._select_optimal_angle([B1, B2], prev_angles[1], 'B')
                C_base = np.degrees(np.arctan2(Ky, Kx))
                C = sel(C_base, 'C', prev_angles[0])
                return C, B

        raise ValueError(f"Unsupported TABLE_SPINDLE_TILTING config {axis1}{axis2} with tool dir {td}")

    def inverse_kinematics(self,
                           tool_position: np.ndarray,
                           tool_orientation: np.ndarray,
                           prev_angles: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        """
        完整的五轴机床逆解计算
        """
        # 验证刀具方向是否已归一化（容差1e-6）
        norm = np.linalg.norm(tool_orientation)
        if not np.isclose(norm, 1.0, atol=1e-3):
            tool_orientation = tool_orientation / norm  # 归一化

        # 获取旋转轴名称
        first_axis = self.config.rotary_axes[0].value
        second_axis = self.config.rotary_axes[1].value

        # 计算所有可能的旋转角度组合
        angle_solutions = self._calculate_rotation_angles(tool_orientation, prev_angles)

        # 计算对应的线性轴位置并验证行程
        valid_solutions = []
        for angles in angle_solutions:
            linear_pos = self._calculate_linear_position(tool_position, angles)
            solution = {
                'X': linear_pos['X'],
                'Y': linear_pos['Y'],
                'Z': linear_pos['Z'],
                first_axis: angles[0],
                second_axis: angles[1]
            }

            # 检查行程限制
            # if self._check_limits(solution):
            valid_solutions.append(solution)

        if not valid_solutions:
            raise ValueError("无法找到满足限制的逆解")

        # 选择最优解（最接近前一个位置）
        best_solution = self._select_best_solution(valid_solutions, prev_angles)

        return best_solution

    def _calculate_rotation_angles(self,
                                   tool_orientation: np.ndarray,
                                   prev_angles: Optional[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        计算所有可能的旋转角度组合
        - 若无轴偏转，尝试使用解析法
        - 若有轴偏转，或解析失败，则使用数值优化法求解
        返回: List of (angle1, angle2)
        """
        # 工具方向可直接用于数值法，无需变换
        tool_dir_local = tool_orientation

        if self.config.primary_deflection is None and self.config.secondary_deflection is None:
            # 优先使用解析法
            try:
                if self.config.machine_type == MachineType.TABLE_TILTING:
                    base_angles = self._solve_table_tilting_angles(tool_dir_local, prev_angles)
                elif self.config.machine_type == MachineType.SPINDLE_TILTING:
                    base_angles = self._solve_spindle_tilting_angles(tool_dir_local, prev_angles)
                else:
                    base_angles = self._solve_hybrid_angles(tool_dir_local, prev_angles)

                return [base_angles]
            except Exception as e:
                # 回退使用数值优化法
                print(f"解析解失败，使用数值解: {e}")

        # 有偏转或解析失败的通用处理
        numeric_angles = solve_rotary_angles_numeric(tool_dir_local, self, prev_angles or (0.0, 0.0))
        return [numeric_angles]

    def _calculate_linear_position(self,
                                   tool_position: np.ndarray,
                                   angles: Tuple[float, float]) -> Dict[str, float]:
        """
        计算线性轴位置（按 forward_kinematics 的矩阵顺序推导）
        给定刀具位置 tool_position（在工件/程序坐标系下，3-vector）
        和旋转角 (a, c)（度），直接求解程序坐标下的线性平移 (X,Y,Z)。

        原理：
            Q = Trans(p) * M_fixed * [0,0,0,1]^T
            则 p = Q - (M_fixed[:,3])[:3]
        """
        angle_fst, angle_sec = angles  # 单位：度

        # 构建旋转矩阵
        R1 = self._build_rotation_matrix(self.primary_axis, angle_fst)
        R2 = self._build_rotation_matrix(self.secondary_axis, angle_sec)

        # M_fixed 应该包括 spindle_swing_matrix
        M_fixed = (
                self.T_base_to_primary @
                R1 @
                self.T_primary_to_secondary @
                R2 @
                self.spindle_swing_matrix @  # 缺失部分
                self.T_secondary_to_tool
        )

        # 从刀具目标位置中减去固定部分的平移量
        s_translation = M_fixed[:3, 3]
        tool_position = np.asarray(tool_position, dtype=float).reshape(3)
        linear_pos = tool_position - s_translation

        return {
            'X': float(linear_pos[0]),
            'Y': float(linear_pos[1]),
            'Z': float(linear_pos[2])
        }

    def _check_limits(self, joint_values: Dict[str, float]) -> bool:
        """检查各轴是否在限制范围内"""
        # 检查线性轴
        for axis in ['X', 'Y', 'Z']:
            if axis in self.config.linear_limits:
                min_val, max_val = self.config.linear_limits[axis]
                if not (min_val <= joint_values[axis] <= max_val):
                    return False

        # 检查旋转轴
        first_axis = self.config.rotary_axes[0].value
        second_axis = self.config.rotary_axes[1].value

        if first_axis in self.config.rotary_limits:
            min_val, max_val = self.config.rotary_limits[first_axis]
            angle = joint_values[first_axis]
            # 处理角度环绕
            normalized_angle = angle % 360
            if normalized_angle > 180:
                normalized_angle -= 360
            if not (min_val <= normalized_angle <= max_val):
                return False

        if second_axis in self.config.rotary_limits:
            min_val, max_val = self.config.rotary_limits[second_axis]
            angle = joint_values[second_axis]
            normalized_angle = angle % 360
            if normalized_angle > 180:
                normalized_angle -= 360
            if not (min_val <= normalized_angle <= max_val):
                return False

        return True

    def _select_best_solution(self,
                              solutions: List[Dict[str, float]],
                              prev_angles: Optional[Tuple[float, float]]) -> Dict[str, float]:
        """从多个解中选择最优解"""
        if prev_angles is None or not solutions:
            return solutions[0]

        # 计算与前一个位置的距离（优先考虑旋转轴连续性）
        def distance(solution):
            rot_dist = 0
            first_axis = self.config.rotary_axes[0].value
            second_axis = self.config.rotary_axes[1].value

            # 旋转轴距离（考虑角度环绕）
            a_diff = abs(solution[first_axis] - prev_angles[0])
            a_diff = min(a_diff, 360 - a_diff)

            c_diff = abs(solution[second_axis] - prev_angles[1])
            c_diff = min(c_diff, 360 - c_diff)

            return a_diff + c_diff

        return min(solutions, key=distance)

if __name__ == '__main__':
    # 创建配置
    config = MachineConfig()
    config.machine_type = MachineType.TABLE_SPINDLE_TILTING
    config.rotary_axes = (RotaryAxis.A, RotaryAxis.B)
    config.primary_rotary_center = np.array([455, 0, 650])
    config.secondary_rotary_offset = np.array([0, 429.813, 457.164])
    config.secondary_rotary_offset = np.array([-455, 250, -850])
    config.secondary_deflection = np.array([0, 0.7660, 0.643])
    config.spindle_swing_offset = np.array([0, -250, 200])
    config.tool_length = 0
    config.linear_limits = {'X':(-245,455), 'Y':(-200,200), 'Z':(250,650)}
    # config.linear_limits = {'X': (-245, 455), 'Y': (-500, 500), 'Z': (0, 650)}
    config.rotary_limits = {'A': (0, 360), 'B': (-87, 87)}
    # config.rotary_limits = {'A': (0, 360), 'B': (0, 360)}

    # 创建求解器
    solver = KinematicSolver(config)

    print(solver.forward_kinematics({'X':0, 'Y':0, 'Z':0, 'A':0, 'B':0}))

    # X = 114.98085
    # Y = - 258.54103
    # Z = 9.94968
    # A3 = .245642045
    # B3 = -.943509144
    # C3 = .222374640
    X = 143.74231
    Y = - 367.55630
    Z = 26.95157
    A3 = .007423059
    B3 = -.993936722
    C3 = .109702739

    # 测试目标点与姿态
    target_position = np.array([X, Y, Z])  # 目标刀尖坐标（程序坐标系下）
    target_orientation = np.array([A3, B3, C3])  # 刀具方向（例如向 ZY 方向倾斜）

    # 逆向解算
    ik_result = solver.inverse_kinematics(
        tool_position=target_position,
        tool_orientation=target_orientation,
        prev_angles=(0.0, 0.0)  # 初始角度（可选）
    )
    print("逆向解算结果 (Inverse Kinematics):", ik_result)

    # 正向解算验证
    fk_pos, fk_dir = solver.forward_kinematics(ik_result)
    print("正向解算结果位置:", fk_pos)
    print("正向解算结果方向:", fk_dir)