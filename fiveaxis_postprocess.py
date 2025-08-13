import re
import os
import numpy as np
from fiveaxis_kins import KinematicSolver, MachineConfig, MachineType, RotaryAxis

class GCodeParser:
    """
    G代码解析器类
    用于解析和转换G代码文件
    功能1：
        实现ABC(G43.4 欧拉角) 和 IJK(G43.5 方向余弦)的互相变换
    功能2：
        LinuxCNC后处理,根据 G43.4/G43.5结果计算 ABC转轴数据
    """
    
    def __init__(self, file_path=None, kinematics_solver: KinematicSolver = None):
        """
        初始化解析器

        参数:
        file_path: G代码文件路径（可选）
        """
        self.file_path = file_path
        self.original_lines = []
        self.parsed_lines = []
        self.start_line = None
        self.end_line = None


        self.pre_a = 0
        self.pre_b = 0

        self.slover = kinematics_solver

        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """
        加载G代码文件
        
        参数:
        file_path: 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.original_lines = file.readlines()
            self.file_path = file_path
            print(f"成功加载文件: {file_path}")
            print(f"总行数: {len(self.original_lines)}")
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
        except Exception as e:
            print(f"错误: 加载文件时出现问题 - {e}")
    
    def load_from_string(self, content):
        """
        从字符串加载G代码内容
        
        参数:
        content: G代码字符串内容
        """
        self.original_lines = content.strip().split('\n')
        print(f"从字符串加载内容，总行数: {len(self.original_lines)}")
    
    def set_range(self, start_line=None, end_line=None):
        """
        设置解析范围
        
        参数:
        start_line: 起始行号（从1开始，None表示从第一行开始）
        end_line: 结束行号（从1开始，None表示到最后一行）
        """
        self.start_line = start_line
        self.end_line = end_line
        
        if start_line is not None:
            print(f"设置起始行: {start_line}")
        if end_line is not None:
            print(f"设置结束行: {end_line}")
    
    def parse(self) -> int:
        """
        解析G代码
        转换规则:
        1. 在行首遇到了; 就转换为（
        2. 如果在行首遇到NXX就把NXX去除
        """
        self.pre_a = 0
        self.pre_b = 0
        if len(self.original_lines) == 0:
            print("错误: 没有可解析的内容")
            return 1
        
        # 确定解析范围
        start_idx = (self.start_line - 1) if self.start_line is not None else 0
        end_idx = self.end_line if self.end_line is not None else len(self.original_lines)
        
        # 确保索引有效
        start_idx = max(0, start_idx)
        end_idx = min(len(self.original_lines), end_idx)

        # 重置解析行
        self.parsed_lines = []

        lines_num = end_idx - start_idx
        delta_line = (end_idx - start_idx) / 100
        percent = 0

        # 开始解析
        for i in range(start_idx, end_idx):
            line = self.original_lines[i].strip()
            
            # 计算进度百分比
            if delta_line > 0:
                current_percent = int((i - start_idx) / delta_line)
                if current_percent > percent:
                    percent = current_percent
                    print(f"解析进度: {percent}% ({i - start_idx + 1}/{lines_num})")
            
            if not line:  # 空行保持不变
                continue

            if line[0] == ';':  # 如果行首是分号，跳过
                continue
            
            # 应用转换规则
            converted_line = self._apply_conversion_rules(line)
            if line:
                self.parsed_lines.append(converted_line)
        
        print(f"解析完成，处理了 {len(self.parsed_lines)} 行")
        return self.parsed_lines

    def _ijk_to_abc(self, x, y, z, i, j, k):
        """
        统一的 IJK → ABC 转换接口，根据机床构型自动选择解算方式

        参数：
            x, y, z (float): 刀具位置坐标（用于支持混合/偏心结构的解）
            i, j, k (float): 刀具方向向量
            pre_a, pre_b (float): 上一次的旋转角（用于连续性）

        返回：
            A, B, C: 三个角度中任意两个（或三个），按机床配置返回
        """
        tool_orientation = np.array([i, j, k], dtype=float)
        tool_orientation /= np.linalg.norm(tool_orientation)

        tool_position = np.array([x, y, z], dtype=float)

        # 调用逆解求解两个旋转角（一般是 A 和 C 或 A 和 B）
        ik_result = self.slover.inverse_kinematics(
            tool_position=tool_position,
            tool_orientation=tool_orientation,
            prev_angles=(self.pre_a, self.pre_b)
        )

        # 提取当前配置的旋转轴（例：['A', 'C']）
        rotary_axes = self.slover.config.rotary_axes

        # 初始化为空
        A = B = C = None

        # 根据轴名称映射对应角度
        for axis in rotary_axes:
            angle = ik_result.get(axis.value)
            if axis == RotaryAxis.A and angle is not None:
                A = round(angle, 6)
            elif axis == RotaryAxis.B and angle is not None:
                B = round(angle, 6)
            elif axis == RotaryAxis.C and angle is not None:
                C = round(angle, 6)

        # 更新预置角度（用于连续性处理）
        if len(rotary_axes) >= 2:
            self.pre_a = ik_result[rotary_axes[0].value]
            self.pre_b = ik_result[rotary_axes[1].value]

        return A, B, C

    def convert_line(self, line):
        """
        转换单行程序：提取XYZIJK并替换为XYZAB
        """
        
        result = {}
        for key in 'GXYZIJKF':
            m = re.search(rf'{key}([-+]?\d*\.?\d+)', line)
            result[key] = m.group(1) if m else None
        
        g_match = re.search(r'G(\d+)', line)
        g = g_match.group(1) if g_match else None


        x = result['X']
        y = result['Y']
        z = result['Z']
        i = result['I']
        j = result['J']
        k = result['K']
        f = result['F']
        
        a = None
        b = None
        c = None
        if i and j and k:
            i, j, k = float(i), float(j), float(k)
            a, b, c = self._ijk_to_abc(x, y, z, i, j, k)

        new_line = ""
        if g is not None:
            new_line += f"G{g}"
        if x is not None:
            new_line += f"X{x}"
        if y is not None:
            new_line += f"Y{y}"
        if z is not None:
            new_line += f"Z{z}"
        if a is not None:
            new_line += f"A{a:.6f}"
        if b is not None:
            new_line += f"B{b:.6f}"
        if c is not None:
            new_line += f"C{c:.6f}"
        if f is not None:
            new_line += f"F{f}"

        return new_line
    
    def _apply_conversion_rules(self, line):
        """
        应用转换规则
        
        参数:
        line: 待转换的行
        
        返回:
        转换后的行
        """
        # 规则1：跳过注释
        if line.startswith(';'):
            line = ""
        elif line.startswith('('):
            line = ""

        if line == "":
            return line
        
        # 规则2：如果在行首遇到NXX就把NXX去除
        # 使用正则表达式匹配N后跟数字的模式
        pattern = r'^N\d+'
        if re.match(pattern, line):
            line = re.sub(pattern, '', line)

        abc_num_pattern = r'([ABC])(-?)\d+'  # 把负号单独分组
        line = re.sub(abc_num_pattern, r'\1', line)  # 保留字母和负号
        
        #A=-.088284415B=-.990423696C=-.106145012
        # 规则3：A=-.005 转换为 A-0.005，对A、B、C都适用
        # 匹配模式：[ABC]=-. 转换为 [ABC]-0.（去掉等号，负号后面加0）
        # abc_pattern = r'([ABC])(=)(-)\.\d+)'
        # line = re.sub(abc_pattern, r'\1\3\'0', line)
        line = re.sub(r'=-\.', r'-0.', line)  # 替换 "=-." 为 "-0."
        line = re.sub(r'=\.', r'0.', line)  # 替换 "=-." 为 "-0."
        
        # # 处理其他负数情况 A=-123.456 转换为 A-123.456
        # abc_negative_pattern = r'([ABC])=(-\d+\.?\d*)'
        # line = re.sub(abc_negative_pattern, r'\1\2', line)
        line = re.sub('A', 'I', line)
        line = re.sub('B', 'J', line)
        line = re.sub('C', 'K', line)

        line = self.convert_line(line)  # 调用转换函数处理IJK到AB的转换

        return line
    
    def get_parsed_content(self):
        """
        获取解析后的内容
        
        返回:
        解析后的行列表
        """
        return self.parsed_lines
    
    def preview(self, num_lines=10):
        """
        预览解析结果
        
        参数:
        num_lines: 预览的行数
        """
        if not self.parsed_lines:
            print("请先执行解析操作")
            return
        
        print(f"\n解析结果预览（前{num_lines}行）:")
        print("-" * 50)
        for i, line in enumerate(self.parsed_lines[:num_lines]):
            print(f"{i+1:3d}: {line}")
        
        if len(self.parsed_lines) > num_lines:
            print(f"... 还有 {len(self.parsed_lines) - num_lines} 行")
    
    def save_to_file(self, output_file_name=None):
        """
        保存解析结果到文件
        
        参数:
        output_file_name: 输出文件名（可选）
        """
        if not self.parsed_lines:
            print("错误: 没有可保存的内容，请先执行解析操作")
            return
        
        # 确定输出文件名
        if output_file_name is None:
            if self.file_path:
                # 基于原文件名生成新文件名
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                output_file_name = f"{base_name}_converted.txt"
            else:
                output_file_name = "gcode_converted.txt"
        
        try:
            with open(output_file_name, 'w', encoding='utf-8') as file:
                for line in self.parsed_lines:
                    file.write(line + '\n')
            
            print(f"文件已保存到: {output_file_name}")
            print(f"保存了 {len(self.parsed_lines)} 行")
            
        except Exception as e:
            print(f"错误: 保存文件时出现问题 - {e}")
    
    def get_statistics(self):
        """
        获取解析统计信息
        """
        if not self.original_lines:
            print("没有加载任何内容")
            return
        
        print(f"\n统计信息:")
        print(f"原始总行数: {len(self.original_lines)}")
        print(f"解析行数: {len(self.parsed_lines) if self.parsed_lines else 0}")
        
        if self.start_line or self.end_line:
            start = self.start_line or 1
            end = self.end_line or len(self.original_lines)
            print(f"解析范围: 第{start}行 到 第{end}行")


# 使用示例
if __name__ == "__main__":
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
    config.linear_limits = {'X': (-245, 455), 'Y': (-200, 200), 'Z': (250, 650)}
    # config.linear_limits = {'X': (-245, 455), 'Y': (-500, 500), 'Z': (0, 650)}
    config.rotary_limits = {'A': (0, 360), 'B': (-87, 87)}
    # config.rotary_limits = {'A': (0, 360), 'B': (0, 360)}

    # 创建求解器
    solver = KinematicSolver(config)
    default_filename = "007020.SPF"
    parser = GCodeParser(default_filename, solver)

    # parser.load_from_string("""
    #     N7390X136.25668Y-225.16376Z-41.57356A3=.577B3=.5777C3=.5777
    #     N7391X136.62755Y-225.14299Z-41.68616A3=.086927128B3=-.990254387C3=-.108811415
    # """)
    parser.parse()
    # print(parser.get_parsed_content())
    parser.save_to_file("007020_AB.nc")