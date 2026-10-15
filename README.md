# XRI：薄膜标定与三维散射强度联合反演

输入薄膜 NeXus、样品 NeXus 和晶体 ROI 表，输出弹性峰积分后的三维数组 **I(X,Y,Z)**。代码自成一套，不导入上一级 `xrsabre`，也不依赖原来的处理脚本、缓存或 `reconstruction.py`。

**保留 TV 正则化。** 不包含重建后的空间裁剪、强度阈值分割、对称性清理、底座去除、插值加密、网格平滑或曲面导出。ROI 提取和标定质量检查属于测量处理；TV 属于反演目标函数。它们的参数与几何后处理分开说明。

## 运行

在当前项目根目录的 PowerShell 中：

```powershell
# 只检查文件、坐标、帧数、ROI 和能量范围，不生成结果
.venv\Scripts\python.exe XRI/run_xri.py --config XRI/config.15469.json --validate-only

# 从两个原始 NeXus 文件开始完整处理
.venv\Scripts\python.exe XRI/run_xri.py --config XRI/config.15469.json

# 独立测试，不需要实验数据
.venv\Scripts\python.exe -m unittest discover -s XRI/tests -v
```

移到另一台机器时，安装 Python 3.10 或以上版本，在该目录执行 `python -m pip install -r requirements.txt`，修改 JSON 内三个输入路径和输出目录，然后执行：

```powershell
python run_xri.py --config config.15469.json
```

所有相对路径以 **JSON 配置所在目录** 为基准，与启动命令时的工作目录无关。程序只读原始数据。输出目录必须不存在；重新运行应指定一个新的 `runtime.output_dir`，避免混用旧缓存。失败时会保留已生成的中间文件和日志供检查。

## 当前数据配置

| 项目 | 已确认的设置 |
|---|---|
| 薄膜扫描 | `15341_grid_scan_20261013220247_1.nxs` |
| 成像扫描 | `15469_grid_scan_20261014231918_1.nxs` |
| ROI | 本目录 `examples/DT_Be_ROI.txt` 的快照 |
| 曝光时间 | 薄膜、样品均为 0.5 s |
| 入射监测 | `D_SiC_Current1` |
| 探测器 | `D_LAMBDA`，55 μm 物理像素间距 |
| 薄膜 | Kapton，厚度 0.05 mm |
| ROI 修正 | 原编号 35 → VU-A3；31、44 明确排除 |
| 薄膜轴 | X：−5～5 mm，21 点；能量：9.680～9.694 keV，72 点 |
| 样品轴 | Y：−0.96～0.19 mm，21 点；Z：−1.62～−0.54 mm，21 点 |
| 样品能量 | 9.680～9.694 keV，16 点，梯形积分 |

坐标和点数从文件的扫描计划读取，再用逐帧 setpoint 核对。样品实际 Y 步长为 **57.5 μm**，Z 步长为 **54 μm**，能量步长约 **0.9333 eV**。程序不会用口头估计的步长重建坐标。

**55 μm 是探测器像素间距。** 各晶体通过薄膜位移与光斑位移标定到样品 X 坐标；默认使用有效晶体的“每探测器像素对应样品长度”的中位数作 X 网格间距。21 个薄膜位置是标定观测数，不是恢复深度数。当前数据实测运行的 X 间距约 64.09 μm，完整网格为 157×21×21。网格间距不等于测得的空间分辨率。

## 处理顺序与结果

```text
薄膜 NXS + ROI → 像素计数/剂量缓存 ─┐
                                  ├→ 匹配能量积分 → 连续移动 PSF 标定
样品 NXS → 按 setpoint 建立 YZE 网格 ┘
       → 构造各晶体 A[p,X] → 联合 NNLS / 晶体相对增益
       → 固定增益的非负 TV 反演 → I(X,Y,Z) + 轴 + 残差 + 可观测性诊断
```

默认输出在 `XRI/results/15469/`：

- `reconstruction.h5`：主结果。TV 收敛时采用 TV；未收敛时采用 NNLS 并明确记录。
- `reconstruction_nnls.h5`：非负最小二乘基线。
- `reconstruction_tv.h5`：启用 TV 时保存其迭代结果，必须同时检查收敛标志。
- `summary.json`、`config.resolved.json`、`provenance.json`：参数、拟合质量、数值状态、输入文件与软件版本。
- `calibration.h5`、`tomography_inputs.h5`：保留像素级标定、样品积分及计数方差。
- `psf_models.json`、`solver_diagnostics.h5`、`diagnostics.png`、`run.log`：模型、残差、灵敏度及运行记录。

主数组保留参考文件的数据集名 `reinterp_insect_ck` 和 float32 类型，轴明确为 **X、Y、Z**。强度是数组的值，不是第四维；不补造参考文件中的 200 点，也不强制复制参考文件尺寸。

```python
import h5py
with h5py.File("results/15469/reconstruction.h5", "r") as f:
    intensity_xyz = f["reinterp_insect_ck"][:]
    x_mm, y_mm, z_mm = (f[name][:] for name in ("x_mm", "y_mm", "z_mm"))
    sensitivity_x = f["sensitivity_x"][:]
# intensity_xyz[ix, iy, iz] 对应 (x_mm[ix], y_mm[iy], z_mm[iz])
```

完整定义见 [输入输出参数.md](输入输出参数.md) 和 [处理逻辑.md](处理逻辑.md)。实验验证记录见 [验证记录.md](验证记录.md)。

本次从原始文件开始的实测运行已完成：增益拟合、NNLS 与 TV 均满足设定的数值收敛条件，TV 在 39586 次迭代后收敛，主结果为 TV。NNLS/TV 相对 L2 残差分别约 42.04% / 42.10%，表明当前响应模型和背景仍有明显未解释的成分，不能将数值收敛当作实物重建准确度。

## 使用边界

该实现针对当前 HEPS `P.grid_scan` 数据结构，要求完整薄膜网格和完整样品 YZE 网格；不完整扫描会报错，不以零填充缺测。支持按 setpoint 重排帧和合并重复测量。

输出是相对 Kapton 等效弹性散射密度，尚未扣除样品背景或校正吸收、组成差异。默认保存完整 X 范围；其中响应很弱的区域不具备可靠定量能力，须结合 `sensitivity_x` 和响应矩阵秩解释。数值为零不能据此判定该处没有材料。TV 和连续 PSF 模型也不能替代额外的测量信息。
