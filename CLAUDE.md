# StaMPS Matlab → Python 翻译项目

## 项目概述
将 StaMPS (Stanford Method for Persistent Scatterers) InSAR 处理软件从 Matlab 翻译为 Python。用户为项目负责人，Claude 负责验收和审查翻译后的 Python 代码。

## 目录结构
- `stamps_matlab/` — Matlab 原始源码，**只读参考，禁止修改**
- `stamps_python/` — Python 翻译代码，主要工作区
- `test_data/` — 测试数据（含 .mat 和 .h5 对照文件）

## 处理流程（8步 + 导出）
| Step | 描述 | Python 文件 | Matlab 对应 |
|------|------|-------------|-------------|
| 1 | 数据加载 | data_loader.py | ps_load_initial_isce.m |
| 2 | Gamma 估算 | gamma_est.py | ps_est_gamma_quick.m |
| 3 | PS 选择 | ps_selection_final.py | ps_select.m |
| 4 | 剔除噪声点 | ps_weeding.py | ps_weed.m |
| 5 | 相位校正+合并 | ps_correct_phase.py, ps_merge_patches.py | ps_correct_phase.m, ps_merge_patches.m |
| 6 | 相位解缠 | phase_unwrapping.py, uw_core.py | ps_unwrap.m, uw_3d.m |
| 7 | SCLA 估算 | scla_estimation.py | ps_calc_scla.m, ps_smooth_scla.m |
| 8 | SCN 滤波 | scn_filt.py | ps_scn_filt.m |
| 导出 | 结果导出 | export_results.py | ps_plot.m |

## 入口和运行方式
```bash
# 运行指定步骤
D:\env\python311\python.exe stamps_main.py --start N --end N --config d:/coding/Stamps_Refactor_Project/test_data

# 导出结果
D:\env\python311\python.exe export_results.py --input_path <h5目录> --output_dir <输出路径> --mode ps
```

## 验收标准
代码验收时需检查以下方面：

### 1. 算法正确性
- **必须对标 Matlab 源码**：每个步骤的 Python 实现逻辑需与 `stamps_matlab/matlab/` 下的对应 .m 文件一致
- 数值结果应与 Matlab `.mat` 文件对照验证（test_data 中同时有 .mat 和 .h5）
- 特别注意：Matlab 1-based 索引 vs Python 0-based 索引、Matlab datenum vs Python ordinal 日期转换

### 2. 数据 I/O
- 使用 HDF5 (.h5) 替代 Matlab .mat 文件
- 复数相位数据（complex64）需正确存储/读取
- 大型数组使用 gzip 压缩
- 文件命名约定：`ps{ver}.h5`, `pm{ver}.h5`, `rc{ver}.h5`, `bp{ver}.h5`, `phuw{ver}.h5`, `scla{ver}.h5`, `scn{ver}.h5` 等

### 3. 性能要求
- **必须使用 numpy 矢量化**，禁止在数百万像素上使用 Python 循环
- 空间搜索使用 scipy.spatial.cKDTree 等高效结构
- 矩阵求解使用 numpy.linalg.lstsq 或 scipy 线性代数工具
- 滤波使用 scipy.signal 或 scipy.ndimage

### 4. 架构约定
- `stamps_main.py` 的 `StampsRunner` 类统一调度各步骤
- 每个步骤有独立的 Python 类（如 GammaEstimator, PSSelector, PSWeeder 等）
- 配置参数通过 `StampsConfig`（getparm.py）单例获取
- 使用 Python logging（`logging.getLogger("stamps")`）记录关键信息
- 每步需记录处理前后的点数变化

### 5. 代码质量
- 类型注解：使用 `from __future__ import annotations` + typing
- 文档：类和方法需有 docstring，说明 Matlab 对应关系
- 错误处理：文件不存在、数据维度不匹配等边界情况
- 无 GUI 依赖，纯数据处理

## 技术栈
- Python 3.11（路径：`D:\env\python311\python.exe`）
- numpy, scipy, h5py, geopandas
- 外部工具：snaphu v2.0.7（相位解缠），triangle（Delaunay 三角剖分）

## 注意事项
- 仅实现 ISCE 处理器 + PS 模式（非 SBAS）
- `stamps_matlab/` 目录为只读参考，禁止任何修改
- `test_data/` 中 `.mat` 文件为 Matlab 运行结果，用于与 Python `.h5` 输出对照
- Windows 环境，路径使用正斜杠或 Path 对象
