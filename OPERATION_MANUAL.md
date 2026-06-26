# StaMPS Python 操作手册

> StaMPS (Stanford Method for Persistent Scatterers) InSAR 处理软件 Python 版本操作手册

---

## 目录

1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [处理流程](#3-处理流程)
   - [Step 1: 数据加载](#step-1-数据加载)
   - [Step 2: Gamma 估算](#step-2-gamma-估算)
   - [Step 3: PS 选择](#step-3-ps-选择)
   - [Step 4: 剔除噪声点](#step-4-剔除噪声点)
   - [Step 5: 相位校正与合并](#step-5-相位校正与合并)
   - [Step 6: 相位解缠](#step-6-相位解缠)
   - [Step 7: SCLA 估算](#step-7-scla-估算)
   - [Step 8: SCN 滤波](#step-8-scn-滤波)
4. [结果导出](#4-结果导出)
5. [配置参数说明](#5-配置参数说明)
6. [常见问题](#6-常见问题)

---

## 1. 环境配置

### 1.1 Python 环境

```
Python 版本: 3.11
解释器路径: D:\env\python311\python.exe
```

### 1.2 依赖库

```bash
pip install numpy scipy h5py geopandas gdal
```

### 1.3 外部工具

| 工具 | 版本 | 用途 | 配置方式 |
|------|------|------|----------|
| Snaphu | v2.0.7 | 相位解缠 | 设置 `snaphu_path` 环境变量或在配置中指定路径 |
| Triangle | - | Delaunay 三角剖分 | Windows 下使用 scipy 内置实现 |

### 1.4 目录结构

```
project_root/
├── stamps_matlab/          # Matlab 源码（只读参考）
├── stamps_python/          # Python 源码
│   ├── stamps_main.py      # 主入口
│   ├── data_loader.py      # Step 1
│   ├── gamma_est.py        # Step 2
│   ├── ps_selection_final.py  # Step 3
│   ├── ps_weeding.py       # Step 4
│   ├── ps_correct_phase.py # Step 5a
│   ├── ps_merge_patches.py # Step 5b
│   ├── phase_unwrapping.py # Step 6
│   ├── uw_core.py          # 解缠核心算法
│   ├── scla_estimation.py  # Step 7
│   ├── scn_filt.py         # Step 8
│   └── export_results.py   # 结果导出
├── test_data/              # 测试数据目录
│   ├── parms.mat           # 配置参数
│   ├── patch.list          # Patch 列表
│   ├── PATCH_437/          # Patch 数据
│   ├── PATCH_438/          # Patch 数据
│   └── *.raw, *.in, ...    # ISCE 原始数据
```

---

## 2. 数据准备

### 2.1 ISCE 预处理输出文件

处理前需确保以下 ISCE 输出文件存在：

| 文件 | 说明 | 格式 |
|------|------|------|
| `pscands.1.ij` | PS 候选点坐标 (ID, Az, Rg) | 文本，每行 3 列 |
| `pscands.1.ph` | 复数相位数据 | 二进制，float32 交错 |
| `pscands.1.ll` | 经纬度坐标 | 二进制，float32 × 2 |
| `pscands.1.da` | 振幅离差指数 | 二进制，float32 |
| `pscands.1.hgt` | 高程数据 | 二进制，float32 |
| `bperp.1.in` | 垂直基线 | 文本，每景一个值 |
| `day.1.in` | 获取日期 | 文本，YYYYMMDD 格式 |
| `master_day.1.in` | 参考影像日期（legacy-compatible 字段名） | 文本，YYYYMMDD 格式 |
| `heading.1.in` | 卫星航向角 | 文本，单值 |
| `lambda.1.in` | 雷达波长 | 文本，单值 |
| `width.txt` | 影像宽度 | 文本 |
| `len.txt` | 影像长度 | 文本 |
| `calamp.out` | 振幅定标系数 | 文本 |
| `inc_angle.raw` 或 `look_angle.1.in` | 入射角/视角 | 二进制/文本 |

### 2.2 配置文件

**parms.mat** 必须存在于数据目录根目录，包含处理参数。首次运行时会自动创建默认参数。

### 2.3 Patch 配置

**patch.list** 文件指定要处理的 Patch 目录：
```
PATCH_437
PATCH_438
```

---

## 3. 处理流程

### 运行命令

```bash
# 运行指定步骤
D:\env\python311\python.exe stamps_main.py --start <N> --end <N> --config <数据目录>

# 运行多个步骤
D:\env\python311\python.exe stamps_main.py --start 1 --end 8 --config D:/coding/Stamps_Refactor_Project/test_data

# 自动从上次完成步骤继续
D:\env\python311\python.exe stamps_main.py --start 0 --end 8 --config D:/coding/Stamps_Refactor_Project/test_data
```

### Step 编号对照

| Step | 功能 | Python 模块 | Matlab 对应 |
|------|------|-------------|-------------|
| 0 | 从上次完成步骤继续 | - | - |
| 1 | 数据加载 | data_loader.py | ps_load_initial_isce.m |
| 2 | Gamma 估算 | gamma_est.py | ps_est_gamma_quick.m |
| 3 | PS 选择 | ps_selection_final.py | ps_select.m |
| 4 | 剔除噪声点 | ps_weeding.py | ps_weed.m |
| 5 | 相位校正+合并 | ps_correct_phase.py, ps_merge_patches.py | ps_correct_phase.m, ps_merge_patches.m |
| 6 | 相位解缠 | phase_unwrapping.py, uw_core.py | ps_unwrap.m, uw_3d.m |
| 7 | SCLA 估算 | scla_estimation.py | ps_calc_scla.m, ps_smooth_scla.m |
| 8 | SCN 滤波 | scn_filt.py | ps_scn_filt.m |

---

### Step 1: 数据加载

**功能**: 从 ISCE 格式文件加载 PS 候选点数据，转换为 HDF5 格式

**运行命令**:
```bash
python stamps_main.py --start 1 --end 1 --config <数据目录>
```

**输入文件** (位于各 PATCH 目录或父目录):
| 文件 | 说明 |
|------|------|
| pscands.1.ij | PS 候选点坐标 |
| pscands.1.ph | 复数相位 |
| pscands.1.ll | 经纬度 |
| pscands.1.da | 振幅离差 |
| pscands.1.hgt | 高程 |
| bperp.1.in | 垂直基线 |
| day.1.in, master_day.1.in | 日期（`master_day` 为兼容字段名，含义是参考日期） |
| heading.1.in, lambda.1.in | 参数 |
| calamp.out | 定标系数 |
| width.txt, len.txt | 影像尺寸 |

**输出文件** (每个 PATCH 目录):
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| ps1.h5 | ph | (n_ps, n_ifg) complex64 | 相位矩阵 |
| | ij | (n_ps, 3) int32 | [ID, Az, Rg] |
| | lonlat | (n_ps, 2) float64 | 经纬度 |
| | xy | (n_ps, 3) float32 | [ID, x, y] 局部坐标 |
| | bperp | (n_ifg,) float64 | 垂直基线 |
| | bperp_mat | (n_ps, n_ifg-1) float32 | 逐点基线矩阵 |
| | day, master_day | int32 | 日期 (Matlab datenum；`master_day` 为兼容字段名，含义是参考日期) |
| | master_ix | int32 | 参考影像索引 (1-based；legacy-compatible 字段名) |
| | sort_ix | (n_ps,) int32 | 排序索引 |
| | D_A | (n_ps,) float64 | 振幅离差 |
| | hgt | (n_ps,) float32 | 高程 |
| no_ps_info.h5 | stamps_step_no_ps | (5,) int32 | 步骤状态标记 |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| insar_processor | isce | InSAR 处理器 |
| small_baseline_flag | n | 是否使用小基线模式 |

**注意事项**:
- 日期格式转换: Matlab datenum = Python ordinal + 366
- 索引格式: HDF5 中 `master_ix`（参考影像索引，legacy-compatible 字段名）、`sort_ix` 为 1-based，内部使用时转为 0-based
- 若存在 `baselineGRID_*` 目录，将加载逐像素基线网格

---

### Step 2: Gamma 估算

**功能**: 估算每个 PS 候选点的相干性 (gamma) 和 DEM 误差参数

**运行命令**:
```bash
python stamps_main.py --start 2 --end 2 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| ps1.h5 | Step 1 输出 |
| inc1.mat/h5 或 la1.mat/h5 | 入射角/视角 |

**输出文件**:
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| pm1.h5 | K_ps | (n_ps,) float64 | DEM 误差系数 |
| | C_ps | (n_ps,) float64 | 参考影像相位偏差 |
| | coh_ps | (n_ps,) float64 | 相干性 (gamma) |
| | ph_patch | (n_ps, n_ifg-1) complex64 | 空间滤波后相位 |
| | ph_res | (n_ps, n_ifg-1) float32 | 残余相位 |
| | low_pass | (32, 32) float64 | 低通滤波器 |
| | grid_ij | (n_ps, 2) float32 | 网格坐标 |
| | Nr, coh_bins | (100,) | 随机相位分布直方图 |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| filter_grid_size | 50 | 网格大小 (m) |
| filter_weighting | P-square | 权重策略 |
| clap_win | 32 | CLAP 窗口大小 |
| clap_alpha | 0.5 | CLAP 自适应参数 |
| clap_beta | 0.1 | CLAP 低通权重 |
| max_topo_err | 20 | 最大地形误差 (m) |
| gamma_change_convergence | 0.005 | 收敛阈值 |
| gamma_max_iterations | 3 | 最大迭代次数 |

**注意事项**:
- 迭代直到 gamma_change < gamma_change_convergence
- CLAP 滤波器在频域实现空间自适应滤波
- Nr 直方图由 Monte Carlo 随机模拟生成，每次运行略有不同

---

### Step 3: PS 选择

**功能**: 根据相干性阈值筛选最终 PS 点

**运行命令**:
```bash
python stamps_main.py --start 3 --end 3 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| pm1.h5 | Step 2 输出 |
| ps1.h5 | Step 1 输出 |

**输出文件**:
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| select1.h5 | ix | (n_selected,) uint16 | 选中的 PS 索引 |
| | keep_ix | (n_selected,) uint8 | 最终保留标记 |
| | coh_ps2 | (n_selected,) float64 | 重估后相干性 |
| | K_ps2, C_ps2 | (n_selected,) | 重估后参数 |
| | ph_patch2, ph_res2 | (n_selected, n_ifg-1) | 重估后相位 |
| | coh_thresh | float or (n_ps,) | 相干性阈值 |
| | ifg_index | (n_used_ifg,) int32 | 使用的干涉图索引 |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| select_method | PERCENT | 阈值计算方法 |
| max_percent_rand | 100 | 允许的随机相位百分比 |
| select_reest_gamma_flag | y | 是否重估 gamma |
| gamma_stdev_reject | 0 | 相干性标准差阈值 (bootstrap) |
| drop_ifg_index | [] | 排除的干涉图索引 |

**注意事项**:
- 阈值计算基于 D_A 分箱的直方图分析
- 重估时逐点从网格移除后重新滤波
- ifg_index 保存为 1-based 调整后索引 (范围 1~n_ifg-1)

---

### Step 4: 剔除噪声点

**功能**: 剔除邻近噪声点和空间异常点

**运行命令**:
```bash
python stamps_main.py --start 4 --end 4 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| ps1.h5 | Step 1 输出 |
| select1.h5 | Step 3 输出 |
| pm1.h5 | Step 2 输出 |

**输出文件**:
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| weed1.h5 | ix_weed | (n_selected,) uint8 | 保留标记 |
| | ps_std | (n_selected,) float32 | 噪声标准差 |
| | ps_max | (n_selected,) float32 | 噪声最大值 |
| ps2.h5 | ij, lonlat, xy, ... | (n_final,) | 剔除后的 PS 数据 |
| pm2.h5 | K_ps, C_ps, coh_ps, ... | (n_final,) | 剔除后的参数 |
| ph2.h5 | ph | (n_final, n_ifg) | 剔除后的相位 |
| bp2.h5 | bperp_mat | (n_final, n_ifg-1) | 剔除后的基线 |
| hgt2.h5, inc2.h5 | hgt, inc | (n_final,) | 高程、入射角 |
| psver.h5 | psver | int32 | 版本号 (升级为 2) |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| weed_standard_dev | 1.0 | 噪声标准差阈值 |
| weed_max_noise | 2.0 | 噪声最大值阈值 |
| weed_time_win | 365 | 时间窗口 (天) |
| weed_zero_elevation | n | 是否剔除零高程点 |
| weed_neighbours | y | 是否进行邻域剔除 |

**剔除流程**:
1. **邻域剔除**: 同一分辨率单元内保留最高相干性点
2. **零高程剔除**: 可选移除高程 < 1e-6 的点
3. **重复坐标剔除**: 相同 xy 坐标保留最高相干性点
4. **噪声剔除**: Delaunay 三角网边缘噪声估计

**注意事项**:
- psver 从 1 升级为 2，后续文件命名为 *2.h5
- SB 模式下跳过邻域剔除

---

### Step 5: 相位校正与合并

**功能**: 校正空间不相关视角误差，合并多个 Patch

**运行命令**:
```bash
python stamps_main.py --start 5 --end 5 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| ps2.h5, pm2.h5, bp2.h5, ph2.h5 | Step 4 输出 |
| patch_noover.in | 各 Patch 无重叠区域边界 |

**输出文件** (每个 PATCH 目录):
| 文件 | 数据集 | 说明 |
|------|--------|------|
| rc2.h5 | ph_rc | 校正后相位 |
| | ph_reref | 重参考相位 |

**输出文件** (项目根目录，合并后):
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| ps2.h5 | n_ps, ij, lonlat, xy, ... | (n_merged,) | 合并后 PS 数据 |
| pm2.h5 | K_ps, C_ps, coh_ps, ... | (n_merged,) | 合并后参数 |
| ph2.h5 | ph | (n_merged, n_ifg) | 合并后相位 |
| bp2.h5 | bperp_mat | (n_merged, n_ifg-1) | 合并后基线 |
| rc2.h5 | ph_rc, ph_reref | (n_merged, n_ifg) | 校正后相位 |
| hgt2.h5, inc2.h5 | hgt, inc | (n_merged,) | 高程、入射角 |
| ifgstd2.h5 | ifg_std | (n_ifg,) | 各干涉图噪声标准差 (度) |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| merge_resample_size | 0 | 重采样网格大小 (0=不重采样) |
| merge_standard_dev | 1.0 | 合并标准差阈值 |

**校正公式**:
```
ph_rc = ph * exp(-j * (K_ps * bperp_mat + C_ps))
ph_reref = ph_patch (with reference column = 1)
```

**注意事项**:
- 合并时根据 patch_noover.in 去除重叠区域
- 若重采样 (merge_resample_size > 0)，使用 SNR 加权平均
- ifg_std 用于后续步骤的方差-协方差矩阵

---

### Step 6: 相位解缠

**功能**: 三维相位解缠，恢复绝对相位

**运行命令**:
```bash
python stamps_main.py --start 6 --end 6 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| ps2.h5, pm2.h5, bp2.h5, rc2.h5 | Step 5 输出 |
| scla_smooth2.h5 | 需要先运行 Step 7 (或为空) |

**输出文件**:
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| phuw2.h5 | ph_uw | (n_ps, n_ifg) float32 | 解缠后相位 (弧度) |
| | msd | (n_ifg,) float32 | 参考-次影像差质量指标 |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| unwrap_method | 3D_QUICK | 解缠方法 |
| unwrap_grid_size | 200 | 解缠网格大小 (m) |
| unwrap_time_win | 730 | 时间窗口 (天) |
| unwrap_prefilter_flag | n | 是否预滤波 |
| unwrap_gold_n_win | 32 | Goldstein 滤波窗口 |
| unwrap_gold_alpha | 0.8 | Goldstein alpha 参数 |
| scla_deramp | n | 是否在解缠前去轨道斜坡 |

**解缠流程**:
1. 加载并归一化相位
2. 减除 SCLA (若存在)
3. 网格化: uw_grid_wrapped
4. 插值: uw_interp
5. 时空解缠: uw_sb_unwrap_space_time
6. 统计代价: uw_stat_costs → 调用 snaphu
7. 从网格恢复: uw_unwrap_from_grid
8. 加回 SCLA

**注意事项**:
- 需要 snaphu v2.0.7 在 PATH 或配置中指定
- reference 列相位保持为零
- 若 SB 模式，需运行 sb_invert_uw 转换单参考影像序列

---

### Step 7: SCLA 估算

**功能**: 估计空间相关视角误差 (DEM 误差) 和参考影像大气/轨道误差

**运行命令**:
```bash
python stamps_main.py --start 7 --end 7 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| ps2.h5, phuw2.h5, bp2.h5 | 前序步骤 |
| ifgstd2.h5 | Step 5 输出 |

**输出文件**:
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| scla2.h5 | K_ps_uw | (n_ps,) float32 | DEM 误差系数 |
| | C_ps_uw | (n_ps,) float32 | 参考影像大气误差 |
| | ph_scla | (n_ps, n_ifg) float32 | SCLA 相位校正量 |
| | ifg_vcm | (n_ifg, n_ifg) float32 | 干涉图方差-协方差矩阵 |
| scla_smooth2.h5 | K_ps_uw | (n_ps,) | 平滑后参数 |
| | C_ps_uw | (n_ps,) | 平滑后参数 |
| | ph_scla | (n_ps, n_ifg) | 平滑后校正量 |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| scla_method | L2 | 估计方法 (L1/L2) |
| scla_deramp | n | 是否去轨道斜坡 |
| scla_drop_index | [] | 排除的干涉图 |
| subtr_tropo | n | 是否减除大气校正 |

**估计公式**:
```
G = [ones, mean(bperp)']  (或加 [day] 列估计速度)
K_ps_uw = lscov(G, ph', ifg_vcm)
ph_scla = K_ps_uw * bperp_mat
C_ps_uw = mean(ph_uw - ph_scla)
```

**平滑方法**:
- Delaunay 三角网邻域裁剪
- K_ps_uw 超出邻域范围的值被裁剪到邻域 min/max

---

### Step 8: SCN 滤波

**功能**: 空间相关噪声滤波，提取形变信号

**运行命令**:
```bash
python stamps_main.py --start 8 --end 8 --config <数据目录>
```

**输入文件**:
| 文件 | 来源 |
|------|------|
| ps2.h5, phuw2.h5, scla_smooth2.h5 | 前序步骤 |

**输出文件**:
| 文件 | 数据集 | 形状 | 说明 |
|------|--------|------|------|
| scn2.h5 | ph_scn_slave | (n_ps, n_ifg) float32 | 空间低通滤波后相位（legacy-compatible 数据集名） |
| | ph_hpt | (n_ps, n_ifg) float32 | 高通时域相位 |
| | ph_ramp | (n_ps, 0) or (n_ps, n_ifg) | 轨道斜坡 |

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| scn_time_win | 365 | 时域滤波窗口 (天) |
| scn_wavelength | 100 | 空间滤波波长 (m) |
| scn_deramp_ifg | [] | 需要去斜坡的干涉图 |
| unwrap_grid_size | 200 | 解缠网格大小 |

**滤波流程**:
1. ph_input = ph_uw - ph_scla
2. Delaunay 边缘相位差 dph
3. 时域高斯低通滤波 → dph_lpt
4. 高通时域相位: dph_hpt = dph - dph_lpt
5. 稀疏求解恢复逐像素 ph_hpt
6. 空间高斯低通滤波 → ph_scn_slave（legacy-compatible 数据集名）

**注意事项**:
- reference 列保持为零
- 空间滤波半径 = 4 × scn_wavelength

---

## 4. 结果导出

**运行命令**:
```bash
python export_results.py --input_path <H5目录> --output_dir <输出目录> --mode ps
```

**输入文件**:
| 文件 | 说明 |
|------|------|
| ps2.h5 | PS 点坐标 |
| phuw2.h5 | 解缠相位 |
| pm2.h5 | 相干性等参数 |
| bp2.h5 | 基线矩阵 |

**输出文件**:
| 文件 | 数据集 | 说明 |
|------|--------|------|
| ps_plot_v.h5 | ph_disp | 形变速率 (mm/yr) |
| ps_plot_ts_v.h5 | ph_mm | 时间序列形变 (mm) |
| | day | 日期 |
| *.shp, *.dbf, *.shx | - | ESRI Shapefile |

**Shapefile 字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| lon | float | 经度 |
| lat | float | 纬度 |
| velocity | float | 形变速率 (mm/yr) |
| coh | float | 相干性 |
| K_ps | float | DEM 误差系数 |
| hgt | float | 高程 (m) |
| ts_YYYYMMDD | float | 时序形变 (mm) |

---

## 5. 配置参数说明

### 5.1 常用参数

| 参数 | 默认值 | Step | 说明 |
|------|--------|------|------|
| insar_processor | isce | 1 | InSAR 处理器 (isce/gamma/doris) |
| small_baseline_flag | n | 全部 | 是否小基线模式 |
| lambda | 0.056 | 1, 2 | 雷达波长 (m) |
| heading | - | 1 | 卫星航向角 (度) |
| max_topo_err | 20 | 2 | 最大地形误差 (m) |
| filter_grid_size | 50 | 2 | CLAP 网格大小 (m) |
| gamma_change_convergence | 0.005 | 2 | Gamma 收敛阈值 |
| select_method | PERCENT | 3 | PS 选择方法 |
| weed_standard_dev | 1.0 | 4 | 噪声剔除标准差阈值 |
| unwrap_grid_size | 200 | 6, 8 | 解缠网格大小 (m) |
| scn_time_win | 365 | 8 | SCN 时域窗口 (天) |
| scn_wavelength | 100 | 8 | SCN 空间波长 (m) |

### 5.2 参数修改方式

方式一: 修改 `parms.mat` 文件 (Matlab 格式)

方式二: 通过代码修改:
```python
from getparm import StampsConfig
cfg = StampsConfig(work_dir=data_dir)
cfg.setparm('parameter_name', new_value)
```

---

## 6. 常见问题

### Q1: Step 1 报错 "heading.1.in is empty"
**原因**: ISCE 预处理未生成航向角文件
**解决**: 检查 ISCE 处理流程，确保生成 `heading.1.in`

### Q2: Step 2 迭代不收敛
**原因**: 数据质量差或参数设置不当
**解决**: 
- 检查相位数据是否有大量零值
- 增大 `gamma_change_convergence` 阈值
- 增大 `gamma_max_iterations`

### Q3: Step 3 选择后 PS 数量骤减
**原因**: 相干性普遍偏低，阈值过高
**解决**:
- 检查 `coh_ps` 分布
- 调整 `max_percent_rand` 参数
- 检查输入数据质量

### Q4: Step 6 snaphu 执行失败
**原因**: snaphu 路径未配置或版本不兼容
**解决**:
- 确保安装 snaphu v2.0.7
- 设置环境变量或在配置中指定 `snaphu_path`
- Windows 下可能需要修改解缠方法

### Q5: HDF5 文件维度与 Matlab 不一致
**原因**: 不同的处理配置或随机过程差异
**说明**: 
- Matlab 的顶层 .mat 可能来自不同的处理配置
- 验证时使用同一次运行的数据对比
- 重点关注公式验证而非绝对数值

### Q6: 内存不足
**原因**: PS 点数量过多，大型矩阵占用内存
**解决**:
- 增大系统内存
- 分 Patch 处理
- 使用 `float32` 而非 `float64`

---

## 附录: 文件命名约定

| 阶段 | 文件前缀 | 版本号 | 示例 |
|------|----------|--------|------|
| 加载后 | ps, ph, bp, hgt, da | 1 | ps1.h5 |
| Gamma 估算后 | pm | 1 | pm1.h5 |
| 选择后 | select | 1 | select1.h5 |
| 剔除后 | ps, pm, ph, bp, hgt, inc, weed | 2 | ps2.h5, weed1.h5 |
| 校正后 | rc | 2 | rc2.h5 |
| 解缠后 | phuw | 2 | phuw2.h5 |
| SCLA 估算后 | scla, scla_smooth | 2 | scla2.h5 |
| SCN 滤波后 | scn | 2 | scn2.h5 |
| 导出结果 | ps_plot | v, ts_v | ps_plot_v.h5 |

---

*文档版本: 1.0*
*最后更新: 2026-04-08*
