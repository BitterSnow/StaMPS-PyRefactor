# StaMPS Python 翻译验收计划

> 按步骤逐项验收，每步分三个维度：**算法对照**、**数据一致性**、**代码质量**

---

## 通用检查项（每步必查）

| # | 检查项 | 说明 |
|---|--------|------|
| G1 | 索引转换 | Matlab 1-based → Python 0-based 是否正确，尤其是 day_ix、master_ix、ij 的第一列 |
| G2 | 日期转换 | Matlab datenum（0000-01-01基准）vs Python ordinal（0001-01-01基准），差值=366 |
| G3 | 矢量化 | 核心计算是否使用 numpy 矢量化，是否有不必要的百万级 Python 循环 |
| G4 | 复数处理 | complex64 相位数据的存储/读取是否正确（h5py 行为） |
| G5 | 日志输出 | 处理前后点数变化、关键中间结果是否记录 |
| G6 | 内存管理 | 大数组是否及时释放，是否使用 float32 而非 float64（在精度允许时） |

---

## Step 1: 数据加载 (data_loader.py)

**Matlab 对照**: ps_load_initial_isce.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 1.1 | 原始文件解析 | 对比 ps1.h5 与 ps1.mat 中各变量维度：ph(50468,108)、ij、lonlat、bperp、bperp_mat、day 等 |
| 1.2 | 复数相位 | 检查 ph 的 dtype 是否为 complex64/complex128，数值是否与 Matlab 一致 |
| 1.3 | 垂直基线矩阵 | bperp_mat 的 shape 应为 (n_ps, n_slave)，与 Matlab (n_ps, n_image-1) 对应 |
| 1.4 | 排序索引 | sort_ix、day_ix 排列逻辑是否与 Matlab 一致 |
| 1.5 | SB 加载器 | ISCESBLoader 是否正确处理 ifgday/ifgday_ix |
| 1.6 | 跨平台路径 | Windows 路径下 raw 文件读取是否正确（二进制 endianness） |

**数值验证**: 运行 Step 1 → 对比 ps1.h5 与 ps1.mat 各字段前 100 个值

---

## Step 2: Gamma 估算 (gamma_est.py)

**Matlab 对照**: ps_est_gamma_quick.m, ps_topofit.m, clap_filt.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 2.1 | CLAP 滤波器 | clap_filt 实现与 Matlab clap_filt.m 对比，检查 FFT 卷积边界处理 |
| 2.2 | TopoFit 算法 | ps_topofit_batch 与 ps_topofit.m 对比：网格化→拟合→相位斜率提取 |
| 2.3 | 迭代收敛 | gamma_change_save 的收敛曲线是否合理（应逐步下降） |
| 2.4 | 输出变量 | pm1.h5 中 K_ps, C_ps, coh_ps, ph_patch, ph_res 的维度和数值 |
| 2.5 | 网格参数 | grid_size, low_pass 尺寸是否与 Matlab 一致 |
| 2.6 | 重启逻辑 | restart_flag 参数是否能正确恢复迭代 |

**数值验证**: 对比 pm1.h5 与 Matlab pm1.mat → coh_ps 分布直方图、K_ps 均值/标准差

---

## Step 3: PS 选择 (ps_selection_final.py)

**Matlab 对照**: ps_select.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 3.1 | 选择阈值 | coh_thresh / D_A 阈值筛选逻辑与 ps_select.m 一致 |
| 3.2 | 直方图匹配 | 随机相位分布直方图拟合（Monte Carlo）是否正确 |
| 3.3 | 重估分支 | reest_flag=0/1 两条路径是否都正确处理 |
| 3.4 | 输出变量 | select1.h5 中 kept_ix 的维度，筛选后点数是否合理 |
| 3.5 | psver 更新 | 选择后 psver 是否从 1 变为 2，后续文件命名是否正确 |

**数值验证**: 对比筛选后点数（Python vs Matlab），kept_ix 是否一致

---

## Step 4: 剔除噪声点 (ps_weeding.py)

**Matlab 对照**: ps_weed.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 4.1 | Delaunay 三角剖分 | scipy.spatial.Delaunay 与 Matlab delaunay 结果是否一致 |
| 4.2 | 邻域搜索 | cKDTree 邻近点搜索的距离计算是否正确（geographic→local坐标） |
| 4.3 | 标准差剔除 | weed_standard_dev 阈值的应用逻辑 |
| 4.4 | 旁瓣剔除 | 同一分辨率单元内冗余点的识别与移除 |
| 4.5 | SB 分支 | small_baseline_flag='y' 时 no_weed_adjacent=1 是否跳过邻域剔除 |
| 4.6 | weed1 输出 | weed1.h5 中 kept_ix 维度、ps2.h5 各字段维度 |

**数值验证**: 对比 weed1.h5 vs weed1.mat → 点数差异 <5%，检查 pm2.h5 中 K_ps/coh_ps 数值分布

---

## Step 5: 相位校正 + 合并 (ps_correct_phase.py + ps_merge_patches.py)

**Matlab 对照**: ps_correct_phase.m, ps_merge_patches.m, ps_calc_ifg_std.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 5.1 | SCLA 相位校正 | K_ps * bperp 相位去除是否与 ps_correct_phase.m 一致 |
| 5.2 | rc2 输出 | ph_rc（校正后相位）维度和数值 |
| 5.3 | 多 Patch 合并 | PATCH_437 + PATCH_438 合并后总数 = 49457+63144=112601，与顶层 ps_plot 文件一致 |
| 5.4 | 重叠处理 | 两个 Patch 间重叠区域（overlap pixels）的去重逻辑 |
| 5.5 | ifg_std 计算 | calc_ifg_std 计算每幅干涉图相位标准差的逻辑 |
| 5.6 | 合并后文件 | 顶层 ps2.h5、ph2.h5、pm2.h5、bp2.h5、rc2.h5 的维度 |

**数值验证**: 对比 rc2.h5 vs rc2.mat、ifgstd2.h5 vs ifgstd2.mat

---

## Step 6: 相位解缠 (phase_unwrapping.py + uw_core.py)

**Matlab 对照**: ps_unwrap.m, uw_3d.m, uw_unwrap_time.m, sb_invert_uw.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 6.1 | 网格化 | UwGrid：Delaunay 三角剖分 + 网格化相位，与 uw_grid.m 对比 |
| 6.2 | 插值 | UwInterp：从网格插值回 PS 点，与 uw_interp.m 对比 |
| 6.3 | 时空解缠 | UwSpaceTime：3D 解缠核心逻辑，与 uw_3d.m / uw_space_time.m 对比 |
| 6.4 | Snaphu 调用 | snaphu v2.0.7 接口：输入文件格式、命令行参数、输出解析 |
| 6.5 | msd 计算 | 主从差（master-slave difference）质量指标 |
| 6.6 | phuw2 输出 | ph_uw 维度 (n_ps, n_ifg)、数值范围是否合理（非缠绕 2π 跳变） |
| 6.7 | SB 转换 | small_baseline_flag='y' 时 sb_invert_uw 分支 |

**数值验证**: 对比 phuw2.h5 vs phuw2.mat → 解缠相位差异统计（均值、标准差、异常值比例）

---

## Step 7: SCLA 估算 (scla_estimation.py)

**Matlab 对照**: ps_calc_scla.m, ps_smooth_scla.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 7.1 | DEM 误差估算 | K_ps_uw (DEM error) 计算与 ps_calc_scla.m 对比 |
| 7.2 | 主影像 APS | C_ps_uw (master APS) 提取逻辑 |
| 7.3 | 最小二乘求解 | np.linalg.lstsq 的使用是否正确（权重、归一化） |
| 7.4 | 平滑处理 | ps_smooth_scla 的 Delaunay 邻域裁剪逻辑 |
| 7.5 | 迭代 | SCLA 估算是否支持多轮迭代 |
| 7.6 | scla2 输出 | K_ps_uw, C_ps_uw, ph_scla, ifg_vcm 的维度和数值 |

**数值验证**: 对比 scla2.h5 vs scla2.mat、scla_smooth2.h5 vs scla_smooth2.mat

---

## Step 8: SCN 滤波 (scn_filt.py)

**Matlab 对照**: ps_scn_filt.m

| # | 检查项 | 验证方法 |
|---|--------|----------|
| 8.1 | 高斯滤波器 | 空间+时间低通滤波参数与 ps_scn_filt.m 对比（sigma、截断半径） |
| 8.2 | Goldstein 滤波 | 频域滤波实现与 goldstein_filt.m 对比 |
| 8.3 | 滤波前后对比 | 相位标准差变化是否合理（应下降） |
| 8.4 | scn2 输出 | ph_scn_slave, ph_hpt, ph_ramp 的维度 |
| 8.5 | 权重计算 | Gaussian 窗口或 Butterworth 参数与 Matlab 源码一致 |

**数值验证**: 对比 scn2.h5 vs scn2.mat → 滤波后相位标准差

---

## 导出模块 (export_results.py)

**Matlab 对照**: ps_plot.m, Stamps2ShpWithDV73.py

| # | 检查项 | 验证方法 |
|---|--------|----------|
| E1 | 速率计算 | 形变速率 (mm/yr) 公式与 ps_plot.m 一致：相位→径向位移→年均速率 |
| E2 | 时间序列 | ts 位移量计算，master_date 为零点 |
| E3 | 波长转换 | lambda 相位转距离的系数是否正确 |
| E4 | ps_plot_v.h5 | ph_disp 维度与 Matlab ps_plot_v.mat 对比 |
| E5 | ps_plot_ts_v.h5 | ph_mm 维度与 Matlab ps_plot_ts_v.mat 对比 |
| E6 | Shapefile 输出 | 字段定义与 Stamps2ShpWithDV73.py 一致，坐标系正确 |
| E7 | 命令行接口 | --input_path / --output_dir / --mode 参数正常工作 |

---

## 执行顺序

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 → Step 8 → 导出
  ↓        ↓        ↓        ↓        ↓        ↓        ↓        ↓       ↓
 算法对照  算法对照  算法对照  算法对照  算法对照  算法对照  算法对照  算法对照  算法对照
 数值验证  数值验证  数值验证  数值验证  数值验证  数值验证  数值验证  数值验证  数值验证
 代码质量  代码质量  代码质量  代码质量  代码质量  代码质量  代码质量  代码质量  代码质量
```

每步验收输出格式：
1. **通过/不通过** — 总体结论
2. **问题清单** — 具体问题编号、严重程度（高/中/低）、描述
3. **数值对比表** — 关键变量 Python vs Matlab 的统计量对比
4. **建议改进** — 非必须但可优化的点
