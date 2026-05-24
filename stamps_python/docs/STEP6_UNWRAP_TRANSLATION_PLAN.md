# Step 6 相位解缠 (Phase Unwrap) 翻译思路

本文档整理 MATLAB Step 6 的调用链、数据流和算法要点，作为 Python 实现的路线图。**先明确思路再编码**。

---

## 0. 数据与 IO 原则（必读）

- **计算数据来源**：Step 6 的**全部输入数据**仅来自前序 Step 1–5 生成的 **HDF5（.h5）文件**。不依赖任何 .mat 文件参与计算。
- **.mat 的用途**：仅作为**参考与对比**——例如用 MATLAB 跑出的 phuw_sb.mat、uw_grid.mat 等与 Python 结果做数值对比、回归测试时，可读取 .mat；**正常流水线中不读 .mat、不写 .mat**。
- **输出**：Step 6 写出 **phuw_*.h5**（及 SB 时的 phuw_sb_res.h5 等），与项目统一使用 .h5 的约定一致。若需与 MATLAB 对比，可单独提供“导出为 .mat”的脚本，而不作为主流程。

---

## 1. MATLAB 入口与总体流程

### 1.1 入口 (stamps.m)

```matlab
% stamps.m 约 494-507 行
if start_step<=6 & end_step >=6
    ps_unwrap
    if strcmpi(small_baseline_flag,'y')
        sb_invert_uw
    end
end
```

- **Step 6 必做**：`ps_unwrap()` —— 在网格上做 3D 相位解缠，得到每个 ifg 的 unwrapped phase，再映射回 PS 点。
- **仅 SB 模式**：`sb_invert_uw()` —— 将 SB 的 ifg 相位反演为单主影像时间序列 `ph_uw`（及残差、协方差），并保存 `phuw.mat` / `phuw_sb_res.mat`。

### 1.2 ps_unwrap() 流程概览

| 阶段 | 作用 | 涉及 .m |
|------|------|---------|
| 准备 | 读 ps/rc/pm（及 bperp）——**Python 仅从 .h5 读**，算 bperp_mat，构造 `ph_w`，可选减 scla/ramp/aps（scla/aps 若有则来自后续步骤 .h5） | ps_unwrap 自身 |
| 选项 | unwrap_hold_good_values 时加载已有 phuw、good_pixels，做 ph_uw_predef | sb_identify_good_pixels |
| 解缠核心 | 网格化 + 时间平滑 + 空间解缠 + 回插 | uw_3d → 见下 |
| 后处理 | 加回 scla/ramp/aps；若 unwrap_patch_phase 则加 angle(rc*conj(ph_w)) | ps_unwrap 自身 |
| 输出 | 存 phuw（MATLAB 为 .mat；Python 为 phuw_*.h5），未解缠 ifg 列置 0 | 前序 .h5 仅写 .h5 |

**平台分支**：  
- 非 Windows：`uw_3d(...)`（完整 3D + 统计代价 + Snaphu）。  
- Windows：`uw_nosnaphu(...)`（仅时间解缠 + 简单空间解缠，不调 Snaphu）。

---

## 2. uw_3d 内部调用链（核心）

```
uw_3d(ph, xy, day, ifgday_ix, bperp, options)
  │
  ├─► uw_grid_wrapped(ph, xy, grid_size, prefilt_win, goldfilt_flag, lowfilt_flag, gold_alpha, ph_uw_predef)
  │      将 PS 点相位重采样到网格，可选 Goldstein/低通滤波，写出 uw_grid（stamps_save 'uw_grid'）
  │      内部用 wrap_filt(ph_grid, prefilt_win, gold_alpha, [], lowfilt_flag)
  │
  ├─► uw_interp
  │      读 uw_grid、建三角网（Delaunay 或 triangle），得到网格边、rowix/colix、Z（网格→节点）
  │      写出 uw_interp（edgs, n_edge, rowix, colix, Z）
  │
  ├─► uw_sb_unwrap_space_time(day, ifgday_ix, unwrap_method, time_win, la_flag, bperp, n_trial_wraps, prefilt_win, scf_flag, temp, ...)
  │      在边上做：弧段 DEM/温度校正(K/Kt)、时间维平滑、得到 dph_space_uw + dph_noise + spread
  │      写出 uw_space_time
  │
  ├─► uw_stat_costs(unwrap_method, variance)
  │      读 uw_grid、uw_interp、uw_space_time，按 DEFO 统计代价写 snaphu.costinfile，调 Snaphu 做 2D 解缠
  │      写出 uw_phaseuw（ph_uw 网格上、msd）
  │
  └─► [ph_uw, msd] = uw_unwrap_from_grid(xy, grid_size)
         读 uw_phaseuw、uw_grid，将网格解缠结果 + 原始 wrapped 相位差回插到每个 PS 点
         返回的 ph_uw 是 (n_ps_orig, n_ifg)，供 ps_unwrap 写 phuw（MATLAB 写 .mat；Python 写 phuw_*.h5）
```

**数据传递方式**：MATLAB 用 `stamps_save` / `load` 写读中间 .mat（uw_grid, uw_interp, uw_space_time, uw_phaseuw）。Python **不写读 .mat**：中间量用内存对象传递，或写 patch 内的 .h5（如 uw_grid.h5）供调试；与 MATLAB 对比时仅**只读**已有 .mat 做参考。

---

## 3. 涉及函数清单与职责

| 文件 | 职责 | 输入/输出 |
|------|------|-----------|
| **ps_unwrap.m** | 入口：读配置与数据，预处理 ph_w，调 uw_3d，后处理，写 phuw | MATLAB 读 ps/rc/pm/bp/scla/tca(.mat)；**Python 仅读对应 .h5**，写 phuw_*.h5 |
| **uw_3d.m** | 编排：grid → interp → time-space → stat_costs → from_grid | 入：ph, xy, day, ifgday_ix, bperp, options；出：ph_uw, msd |
| **uw_grid_wrapped.m** | 网格化 + 可选 Goldstein/低通 | 入：ph_in, xy_in, pix_size, ...；MATLAB 写 uw_grid；**Python 用内存或 uw_grid.h5** |
| **wrap_filt.m** | Goldstein 自适应 + 低通滤波（2D 复数相位） | 入：ph(n_i,n_j), n_win, alpha, n_pad, low_flag；出：ph_out[, ph_out_low] |
| **uw_interp.m** | 网格三角化、边列表、row/col 边索引、Z | MATLAB 读 uw_grid；**Python 读上步内存/uw_grid.h5**；输出内存或 .h5 |
| **uw_sb_unwrap_space_time.m** | 弧段时间解缠：K/Kt 估计、时间平滑、dph_noise、可选 scf 空间梯度 | MATLAB 读 uw_grid, uw_interp；**Python 读前序步骤产出（内存/.h5）**；输出内存或 .h5 |
| **uw_sb_smooth_unwrap.m** | 3D 方法时对高噪声弧做平滑解缠（模拟退火等） | 被 uw_sb_unwrap_space_time 调用 |
| **gradient_filt.m** | 局部相位梯度（ifreq, jfreq）用于 scf | 被 uw_sb_unwrap_space_time 在 scf_flag='y' 时调用 |
| **uw_stat_costs.m** | 构造 Snaphu 代价、写 snaphu.in/costinfile/conf、调 snaphu、读结果 | MATLAB 读 uw_grid, uw_interp, uw_space_time；**Python 读前序内存/.h5**；写 uw_phaseuw（内存或 .h5）, snaphu.* |
| **uw_unwrap_from_grid.m** | 网格解缠结果回插到原始 PS 点 | 读 uw_phaseuw, uw_grid；出：ph_uw, msd |
| **uw_nosnaphu.m** | Windows 简化路径：uw_triangulate → uw_unwrap_time → uw_unwrap_space | 不写 uw_phaseuw，直接返回 ph_uw |
| **uw_triangulate.m** | 仅 nosnaphu 路径：Delaunay + 边/单元索引 | 写 uw_triangulate 相关 |
| **uw_unwrap_time.m** | 仅 nosnaphu：时间维解缠 | 读 triangulate 结果 |
| **uw_unwrap_space.m** | 仅 nosnaphu：2D 空间解缠（读 snaphu 或简单算法） | 出 ph_uw |
| **sb_invert_uw.m** | SB：ifg 相位 → 单主影像时间序列 + 残差 + 协方差 | MATLAB 读 phuw_sb, ps, (pm, rc).mat 写 .mat；**Python 读 phuw_sb.h5, ps*.h5, pm*.h5, rc*.h5，写 phuw.h5, phuw_sb_res.h5** |

---

## 4. 算法要点（便于逐块翻译）

### 4.1 网格化 (uw_grid_wrapped)

- 用 `xy` 的 2、3 列（或 1、2）算 `grid_ij`，把每个 PS 点归属到网格格点；若 `pix_size==0` 则用原始 ij 不重采样。
- 每 ifg：`ph_this = ph_in(:,i)`（实相位则 `exp(1i*ph_this)`），按 grid_ij 累加到 `ph_grid`；可选 `wrap_filt` 得 gold/低通，再取非零格点组成 `ph`（n_ps_grid × n_ifg）。
- 输出：ph, ph_in, ph_lowpass（可选）, ph_uw_predef（可选）, xy, ij, nzix, grid_ij, n_i, n_j, n_ifg, n_ps, grid_x_min, grid_y_min, pix_size。

### 4.2 插值/三角化 (uw_interp)

- 由 `nzix` 得到网格上非零点的 (x,y)，Delaunay 三角化（或调用 `triangle` 生成 .node/.edge/.ele）。
- 得到边表 `edgs`（边号, 节点1, 节点2），以及网格相邻格点对应的边索引 `rowix`、`colix`，和网格点到节点编号的映射 `Z`（用于后续 Snaphu 输出格点→节点）。

### 4.3 时间维 (uw_sb_unwrap_space_time)

- **G 矩阵**：n_ifg × n_image，G(i, master_ix)=-1, G(i, slave_ix)=1；去掉全零列得到有效 day。
- **弧段相位差**：`dph_space = ph(edgs(:,3),:).*conj(ph(edgs(:,2),:))` 再单位化。
- **可选温度**：用 temp 与 trial 相位估计 Kt，`dph_space *= exp(-1i*Kt*temp')`。
- **Look angle (DEM) 误差**：用 bperp 和 trial 估计每弧 K，`dph_space *= exp(-1i*K*bperp')`。
- **时间平滑**（3D_FULL / 3D 等）：
  - 3D_FULL：对每个 image 做子网、按 day 排序、高斯加权平滑、累积得到 dph_smooth_ifg。
  - 否则：G\angle(dph_space) 得 dph_space_series，再时间窗平滑得 dph_smooth_series，dph_smooth_ifg = G*dph_smooth_series；高噪声弧可选 uw_sb_smooth_unwrap。
- **噪声**：`dph_noise = angle(dph_space .* exp(-1i*dph_smooth_ifg))`，部分弧可置 nan。
- **解缠相位**：`dph_space_uw = dph_smooth_ifg + dph_noise`，再加回 K*bperp、Kt*temp。
- **scf_flag**：gradient_filt 算 ifreq_ij, jfreq_ij，边上的梯度预测 dph_smooth_uw2，与时间平滑比较，选更稳的。
- 写出：dph_space_uw, dph_noise, spread, G, predef_ix（可选）, ifreq_ij, jfreq_ij, shaky_ix（可选）。

### 4.4 空间解缠 (uw_stat_costs)

- 由 dph_noise 的 std 得弧方差 sigsq_noise；2D 方法用边长/APS 模型。
- 对每个 ifg：用 dph_space_uw、dph_smooth、spread 构造 Snaphu 的 offset/sigsq/dzmax/laycost，写 snaphu.costinfile；uw.ph 按 Z 重排成 2D 写 snaphu.in（复数）；调 `snaphu -d -f snaphu.conf ncol`；读 snaphu.out 浮点解缠相位，填 ph_uw(:,i1)，并算 msd(i1)。
- 最后 save uw_phaseuw ph_uw msd。

### 4.5 回插 (uw_unwrap_from_grid)

- 对每个原始 PS 点 i：通过 grid_ij(i) 找到网格节点 ix，取 uu.ph_uw(ix,:)；再与 uw.ph_in(i,:) 的相位差（angle）对齐到同一 2π 周期，得到该点的 ph_uw(i,:)。若有 ph_in_predef，则预定义点用预定义值并做整体 2π 对齐。
- msd 直接来自 uu.msd。

### 4.6 SB 反演 (sb_invert_uw)

- 读 phuw_sb（ph_uw 仅 unwrap_ifg_index 列）；**Python 从 phuw_sb.h5 读**，用 drop_ifg_index 得到 unwrap_ifg_index。
- 若存在 pm/rc，用 ph_noise=angle(rc*conj(pm.ph_patch)) 估计 sb_cov；**Python 从 pm*.h5、rc*.h5 读**；否则单位阵。C = sb_cov(unwrap_ifg_index, unwrap_ifg_index)。
- 参考点 ref_ps = ps_setref；ph_uw_sb 减去 ref 均值。
- G2 = G(unwrap_ifg_index, :)，去掉全零列；master 列置 0。lscov(G2, ph_uw_sb', C)' 得 ph_uw（n_ps × n_image），ph_res = G*ph_uw'，sm_cov 由 G2' inv(C) G2 的逆得到。
- **Python 写 phuw.h5（ph_uw, unwrap_ifg_index_sm）、phuw_sb_res.h5（ph_res, sb_cov, sm_cov）**；不写 .mat。

---

## 5. 与现有 Python 的衔接（仅用 .h5）

- **配置**：所有 getparm（如 unwrap_method, unwrap_time_win, unwrap_grid_size, unwrap_prefilter_flag, drop_ifg_index 等）从 `StampsConfig` / getparm 读，与 Step 1–5 一致。
- **数据来源（全部来自前序 .h5，不读 .mat）**：
  - **ps**：ps2.h5（Step 4 后）或 ps1.h5；n_ps, n_ifg, xy, bperp, day, master_ix, ifgday_ix 等。
  - **rc**：rc2.h5 或 rc1.h5（ph_rc）；**pm**：pm2.h5（ph_patch, K_ps）；**bperp_mat**：由 ps 的 bperp 按 MATLAB 逻辑扩展得到（或若将来有 bp*.h5 则从 bp*.h5 读），**不读 bp*.mat**。
  - **scla / ramp / aps**：若实现“减 scla/加回”等，数据应来自 Step 7/8 的 .h5 产物（若项目有对应 .h5）；否则该可选分支不读任何 .mat。
- **IO**：
  - **输入**：仅读 patch_dir 下的 .h5（ps*.h5, rc*.h5, pm*.h5 等）。
  - **中间量**：方案 A 全部内存传递；方案 B 可写 uw_grid.h5、uw_space_time.h5 等于 patch 内便于调试；**不写 .mat**。
  - **输出**：phuw_*.h5（及 SB 的 phuw_sb_res.h5 等）；与 MATLAB 对比时**仅**在单独脚本中读取已有 .mat 做参考，不写入 .mat。
- **patch**：Step 6 在 merge 之后通常对整幅或每个 patch 跑一次；若保留 per-patch，则每个 patch_dir 内只读写该目录下的 .h5。
- **small_baseline_flag**：从 config 读；为 'y' 时在 run_step_6 末尾调 sb_invert_uw 等价逻辑（读/写均为 .h5）。

---

## 6. 外部依赖与兼容策略

| 依赖 | 用途 | 策略 |
|------|------|------|
| **Snaphu** | 2D/3D 统计代价解缠 | 与 MATLAB 一致：生成 snaphu.in / snaphu.costinfile / snaphu.conf，subprocess 调 snaphu，读 snaphu.out。若未安装则退化为“仅时间解缠 + 简单空间”或报错提示。 |
| **triangle** | 高质量 Delaunay（Linux/Mac） | uw_interp 中可选；无则用 scipy.spatial.Delaunay 生成边（与 MATLAB 无 triangle 时一致）。 |
| **writecpx/read** | Snaphu 复数输入格式 | Python 用 numpy 写二进制（实部+虚部 float32 或约定格式），与 Snaphu 文档一致即可。 |

---

## 7. Python 模块与类划分建议

- **phase_unwrapping.py**（或拆成子模块）：
  - **UnwrapPipeline**：总控，实现 ps_unwrap 逻辑（读 ph_w、预处理、调各子步、后处理、写 phuw）。
  - **GridWrapped**：uw_grid_wrapped + wrap_filt（可单独函数 wrap_filt(ph_grid, n_win, alpha, low_flag)）。
  - **UnwrapInterp**：uw_interp，输出 edgs, rowix, colix, Z, n_edge。
  - **UnwrapSpaceTime**：uw_sb_unwrap_space_time（时间平滑、K/Kt、dph_space_uw/dph_noise/spread）；可选依赖 **UwSbSmoothUnwrap**（uw_sb_smooth_unwrap）和 **gradient_filt**。
  - **UnwrapStatCosts**：uw_stat_costs（构造代价、调 Snaphu、填 ph_uw 网格、msd）。
  - **UnwrapFromGrid**：uw_unwrap_from_grid（回插到 PS 点）。
- **sb_invert_uw.py**（或放在 phase_unwrapping.py 末尾）：  
  **SBInvertUw**：从 **phuw_sb.h5、ps*.h5、可选 pm*.h5/rc*.h5** 读入；算 sb_cov、G2、lscov、ph_uw、ph_res、sm_cov；**仅写 phuw.h5、phuw_sb_res.h5**（不写 .mat）。

**数据在管道中的形态**：  
用 dataclass 或简单 dict 表示“当前 patch 的网格/边/时间解缠/空间解缠”等，避免全局。与 MATLAB 对照时**仅读取** MATLAB 生成的 .mat 做参考，不依赖 .mat 参与计算；可选提供“将当前结果导出为 .mat”的独立工具便于对比。

---

## 8. 实现顺序建议

1. **先实现“最小可跑”路径**（便于与 test_data 对照）：  
   - 仅 SB + 已有 test_data 的 PATCH；  
   - unwrap_method 先支持 `3D_QUICK` 或 `2D`（少依赖 uw_sb_smooth_unwrap / 3D_FULL 的复杂分支）；  
   - la_flag='n', scf_flag='n'，不读 scla/tca（或读但先不减），减少依赖 Step 7/8 产物。

2. **再补全**：  
   - uw_grid_wrapped + wrap_filt；  
   - uw_interp（Delaunay + rowix/colix/Z）；  
   - uw_sb_unwrap_space_time 的 3D_QUICK/2D/3D_FULL 分支；  
   - uw_stat_costs（Snaphu 调用 + 读写格式）；  
   - uw_unwrap_from_grid；  
   - ps_unwrap 的 scla/ramp/aps 加减、unwrap_patch_phase、hold_good_values；  
   - sb_invert_uw；  
   - 最后考虑 uw_nosnaphu 路径（Windows 或无 Snaphu 时）。

3. **测试**：  
   - 用 test_data 的 SMALL_BASELINES 下某 PATCH，跑 Step 1–5 再 Step 6，**计算完全基于生成的 .h5**；若需验证，可**仅读取** MATLAB 生成的 phuw_sb.mat 等做数值对比（.mat 仅作参考）。  
   - 单元测试：wrap_filt、Delaunay 边、G 矩阵构造、lscov 与 MATLAB 一致。

---

## 9. 数值与类型注意

- **相位**：MATLAB 用 single/float；Python 统一 float32 或 float64 与 HDF5 一致，角度弧度。
- **复数**：HDF5 可能存成 compound (real, imag)；读入后用现有 _ensure_complex64 转成 np.complex64 再运算。
- **lscov**：可用 scipy.linalg 或 numpy 解加权最小二乘；注意 MATLAB 的 lscov(A, B, W) 是 B = A*X 的 W 加权，对应 `X = (A' W A)^{-1} A' W B`。
- **Snaphu 二进制**：确认 INFILEFORMAT/OUTFILEFORMAT 与写的 float/complex 格式一致（如 4 字节 float、行优先等）。

---

## 10. 文档对照小结（MATLAB → Python）

| MATLAB | Python |
|--------|--------|
| load/save uw_grid, uw_interp, uw_space_time, uw_phaseuw | **不读 .mat**；中间量用内存或 patch 内 .h5；StampsConfig.getparm 替代 getparm |
| load ps/rc/pm/bp | **仅读前序 .h5**：ps*.h5, rc*.h5, pm*.h5；bperp_mat 由 bperp 扩展或 bp*.h5 |
| stamps_save(phuwname, ph_uw, msd) | **仅写 phuw_*.h5**（不写 .mat）；.mat 仅作对比时由单独脚本读取参考 |
| setdiff([1:ps.n_ifg], drop_ifg_index) | unwrap_ifg_index = 索引数组，从 getparm('drop_ifg_index') 解析 |
| G 矩阵 n_ifg×n_image | 同构 numpy 数组，ifgday_ix 从 ps*.h5 来 |
| lscov(G2, ph_uw_sb', C)' | 加权最小二乘，可封装 lscov(G, B, W) 函数 |
| angle(ph), exp(1i*ph) | np.angle, np.exp(1j*ph)；相位 array 用 _ensure_complex64 若来自 HDF5 |

按此思路，可先实现 **phase_unwrapping.py** 的骨架（UnwrapPipeline + GridWrapped + UnwrapInterp + UnwrapSpaceTime 简化版 + UnwrapStatCosts + UnwrapFromGrid），**全部输入来自前序 .h5、全部输出写入 .h5**；再在 **stamps_main.py** 的 `run_step_6` 里调用并接 sb_invert_uw，最后补全分支与可选功能。
