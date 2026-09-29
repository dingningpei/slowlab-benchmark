# Phase 1：深土壤边界审计

GreenLight 2.0.5 的 Katzin 2021 / Vanthoor 2011 定义中，`tSoOut` 默认恒定 20°C，是第五层土壤与外侧土层的热边界。`hSo5=0.64 m`、`hSoOut=1.28 m`、`lambdaSo=0.85 W m⁻¹ K⁻¹`，接口导热系数为 0.8854 W m⁻² K⁻¹。GreenLight 的 EnergyPlus 转换器明确称 `tSoOut` 为约 2 m 深土温，并从 EnergyPlus 每月深土温拟合逐时正弦曲线。因此原生默认 20°C 不是 Cabauw 实测边界，也不能自动称为全年现实驱动。

KNMI Cabauw [soil heat lb1](https://dataplatform.knmi.nl/dataset/cesar-soil-heat-lb1-t10-v1-0) 提供露天田地 0–50 cm 土温，不覆盖约 2 m 深温室下方的边界；即使下载全部数据也不能直接校准此输入。官方文件目录元数据表明 2016-12 边界月至 2020-12 可选 49 个月合计 11,552,807 字节，但**没有下载**。目前不应为了解决该问题盲目下载这批数据。

已把 `ReusableGreenLight(..., soil_boundary_c=...)` 加成显式、有限范围的开发情景输入。原默认行为不变。`scripts/check_v22_online_weather.py --soil-boundary-c` 在已批准的 Cabauw lc1 天气上比较 8°C 与 20°C、各 6×300 s；结果保存于 `configs/v22_deep_soil_boundary_short_sensitivity.json`。两个情景都通过时间因果与数值检查，20°C 显式输入与原默认六步室内记录在 1e-7 容差内一致。30 分钟后室温差约 7.5e-12°C，这是深层土壤热惯性在短窗内的结果，**不能**外推为年度无影响。

正式活动前需冻结深土温场景来源或有根据的年度曲线，并在独立开发天气年中测量全年气候、能耗和结果对该假设的敏感性。无温室下方实测土温时，论文只能称这是未校准边界的情景分析，不是温室实测验证。
