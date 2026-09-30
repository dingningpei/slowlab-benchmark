# V2.2 Phase 0：GreenLight 气候—作物组合选择

日期：2026-09-27。结论：**选定模型族；设施尺度及完整任务合同尚未冻结。**

> 现状（2026-09-29）：文末提出的设施尺度问题已解决。实验单元定为 96 m² 独立模拟隔间（`docs/experimental_unit_scale.md`），几何、边界和设备容量逐项写入任务合同，现行版本为 `configs/task_contract_v4.json`。下文保留当时的记录。

## 选择

采用官方 GreenLight **v2.0.5 / fa502eddae5f9eff7b3380c88037d9b5f3f14bf5** 的 `main_katzin_2021.json` 组合，保持官方加载顺序：

1. `vanthoor_2011/greenhouse_vanthoor_2011_chapter_8.json`：温室热量、水汽与 CO₂ 动态。
2. `vanthoor_2011/crop_vanthoor_2011_chapter_9_simplified.json`：Vanthoor 番茄作物模型的 Katzin 简化实现。
3. `extension_greenhouse_katzin_2021_vanthoor_2011.json`：Katzin 补光、热交换和控制扩展。

配置与 SHA-256 记录在 `configs/greenlight_model_family.json`。这是模型族选择，不是可运行任务配置或现实校准证书。现有 TOMGRO 主环境尚未迁移。

选择该组合的理由是作者已经定义了共同状态与方程接口，不需要自行把另一作物族接入气候模型。`main_katzin_2020.json` 是历史验证回放入口，要求观测管温或供暖设定等附加输入，不作为本次自主实验入口。2021 入口默认天气为常数；正式实验必须另行提供天气，不能直接使用默认天气生成主结果。

## 双向耦合的源码证据

- 实际气候影响作物：冠层温度 `tCan`、吸收光照 `parCan`、气孔 CO₂ `co2Stom` 等进入光合、生长与呼吸计算。
- 作物反馈气候：`lai = sla * cLeaf`；`mcAirCan = mCo2 / mCh2o * (mcAirBuf - mcBufAir - mcOrgAir)`。作物文件覆盖基础温室的无作物占位项。
- 冠层与温室通过热交换和蒸腾相连，含 `hCanAir`、`mvCanAir`。这不表示已有根区缺水/盐害反馈。
- 收获通量 `mcFruitHar` 单位为 `mg{CH2O} m^-2 s^-1`，依照简化阈值函数连续产生；`cFruit` 是仍在植株上的库存。累计收获应对通量积分，不能把末期库存当作累计产量。

这些是静态方程审计结论，尚未验证断点续跑、控制响应、长期数值稳定或真实轨迹误差。

## 第一版的范围

作物使用该原生番茄模型族，不宣称对应 AGC 矮生番茄或某一已校准品种。初始植株状态沿用该模型配套来源，后续冻结时核实单位与移栽定义。

| 项目 | 决定 |
| --- | --- |
| 加热 | 支持管理温度设定，经共同控制器和供暖动态得到实际温度；范围待合同冻结 |
| CO₂ | 支持设定与施用控制，记录实际浓度和消耗；不承诺设定必达 |
| 补光 | 使用原生顶部 LED 配置；不添加第二组 LED 的株间灯解释 |
| 通风/湿度 | 使用原生通风和湿度控制路径；不是任意可达的独立湿度设定，更不等于验证了湿度相关病害 |
| 幕布 | 保留配套物理模型，第一版共同固定控制规则 |
| 根区 | 固定充分供水供肥假设；不将现有独立 EC 模块接入产量，也不开放灌溉/EC 处理 |
| 产量与收益 | 先保留原生收获通量；鲜重、商品率、售价换算必须显式冻结，不能直接称真实利润 |

AGC 2023/2024 可按既有证据角色使用，但不跨作物校准该模型的产量。历史失败的真实轨迹验证仍然失败，选择新的原生组合不会改变它。

## 新发现的设施尺度问题与停止点

Katzin 扩展的 `aFlr = 4e4 m²`，`aCov = 4.84e4 m²`，是大型温室模板。不能把它称为 12 m² 或 96 m² 隔间，也不能只替换面积而保留所有通风、表面积、泄漏和设备容量参数。

按用户要求遇到问题暂停。下一步建议先确定：将实验单元定义为具有原生几何的独立模拟温室，还是建立有来源的小型隔间模板。前者可先保留原生物理一致性，但必须如实说明每个处理占用的设施尺度；后者必须重新核对围护面积、通风几何、设备容量与单位面积通量。本轮不替用户悄悄更改实验单元。

## 检查与来源

本轮利用已有源码 checkout，核对 commit、计算四文件哈希，并检查官方加载顺序、作物覆盖项、收获单位、设施面积及控制符号。未运行数值模拟、未调用 LLM、未下载数据集。静态检查不是动态集成测试。

- [官方锁定入口](https://github.com/davkat1/GreenLight/blob/fa502eddae5f9eff7b3380c88037d9b5f3f14bf5/greenlight/models/katzin_2021/definition/main_katzin_2021.json)
- [官方作物方程](https://github.com/davkat1/GreenLight/blob/fa502eddae5f9eff7b3380c88037d9b5f3f14bf5/greenlight/models/katzin_2021/definition/vanthoor_2011/crop_vanthoor_2011_chapter_9_simplified.json)
- [官方气候与控制扩展](https://github.com/davkat1/GreenLight/blob/fa502eddae5f9eff7b3380c88037d9b5f3f14bf5/greenlight/models/katzin_2021/definition/extension_greenhouse_katzin_2021_vanthoor_2011.json)
