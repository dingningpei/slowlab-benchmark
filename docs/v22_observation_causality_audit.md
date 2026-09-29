# Phase 0：GreenLight 观测时间因果性检查

日期：2026-09-27。状态：**已实现并测试新的在线观测路径，隔离上游重采样风险；完整 V2.2 执行器接线与端到端验证待 Phase 1。** 未证明既有正式结果受影响。

## 已确认的索引行为

锁定源码 `greenlight/_save/core.py:81–84` 的非 linear 分支注释为 left，但代码使用：

```python
rows = mdl.full_sol['Time'].searchsorted(interpolated_time_stamps)
rows = np.clip(rows, 0, mdl.full_sol.shape[0] - 1)
```

没有 `side='right'` 后减一。对递增时间轴，非采样时刻会取后一个点，并在输出中赋予较早的目标时间。

使用 NumPy 按同一索引表达式做最小复现（没有运行完整 save_sim 或数值求解器）：原记录时间 [0,300,600] 秒，值 [10,20,30]。

| 查询/重采样时刻 s | 上游表达式选中的原记录 s | 输出值 | 只能读已产生记录时应选择 s |
| --- | --- | --- | --- |
| 0 | 0 | 10 | 0 |
| 150 | 300 | 20 | 0 |
| 300 | 300 | 20 | 300 |
| 450 | 600 | 30 | 300 |

断言确认非对齐时刻选中了未来点。该问题不能单靠把全局 interpolation 切到 left 解决。对离线数值结果插值本身不应一概称为数据泄漏；风险在于将这种输出时间戳直接当成已到达的在线传感器记录，交给控制器/Reader。

## 相关、但需要区分的风险

官方 `energy_plus.py:96–98,184–187` 明确将整天太阳辐射总和重复到当天所有时刻，且说明其需要未来信息。原生灯控引用 dayRadSum，因此不能照搬这种“完美预知天气”的输入作为自主实验控制信息。

当前 SlowLab 的 `slowlab/agc_greenlight_weather.py` 已将该字段定义为 observed-so-far 积分；不能误报现有适配器仍在输出整日未来总量。但累计已发生量与原生描述的预测全天量语义不同，主任务必须显式定义控制规则，不能静默互换。

内部物理模型可以使用预生成天气驱动；代理可见观测与控制器可用测量必须分别受时间约束。不能因为物理求解器使用插值就判定所有仿真无效。

## 建议的下一步修复

1. 单独实现 SlowLab 在线观测适配器，使用 measurement_time 与 available_at；只选择同时满足已产生和已到达的记录。不从离线重采样 CSV 反推在线采样。
2. 对过去记录查询采用 last-available 语义；查询早于首条可用记录返回空，不夹取首条未来记录。保留原 measurement_time，不将它改成查询时刻。
3. 控制器只读当时可用信息。灯控可先使用固定时段、当时光照及截至当前累计辐射；不使用未来整日真值。正式规则还需 treatment schema 明确。
4. 增加相同历史前缀/不同未来数据的非干扰测试，覆盖控制器、Reader、缓存和日志；再继续冻结任务合同。

这属于可修复的时间接口问题，不要求换掉 GreenLight、增加数据下载或开展真实温室实验。按用户要求本轮遇到新问题后暂停；没有悄悄修改第三方 checkout。

本轮验证：同索引表达式的轻量 NumPy 复现和源码检查。无 LLM API、无数据下载、无正式实验、未查看新盲测输出。

锁定版本和路径：`configs/v22_greenlight_model_family.json`。可复现索引检查：`scripts/audit_v22_time_index.py`。

## 修复结果（2026-09-27）

新增 `slowlab/online_observations.py`：

- executor 拥有单调时间钟与传感器白名单；代理只能获得查询返回值，不能获得此对象或其写入/推进方法。
- 每条记录保留 measurement_time、available_at、variable、unit、compartment、value；必须先发生测量，才能由 executor 记录。允许延迟交付；禁止事后回填更早的可见时间。
- history/latest 同时按历史知识截止点与到达时间筛选；不插值、不把时间戳重写成查询时间、不夹取首条未来记录。
- 相同隔间/变量/测量时刻的记录不可覆盖。噪声应在 executor 生成一次后记录，查询不生成噪声，返回副本避免调用者修改存储。

新增 `greenlight_raw_endpoint`：只接受成功的、原始 states_sol 结构，要求末端恰好等于采样事件时间，校验状态顺序/维度/有限值，只返回明确选择的状态变量。旧 `read_greenlight_series` 明确为离线分析入口。无需修改第三方源码或重写历史数据。

未来 executor 的接线顺序：推进求解器恰好到传感器事件 → 检查 raw endpoint → 转换公开传感器/施加一次噪声 → executor 时间推进到事件 → 记录及计划到达 → 代理查询 payload。不能先运行到未来再将离线重采样结果当作在线传感器。完整场景在 Phase 1 接线时测试。

29 项相关测试通过（1.15 秒，单线程）：`tests/test_online_observations.py`、`tests/test_greenlight_adapter.py`、`tests/test_agc_greenlight_weather.py`。覆盖非对齐查询、精确端点、延迟/乱序到达、无可用记录、未来查询拒绝、不同未来相同前缀、重复读取及返回值修改、重复身份、回填到达时间、错误求解端点/形状/非有限状态。测试使用轻量合成数据与模拟 solver-result 对象，没有执行完整 GreenLight 仿真。

边界：这些接口不验证传感器生产者提供的数据是否真实，也不是 Python 对象级安全沙箱；新执行器必须控制写入权和时间推进。Full/Endpoint 的信息权限还需在代理工具层实现。控制器全天辐射预知风险及语义合同也仍需单独处理，不能由观测存储测试宣称已解决。现有主环境仍为 TOMGRO，未为完成此修复而强行迁移旧实验。
