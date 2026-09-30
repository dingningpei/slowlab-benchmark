# SlowLab v2.3 To-do

对应 `RESEARCH_PLAN.md` 的阶段。修改规则见 `CLAUDE.md`：勾选自己刚完成的项目不用问；其它改动先在对话里提出。
完成一项时在同一行末尾用 `→ 证据文件` 指出结果或审计文件。

## Phase 1：可验证的执行环境（收尾）

已通过：模型复用、编译 RHS、空隔间模式、因果观测、四隔间同步、单/四隔间全年、事件驱动全年活动、深土壤全年配对、2014 天气读取。剩余：

- [x] 冻结独立于 site/天气真值的测量噪声与缺测假设；实现同一 (隔间, 通道, 时刻) 只采样一次、可重复查询的缓存与审计 → `configs/sensor_noise_v0.json`、`results/sensor_noise_v0_prefix_check.json`
- [x] 冻结深土壤默认边界（8 或 20 °C）及其披露方式；写入合同 v4 → `configs/task_contract_v4.json`（开发期沿用原生 20 °C；Phase 3 起为 8–20 °C 私有 site 参数）
- [x] 执行器去掉四单元硬编码，policy 改为 schema 对象；从合同读取隔间数 → `slowlab/policy.py`、`tests/test_policy_and_units.py`
- [ ] 范围审计改为区分物理错误、来源外推、声明假设三类，不靠删 site 通过
- [ ] 在 `pyproject.toml` 声明 GreenLight 为 pinned 可选依赖（git+commit）；保留 `--source` 路径方式
- [ ] 清点 `scripts/`：删除已被全年运行取代的一次性短测脚本，保留 verify/audit/annual 三类

验收：不用 LLM，脚本可完成含中途观察、改变下一实验、停止/重种的整场活动，并在不利结果下完整结算（已通过）；噪声与通用执行器补齐后重新跑一次四隔间全年并审计。

## Phase 2：算力预算、代理接口与强基线

- [ ] 在租用多核 Linux 上测四隔间全年耗时；比较 600 s 与 300 s 步长的状态差异；给出 N × 隔间 × 副本的可行组合与费用表
- [ ] `slowlab/agent_api.py`：子进程 + JSON 行协议包住 `CampaignExecutor.dispatch`；代理拿不到任何 Python 对象、文件、网络
- [ ] 公共工具集（查询、汇总、GP 拟合、候选预测）：独立工具随机种子；输入输出与数值预算冻结
- [ ] 测试：公开历史相同、私有 seed/latent/future 不同 → prompt 与工具输出逐字节相同
- [ ] 防火墙阻断改为硬错误并持久化；出站 payload 审计副本、hash、协议版本
- [ ] 实现支持过程观测、错峰启动/停止和资源约束的 GP/BO 策略；只在开发集调参
- [ ] 实现实验前推荐与固定管理参照；不调用私有 oracle
- [ ] 实现同一初始快照的 Full/Endpoint 分支与匹配预算；科学观测严格隔离
- [ ] 新的 LLM 循环（不复用 `legacy/v2.1` 的 TOMGRO harness）：决策点调用、上下文管理、失败处理

验收：完全相同的公开历史下更换隐藏真值不改变 prompt/tool output；全部方法通过相同设施验证器。

## Phase 3：site 分布、生成与评分协议冻结

- [ ] 定义 site = {设施参数扰动, 作物参数扰动, 深土壤边界温度（8–20 °C，合同 v4）, 价格情景, 天气年份块, 噪声种子}；写 `configs/site_distribution_v0.json`；保留 prior 很强的合理 sites
- [ ] 分开开发 / pilot / test sites；记录全部生成、排除与版本信息
- [ ] 私有 seed 带盐 commitment；artifact、文件名、日志附件防泄漏检查
- [ ] 隔离的推荐评测器：独立天气副本；Monte Carlo 误差小于要报告的差异
- [ ] best-known feasible reference 的数值搜索与搜索误差报告
- [ ] 冻结统计单位、主检验族、次要指标、CI、失败与缺失处理
- [ ] 用开发/pilot 方差做功效或精度分析，确定 N；冻结 E3 packet 选择规则与诊断 Reader

验收：在看正式结果前，能写出每个 figure 的输入、统计量与解释边界。

## Phase 4：小型 pilot 与运行预算

- [ ] fake-model 全路径测试，覆盖所有动作与失败路径
- [ ] 核验候选模型精确 ID、provider 路由、数据发送设置；禁用 provider 端隐藏推理
- [ ] 在少量 pilot sites 上核算每 campaign 调用、token、费用、格式失败与执行耗时
- [ ] 检查过程反馈是否有可操作的决策机会；只按接口/任务逻辑修复，不按 arm 输赢调参
- [ ] 预算表包含 pilot、重试、Reader 会话、初始推荐、离线评测；硬暂停策略；取得执行授权
- [ ] 锁定代码、配置、prompts、模型、工具与预注册文档（hash）

验收：预算表可由 pilot 日志重算；每个正式 identity 与暂停/恢复规则都已确定。

## Phase 5：一次性执行 E1、E2

- [ ] 按冻结配对与随机顺序运行正式 sites：先记录初始推荐，再从共同起点分支
- [ ] 保存 Design、Executor、Reader、API、费用与 blinded-payload 记录
- [ ] 审计缺失/重复 identity、实际模型/provider、资源会计与结果可重放性
- [ ] 对最终/初始/固定方案做独立离线评测；保存全部不利与失败结果
- [ ] 数据锁定后一次性计算主结果；报告协议偏离而不静默补改

验收：E1 Full 与 E2 Full 是同一批数据；无重复计数、事后排除或跨 provider 偷换。

## Phase 6：E3 与必要的机制核验

- [ ] 按预注册规则抽取事实 history packets（含提前停止与未完成结果）
- [ ] 两个 Reader 使用逐字节相同证据，独立会话、随机顺序，不见来源方法与评测分数
- [ ] 分析 Reader 替换与固定 Reader 的 history 效用；不强行做加性归因
- [ ] 完成已注册的 Executor 敏感性；无 LLM 鲁棒性实验支持的范围不作稳健性主张

## Phase 7：论文与复现材料

- [ ] 新建 `paper/`（v2.3），按 `RESEARCH_PLAN.md` §6 的图表与证据映射写作
- [ ] 每项主张可追溯到协议、日志与分析输出；`verify_results_claims` 式脚本
- [ ] 明确新稿是模拟自动实验研究；保留 GreenLight 外部验证失败与未校准执行器说明
- [ ] 数据锁定后发布可复现代码、seed reveal 与必要数据包；不发布密钥或私人 reviews
- [ ] `legacy/v2.1` 只作先导研究引用，不混入新版主统计
