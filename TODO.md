# SlowLab v2.3 To-do

对应 `RESEARCH_PLAN.md` 的阶段。修改规则见 `CLAUDE.md`：勾选自己刚完成的项目不用问；其它改动先在对话里提出。
完成一项时在同一行末尾用 `→ 证据文件` 指出结果或审计文件。

## Phase 1：可验证的执行环境（收尾）

已通过：模型复用、编译 RHS、空隔间模式、因果观测、四隔间同步、单/四隔间全年、事件驱动全年活动、深土壤全年配对、2014 天气读取。剩余：

- [x] 冻结独立于 site/天气真值的测量噪声与缺测假设；实现同一 (隔间, 通道, 时刻) 只采样一次、可重复查询的缓存与审计 → `configs/sensor_noise_v0.json`、`results/sensor_noise_v0_prefix_check.json`
- [x] 冻结深土壤默认边界（8 或 20 °C）及其披露方式；写入合同 v4 → `configs/task_contract_v4.json`（开发期沿用原生 20 °C；Phase 3 起为 8–20 °C 私有 site 参数）
- [x] 执行器去掉四单元硬编码，policy 改为 schema 对象；从合同读取隔间数 → `slowlab/policy.py`、`tests/test_policy_and_units.py`
- [x] 范围审计改为区分物理错误、来源外推、声明假设三类，不靠删 site 通过 → `configs/reality_audit_policy_v1.json`、`tests/test_reality_constraints.py`
- [x] 在 `pyproject.toml` 声明 GreenLight 为 pinned 可选依赖（git+commit）；保留 `--source` 路径方式 → `slowlab/greenlight_source.py`、`tests/test_greenlight_source.py`
- [x] 清点 `scripts/`：删除已被全年运行取代的一次性短测脚本，保留 verify/audit/annual 三类 → 45 个减为 23 个，`docs/` 14 篇减为 12 篇

验收：不用 LLM，脚本可完成含中途观察、改变下一实验、停止/重种的整场活动，并在不利结果下完整结算（已通过）；噪声与通用执行器补齐后重新跑一次四隔间全年并审计（**已通过，2026-09-30** → `results/noise_annual_result.json`、`results/noise_annual_audit.json`）。**Phase 1 完成。**

## Phase 2：算力预算、代理接口与强基线

- [x] 实测四隔间全年耗时与并行扩展（改为本地 i7-10700K WSL2，不租服务器）：两项不改数值的优化后每场约 0.52 h，12 worker 约 23 场/小时，1,024 场约 1.84 天 → `results/annual12_wsl_20260930_0df43c9.json`、`results/compute_scaling_wsl_20260930_0df43c9.json`
- [x] Phase 3 定出 N 后，给出 N × 隔间 × 副本（含评测副本、pilot、重跑）的算力表 → 约 2,700 核时，见决定记录 2026-10-04
- [x] `slowlab/agent_api.py`：子进程 + JSON 行协议包住 `CampaignExecutor.dispatch`；代理拿不到任何 Python 对象、文件、网络 → `slowlab/agent_protocol.py`、`executor_server.py`、`agent_client.py`，`tests/test_agent_api.py`，`results/agent_api_prefix_check.json`
- [x] 公共工具集（查询、汇总、GP 拟合、候选预测）：独立工具随机种子；输入输出与数值预算冻结 → `slowlab/tools.py`（analysis-tools-v1，最终锁定在 Phase 4）、`tests/test_tools.py`
- [x] 测试：公开历史相同、私有 seed/latent/future 不同 → prompt 与工具输出逐字节相同 → `tests/test_llm_agent.py`、`tests/test_tools.py`、`tests/test_agent_api.py`
- [x] 防火墙阻断改为硬错误并持久化；出站 payload 审计副本、hash、协议版本 → `slowlab/outbound_audit.py`
- [ ] 实现支持过程观测、错峰启动/停止和资源约束的 GP/BO 策略；只在开发集调参
  - [x] gp-bo-v2：两波与错峰两种调度，种植季节入 GP，EI 与推荐针对合同 v5 的两季平均评分 → `slowlab/bo_agent.py`、`results/bo_dev_runs_20260930.json`
  - [ ] 过程感知部分：用 Phase 3 开发 site 训练过程预测器后接入错峰调度（2026-09-30 决定；「至今毛利」在第 90 天几乎无信息）
  - [x] 主基线改为带先验的局部 BO（决定记录 2026-10-03）：在开发 site 数据上拟合并冻结 GP 超参数；实现以固定参照为起点、信赖域内求 EI、只在已种方案中推荐；按预定规则在开发 site 上选初始半径并检查合格线 → `configs/prior_bo_v1.json`、`results/prior_bo_radius_selection_20261004.json`
- [ ] 实现实验前推荐与固定管理参照；不调用私有 oracle
- [x] 实现同一初始快照的 Full/Endpoint 分支与匹配预算；科学观测严格隔离 → `slowlab/branching.py`、`tests/test_branching.py`（选项 A）
- [x] 新的 LLM 循环（不复用 `legacy/v2.1` 的 TOMGRO harness）：决策点调用、上下文管理、失败处理 → `slowlab/llm_agent.py`、`scripts/run_llm_campaign.py`

验收：完全相同的公开历史下更换隐藏真值不改变 prompt/tool output；全部方法通过相同设施验证器。

## Phase 3：site 分布、生成与评分协议冻结

- [x] 定义 site = {设施参数扰动, 作物参数扰动, 深土壤边界温度（8–20 °C，合同 v4）, 价格情景, 天气年份块, 噪声种子}；写 `configs/site_distribution_v0.json`；分布只依据物理与经济合理性事先定义；生成后不按任何与结果相关的性质（prior 强弱、学习空间、方法表现）筛选或调整；排除只按预注册的数值或机械规则进行，并全部报告。定稿为 `configs/site_distribution_v1.json`（决定记录 2026-10-01）
  - [x] 每个参数范围对照原始文献注明来源；无来源者标为声明假设。证据：`configs/site_distribution_v1.json`（每项 evidence 字段）
  - [x] 新的未触碰天气年份（2002–2011、2015，另含 2001-12 边界月；2021–2024 在 lc1 目录中缺月，不可用）：清单见 `configs/weather_formal_acquisition_proposal_v0.json`，已获批；样本月审计发现 2015-01 一条损坏记录，修复规则 v0 已获批（见决定记录），按该规则下载其余并审计。证据：`results/weather_formal_year_audit_v0.json`
- [x] 用开发 site 训练过程预测器（在季内记录→整季毛利），并作为公共工具提供给 LLM；BO 的错峰调度使用同一预测器（数据与验证已完成：40 个开发 site、788 茬；保留简单岭回归，见决定记录 2026-10-01；剩余：公共工具与 BO 接入）。证据：`results/process_predictor_validation_v2_20261001.json`、`results/bo_staggered_predictor_check_20261001.json`
- [x] 合同 v7 与 LLM 框架 v2（读取不计决策次数、最近 6 轮保留原始结果、2,000 字符笔记、按日汇总读取；见决定记录 2026-10-01），并用节制与大量读取两种 scripted 模型校准预算；校准后定为合同 v8（每场 500 万字符）。证据：`results/context_budget_calibration_20261001.json`
- [x] 分开开发 / pilot / test sites；记录全部生成、排除与版本信息。证据：`configs/site_partition_v1.json`（test 的 site 数待功效分析确定）
- [x] 私有 seed 带盐 commitment；artifact、文件名、日志附件防泄漏检查。证据：`configs/seed_commitment_pilot_v1.json`、`configs/seed_commitment_test_v1.json`；防泄漏扫描 `scripts/check_public_leaks.py`
- [x] 隔离的推荐评测器：独立天气副本；Monte Carlo 误差小于要报告的差异（设计已定，见决定记录 2026-09-30）；评测年数定为 3 年。证据：`results/evaluator_check_20260930.json`、`results/evaluation_error_study_20260930.json`
- [x] best-known feasible reference 的数值搜索与搜索误差报告（随机 8 个 test site，其中 2 个做两次；决定记录 2026-10-03） → 8 个 test site，搜索误差 1.6–2.1 EUR/m²；`results/phase6_analysis_20261005.json`
- [x] 冻结统计单位、主检验族、次要指标、CI、失败与缺失处理。见 RESEARCH_PLAN §5 与决定记录 2026-10-01
- [x] 用开发/pilot 方差做功效或精度分析，确定 N；冻结 E3 packet 选择规则与诊断 Reader（E3 规则已冻结，见决定记录 2026-10-01 与 `slowlab/history_packet.py`；开发方差研究运行中，pilot 的 LLM 方差已得（`results/pilot_evaluation_analysis_20261003.json`）；最终 N 待主基线在 pilot site 上补跑后的方差） → N = 48，见决定记录 2026-10-04 与 `results/pilot_evaluation_analysis_pbo_20261004.json`

验收：在看正式结果前，能写出每个 figure 的输入、统计量与解释边界。

## Phase 4：小型 pilot 与运行预算

- [x] fake-model 全路径测试，覆盖所有动作与失败路径。证据：`tests/test_full_path.py`（经 pilot 入口 `scripts/run_campaign_job.py`）
- [x] 核验候选模型精确 ID、provider 路由、数据发送设置；禁用 provider 端隐藏推理（模型与精确 ID 已定并按 OpenRouter 公开列表核对，见 `configs/pilot_models_v3.json`（DeepSeek 官方 API；MiMo、GLM 固定官方端点；GLM 推理必选、上限 1,024 token）；连通检查见 pilot 运行目录的 model_check 记录）。证据：`results/pilot_model_checks_20261001.json`、`results/pilot_provider_data_policies_20261002.json`
- [x] 在少量 pilot sites 上核算每 campaign 调用、token、费用、格式失败与执行耗时。证据：`results/pilot_accounting_20261002.json`
- [x] 检查过程反馈是否有可操作的决策机会；只按接口/任务逻辑修复，不按 arm 输赢调参 → 正式运行前未单独完成；用户 2026-10-05 决定以事后描述关闭：`results/formal_process_feedback_20261005.json`（无模型中途停种；约一半 campaign 的 Full 与 Endpoint 后续安排相同）
- [x] 在租用服务器上建立运行环境：核对软件版本与 GreenLight 哈希，跑全部测试，单个评测计时并与台式机结果对比（记录差异）；用户放置新的专用 API key → `results/server_parity_check_20261004.json`；服务器全部测试通过；用户已放置专用 key（连通检查见正式运行目录 model_check.json）
- [x] 冻结后的主基线在 8 个 pilot site 上各跑 2 个种子 × Full/Endpoint 并评测，用于确定 N（只花算力，不花 API 费用） → `results/pilot_evaluation_analysis_pbo_20261004.json`
- [x] 预算表包含 pilot、重试、Reader 会话、初始推荐、离线评测；硬暂停策略；取得执行授权 → 决定记录 2026-10-04；用户于 2026-10-04 授权，API 硬上限 30 美元
- [x] 锁定代码、配置、prompts、模型、工具与预注册文档（hash） → `configs/formal_lock_v1.json`（运行前）、`configs/formal_lock_v2.json`（分析代码）、`configs/formal_lock_v3.json`（Phase 6 代码）

验收：预算表可由 pilot 日志重算；每个正式 identity 与暂停/恢复规则都已确定。

## Phase 5：一次性执行 E1、E2

- [x] 按冻结配对与随机顺序运行正式 sites：先记录初始推荐，再从共同起点分支 → 480 场全部完成；运行顺序为 site 编号顺序而非随机顺序，见决定记录 2026-10-05（协议偏离）
- [x] 保存 Design、Executor、Reader、API、费用与 blinded-payload 记录 → 正式运行目录（campaign 记录与对话、出站审计、逐次调用记录、费用账本、结算、E3 Reader 记录）已拷回 `~/slowlab-data/formal-20261004/` 并核对哈希；私有模拟轨迹（60 GB）未拷回，可确定性重算（已验证：`results/formal_trace_replay_check_20261005.json`）
- [x] 审计缺失/重复 identity、实际模型/provider、资源会计与结果可重放性 → `results/formal_audit_20261005.json`（全部通过）、`results/formal_replay_check_20261005.json`（8 个随机评测重跑逐字节一致）
- [x] 对最终/初始/固定方案做独立离线评测；保存全部不利与失败结果 → 2,880 个评测全部完成、无失败；`results/formal_analysis_e1e2_20261005.json`
- [x] 数据锁定后一次性计算主结果；报告协议偏离而不静默补改 → `results/formal_analysis_e1e2_20261005.json`（锁定代码 v2，运行一次）；偏离见决定记录 2026-10-05
- [x] 服务器格式化前，把全部结果、费用账本与审计记录复制回本地并核对哈希 → `~/slowlab-data/formal-20261004/`（22,007 个文件，整体哈希与服务器一致，清单 MANIFEST.sha256）；代码在 Git 中；私有模拟轨迹 60 GB 未拷回（确定性可重算）（已验证：`results/formal_trace_replay_check_20261005.json`）
- [x] 字段顺序错误的修复与重跑（决定记录 2026-10-05） → `results/repair_run_record_20261006.json`
  - [x] 修复与测试 → `ac75ed0`（`tests/test_field_order.py`）
  - [x] 新主机逐字节一致性检查 → `results/server2_parity_check_20261005.json`
  - [x] 开发 site 上重选半径 → `results/prior_bo_radius_selection_repair_20261006.json`（0.35）
  - [x] 26 个 LLM Full 分支断点续跑（含逐字节重放验证） → `results/repair_run_record_20261006.json`
  - [x] 主基线 192 场重跑并评测 → `results/repair_run_record_20261006.json`
  - [x] E3 受影响的数据包与边界传热敏感性重做 → `results/repaired_phase6_analysis_20261006.json`
  - [x] lock v5；用锁定分析代码在修复后数据上运行一次 → `configs/formal_lock_v8.json`；`results/repaired_formal_analysis_e1e2_20261006.json`
  - [x] 论文：主结果改用修复后的数据，附录报告原始运行与错误 → `7f197aa`（附录 `app:original`）

验收：E1 Full 与 E2 Full 是同一批数据；无重复计数、事后排除或跨 provider 偷换。

## Phase 6：E3 与必要的机制核验

- [x] 按预注册规则抽取事实 history packets（含提前停止与未完成结果）（64 个，4 个方法各 16 个，全部 Reader 读同一批；决定记录 2026-10-03） → 64 个（承诺种子，`scripts/e3_readers.py`）；`results/phase6_analysis_20261005.json`
- [x] 两个 Reader 使用逐字节相同证据，独立会话、随机顺序，不见来源方法与评测分数 → 同一数据包字节送给全部 Reader，每次读取为独立无状态会话，不含方法名与得分；读取按模型 × 数据包的固定顺序并行进行（会话之间无状态，顺序不影响结果），未随机化
- [x] 分析 Reader 替换与固定 Reader 的 history 效用；不强行做加性归因 → `results/phase6_analysis_20261005.json`（锁定代码 v3，运行一次）
- [x] 完成已注册的敏感性：边界传热（16 个 test site × 5 个 Full 最终推荐，Ueff 0 与 4）与干物质比例（由已有结果换算）；其余敏感性不做，论文不作相应的稳健性主张 → `results/phase6_analysis_20261005.json`；最优参照搜索见同一文件
  - [ ] 控制步长 600 s 对 300 s（2026-09-30 自 Phase 2 移入：算力已不需要更长步长；需先把写死的 300 s 改为合同参数，并区分控制频率与观测频率，用评测器比较得分与方案排序）（2026-10-03 决定不做）

## Phase 7：论文与复现材料

- [ ] 新建 `paper/`（v2.3），按 `RESEARCH_PLAN.md` §6 的图表与证据映射写作
- [x] 每项主张可追溯到协议、日志与分析输出；`verify_results_claims` 式脚本 → `scripts/make_paper_numbers.py` 从结果文件生成全部数字，`scripts/verify_results_claims.py` 检查无手写或过时数字
- [x] 明确新稿是模拟自动实验研究；保留 GreenLight 外部验证失败与未校准执行器说明 → `paper/sections/07_limitations.tex` 与附录 B（AGC 2019 外部检验失败）
- [ ] 数据锁定后发布可复现代码、seed reveal 与必要数据包；不发布密钥或私人 reviews（2026-10-05 用户决定：种子开封与数据包在录用后公开；投稿时按 D&B 要求向审稿人提供匿名访问，不含种子开封文件）
  - [ ] arXiv 预印本（NeurIPS 模板 preprint 选项）与代码仓库同时公开，公开前运行防泄漏检查（决定记录 2026-10-06）
  - [ ] 投稿 NeurIPS 2027 E&D：数据托管于指定平台并附 Croissant，仅审稿人可访问，不含种子开封文件
- [ ] `legacy/v2.1` 只作先导研究引用，不混入新版主统计
