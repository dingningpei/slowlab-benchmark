# 四隔间第 240 天 native RHS 溢出：精确定位与修复验证

日期：2026-09-28。首次四隔间全年 pilot 在第 240.756944 天、unit 1、tick 69,338 失败，旧 native RHS 报 FE_OVERFLOW=8；已保存完整试探状态、当前输入、提交状态和两个 RHS 导数，见 `results/v22/remote_four_unit_annual_diagnostic_result.json`。该重放用 2017 Cabauw 天气和开发合同，不是正式 LLM 实验。失败进程并非触及内存/时间限制。

同一 LSODA 试探状态下，参考 Python RHS 对 `exp` 发出 overflow warning，但最终 28 个导数均有限；旧 native RHS 最终导数也均有限，两者最大绝对差为 7.63e-17。逐命令诊断仅找到一处溢出，见 `results/v22/native_exp_fault_diagnosis.json`：凝结通量中 `1/(1+exp(x))` 的指数参数极大，完整 sigmoid 仍数学上趋于零。这是实现层中间项的数值稳定性问题，不能概括为 GreenLight 物理模型已失效，也不能因此宣称全年环境稳定。

`slowlab/v22/native_rhs.py` 的受限 AST 编译器现在把精确模式 `1/(1+np.exp(x))` 映射为按 `x` 符号分支的代数等价稳定计算：`x>=0` 时用 `exp(-x)/(1+exp(-x))`，否则用 `1/(1+exp(x))`。其他非法/除零/非有限导数仍保持失败即停。极端正负输入编译回归通过。在原始故障试探状态上，修复后的 native RHS 不再报错，最终导数与参考 RHS 最大差 7.63e-17、与旧 native 实际导数一致，见 `results/v22/native_fault_fix_replay.json`。

这只解决**一处已观察到的试探状态**。必须进一步完成短程 native/reference 轨迹回归，再在同一固定天气、策略、设施和资源上限下重跑四隔间全年 pilot；若出现其他不同的数值故障，保留身份与状态后逐项处理。年度深土壤敏感性和正式代理矩阵仍以全年稳定性为先决条件。
