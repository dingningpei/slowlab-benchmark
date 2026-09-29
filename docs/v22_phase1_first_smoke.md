# V2.2 Phase 1：首次短时接入测试

日期：2026-09-27。状态：**最小接入通过，暂停在重复模型加载的性能问题；Phase 1 未完成。**

## 实现范围

`slowlab/v22_greenlight_smoke.py` 检查四个锁定模型文件哈希，从合同构建 96 m² 几何/设备覆盖与显热边界修正，并计算采样保持的供暖、通风、CO₂ 和灯光指令。

`scripts/run_v22_greenlight_smoke.py` 使用一个隔间、固定合成天气、两个 300 秒步长。第一条控制从加载后的实际初始状态计算；第二步使用前一步末端状态。保留并传递所有作物/气候状态，直接读取 `states_sol`，不调用离线 save()/run() 重采样路径。温度记录进入 `OnlineObservations`。

这只是接入烟测原型，不是完整 campaign executor。当前气候传感器转换无噪声、无延迟，符合 v0；温度接入公开记录，RH/CO₂ 用于当前控制。Full/Endpoint 权限、多隔间调度、收获资源积分、停止/清理/重种尚未接入，不应从本次测试推断这些能力完成。

## 依赖与重现

原临时虚拟环境不能直接加载所需包。使用现有 Python 3.12 运行时，加 `/tmp/slowlab-v22-runtime-deps` 中的软件依赖及锁定源码。numexpr、SciPy、NumPy 从本地 pip 缓存补齐，无数据集下载。

最终版本：NumPy 1.26.4、SciPy 1.14.1、pandas 2.2.3、numexpr 2.10.2。最初尝试出现 NumPy/SciPy 版本警告，不作为最终验收数据；固定版本后重新运行，最终结果保存在 `configs/v22_phase1_first_smoke_result.json`。

```sh
PYTHONPATH=/tmp/slowlab-v22-runtime-deps:/private/tmp/greenlight-v2.0.5 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /Users/dingningpei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/run_v22_greenlight_smoke.py --source /private/tmp/greenlight-v2.0.5 --out /tmp/v22_phase1_smoke_result_pinned.json
```

以上是本机命令，不是可移植的已打包运行环境；未来持久运行需固定依赖环境。命令调用官方求解器但不调用 LLM API。

## 测量结果

- 模拟时间：06:00–06:10，两个控制步长；固定天气仅用于诊断。
- 总耗时 11.7613 秒；峰值 RSS 104611840 bytes，约 99.77 MiB（macOS ru_maxrss 字节口径）。
- 第一步加载 7.8196 秒（为取得初始观测加载两次），求解 0.1095 秒。
- 第二步加载 3.7766 秒，求解 0.0553 秒。
- 室温端点：18.3324、19.1714°C；RH：75.5506%、70.2783%；CO₂：461.5970、531.1236 ppm。
- 两个原始求解端点成功，输出有限，落在烟测宽松范围内。这不是整季物理稳定性、真实轨迹或控制精度验收。
- 40 项相关轻量测试通过：控制指令边界/方向/未来天气不干扰，既有时间观测、GreenLight 适配及合同约束。

## 停止原因与下一步

当前原型每个 5 分钟 tick 都重新解析/构造整个模型，而加载成本远高于积分成本。按第二步仅加载耗时直线外推，单隔间 180 天的 51840 个 tick 会花约 54.4 小时在加载上；这只是说明当前架构不能规模化，不是正式运行时预测，也不能据此宣称优化后会达到某个速度。

推荐下一步：每隔间只加载/编译一次，将控制命令转为运行期可更新的输入；在传感器边界续接状态并保留状态次序，避免每步重建表达式。先验证与本次同条件短轨迹等价、指令保持及未来数据不干扰，再重新测性能。不能为提速取消中途观察、偷偷把控制步长改成一天，或跳过因果接口。

依用户“完成阶段或遇到问题停下”要求，本轮停在此处。未启动 180 天或 365 天模拟，不进行持续高负载运行。Phase 1 其余项目保持未完成。
