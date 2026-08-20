# Quintet-Verify (QV)

基于锚定认知架构的多智能体认知控制框架 —— 采用 A/B/C/D/E 五角色庭审式分工与私有认知轨迹（✓ / ? / ✗），在推理时对智能体施加刚性制衡：不依赖微调。

与模型无关 · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 线上 Demo（已通过 workbuddy 部署）：https://a524b836ec13d5d9f.app.workbuddy.link

---

目录

- [项目简介](#项目简介)
- [关键特性](#关键特性)
- [谁应该使用](#谁应该使用)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [最小示例（真实 LLM）](#最小示例真实-llm)
  - [无 LLM 的离线演示（stub）](#无-llm-的离线演示stub)
  - [Web 控制台](#web-控制台)
- [实跑数据](#实跑数据)
- [参与贡献](#参与贡献)
- [理论溯源与独立性声明](#理论溯源与独立性声明)
- [License](#license)

---

项目简介

Quintet-Verify（QV）是一套面向可审计 AI 决策的多智能体控制框架。框架通过五角庭审式角色分工与“私有认知轨迹（Dense Track）”的物理隔离，实现推理时的刚性制衡，目标是降低幻觉风险、提升输出可信度并支持精细化增量修改以节省 token。

关键特性

- 双通道认知架构：public（路由可见）与 private Dense Track（✓ / ? / ✗）物理隔离，三层防护（结构／类型／运行期断言）保证私有认知不进入路由。
- 五角色庭审流程（A/B/C/D/E）：职责分明，监督与被监督不共享立场，强化他律制衡。
- 可审计私有轨迹：每条私有标记可追溯验证依据（anchor_id + checksum）。
- 增量修改：精确定位修改坐标，平均节省大量上下文 token（实测 ~80% 优化示例在框架设计中）。
- 模型无关：可接入 OpenAI 兼容接口或其它支持的 LLM 服务。

Fable 5 盲评（节选）

- Fable 5 对本项目输出的盲评综合评分为 9.17/10，参考对象（盲评选手）评分 8.0/10，差距 +1.17。
- 三个任务均领先：代码审查、技术方案评估、法律文书任务。（详细对比见仓库内 benchmark/ 或 docs/）

---

谁应该使用

- 正在做 RAG 应用但被幻觉困扰的开发者
- 需要可审计 AI 决策的法律、金融、医疗场景
- 想体验多智能体制衡但不想写复杂编排的研究者

快速开始

安装

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

最小示例（真实 LLM）

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. 配置模型（OpenAI 兼容接口，五角共用或分角色配置）
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # 返回 (A, B, C, D, E)

# 2. 构图并运行（interrupt_before_e=False 表示不在 E 前挂起等待用户反馈）
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="评估某技术方案的可行性，给出结论。",
            user_satisfied=True,   # 满意则 E 同意判决，流程终止
            intensity="high",      # 强度分档：low / high / max
            flags=dict(DEFAULT_FLAGS),  # 插件开关，默认全关
        )
    )
)

# 3. 读取结果
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # D 的裁决与置信度
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

无 LLM 的离线演示（stub agents）

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="演示任务", user_satisfied=True))
)
print(final["public"].verdict)
```

Web 控制台

```bash
python server.py  # 默认 8000 端口，浏览器打开即可
```

控制台支持：标准 / Fast / 自定义 三种流水线、强度分档、插件开关、分角色 API 配置、流程定义编辑与实时校验、角色提示词覆盖、运行过程可视化等功能。

实跑数据（示例）

Full 模式真实运行（2026-08，DeepSeek-V4-Pro-0813 模型，法律文书审查评估任务）：

- Token 消耗：约 13 万
- API 调用：19 次
- 费用：约 ¥2.3
- 最终裁决：D 置信度 90/100（通过）

产出结论为“混合架构条件性可行”：初稿中无数据支撑的经验断言在两轮打回后被全部清除或降级为待验证假设，验收标准被改写为可审计口径。

参与贡献

欢迎 PR 与 Issue。提交前建议：

1. 在本地跑通 `python server.py` 与离线演示，并在 PR 中说明改动动机与实测结果。
2. 新手可从标记为 `good first issue` 的条目入手。

---

理论溯源与独立性声明（节选）

- 认知标记符号（✓ / ? / ✗）与“私有认知轨迹（Dense Track）”的命名：受 J-Space Cognition Suite V3.6（doi:10.5281/zenodo.21971181）启发。
- 尽管存在理论参照，Quintet-Verify 在架构、实现与控制哲学上保持独立（详见仓库文档/论文）。

License

本项目采用 MIT License（如需其他许可请在仓库中调整）。

---

如果你希望我把 README 翻译为英文版、添加 badges（CI / PyPI / License / Python version）或补充更详细的贡献指南/PR 模板、示例输出（截图或文本），我可以继续提交针对性的 PR。