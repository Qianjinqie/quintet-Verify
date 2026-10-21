[English](docs/README.en.md) | **简体中文** | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [Español](docs/README.es.md) | [Français](docs/README.fr.md) | [Deutsch](docs/README.de.md) | [Português (BR)](docs/README.pt-BR.md) | [Русский](docs/README.ru.md)

# Quintet-Verify (QV)

**硬约束多智能体推理** · 9.17/10 盲评 · ~$0.36/次 · Python 3.11+

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Stars](https://img.shields.io/github/stars/Qianjinqie/quintet-Verify?style=social)](https://github.com/Qianjinqie/quintet-Verify/stargazers)
![Cost](https://img.shields.io/badge/cost-~$0.36%2Frun-yellowgreen)
![BlindEval](https://img.shields.io/badge/Fable%205%20Blind%20Eval-9.17%2F10-brightgreen)

一个基于锚定认知架构的多智能体认知控制框架——通过五角色（A/B/C/D/E）法庭式分工与私有认知痕迹（✓/?/✗），在推理时对智能体施加刚性制衡：不做微调、不改权重，只约束“谁在何时发言、以何种权限发言、基于何种证据发言”。

- 模型无关 · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 在线演示（通过 workbuddy 部署）：https://a524b836ec13d5d9f.app.workbuddy.link

---

## 目录

- [结果先行](#结果先行)
  - [Fable 5 盲评结论](#fable-5-盲评结论)
- [关键约束](#关键约束)
- [为什么不直接使用一个大模型？](#为什么不直接使用一个大模型)
- [核心特性](#核心特性)
- [它与其他多智能体框架有何不同？](#它与其他多智能体框架有何不同)
- [质量说明](#质量说明)
- [实测数据](#实测数据)
- [谁适合使用它？](#谁适合使用它)
- [插件](#插件)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [最小示例（真实 LLM）](#最小示例真实-llm)
  - [无 LLM 的离线演示](#无-llm-的离线演示)
  - [Web 控制台](#web-控制台)
- [理论来源与独立性声明](#理论来源与独立性声明)
  - [灵感来源](#灵感来源)
  - [根本架构差异](#根本架构差异)
  - [核心差异总结](#核心差异总结)
  - [独立性声明](#独立性声明)

---

## 结果先行

📊 **Fable 5 盲评结论**

**评测设置**——Quintet-Verify 运行在 DeepSeek-V4-Flash-0731 上（一个拥有 130 亿激活参数的轻量版本；法律文书任务使用 DeepSeek-V4-Pro-0813），而参赛者为Fable5，需要特别注意的是，评委Fable5和参赛者Fable5之间并不知道），不借助任何框架直接运行。评测模型只能看到 A、B 两个匿名标签——换句话说，在强模型打分之下，由 Quintet-Verify 驱动的轻量模型胜过了 F5。

Fable 5 的盲评给本项目打出 9.17/10 的总分，而参赛者（Fable 5）得到 8.0/10——领先 +1.17。

三项任务全部获胜：

- 代码评审：9.2 对 8.5（参赛者 F5 漏掉了 rpop 消息丢失、state 字段被覆盖、task_id 可预测泄露等细微问题）  
- 技术方案评估：9.5 对 8.0（F5 在排序层把 pairwise 误算成了 pointwise——致命错误）  
- 法律文书评审：8.8 对 7.5（F5 未提供具体的核验方案）

注：法律文书任务使用 DeepSeek‑V4‑Pro‑0813；代码评审与技术评估使用 DeepSeek‑V4‑Flash‑0731。框架以标准模式——高强度运行。

评测方（F5）给出的定性结论：

> 被测系统在技术深度、计算准确性、风险识别、数据支撑与专业严谨性五个维度上均优于被评对象。适用于投资决策、技术尽职调查、预生产部署评审、合规审计等高风险场景。

注：实际使用中，QV 会发起大量 API 调用，产生较高成本——它以金钱和时间换取质量。

---

## 关键约束

- A 不撰写正文，B 不改动框架，C 不给出全文修改，D 不代他人执笔，E 不直接推翻裁定。  
- 认知标记 ✓/?/✗ 以私有痕迹的形式存在，绝不进入路由决策。  
- 发回重审必须附带锚点坐标；返工仅限指定区域（涟漪声明约束修改范围）。

---

## 为什么不直接使用一个大模型？

单个模型“生成即定稿”是幻觉直达终点的根源：它自己生成、自己评审、自己确认——监督者与被监督者立场同一。Quintet‑Verify 把这一闭环拆解为五个权限相互锁定的角色——

制衡从“自律”演化为“他律”：监督者与被监督者立场不再同一；每次发回重审都被约束在锚点坐标之内，返工绝不会导致全文漂移。

---

## 核心特性

- **双通道认知架构**：公开通道（对路由可见）与私有 Dense Track（✓/?/✗）物理隔离，三层防护（结构 / 类型 / 运行时断言）确保私有认知绝不进入路由。  
- **五角色刚性制衡**：A（架构师）· B（执行者）· C（核验者）· D（裁决者）· E（陪审员）——每个角色都有锁定的权限与禁区。  
- **三条铁律（由代码强制执行）**：① 置信隔离 ② 增量修改（锚点坐标 + 涟漪声明）③ 熔断禁言令（max_iterations: 1–10，默认 5；D 可下达禁言令）。  
- **强度分级**：low / high（默认）/ max；在严苛模式下，缺陷必须附带影响链，且 D 在置信不足时不得放行。  
- **快速模式**：仅 B/C/D 三角，无大纲、无申诉分支——适合简单任务与低成本场景。  
- **过拟合治理**：C 没有强制配额（允许零缺陷，但必须附上覆盖面声明）；D 可认定 C 过于吹毛求疵并将其发回复审（最多 2 次）。  
- **插件机制**：PLUGINS 注册表 + flags（默认全部关闭）；可选的 qv_extras.py 可通过标准化的 pre/post_act_hooks 挂载——见[插件](#插件)。  
- **可自定义工作流（v1.4）**：前端编辑工作流定义、创建新角色并继承 AE 权限、静态校验，以及每角色独立的 LLM 配置。

---

## 它与其他多智能体框架有何不同？

- **AutoGen / CrewAI / MetaGPT**：均支持多智能体协作且模型无关，但缺少三种能力：推理时刚性制衡、私有认知痕迹审计、增量修改（节省 token）。  
- **Quintet‑Verify**：在支持多智能体协作与模型无关的基础上，额外提供：  
  - 推理时刚性制衡（法庭式权限互锁）  
  - 可审计的私有认知痕迹（Dense Track）  
  - 增量修改（锚点坐标 + 涟漪声明，平均节省 ~80% 上下文）

大多数框架做的是“让 AI 协作”；Quintet‑Verify 做的是“让 AI 相互制衡”——前者追求输出效率，后者追求结论可信度。

---

## 质量说明

与直接使用单个模型相比，Quintet‑Verify 的收益来自结构性制衡，而非更大的参数量：

- **与纯模型相比**：幻觉不再直达终点——必须先经过 C 的评审与 D 的裁定，且每次发回重审都被约束在锚点坐标之内。C 的零缺陷结论与 D 的过拟合判定同时防止了“为挑刺而挑刺”的内耗。  
- **与纯 J‑Space 相比**：J‑Space 的密集认知管理运行在单个模型内部，自我监督的天花板是“自己抓到自己”。Quintet‑Verify 保留 ✓/?/✗ 作为私有认知痕迹，同时把裁决权交给独立角色——将制衡从自律升级为他律。

以上是机制层面的定性比较；实际效果通常远优于其底座模型——详见“结果先行”一节。

---

## 实测数据

完整模式实测运行（2026 年 8 月，DeepSeek‑V4‑Pro‑0813，法律文书评审任务）：

- Token 消耗：~130,000  
- API 调用：19  
- 成本：~¥2.3（~$0.36）  
- 最终裁定：D 以 90/100 的置信度放行

输出结论为“混合架构下有条件可行”：初稿中缺乏数据支撑的经验性断言，经过两轮发回重审后被整体删除或降级为待验证假设；验收标准被改写为可审计的条款。

---

## 谁适合使用它？

- 正在构建 RAG 应用却被幻觉困扰的开发者  
- 需要 AI 决策可审计的法律、金融与医疗场景  
- 想体验多智能体制衡又不想编写复杂编排的研究者  
- 对 AutoGen/CrewAI 的协作效率感到满意、但对结论可信度要求更高的团队  
- 使用 AI 编写高质量代码的程序员  
- 使用 AI 辅助数学研究的研究者

---

## 插件

- **插件包**：可选的 `qv_extras.py` 在 [Quintet-Verify-plugin](https://github.com/Qianjinqie/Quintet-Verify-plugin) 单独发布——把它放到 `quintet_verify.py` 旁边，导入时会自动挂载（无需配置）。  
- **没有插件包时**：核心工作流照常运行，7 个插件 flag 只是保持关闭。

---

## 快速开始

### 安装

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt
```

### 最小示例（真实 LLM）

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. 配置模型（OpenAI 兼容端点；可共享或按角色配置）
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # 返回五个角色（A、B、C、D、E）

# 2. 构建图并运行（interrupt_before_e=False 表示在 E 之前不暂停等待用户反馈）
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="Evaluate the feasibility of a technical solution and provide a conclusion.",
            user_satisfied=True,   # 满意 → E 同意裁定，工作流结束
            intensity="high",      # 强度分级：low / high / max
            flags=dict(DEFAULT_FLAGS),  # 插件开关，默认全部关闭
        )
    )
)

# 3. 读取结果
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # D 的裁定与置信度
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

### 无 LLM 的离线演示

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="Demo task", user_satisfied=True))
)
print(final["public"].verdict)
```

### Web 控制台

```bash
python server.py  # 默认端口 8000，在浏览器中打开
```

控制台支持：标准 / 快速 / 自定义流水线、强度分级、插件开关、按角色的 API 配置、工作流定义编辑与实时校验、角色提示词覆盖，以及运行可视化（事件流 / 草稿 / Dense Track）。

---

## 理论来源与独立性声明

### 灵感来源

Quintet‑Verify 在以下方面汲取了先前工作的灵感：

· 认知标记符号（✓/?/✗）与“私有认知痕迹”（Dense Track）的命名：源自 J‑Space Cognition Suite V3.6（doi:10.5281/zenodo.21971181）。该工作首次系统性地提出在单个模型内部使用紧凑符号管理认知状态，并验证了“内省标记”对推理质量的正向作用。
· 多角色法庭式制衡的组织形式：受 b23.tv/z3C0vVM 展示的 Quintet Verification 多模型辩论系统启发，该系统展示了“通过角色分离与异议程序将生成幻觉转化为共识博弈”的可行性。

我们对上述工作的原创贡献致以敬意与感谢。

### 根本架构差异

尽管 Quintet‑Verify 借用了 J‑Space 的符号命名与 Quintet Verification 的组织形式，但其底层认知架构与两者存在不可还原的差异。这些差异并非渐进式改进，而是对三个根本假设的重新定义：认知主体、信息拓扑与控制性质。

| 维度 | J‑Space（全局工作空间理论） | Quintet Verification（多模型辩论） | Quintet‑Verify（锚定认知理论） |
|---|---|---|---|
| 认知主体 | 单个模型内部 | 多个独立模型（无状态） | 多个独立智能体（各自拥有私有状态） |
| 认知状态的载体 | 全局上下文窗口 | 各模型的独立输出 | 公开共享状态 + 私有痕迹（物理隔离） |
| 信息流 | 全局广播 + 竞争性访问 | 辩论式全量传输 | 锚定引用（共享事实经由 anchor_id + 校验和） |
| 修改机制 | 全量重写（依赖内省修正） | 全量重写（依赖辩论修正） | 增量修改（精确坐标 + 涟漪声明） |
| 控制性质 | 提示词引导的软约束 | 提示词引导的软约束 | 编译期校验 + 路由守卫（硬约束） |
| 恢复机制 | 文本台账回滚 | 无标准化恢复 | 原子状态检查点（精确回滚到任意轮次） |
| 制衡 | 自我监督（立场同一） | 跨模型辩论（立场独立） | 权限互锁 + 法庭程序（立场分离 + 程序刚性） |
| 可审计性 | 思维链文本（非结构化回滚） | 辩论文本（非结构化回滚） | 结构化 Dense Track 审计（✓ 的核验证据与 ✗ 的证伪证据均可追溯） |

### 核心差异总结

· J‑Space 解决“一个人如何思考得更好”（单一意识的深度内省）；
· Quintet Verification 解决“一个群体如何通过辩论达成共识”（借助多个立场实现外部制衡）；
· Quintet‑Verify 解决“一群拥有独立认知状态的智能体如何通过刚性规则收敛到可验证的结论”（分布式认知的宪制治理）。

Quintet‑Verify 的独特贡献在于首次将“私有认知痕迹（Dense Track）”与“刚性权限矩阵”相结合，使多智能体系统在保持个体推理深度的同时，实现可审计、可追溯、可强制终止的系统级可靠性。

### 独立性声明

基于上述根本架构差异，Quintet‑Verify 在此声明：

· 非衍生作品：Quintet‑Verify 不是 J‑Space Cognition Suite V3.6 的衍生作品或 fork。两者采用完全不同的代码实现、认知拓扑与控制哲学。
· 独立知识产权：Quintet‑Verify 的全部代码由我们团队独立编写；未复制或翻译任何来自 J‑Space 或 Quintet Verification 的源代码、提示词模板或协议文本。
· 引用而非依赖：本系统将 J‑Space 与 Quintet Verification 列为“理论参考”而非“技术依赖”。移除这些引用不影响 Quintet‑Verify 核心工作流的完整运行。
· 学术诚信：本声明旨在准确追溯思想来源，同时清晰界定我们原创贡献的边界。我们鼓励读者在上述工作与本系统之间开展比较研究，以理解分布式认知架构的不同演化路径。

给读者的建议：

· 如果你对“单个模型如何通过内省标记改进推理”感兴趣，请参阅 J‑Space Cognition Suite V3.6。
· 如果你对“多个模型如何通过辩论达成共识”感兴趣，请参阅 Quintet Verification 系统。
· 如果你对“如何让多智能体协作通过刚性规则产出可审计、经过核验的结论”感兴趣，请继续使用 Quintet‑Verify。

**注：其他语言版本如有翻译错误，请以本版本（简体中文）为准**
