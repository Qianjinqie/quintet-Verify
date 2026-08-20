Quintet-Verify (QV)

基于锚定认知架构的多智能体认知控制框架——用 A/B/C/D/E 五角色庭审式分工与私有认知轨迹（✓/?/✗），在推理时对智能体施加刚性制衡：不靠微调、不改权重，只约束"谁在什么时候、以什么权限、基于什么证据说话"。

与模型无关 · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 线上 Demo(已通过 workbuddy 部署)：https://a524b836ec13d5d9f.app.workbuddy.link
先行效果

📊 Fable 5 盲评结论

Fable 5 对本项目输出的盲评综合评分为 9.17/10，对选手 Fable5(评委不知道) 评分为 8.0/10，差距 +1.17。

三个任务全部领先：

· 代码审查：9.2 vs 8.5（选手 F5 遗漏了 rpop 丢消息、状态字段覆盖、task_id 可预测泄露等隐蔽问题） · 技术方案评估：9.5 vs 8.0（F5 在精排层将 pairwise 误算为 pointwise，误差致命） · 法律文书审查：8.8 vs 7.5（F5 未给出具体验证方案）

注：法律文书任务使用 DeepSeek-V4 Pro-0812；代码审查与技术方案评估使用 DeepSeek-V4 Flash-0731。框架采用标准模式-high 强度。

评委 F5 对本项目的定性结论：

技术深度、计算准确性、风险识别、数据支撑、专业严谨性五个维度上系统性优于被评价对象。适合投资决策、技术尽调、生产部署前审查、合规审计等高要求场景。

一眼看懂：五角色是怎么协作的？

￼
关键约束：

· A 不写正文，B 不改框架，C 不提供修改全文，D 不打回代笔，E 不直接改判 · 认知标记 ✓/?/✗ 作为私有轨迹存在，永远不进路由决策 · 打回必须带锚点坐标，返工只改指定区域（涟漪声明约束波及范围）

为什么不直接用一个大模型？

单一模型的"生成即定论"是幻觉直达终点的根因：它自己生成、自己审查、自己确认，监督者与被监督者是同一个立场。Quintet-Verify 把这套闭环拆成五个权限互锁的角色——

制衡从"自律"升级为"他律"：监督者与被监督者不共享立场，每一轮打回都被约束在锚点坐标内，返工不会引发全文漂移。

核心特性

· 双通道认知架构：public（路由可见）与 private Dense Track（✓/?/✗）物理隔离，三层防护（结构 / 类型 / 运行期断言）保证私有认知不进路由。 · 五角色刚性制衡：A 架构师 · B 执行者 · C 验证者 · D 裁判 · E 陪审员，权限互锁、各有权限禁区。 · 三条铁律（代码强制）：① 置信度隔离 ② 增量修改（锚点坐标 + 涟漪声明）③ 熔断禁言（max_iterations 110，D 可签 gag_order）。 · 强度分档：low / high（默认）/ max；从严时缺陷须附影响链、D 置信不足不得通过。 · Fast 模式：仅 B/C/D 三角，无提纲、无上诉支线，适合简单任务与低成本场景。 · 过拟合治理：C 无强制配额（允许零缺陷，须附覆盖说明）；D 可判定 C 吹毛求疵并打回重审（上限 2 次）。 · 插件化机制：PLUGINS 注册表 + flags 开关（默认全关），可选包 qv_extras.py 经 pre/post_act_hooks 标准化挂载。 · 自定义流程（v1.4）：前端编辑流程定义、新建角色并继承 AE 权限、静态校验、分角色独立 LLM 配置。

和别的多智能体框架有什么不同？

· AutoGen / CrewAI / MetaGPT：均支持多智能体协作、模型无关，但不具备推理时刚性制衡、私有认知轨迹审计、增量修改（省 Token）三大能力。 · Quintet-Verify：在同样支持多智能体协作与模型无关的基础上，额外提供： · 推理时刚性制衡（庭审式权限互锁） · 私有认知轨迹（Dense Track）可审计 · 增量修改（锚点坐标 + 涟漪声明，平均省 80% 上下文）

大多数框架在做"让 AI 协作"，Quintet-Verify 在做"让 AI 互相制衡"——前者追求产出效率，后者追求结论可信。

质量说明

相比直接使用单一模型，Quintet-Verify 的收益来自结构性制衡而非更大的参数量：

· 相比纯模型：幻觉不再直接抵达终点——它必须先通过 C 的审查和 D 的裁决，且每一轮打回都被约束在锚点坐标内。C 的零缺陷结论与 D 的过拟合判定同时防止"为挑刺而挑刺"的内耗。 · 相比纯 J-Space：J-Space 的稠密认知管理作用于单一模型内部，自我监督的天花板是"自己抓自己"。Quintet-Verify 把 ✓/?/✗ 保留为私有认知轨迹的同时，将裁决权交给独立角色，制衡从"自律"升级为"他律"。

上述为机制层面的定性对比；实战效果普遍大幅优于其基座模型，具体参见效果先行。

实跑数据

Full 模式真实运行（2026-08，DeepSeek-V4-Pro-0813 模型，法律文书审查评估任务）：

· Token 消耗：约 13 万 · API 调用：19 次 · 费用：约 ¥2.3 · 最终裁决：D 置信度 90/100 通过

产出结论为"混合架构条件性可行"：初稿中无数据支撑的经验断言在两轮打回后被全部清除或降级为待验证假设，验收标准被改写为可审计口径。

谁应该用？

· 正在做 RAG 应用 但被幻觉困扰的开发者 · 需要 可审计的 AI 决策 的法律、金融、医疗场景 · 想体验 多智能体制衡 但不想写复杂编排的研究者 · 对 AutoGen/CrewAI 协作效率满意，但对结论可信度有更高要求的团队 · 用 AI 写高质量代码的程序员 · 用 AI 辅助数学研究的研究者

快速开始

安装

git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt

 最小示例（真实 LLM）

from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

 1. 配置模型（OpenAI 兼容接口，五角共用或分角色配置）
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # 返回 (A, B, C, D, E) 五个角色

2. 构图并运行（interrupt_before_e=False 表示不在 E 前挂起等待用户反馈）
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

3. 读取结果
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # D 的裁决与置信度
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
无 LLM 的离线演示

from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="演示任务", user_satisfied=True))
)
print(final["public"].verdict)
Web 控制台


python server.py  # 默认 8000 端口，浏览器打开即可
控制台支持：标准 / Fast / 自定义三种流水线、强度分档、插件开关、分角色 API 配置、流程定义编辑与实时校验、角色提示词覆盖、运行过程可视化（事件流 / 草稿 / Dense Track）。

参与贡献

PR 与 Issue 均欢迎。提交前请先跑通 python server.py 与离线演示，并在 PR 中说明改动动机与实测结果。新手可从标记为 good first issue 的条目入手。

-🔬 理论溯源与独立性声明-

灵感来源
Quintet-Verify 在以下方面受到前期工作的启发：

· 认知标记符号（✓/?/✗）与"私有认知轨迹"（Dense Track）的命名：源自 J-Space Cognition Suite V3.6（doi:10.5281/zenodo.21971181）。该工作首次系统性地提出了在单一模型内部使用紧凑符号管理认知状态的方法，并验证了"内省标记"对提升推理质量的积极作用。 · 多角色庭审式制衡的组织形式：受到 b23.tv/z3C0vVM 展示的五者验证（Quintet Verification）多模型辩论体系的启发。该体系展示了"通过角色分离与异议申请制，将生成幻觉转化为共识博弈"的可行性。

我们对上述工作的原创贡献表示敬意与感谢。

根本性架构差异
尽管 Quintet-Verify 借用了 J-Space 的符号命名和五者验证的组织形式，其底层认知架构与上述工作存在不可化约的根本性区别。这些区别并非增量改进，而是对"认知主体"、"信息拓扑"和"控制性质"三个基本假设的重新定义：

认知主体

· J-Space（全局工作空间理论）：单一模型内部 · 五者验证（多模型辩论）：多个独立模型（无状态） · Quintet-Verify（锚定认知理论）：多个独立智能体（各有私有状态）

认知状态的载体

· J-Space：全局上下文窗口 · 五者验证：各模型的独立输出 · Quintet-Verify：public 公共状态 + private 私有轨迹（物理隔离）

信息流动方式

· J-Space：全局广播 + 竞争接入 · 五者验证：辩论式全量传输 · Quintet-Verify：锚定引用（通过 anchor_id + checksum 共享事实）

修改机制

· J-Space：全量重写（依赖内省矫正） · 五者验证：全量重写（依赖辩论修正） · Quintet-Verify：增量修改（精准坐标 + 涟漪声明，平均省 80% 上下文）

控制性质

· J-Space：提示词引导的软约束 · 五者验证：提示词引导的软约束 · Quintet-Verify：编译期校验 + 路由守卫的硬约束

恢复机制

· J-Space：账本文本回溯 · 五者验证：无标准化恢复 · Quintet-Verify：原子化状态检查点（可精确回滚至任意轮次）

制衡方式

· J-Space：自我监督（同一立场） · 五者验证：跨模型辩论（独立立场） · Quintet-Verify：权限互锁 + 庭审流程（立场分离 + 流程刚性）

审计能力

· J-Space：思维链文本（不可结构化回溯） · 五者验证：辩论文本（不可结构化回溯） · Quintet-Verify：Dense Track 结构化审计（可逐条追踪 ✓ 的验证依据与 ✗ 的证伪证据）

核心差异总结：

· J-Space 解决的是"一个人如何更好地思考"（单一意识的深度内省）； · 五者验证解决的是"一群人如何通过辩论达成共识"（多立场的外部制衡）； · Quintet-Verify 解决的是"一群有独立认知状态的智能体如何通过刚性规则收敛于可验证结论"（分布式认知的宪法治理）。

Quintet-Verify 的独特贡献在于：首次将"私有认知轨迹（Dense Track）"与"刚性权限矩阵"结合，使多智能体系统在保持个体推理深度的同时，获得可审计、可回溯、可强制终止的系统级可靠性。

独立性声明
基于上述根本性架构差异，Quintet-Verify 特此声明：

非衍生作品：Quintet-Verify 不是 J-Space Cognition Suite V3.6 的衍生作品或分支。两者采用完全不同的代码实现、认知拓扑与控制哲学。
独立知识产权：Quintet-Verify 的全部代码由本团队独立编写，未复制或翻译 J-Space 或五者验证的任何源代码、提示词模板或协议文案。
引用而非依赖：本系统将 J-Space 和五者验证列为"理论参照"而非"技术依赖"。移除这些参照不影响 Quintet-Verify 核心流程的完整运行。
学术诚信：本声明旨在准确追溯思想源头，同时明确界定本项目的原创贡献边界。我们鼓励读者将上述工作与本系统进行对照研究，以理解分布式认知架构的不同演化路径。
建议读者
· 若您关注"单一模型如何通过内省标记提升推理"，请参阅 J-Space Cognition Suite V3.6。 · 若您关注"多模型如何通过辩论达成共识"，请参阅五者验证体系。 · 若您关注"如何通过刚性规则让多智能体协作产生可审计、可制衡的结论"，请继续使用 Quintet-Verify。
