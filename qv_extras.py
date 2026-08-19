"""qv-extras · 认知增强扩展包（可选插件包，v1.3）

设计契约：
- 主代码（quintet_verify.py）不依赖本包；检测到本包时自动 import 并调用 install()。
- 不安装本包，核心流程稳如泰山；安装后按 flags 自动挂载：
    pre_conclusion_bridge → B 的 pre_act_hook（Prompt 前插入"结论前桥接"规则）
    metacog_control       → D 的 post_act_hook（裁决后挂载"低置信度复算"）
    empirical_escape     → C 的 pre_act_hook（验证停滞时拆解候选集 + 差分测试）
    ledger_checkpoint    → D 的 pre_act_hook（终审前叠加编号化开放问题账本 + 接缝刷新，基于 LangGraph checkpointer 跨会话续审）
    cognitive_distillation→ 主代码在节点结束时调用 distill()（默认关闭）
"""

from __future__ import annotations

from quintet_verify import (
    CognitiveMark,
    PrivateCognition,
    QuintetState,
    Verdict,
)

EXTRAS_VERSION = "1.3"

BRIDGE_RULE = (
    "\n【结论前桥接·qv-extras】每个结论性段落必须先显式列出所依赖的中间论点"
    "（以'桥接：'开头标注），严禁先下结论再补理由。"
)

EMPIRICAL_ESCAPE_RULE = (
    "\n【经验逃逸·qv-extras】当你（验证者 C）推导停滞、或无法判定某条缺陷/断言的真伪时，禁止空想或含糊带过。"
    "必须执行经验逃逸：① 把该未知量拆解为【有限的候选断言集合】（2~4 条互斥假设），并【并行】为每条候选安排独立的验证通道；"
    "② 对每条候选给出【可证伪/可验证的检验方法】——用差分测试思路，说明在不同假设下预期会观察到什么差异（差异观测即判据）；"
    "③ 汇总各候选的验证结果后再下结论。任何'无法判定'都必须随附候选集与并行检验方案，否则视为审查不充分。"
)

LEDGER_RULE = (
    "\n【开放问题账本·qv-extras】说明：本框架的状态原子化落盘、可编号、可恢复已由 LangGraph checkpointer 自动保障。"
    "本插件在其之上叠加 J-Space 式的【开放问题编号 + 接缝刷新】层，便于跨会话续审："
    "结案前，从仍未解决的 ? 待决思考中提炼一份【编号化开放问题账本】，逐条编号（Q1 / Q2 / …），"
    "每条写明问题、当前最佳假设，以及【接缝刷新点】——即下次续审应从哪条证据/锚点重新核验。"
    "账本随结案推理一并输出，作为跨会话恢复的语义接缝。"
)

METACOG_FLOOR = 60  # D 通过但置信度低于此值 → 复算


# ----------------------------------------------------------------------
# ① B 的 pre_act_hook：结论前桥接规则注入
# ----------------------------------------------------------------------
def bridge_pre_hook(agent, state: QuintetState, ctx: dict) -> None:
    if state.public.flags.get("pre_conclusion_bridge") and BRIDGE_RULE not in agent.prompt_injections:
        agent.prompt_injections.append(BRIDGE_RULE)


# ----------------------------------------------------------------------
# ① D 的 post_act_hook：低置信度复算（裁决后挂载，保守裁决）
# ----------------------------------------------------------------------
def lowconf_recheck_hook(agent, state: QuintetState, ctx: dict, output):
    if not state.public.flags.get("metacog_control"):
        return output
    if not (isinstance(output, Verdict) and output.approved):
        return output
    if output.confidence_score >= METACOG_FLOOR:
        return output
    agent.think("初审通过但置信度低于阈值，触发独立复算", CognitiveMark.UNCERTAIN)
    try:
        again = agent.act(state, **ctx)
    except Exception as e:  # noqa: BLE001 - 复算失败不阻断主流程
        agent.think(f"复算失败：{type(e).__name__}，维持原判", CognitiveMark.UNCERTAIN)
        return output
    flip = again.approved != output.approved
    agent.think(
        f"复算完成：{'翻转' if flip else '一致'}（{output.confidence_score}→{again.confidence_score}）",
        CognitiveMark.VERIFIED,
        verifier="qv-extras 低置信复算",
    )
    # 保守裁决：任一打回则采纳打回；都通过则取较低置信度
    if not again.approved:
        return again
    if again.confidence_score < output.confidence_score:
        return again
    return output


# ----------------------------------------------------------------------
# ④ C 的 pre_act_hook：经验逃逸与验证（推导停滞时拆解候选集 + 差分测试）
# ----------------------------------------------------------------------
def escape_pre_hook(agent, state: QuintetState, ctx: dict) -> None:
    if state.public.flags.get("empirical_escape") and agent.role == "C":
        if EMPIRICAL_ESCAPE_RULE not in agent.prompt_injections:
            agent.prompt_injections.append(EMPIRICAL_ESCAPE_RULE)


# ----------------------------------------------------------------------
# ⑥ D 的 pre_act_hook：账本与检查点外化（编号化开放问题 + 接缝刷新，支持跨会话续审）
# ----------------------------------------------------------------------
def ledger_pre_hook(agent, state: QuintetState, ctx: dict) -> None:
    if not state.public.flags.get("ledger_checkpoint"):
        return
    if agent.role != "D":
        return
    if ctx.get("mode") != "final":  # 仅在终审推理中要求账本
        return
    if LEDGER_RULE not in agent.prompt_injections:
        agent.prompt_injections.append(LEDGER_RULE)


# ----------------------------------------------------------------------
# ② 认知蒸馏（Cognitive Distillation，默认关闭）
# ----------------------------------------------------------------------
def distill(
    private: PrivateCognition,
    threshold: int = 20,
    keep_recent: int = 5,
) -> PrivateCognition:
    """dense_track 超过阈值时，把历史轨迹凝练为两张清单：

    - 已验证事实清单（✓ → claim + 具名依据）
    - 已证伪假设清单（✗ → claim + 证伪证据）
    归档旧轨迹，仅保留最近 keep_recent 条 ? 待决思考。
    """
    track = private.dense_track
    if len(track) <= threshold:
        return private
    verified = list(private.verified_facts)
    refuted = list(private.refuted_hypotheses)
    pending: list = []
    for atom in track:
        if atom.mark is CognitiveMark.VERIFIED:
            item = f"{atom.claim}（依据：{atom.verifier}）"
            if item not in verified:
                verified.append(item)
        elif atom.mark is CognitiveMark.REFUTED:
            item = f"{atom.claim}（证伪：{atom.killing_evidence}）"
            if item not in refuted:
                refuted.append(item)
        else:
            pending.append(atom)
    return PrivateCognition(
        dense_track=pending[-keep_recent:],
        verified_facts=verified,
        refuted_hypotheses=refuted,
    )


# ----------------------------------------------------------------------
# 安装入口：主代码检测到本包后调用
# ----------------------------------------------------------------------
def install(agents) -> None:
    """给 Agent 实例挂载钩子（幂等）。钩子内部按 state.public.flags 决定是否生效。"""
    for agent in agents:
        if agent.role == "B" and bridge_pre_hook not in agent.pre_act_hooks:
            agent.pre_act_hooks.append(bridge_pre_hook)
        if agent.role == "C" and escape_pre_hook not in agent.pre_act_hooks:
            agent.pre_act_hooks.append(escape_pre_hook)
        if agent.role == "D" and lowconf_recheck_hook not in agent.post_act_hooks:
            agent.post_act_hooks.append(lowconf_recheck_hook)
        if agent.role == "D" and ledger_pre_hook not in agent.pre_act_hooks:
            agent.pre_act_hooks.append(ledger_pre_hook)
