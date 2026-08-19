"""Quintet-Verify · 单文件完整版（合并 models/schemas + agents/base + agents/llm + graph/workflow）

融合内部稠密认知（✓/?/✗）与外部刚性制衡（A/B/C/D/E 五角色庭审式验证）。

三条铁律（代码强制）：
  1. 置信度隔离：private 通道（Dense Track）严禁进入路由边（routing_guard 三层防护）
  2. 增量修改：B 返工必须带精准修改坐标 + 涟漪声明（是否波及 A 的冻结区）
  3. 熔断禁言：max_iterations 默认 5（1~10 有界可调）；D 可签发终审禁言令

运行 Stub 演示（无需 LLM）：
    python3.11 quintet_verify.py
作为库使用：
    from quintet_verify import build_graph, build_llm_agents, LLMConfig, ...
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any, Callable, ClassVar, Literal, TypeVar

import requests
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, model_validator

T = TypeVar("T", bound=BaseModel)


MAX_CRITIC_REWORK = 2  # C 因过拟合被打回重审的上限；超限后 D 必须直接裁决

# 4. 插件化：可选认知增强机制，默认关闭，前端逐项开启
PLUGINS: dict[str, str] = {
    "pre_conclusion_bridge": "结论前桥接：B 的结论段必须先引用已激活的中间论点",
    "broadcast_hub": "广播枢纽：共享事实/约束由 A 推导一次，B/C/D 同源读取",
    "metacog_control": "元认知控制即动作：D 低置信通过时强制 C 独立复核一轮",
    "tiered_gating": "三档门控：简单任务跳过 A 直接进 B",
    "empirical_escape": "经验逃逸与验证：C 推导停滞或无法判定真伪时，拆解为有限候选断言集并行差分验证，而非空想",
    "ledger_checkpoint": "账本与检查点外化：基于 LangGraph checkpointer 已有原子化落盘，叠加编号化开放问题账本与接缝刷新点，支持跨会话续审",
    "cognitive_distillation": "认知蒸馏：dense_track 超阈值自动凝练为已验证/已证伪清单",
}
DEFAULT_FLAGS: dict[str, bool] = {k: False for k in PLUGINS}  # 默认全关，前端逐项开启

# 架构强度分档（①）：追加到角色提示词的强度修饰
INTENSITY_SUFFIX: dict[str, str] = {
    "low": "\n【强度 low】精简执行：只处理高置信的实质问题，从宽裁决，仅打回明确错误。",
    "high": "",
    "max": "\n【强度 max】最高标准：逐句深挖、从严裁决；每条缺陷须附影响链分析；"
           "D 置信度 <85 不得通过；A 必须给出至少两条常规思路和一条非常规新思路。",
}
METACOG_CONFIDENCE_FLOOR = 60  # 插件3：D 通过但置信度低于此值 → 强制 C 独立复核


# ======================================================================
# Part 1 · models/schemas.py —— 全部 Pydantic v2 类（设计文档第 4 节）
# ======================================================================


class CognitiveMark(str, Enum):
    """稠密轨认识状态标记。"""

    VERIFIED = "✓"  # 已验证，须附具名验证依据
    UNCERTAIN = "?"  # 已提出，尚不可作为下游前提
    REFUTED = "✗"  # 已证伪，须保留证伪证据


class ThoughtAtom(BaseModel):
    """稠密轨最小单元，每行必须能无损展开为自然语言。"""

    agent: str  # 单字母角色代号（A~E 或用户自定义角色）
    claim: str
    mark: CognitiveMark
    verifier: str | None = None  # ✓ 必填
    killing_evidence: str | None = None  # ✗ 必填
    expanded: str  # 无损展开的自然语言（可解码性保证）

    @model_validator(mode="after")
    def _evidence_required(self) -> "ThoughtAtom":
        if self.mark is CognitiveMark.VERIFIED and not self.verifier:
            raise ValueError("✓ 必须附带具名验证依据")
        if self.mark is CognitiveMark.REFUTED and not self.killing_evidence:
            raise ValueError("✗ 必须保留证伪证据")
        return self


class PrivateCognition(BaseModel):
    """铁律1载体：此通道【严禁】进入任何路由边。"""

    dense_track: list[ThoughtAtom] = Field(default_factory=list)
    verified_facts: list[str] = Field(default_factory=list)      # 认知蒸馏：已验证事实清单
    refuted_hypotheses: list[str] = Field(default_factory=list)  # 认知蒸馏：已证伪假设清单


class OutlineNode(BaseModel):
    """A 的产出：只有骨架，不写细节。"""

    anchor_id: str
    title: str
    intent: str
    frozen: bool = False
    checksum: str | None = None


class Outline(BaseModel):
    task_restated: str
    nodes: list[OutlineNode]

    def anchor_ids(self) -> set[str]:
        return {n.anchor_id for n in self.nodes}

    def frozen_anchor_ids(self) -> set[str]:
        return {n.anchor_id for n in self.nodes if n.frozen}


class DefectSeverity(str, Enum):
    S = "S"  # 根本性逻辑错误 → D 必须受理
    A = "A"  # 非核心数据瑕疵 → D 酌情
    B = "B"  # 措辞/格式 → D 可当庭驳回


class Defect(BaseModel):
    severity: DefectSeverity
    at_anchor: str
    description: str
    impact_score: Annotated[int, Field(ge=1, le=10)]
    confidence_pct: Annotated[int, Field(ge=0, le=100)]


class CriticReport(BaseModel):
    """C 的产出：尽力审查（允许零缺陷），但必须说明覆盖范围供 D 做过拟合判定。"""

    defects: list[Defect] = Field(default_factory=list)
    coverage: str  # 审查覆盖说明：查了哪些锚点/哪些方面（零缺陷时的问责依据）


class DraftSection(BaseModel):
    anchor_id: str
    content: str


class RevisionCoordinate(BaseModel):
    """精准修改坐标（锚点级定位，不用行号）。"""

    anchor_id: str
    operation: Literal["rewrite", "patch", "delete"]
    instruction: str


class RippleDeclaration(BaseModel):
    """铁律2载体：B 返工必填。"""

    affects_frozen_zone: bool
    impacted_anchor_ids: list[str] = Field(default_factory=list)
    justification: str


class Draft(BaseModel):
    sections: list[DraftSection]
    revision_of: int | None = None  # 初稿 None
    applied_coordinates: list[RevisionCoordinate] = Field(default_factory=list)
    ripple: RippleDeclaration | None = None  # 返工必填

    @model_validator(mode="after")
    def _rework_requirements(self) -> "Draft":
        if self.revision_of is not None:
            if self.ripple is None:
                raise ValueError("返工草稿必须附带涟漪声明（铁律2）")
            if not self.applied_coordinates:
                raise ValueError("返工草稿必须记录实际修改坐标（铁律2）")
        return self


class Verdict(BaseModel):
    """D 的初审判决。打回分两种：给 B 修改坐标，或判 C 过拟合打回重审。"""

    approved: bool
    confidence_score: Annotated[int, Field(ge=0, le=100)]
    remand_coordinates: list[RevisionCoordinate] = Field(default_factory=list)
    critic_overfit: bool = False  # 过拟合判定：C 的缺陷系凑数/吹毛求疵
    overfit_reasoning: str | None = None

    @model_validator(mode="after")
    def _remand_requires_target(self) -> "Verdict":
        if self.critic_overfit and not self.overfit_reasoning:
            raise ValueError("判定 C 过拟合必须给出理由")
        if not self.approved and not self.critic_overfit and not self.remand_coordinates:
            raise ValueError("打回 B 必须指定精准修改坐标")
        if not self.approved and self.critic_overfit and self.remand_coordinates:
            raise ValueError("打回 C（过拟合）与打回 B（坐标）不可同时成立")
        return self


class Appeal(BaseModel):
    """E 的异议书：仅有申请权，不得直接改判。"""

    severity: DefectSeverity
    grounds: str
    new_evidence: bool


class FinalRuling(BaseModel):
    """D 的终审：可签发禁言令（铁律3）。"""

    upheld: bool
    gag_order: bool = False
    reasoning: str
    remand_coordinates: list[RevisionCoordinate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _overrule_requires_coordinates(self) -> "FinalRuling":
        # 改判场景初审往往是"通过"，verdict 无坐标；改判必须自带坐标（铁律2）
        if not self.upheld and not self.remand_coordinates:
            raise ValueError("采纳异议改判时必须指定精准修改坐标")
        return self


class PublicState(BaseModel):
    """路由边【唯一】可见的状态投影。"""

    task: str
    outline: Outline | None = None
    draft: Draft | None = None
    critic_report: CriticReport | None = None
    verdict: Verdict | None = None
    appeal: Appeal | None = None
    final_ruling: FinalRuling | None = None
    iteration: int = 0
    critic_iteration: int = 0  # C 因过拟合被打回重审的次数（上限 MAX_CRITIC_REWORK）
    # ---- 运行配置（前端可调）----
    pipeline: Literal["standard", "fast"] = "standard"  # standard=A~E全流程 / fast=仅B/C/D
    intensity: Literal["low", "high", "max"] = "high"   # 架构强度分档
    flags: dict[str, bool] = Field(default_factory=dict)  # 插件开关（见 PLUGINS）
    hub: str | None = None              # 广播枢纽：共享事实/约束（插件2）
    metacog_rechecked: bool = False     # 元认知复核是否已执行（插件3）
    # 铁律3：熔断上限。默认 5，可调但有界（1~10），超出范围校验失败
    max_iterations: Annotated[int, Field(ge=1, le=10)] = 5
    user_satisfied: bool | None = None


class QuintetState(BaseModel):
    """完整图状态 = 公共通道 + 私有通道（物理隔离）。"""

    public: PublicState
    private: PrivateCognition = Field(default_factory=PrivateCognition)


# ======================================================================
# Part 2 · agents/base.py —— 五角色基类 + 权限校验 + 私有认知通道
# ======================================================================

Role = Literal["A", "B", "C", "D", "E"]


class PermissionViolation(RuntimeError):
    """角色越权：触犯权限禁区。"""


class BaseQuintetAgent(ABC):
    """五角色公共基类：私有认知通道封装 + 权限校验管线 + 标准化挂载点（v1.3）。

    pre_act_hooks:  (agent, state, ctx) -> None       —— act 之前（可注入 prompt_injections）
    post_act_hooks: (agent, state, ctx, output) -> output —— act 之后（可替换产出，如低置信复算）
    """

    role: ClassVar[Role]
    pre_act_hooks: ClassVar[list] = []
    post_act_hooks: ClassVar[list] = []

    def __init__(self) -> None:
        self._dense_track: list[ThoughtAtom] = []
        self.pre_act_hooks = list(type(self).pre_act_hooks)   # 实例级拷贝，防跨实例污染
        self.post_act_hooks = list(type(self).post_act_hooks)
        self.prompt_injections: list[str] = []                # qv-extras 的 Prompt 注入缓冲区

    def think(
        self,
        claim: str,
        mark: CognitiveMark,
        *,
        verifier: str | None = None,
        killing_evidence: str | None = None,
        expanded: str | None = None,
    ) -> ThoughtAtom:
        atom = ThoughtAtom(
            agent=self.role,
            claim=claim,
            mark=mark,
            verifier=verifier,
            killing_evidence=killing_evidence,
            expanded=expanded if expanded is not None else claim,
        )
        self._dense_track.append(atom)
        return atom

    def _drain_thoughts(self) -> list[ThoughtAtom]:
        thoughts, self._dense_track = self._dense_track, []
        return thoughts

    def invoke(self, state: QuintetState, **ctx: Any) -> dict[str, Any]:
        for hook in self.pre_act_hooks:  # v1.3 挂载点
            hook(self, state, ctx)
        output = self.act(state, **ctx)
        for hook in self.post_act_hooks:  # v1.3 挂载点
            output = hook(self, state, ctx, output)
        self.check_permissions(state, output)
        public = self.apply(state.public.model_copy(deep=True), output)
        private = PrivateCognition(
            dense_track=[*state.private.dense_track, *self._drain_thoughts()],
            verified_facts=list(state.private.verified_facts),
            refuted_hypotheses=list(state.private.refuted_hypotheses),
        )
        if _QV_EXTRAS is not None and public.flags.get("cognitive_distillation"):
            private = _QV_EXTRAS.distill(private)  # 认知蒸馏（默认关闭）
        return {"public": public, "private": private}

    @abstractmethod
    def act(self, state: QuintetState, **ctx: Any) -> Any: ...

    @abstractmethod
    def apply(self, public: PublicState, output: Any) -> PublicState: ...

    @abstractmethod
    def check_permissions(self, state: QuintetState, output: Any) -> None: ...


class ArchitectAgent(BaseQuintetAgent):
    """A · 架构师：仅输出提纲/冻结区，不写细节。"""

    role: ClassVar[Role] = "A"


    def act(self, state: QuintetState, **ctx: Any) -> Any:
        raise NotImplementedError("角色基类仅承载权限语义（apply/check_permissions），请使用 LLM/Stub 子类")

    MAX_INTENT_CHARS: ClassVar[int] = 200

    def apply(self, public: PublicState, output: Outline) -> PublicState:
        public.outline = self._seal_freeze_zones(output)
        if public.flags.get("broadcast_hub"):
            # 广播枢纽：共享事实/约束只推导一次，B/C/D 从同一枢纽读取
            frozen = [f"{n.anchor_id}《{n.title}》" for n in output.nodes if n.frozen]
            public.hub = (
                f"任务：{output.task_restated}\n"
                f"锚点序列：{[n.anchor_id for n in output.nodes]}\n"
                f"冻结区（不可擅改）：{frozen or '无'}"
            )
        return public

    def check_permissions(self, state: QuintetState, output: Any) -> None:
        if not isinstance(output, Outline):
            raise PermissionViolation("A 越权：产出必须是 Outline（仅提纲）")
        ids = [n.anchor_id for n in output.nodes]
        if len(set(ids)) != len(ids):
            raise PermissionViolation("A 产出非法：anchor_id 重复")
        for node in output.nodes:
            if len(node.intent) > self.MAX_INTENT_CHARS:
                raise PermissionViolation(
                    f"A 越权：节点 {node.anchor_id} 的 intent 包含细节内容"
                )

    @staticmethod
    def _seal_freeze_zones(outline: Outline) -> Outline:
        for node in outline.nodes:
            if node.frozen:
                payload = f"{node.anchor_id}|{node.title}|{node.intent}"
                node.checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return outline


class BuilderAgent(BaseQuintetAgent):
    """B · 执行者：严格遵循 A 填内容；返工仅限修改坐标 + 涟漪声明。"""

    role: ClassVar[Role] = "B"


    def act(self, state: QuintetState, **ctx: Any) -> Any:
        raise NotImplementedError("角色基类仅承载权限语义（apply/check_permissions），请使用 LLM/Stub 子类")
    @staticmethod
    def remand_coordinates_of(state: QuintetState) -> list[RevisionCoordinate]:
        """B 返工的唯一合法坐标来源：终审改判坐标 > 初审打回坐标。"""
        ruling = state.public.final_ruling
        if ruling is not None and not ruling.upheld and ruling.remand_coordinates:
            return ruling.remand_coordinates
        verdict = state.public.verdict
        if verdict is not None and not verdict.approved:
            return verdict.remand_coordinates
        return []

    def apply(self, public: PublicState, output: Draft) -> PublicState:
        if output.revision_of is not None:
            public.iteration += 1  # 返工计数（铁律3 熔断依据）
            # 返工合并而非整体覆盖：保留旧 draft 中未被本次修改坐标覆盖的章节，
            # 并按 A 提纲顺序重排，避免初稿其他维度在返工后被抹掉（修复"返工覆盖"缺陷）
            old = public.draft
            if old is not None and old.sections:
                new_by_anchor = {s.anchor_id: s for s in output.sections}
                kept = [s for s in old.sections if s.anchor_id not in new_by_anchor]
                merged = kept + list(output.sections)
                order = public.outline.anchor_ids() if public.outline else None
                if order:
                    idx = {a: i for i, a in enumerate(order)}
                    merged.sort(key=lambda s: idx.get(s.anchor_id, len(order)))
                output = Draft(
                    sections=merged,
                    revision_of=output.revision_of,
                    applied_coordinates=output.applied_coordinates,
                    ripple=output.ripple,
                )
        public.draft = output
        return public

    def check_permissions(self, state: QuintetState, output: Any) -> None:
        if not isinstance(output, Draft):
            raise PermissionViolation("B 越权：产出必须是 Draft")
        outline = state.public.outline
        if outline is None:
            # fast 模式无 A：B 自规划段落，豁免锚点校验
            if state.public.pipeline != "fast":
                raise PermissionViolation("B 无法工作：A 的提纲不存在")
        else:
            anchors = outline.anchor_ids()
            for sec in output.sections:
                if sec.anchor_id not in anchors:
                    raise PermissionViolation(
                        f"B 越权：章节 {sec.anchor_id} 不对应 A 的任何锚点"
                    )
        if output.revision_of is not None:
            coordinates = self.remand_coordinates_of(state)
            if not coordinates:
                raise PermissionViolation("B 非法：无打回判决/改判却提交返工草稿")
            allowed = {c.anchor_id for c in coordinates}
            touched = {c.anchor_id for c in output.applied_coordinates}
            if not touched <= allowed:
                raise PermissionViolation(
                    f"B 越权：改动了修改坐标之外的位置 {sorted(touched - allowed)}"
                )
            frozen_hit = touched & (outline.frozen_anchor_ids() if outline else set())
            if frozen_hit:
                raise PermissionViolation(
                    f"B 越权：修改触及 A 的冻结区 {sorted(frozen_hit)}"
                )


class CriticAgent(BaseQuintetAgent):
    """C · 验证者：专找茬（配额由 CriticReport._quota 强制）。"""

    role: ClassVar[Role] = "C"


    def act(self, state: QuintetState, **ctx: Any) -> Any:
        raise NotImplementedError("角色基类仅承载权限语义（apply/check_permissions），请使用 LLM/Stub 子类")
    def apply(self, public: PublicState, output: CriticReport) -> PublicState:
        if public.verdict is not None and public.verdict.critic_overfit:
            public.critic_iteration += 1  # 本次系过拟合打回后的重审
        if (
            public.verdict is not None
            and public.verdict.approved
            and public.verdict.confidence_score < METACOG_CONFIDENCE_FLOOR
            and public.flags.get("metacog_control")
        ):
            # 仅当本次审查确系"低置信通过"触发的复核时才标记，
            # 防止改判返工后的常规审查被误标记而跳过后续元认知动作
            public.metacog_rechecked = True
        public.critic_report = output
        return public

    def check_permissions(self, state: QuintetState, output: Any) -> None:
        if not isinstance(output, CriticReport):
            raise PermissionViolation("C 越权：产出必须是 CriticReport（只提意见）")
        draft = state.public.draft
        if draft is None:
            raise PermissionViolation("C 无法工作：B 的草稿不存在")
        valid = {s.anchor_id for s in draft.sections}
        for d in output.defects:
            if d.at_anchor not in valid:
                raise PermissionViolation(f"C 非法：缺陷锚点 {d.at_anchor} 不存在")
        # 涟漪声明"无波及" → 仅可复核修改坐标（铁律2）
        if (
            draft.revision_of is not None
            and draft.ripple is not None
            and not draft.ripple.affects_frozen_zone
        ):
            allowed = {c.anchor_id for c in draft.applied_coordinates}
            out = [d.at_anchor for d in output.defects if d.at_anchor not in allowed]
            if out:
                raise PermissionViolation(
                    f"C 越权：涟漪声明无波及，仅可复核修改坐标，越界 {out}"
                )


class JudgeAgent(BaseQuintetAgent):
    """D · 裁判：终审权；打回/改判必须给坐标；可签发禁言令。"""

    role: ClassVar[Role] = "D"


    def act(self, state: QuintetState, **ctx: Any) -> Any:
        raise NotImplementedError("角色基类仅承载权限语义（apply/check_permissions），请使用 LLM/Stub 子类")

    MAX_INSTRUCTION_CHARS: ClassVar[int] = 600  # 放宽阈值：避免 D 的打回指令偶发超长被纯长度拦截误判越权而中断 run

    def apply(self, public: PublicState, output: Verdict | FinalRuling) -> PublicState:
        if isinstance(output, Verdict):
            # 新一轮初审：清空上一轮终审与异议残留（状态卫生）
            public.verdict = output
            public.appeal = None
            public.final_ruling = None
        elif isinstance(output, FinalRuling):
            public.final_ruling = output
        else:  # pragma: no cover
            raise PermissionViolation("D 越权：产出必须是 Verdict 或 FinalRuling")
        return public

    def check_permissions(self, state: QuintetState, output: Any) -> None:
        if not isinstance(output, (Verdict, FinalRuling)):
            raise PermissionViolation("D 越权：产出必须是 Verdict 或 FinalRuling")
        frozen = (
            state.public.outline.frozen_anchor_ids() if state.public.outline else set()
        )
        for c in output.remand_coordinates:
            if len(c.instruction) > self.MAX_INSTRUCTION_CHARS:
                raise PermissionViolation("D 越权：修改指令疑似替 B 重写内容")
            if c.anchor_id in frozen:
                raise PermissionViolation(
                    f"D 非法：修改坐标 {c.anchor_id} 指向 A 的冻结区"
                )


class JurorAgent(BaseQuintetAgent):
    """E · 陪审员：仅有异议申请权，不得直接改判。"""

    role: ClassVar[Role] = "E"


    def act(self, state: QuintetState, **ctx: Any) -> Any:
        raise NotImplementedError("角色基类仅承载权限语义（apply/check_permissions），请使用 LLM/Stub 子类")
    def apply(self, public: PublicState, output: Appeal | None) -> PublicState:
        public.appeal = output  # None = 同意判决
        return public

    def check_permissions(self, state: QuintetState, output: Any) -> None:
        if output is not None and not isinstance(output, Appeal):
            raise PermissionViolation("E 越权：仅可提交异议书，不得直接改判")


# ======================================================================
# Part 3 · agents/llm.py —— LLM 驱动五角色（OpenAI 兼容接口）
# ======================================================================

COMMON_RULES = """\
你是 Quintet-Verify 多智能体框架中的角色 {role}（{role_name}）。
铁律：
1. 你的内部思考（thoughts）使用 ✓/?/✗ 标记：✓ 必须给 verifier（具名验证依据），
   ✗ 必须给 killing_evidence（证伪证据），? 表示未验证不可作下游前提。
   thoughts 是私有的，绝不进入 output。
2. 严格遵守你的权限边界，越权输出会被系统拒绝。
输出契约：只输出一个 JSON 对象，结构为
{{"thoughts": [...], "output": <按给定 JSON Schema>}}，不要输出任何其他文字。
"""

ROLE_PROMPTS = {
    "A": (
        "架构师",
        "仅输出提纲（Outline）：每个节点含 anchor_id/title/intent/frozen。"
        "intent 只写'本节要解决什么'，严禁写具体答案。可把核心框架节点标记 frozen=true（冻结区）。",
    ),
    "B": (
        "执行者",
        "严格按 A 的提纲逐锚点填充内容（Draft），不得改动框架。"
        "若输入含【返工指令】：只修改指定坐标，输出 applied_coordinates 与 ripple"
        "（涟漪声明：是否波及冻结区），revision_of 填返工轮次。",
    ),
    "C": (
        "验证者",
        "尽力审查（CriticReport）：找出所有真实缺陷，每条含 severity/at_anchor/description/"
        "impact_score(1-10)/confidence_pct(0-100)。允许零缺陷（defects 为空），但任何情况下"
        "都必须填 coverage 说明审查覆盖范围。严禁为凑数而虚构或夸大缺陷——D 会做过拟合判定，"
        "被判过拟合将打回重审。只提意见，不提供修改后全文。若输入含涟漪声明且 "
        "affects_frozen_zone=false，只审查修改坐标范围内的锚点。",
    ),
    "D": (
        "裁判",
        "初审输出 Verdict：approved + confidence_score(0-100)。"
        "打回 B（approved=false）必须给 remand_coordinates（精准修改坐标），"
        "坐标不得指向冻结区，instruction 是修改指令而非重写全文。"
        "【过拟合判定】若 C 的缺陷属于凑数/吹毛求疵/重复已修复点/无实质影响，"
        "置 critic_overfit=true 并给 overfit_reasoning，此时打回对象是 C 而非 B，"
        "不得同时给 remand_coordinates；若草稿本身可接受，也可直接 approved=true。"
        "终审输出 FinalRuling：upheld/gag_order/reasoning/remand_coordinates；"
        "采纳异议改判（upheld=false）必须给 remand_coordinates；"
        "异议为 B 级或无新证据时可 gag_order=true 签发禁言令。",
    ),
    "E": (
        "陪审员",
        "仅有异议申请权：output 为 Appeal（severity/grounds/new_evidence）或 null（同意判决）。"
        "不得直接改判。",
    ),
}


class LLMConfig(BaseModel):
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    system_prompt: str | None = None  # Beta：自定义系统提示词（覆盖内置角色提示词）
    inherits_from: str | None = None  # v1.4：权限继承（A~E），获得父角色校验/写回语义
    temperature: float = 0.2
    timeout_s: int = 180
    max_retries: int = 2
    backoff_s: float = 2.0  # 网络/解析重试的指数退避基数


def _chat(cfg: LLMConfig, system: str, user: str) -> str:
    resp = requests.post(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        json={
            "model": cfg.model,
            "temperature": cfg.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    # 兼容推理模型（如 deepseek-v4-flash）：content 为空时回落 reasoning_content
    return msg.get("content") or msg.get("reasoning_content") or ""


def _extract_json(raw: str) -> str:
    """容错提取 JSON：模型可能在 JSON 外面裹 markdown 代码围栏。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class _LLMMixin:
    cfg: LLMConfig

    def _ask(self, schema: type[T], user_prompt: str) -> T | None:
        custom = getattr(self.cfg, "system_prompt", None)
        if custom:  # Beta：前端自定义系统提示词（含自定义角色）
            role_name, role_rule = "自定义", custom
        else:
            role_name, role_rule = ROLE_PROMPTS[self.role]  # type: ignore[attr-defined]
        system = COMMON_RULES.format(role=self.role, role_name=role_name) + "\n" + role_rule  # type: ignore[attr-defined]
        system += f"\noutput 的 JSON Schema：{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        if self.role == "E":  # type: ignore[attr-defined]
            system += "\n（E 的 output 允许为 null，表示同意判决不上诉）"

        last_err: Exception | None = None
        prompt = user_prompt
        for attempt in range(self.cfg.max_retries + 1):
            if attempt:
                time.sleep(self.cfg.backoff_s * attempt)  # 退避后再重试
            try:
                raw = _chat(self.cfg, system, prompt)
                data = json.loads(_extract_json(raw))
                self._ingest_thoughts(data.get("thoughts") or [])
                out = data.get("output")
                if out is None:
                    if self.role == "E" or schema is Appeal:  # type: ignore[attr-defined]
                        return None  # E 或继承 E 的角色允许"不上诉"
                    raise ValueError("output 为 null（仅 E 系角色允许）")
                return schema.model_validate(out)
            except requests.RequestException as e:
                last_err = e  # 网络层失败：原样重试，不改提示词
            except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
                last_err = e  # 结构化校验失败：带错反馈给模型重试
                prompt = (
                    f"{user_prompt}\n\n【系统驳回】你上一轮的输出未通过结构化校验：{e}\n"
                    "请修正后重新输出完整 JSON。"
                )
        raise RuntimeError(f"角色 {self.role} 连续失败：{last_err}")  # type: ignore[attr-defined]

    @staticmethod
    def _mark_from_claim(claim: str) -> tuple[CognitiveMark, str]:
        """模型常把 ✓/✗ 写进 claim 文本而非 mark 字段——从文本提取并剥离。"""
        for sym, mark in (("✓", CognitiveMark.VERIFIED), ("✗", CognitiveMark.REFUTED)):
            if claim.lstrip().startswith(sym):
                return mark, claim.lstrip()[1:].strip()
        return CognitiveMark.UNCERTAIN, claim

    def _ingest_thoughts(self, thoughts: list[Any]) -> None:
        """LLM 的内部思考写入私有 Dense Track（容错 + 证据强制不稀释）。"""
        for t in thoughts:
            if isinstance(t, str):
                t = {"claim": t}
            if not isinstance(t, dict):
                continue
            claim = str(t.get("claim", "")).strip()
            if not claim:
                continue
            mark_raw = t.get("mark")
            if mark_raw in ("✓", "?", "✗"):
                mark = CognitiveMark(mark_raw)
            else:
                mark, claim = self._mark_from_claim(claim)
            try:
                self.think(  # type: ignore[attr-defined]
                    claim=claim[:200],
                    mark=mark,
                    verifier=t.get("verifier"),
                    killing_evidence=t.get("killing_evidence"),
                )
            except (ValidationError, ValueError):
                self.think(claim[:200], CognitiveMark.UNCERTAIN)  # type: ignore[attr-defined]

    def _mods(self, state: QuintetState) -> str:
        """强度分档 + 插件对提示词的修饰（在公共状态上读取，不进私有通道）。"""
        mods = INTENSITY_SUFFIX.get(state.public.intensity, "")
        if state.public.pipeline == "fast":
            mods = mods.replace("；A 必须给出至少两条常规思路和一条非常规新思路。", "")  # fast 无 A
        mods += "".join(getattr(self, "prompt_injections", []))  # qv-extras 注入
        return mods


def _state_digest(state: QuintetState) -> str:
    """公共通道完整序列化为 LLM 输入（private 绝不进入提示词，铁律1）。"""
    return state.public.model_dump_json(indent=2)


class LLMArchitect(_LLMMixin, ArchitectAgent):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def act(self, state: QuintetState, **ctx) -> Outline:
        return self._ask(Outline, f"【用户任务】{state.public.task}" + self._mods(state))  # type: ignore[return-value]


class LLMBuilder(_LLMMixin, BuilderAgent):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def act(self, state: QuintetState, **ctx) -> Draft:
        coords = self.remand_coordinates_of(state)
        prompt = f"【公共状态】{_state_digest(state)}"
        if coords:
            prompt += (
                "\n\n【返工指令】仅允许修改以下坐标，并必须输出 applied_coordinates、"
                "ripple（涟漪声明）与 revision_of：\n"
                + json.dumps([c.model_dump() for c in coords], ensure_ascii=False, indent=2)
            )
        return self._ask(Draft, prompt + self._mods(state))  # type: ignore[return-value]


class LLMCritic(_LLMMixin, CriticAgent):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def act(self, state: QuintetState, **ctx) -> CriticReport:
        return self._ask(CriticReport, f"【公共状态】{_state_digest(state)}" + self._mods(state))  # type: ignore[return-value]


class LLMJudge(_LLMMixin, JudgeAgent):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def act(self, state: QuintetState, mode: str = "initial", **ctx) -> Verdict | FinalRuling:
        prompt = f"【公共状态】{_state_digest(state)}" + self._mods(state)
        if mode == "initial":
            return self._ask(Verdict, prompt + "\n\n请输出初审判决 Verdict。")  # type: ignore[return-value]
        return self._ask(FinalRuling, prompt + "\n\n请针对 E 的异议输出终审判决 FinalRuling。")  # type: ignore[return-value]


class LLMJuror(_LLMMixin, JurorAgent):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def act(self, state: QuintetState, **ctx) -> Appeal | None:
        prompt = f"【公共状态】{_state_digest(state)}"
        if state.public.user_satisfied is False:
            prompt += "\n\n【用户反馈】用户对结果不满意，请审阅并决定是否提交异议书（output 可为 null）。"
        else:
            prompt += "\n\n请审阅 D 的判决，决定是否提交异议书（output 可为 null）。"
        return self._ask(Appeal, prompt)


def build_llm_agents(
    cfgs: dict[str, LLMConfig] | LLMConfig,
) -> tuple[LLMArchitect, LLMBuilder, LLMCritic, LLMJudge, LLMJuror]:
    """传单个 LLMConfig → 五角共用；传 dict（键 "A"~"E"）→ 分角色模型。"""
    if isinstance(cfgs, LLMConfig):
        m = {r: cfgs for r in "ABCDE"}
    else:
        default = next(iter(cfgs.values()))
        m = {r: cfgs.get(r, default) for r in "ABCDE"}
    return (
        LLMArchitect(m["A"]),
        LLMBuilder(m["B"]),
        LLMCritic(m["C"]),
        LLMJudge(m["D"]),
        LLMJuror(m["E"]),
    )


# ======================================================================
# Part 4 · graph/workflow.py —— LangGraph 图 + 路由守卫（设计文档第 5 节）
# ======================================================================


def view(state: QuintetState) -> PublicState:
    """状态投影：路由边只拿 public，private 不可达。"""
    return state.public


def routing_guard(fn: Callable[[PublicState], str]) -> Callable[[Any], str]:
    """条件边守卫三层防护：静态源码检查 → 投影 → 运行期断言。

    注意：不能用 @wraps —— 会把 fn 的 PublicState 签名暴露给
    LangGraph 分支类型推断，导致框架按 PublicState 重建入参。
    """
    try:
        src = inspect.getsource(fn)
        if "private" in src or "dense_track" in src:
            raise AssertionError(f"路由函数 {fn.__name__} 源码引用了私有认知通道")
    except (OSError, TypeError):
        pass

    def wrapper(state):  # 无类型标注：LangGraph 原样传入图状态
        quintet = state if isinstance(state, QuintetState) else QuintetState(**state)
        public = view(quintet)
        assert isinstance(public, PublicState), "路由越权访问私有认知通道"
        return fn(public)

    return wrapper


# ---- 插件5：三档门控（简单任务跳过 A） ----
_HEAVY_KEYWORDS = ("评估", "分析", "报告", "方案", "论证", "比较", "设计", "审查", "可行性")


def _is_simple_task(task: str) -> bool:
    return len(task) <= 60 and not any(k in task for k in _HEAVY_KEYWORDS)


def gate_node(state: QuintetState) -> dict:
    """门控：插件开启且任务简单 → 自动生成单锚点提纲直通 B；否则放行到 A。"""
    pub = state.public.model_copy(deep=True)
    if pub.flags.get("tiered_gating") and _is_simple_task(pub.task):
        pub.outline = Outline(
            task_restated=pub.task,
            nodes=[OutlineNode(anchor_id="§1", title="直接回答", intent="覆盖任务全部要点")],
        )
    return {"public": pub, "private": state.private}


@routing_guard
def route_after_gate(public: PublicState) -> str:
    return "b_builder" if public.outline is not None else "a_architect"


@routing_guard
def route_after_verdict(public: PublicState) -> str:
    """D 初审后：通过+满意→END；通过+不满/未表态→E；打回→熔断检查→B。"""
    verdict = public.verdict
    assert verdict is not None, "路由前置条件：verdict 必须存在"
    if verdict.approved:
        # 插件3：低置信通过不是终点，触发动作——C 强制独立复核一轮
        if (
            public.flags.get("metacog_control")
            and not public.metacog_rechecked
            and verdict.confidence_score < METACOG_CONFIDENCE_FLOOR
        ):
            return "c_critic"
        if public.pipeline != "fast":
            return "e_juror"  # Full 模式必经 E 审阅（不满意才上诉，满意则同意判决）
        return END
    if verdict.critic_overfit:
        # D 判 C 过拟合：打回 C 重审；超限则交付当前草稿并标记存疑
        if public.critic_iteration < MAX_CRITIC_REWORK:
            return "c_critic"
        return END
    if public.iteration >= public.max_iterations:  # 铁律3：熔断
        return END
    return "b_builder"


@routing_guard
def route_after_ruling(public: PublicState) -> str:
    """D 终审后：禁言令→END；维持→END；改判→B（受同一熔断约束）。"""
    ruling = public.final_ruling
    assert ruling is not None, "路由前置条件：final_ruling 必须存在"
    if ruling.gag_order:
        return END
    if ruling.upheld:
        return END
    if public.iteration >= public.max_iterations:
        return END
    return "b_builder"


def build_graph(
    architect: ArchitectAgent,
    builder: BuilderAgent,
    critic: CriticAgent,
    judge: JudgeAgent,
    juror: JurorAgent,
    *,
    interrupt_before_e: bool = True,
    checkpointer=None,
):
    """装配 Quintet-Verify 图：START→A→B→C→D初审→(条件边)；E→D终审→(条件边)。

    中断恢复用法（interrupt_before_e=True 时需 checkpointer）：
        g = build_graph(a, b, c, d, e, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "task-001"}}
        g.invoke(init_state, cfg)            # 停在 e_juror 之前
        g.update_state(cfg, {"public": ...}) # 写入 user_satisfied
        g.invoke(None, cfg)                  # 恢复续跑
    """
    if _QV_EXTRAS is not None:
        _QV_EXTRAS.install([architect, builder, critic, judge, juror])
    g = StateGraph(QuintetState)
    g.add_node("gate", gate_node)
    g.add_node("a_architect", lambda s: architect.invoke(s))
    g.add_node("b_builder", lambda s: builder.invoke(s))
    g.add_node("c_critic", lambda s: critic.invoke(s))
    g.add_node("d_judge_initial", lambda s: judge.invoke(s, mode="initial"))
    g.add_node("d_judge_final", lambda s: judge.invoke(s, mode="final"))
    g.add_node("e_juror", lambda s: juror.invoke(s))

    g.add_edge(START, "gate")
    g.add_conditional_edges(
        "gate", route_after_gate,
        {"a_architect": "a_architect", "b_builder": "b_builder"},
    )
    g.add_edge("a_architect", "b_builder")
    g.add_edge("b_builder", "c_critic")
    g.add_edge("c_critic", "d_judge_initial")
    g.add_conditional_edges(
        "d_judge_initial",
        route_after_verdict,
        {"e_juror": "e_juror", "b_builder": "b_builder", "c_critic": "c_critic", END: END},
    )
    g.add_edge("e_juror", "d_judge_final")
    g.add_conditional_edges(
        "d_judge_final",
        route_after_ruling,
        {"b_builder": "b_builder", END: END},
    )
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["e_juror"] if interrupt_before_e else None,
    )


# ======================================================================
# ③ Beta：自定义角色与自定义流程
# ======================================================================

def parse_pipeline(spec: str) -> list[str]:
    """解析流程定义：支持 -、–、—、−、>、＞、→、➔ 等连接符。
    例："A->B->C->D->E"、"A→B→C→F—D→E"。返回去重前的字母序列。
    """
    import re as _re
    parts = _re.split(r"[\s\-‐‑‒–—―−>＞→➔⟶]+", spec.strip())
    seq = [p.upper() for p in parts if p]
    if not seq:
        raise ValueError("流程定义为空")
    for p in seq:
        if len(p) != 1 or not ("A" <= p <= "Z"):
            raise ValueError(f"非法角色代号：{p}（须为单个 A~Z 字母）")
    if len(set(seq)) != len(seq):
        raise ValueError("流程中角色代号不可重复")
    return seq


class GenericOutput(BaseModel):
    """自定义角色的通用产出：一段内容。"""

    content: str


# 父角色的产出 schema（继承时自定义角色的 output 必须匹配父角色类型）
_PARENT_OUTPUT: dict[str, type[BaseModel]] = {}


class GenericRoleAgent(BaseQuintetAgent):
    """用户自定义角色。v1.4：可继承 A~E —— 获得父角色的权限校验与写回语义，
    act 行为仍由用户自定义 System Prompt 决定（产出 schema 随父角色）。"""

    def __init__(self, letter: str, inherits_from: str | None = None) -> None:
        super().__init__()
        self.role = letter  # type: ignore[assignment]
        self.inherits_from = inherits_from
        self._parent: BaseQuintetAgent | None = None
        if inherits_from:
            # object.__new__ 绕过 ABC 抽象限制：委托实例只用于 apply/check_permissions
            # （二者仅依赖类级常量与静态方法），永不调用 act
            self._parent = object.__new__(_PARENT_CLASSES[inherits_from])

    def apply(self, public: PublicState, output: Any) -> PublicState:
        if self._parent is not None:
            return self._parent.apply(public, output)  # 继承父角色的写回语义
        draft = public.draft or Draft(sections=[])
        draft.sections.append(DraftSection(anchor_id=self.role, content=output.content))
        public.draft = draft
        return public

    def check_permissions(self, state: QuintetState, output: Any) -> None:
        if self._parent is not None:
            self._parent.check_permissions(state, output)  # 继承父角色的权限校验
            return
        if not isinstance(output, GenericOutput):
            raise PermissionViolation(f"{self.role} 产出必须是 GenericOutput")


class LLMGenericRole(_LLMMixin, GenericRoleAgent):
    def __init__(self, letter: str, cfg: LLMConfig) -> None:
        super().__init__(letter, inherits_from=cfg.inherits_from)
        self.cfg = cfg

    def act(self, state: QuintetState, **ctx) -> Any:
        schema = _PARENT_OUTPUT.get(self.inherits_from or "", GenericOutput)
        return self._ask(schema, f"【公共状态】{_state_digest(state)}" + self._mods(state))


class StubGenericRole(GenericRoleAgent):
    def act(self, state: QuintetState, **ctx) -> Any:
        self.think("自定义角色处理完毕", CognitiveMark.UNCERTAIN)
        # v1.4：继承父角色时产出父类型（权限校验委托给父角色，产出必须匹配）
        if self.inherits_from == "A":
            return Outline(task_restated=state.public.task,
                           nodes=[OutlineNode(anchor_id="§1", title=f"{self.role} 提纲", intent="Stub 意图")])
        if self.inherits_from == "B":
            anchors = ([n.anchor_id for n in state.public.outline.nodes]
                       if state.public.outline else ["§1"])
            return Draft(sections=[DraftSection(anchor_id=a, content=f"[{self.role}·继承B] {a}") for a in anchors])
        if self.inherits_from == "C":
            return CriticReport(coverage=f"{self.role} 全量审查（Stub）", defects=[])
        if self.inherits_from == "D":
            return Verdict(approved=True, confidence_score=80)
        if self.inherits_from == "E":
            return None
        return GenericOutput(content=f"[{self.role} 自定义角色] 对当前材料的处理结果（Stub）")


def _register_parents() -> None:
    _PARENT_CLASSES.update({"A": ArchitectAgent, "B": BuilderAgent, "C": CriticAgent,
                            "D": JudgeAgent, "E": JurorAgent})
    _PARENT_OUTPUT.update({"A": Outline, "B": Draft, "C": CriticReport,
                           "D": Verdict, "E": Appeal})


_PARENT_CLASSES: dict[str, type[BaseQuintetAgent]] = {}
_register_parents()


# ---- 流程定义模型：边级解析（支持 DAG）----
class FlowDef(BaseModel):
    nodes: list[str]
    edges: list[tuple[str, str]]
    start: str
    end: str


def parse_flow(spec: str) -> FlowDef:
    """边级解析流程定义。支持 -、–、—、>、＞、→ 等连接符。
    线性链 "A->B->C" 展开为边集 [(A,B),(B,C)]。"""
    import re as _re
    tokens = _re.split(r"[\s\-‐‑‒–—―−>＞→➔⟶]+", spec.strip())
    nodes = [t.upper() for t in tokens if t]
    for n in nodes:
        if len(n) != 1 or not ("A" <= n <= "Z"):
            raise ValueError(f"非法角色代号：{n}（须为单个 A~Z 字母）")
    if not nodes:
        raise ValueError("流程定义为空")
    edges = list(zip(nodes, nodes[1:]))
    # 起点 = 无任何前驱的节点；终点 = 无任何后继的节点
    preds = {y for _, y in edges}
    succs = {x for x, _ in edges}
    starts = [n for n in dict.fromkeys(nodes) if n not in preds]
    ends = [n for n in dict.fromkeys(nodes) if n not in succs]
    return FlowDef(nodes=list(dict.fromkeys(nodes)), edges=edges,
                   start=starts[0] if starts else nodes[0],
                   end=ends[0] if ends else nodes[-1])


def validate_flow(definition: str, defined_roles: set[str]) -> list[str]:
    """流程图静态校验（v1.4）。返回错误列表（空 = 通过）。"""
    errors: list[str] = []
    try:
        flow = parse_flow(definition)
    except ValueError as e:
        return [f"解析失败：{e}"]
    # 规则1：所有节点必须已注册
    for n in flow.nodes:
        if n not in defined_roles:
            errors.append(f"节点 {n} 未注册（不在已定义角色集中）")
    # 规则2：循环依赖检测（DFS 三色标记）
    adj: dict[str, list[str]] = {}
    for x, y in flow.edges:
        adj.setdefault(x, []).append(y)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in flow.nodes}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if color.get(v) == GRAY:
                return True
            if color.get(v) == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    if any(color[n] == WHITE and dfs(n) for n in flow.nodes):
        errors.append("存在循环依赖（如 A->B->A）")
    # 规则3：恰好一个起点和一个终点
    preds = {y for _, y in flow.edges}
    succs = {x for x, _ in flow.edges}
    starts = [n for n in flow.nodes if n not in preds]
    ends = [n for n in flow.nodes if n not in succs]
    if len(starts) != 1:
        errors.append(f"起点数量必须为 1（当前 {len(starts)}：{starts or '无'}）")
    if len(ends) != 1:
        errors.append(f"终点数量必须为 1（当前 {len(ends)}：{ends or '无'}）")
    return errors


def build_custom_graph(flow: "FlowDef | list[str]", agents_map: dict[str, BaseQuintetAgent]):
    """自定义流程（v1.4）：支持线性链（list）或 DAG（FlowDef）。无回环无上诉。"""
    if isinstance(flow, list):
        flow = FlowDef(nodes=flow, edges=list(zip(flow, flow[1:])),
                       start=flow[0], end=flow[-1])
    if _QV_EXTRAS is not None:
        _QV_EXTRAS.install(list(agents_map.values()))
    g = StateGraph(QuintetState)
    for letter in flow.nodes:
        agent = agents_map[letter]
        g.add_node(letter, lambda s, a=agent: a.invoke(s))
    g.add_edge(START, flow.start)
    for x, y in flow.edges:
        g.add_edge(x, y)
    g.add_edge(flow.end, END)
    return g.compile()


def build_fast_graph(
    builder: BuilderAgent,
    critic: CriticAgent,
    judge: JudgeAgent,
    *,
    checkpointer=None,
):
    """Fast 模式（②）：仅 B/C/D，无提纲、无上诉支线。强度分档仍生效（max 剔除 A 要求）。"""
    if _QV_EXTRAS is not None:
        _QV_EXTRAS.install([builder, critic, judge])
    g = StateGraph(QuintetState)
    g.add_node("b_builder", lambda s: builder.invoke(s))
    g.add_node("c_critic", lambda s: critic.invoke(s))
    g.add_node("d_judge", lambda s: judge.invoke(s, mode="initial"))
    g.add_edge(START, "b_builder")
    g.add_edge("b_builder", "c_critic")
    g.add_edge("c_critic", "d_judge")
    g.add_conditional_edges(
        "d_judge", route_after_verdict,
        {"b_builder": "b_builder", "c_critic": "c_critic", "e_juror": END, END: END},
    )
    return g.compile(checkpointer=checkpointer)


# ======================================================================
# Part 5 · Stub 五角色（无 LLM 依赖的确定性演示/测试）
# ======================================================================


class StubArchitect(ArchitectAgent):
    def act(self, state: QuintetState, **ctx) -> Outline:
        self.think("任务可分解为 3 个维度", CognitiveMark.VERIFIED, verifier="stub: 结构模板匹配")
        return Outline(
            task_restated=state.public.task,
            nodes=[
                OutlineNode(anchor_id="§1", title="问题定义", intent="界定任务边界与目标", frozen=True),
                OutlineNode(anchor_id="§2", title="方案论证", intent="给出核心论证"),
                OutlineNode(anchor_id="§3", title="结论", intent="收敛结论"),
            ],
        )


class StubBuilder(BuilderAgent):
    def act(self, state: QuintetState, **ctx) -> Draft:
        pub = state.public
        anchors = ([n.anchor_id for n in pub.outline.nodes]
                   if pub.outline else ["§1", "§2", "§3"])
        coords = self.remand_coordinates_of(state)
        if not coords:  # 初稿
            self.think("论据充分性待验证", CognitiveMark.UNCERTAIN)
            return Draft(sections=[DraftSection(anchor_id=a, content=f"[初稿] {a} 内容") for a in anchors])
        self.think("按坐标返工，未触及冻结区", CognitiveMark.VERIFIED, verifier="stub: 坐标白名单校验")
        return Draft(
            sections=[DraftSection(anchor_id=a, content=f"[返工v{pub.iteration + 1}] {a} 内容") for a in anchors],
            revision_of=pub.iteration + 1,
            applied_coordinates=coords,
            ripple=RippleDeclaration(affects_frozen_zone=False, justification="仅修改指定坐标，不涉及冻结区"),
        )


class StubCritic(CriticAgent):
    def act(self, state: QuintetState, **ctx) -> CriticReport:
        pub = state.public
        if pub.draft is not None and pub.draft.revision_of is None:
            return CriticReport(coverage="全锚点初审", defects=[Defect(
                severity=DefectSeverity.S, at_anchor="§2",
                description="核心论证存在循环论证风险", impact_score=9, confidence_pct=85)])
        return CriticReport(coverage="复核修改坐标 §2", defects=[
            Defect(severity=DefectSeverity.A, at_anchor="§2", description="数据引用缺少出处", impact_score=6, confidence_pct=70),
            Defect(severity=DefectSeverity.A, at_anchor="§2", description="边界条件未覆盖", impact_score=5, confidence_pct=65),
        ])


class StubJudge(JudgeAgent):
    def act(self, state: QuintetState, mode: str = "initial", **ctx) -> Verdict | FinalRuling:
        pub = state.public
        if mode == "initial":
            if pub.iteration == 0:
                return Verdict(approved=False, confidence_score=72, remand_coordinates=[
                    RevisionCoordinate(anchor_id="§2", operation="rewrite", instruction="消除循环论证，补充独立论据")])
            return Verdict(approved=True, confidence_score=88)
        if pub.appeal is None:
            return FinalRuling(upheld=True, reasoning="陪审员无异议，维持原判")
        if pub.appeal.severity is DefectSeverity.B or not pub.appeal.new_evidence:
            return FinalRuling(upheld=True, gag_order=True,
                               reasoning="异议无新证据，援引'无理取闹'定义当庭驳回，签发禁言令")
        return FinalRuling(upheld=False, reasoning="采纳异议，发回重审", remand_coordinates=[
            RevisionCoordinate(anchor_id="§2", operation="rewrite", instruction="按陪审团异议重修 §2 论证链")])


class StubJuror(JurorAgent):
    def act(self, state: QuintetState, **ctx) -> Appeal | None:
        if state.public.user_satisfied is False:
            return Appeal(severity=DefectSeverity.S,
                          grounds="用户对结论存疑，主张 §2 仍有根本缺陷", new_evidence=True)
        return None


def build_stub_agents() -> tuple[StubArchitect, StubBuilder, StubCritic, StubJudge, StubJuror]:
    return StubArchitect(), StubBuilder(), StubCritic(), StubJudge(), StubJuror()



# v1.3：可选扩展包 qv-extras —— 检测到即自动挂载，未安装核心流程不受影响
try:
    import qv_extras as _QV_EXTRAS
except ImportError:  # pragma: no cover
    _QV_EXTRAS = None

if __name__ == "__main__":
    graph = build_graph(*build_stub_agents())
    init = QuintetState(public=PublicState(task="演示任务：评估某方案可行性", user_satisfied=True))
    final = graph.invoke(init)
    pub, priv = final["public"], final["private"]
    print("=" * 60)
    print("Quintet-Verify Stub 演示结果")
    print("=" * 60)
    print(f"熔断计数 iteration      : {pub.iteration} / {pub.max_iterations}")
    print(f"冻结区 checksum (§1)    : {pub.outline.nodes[0].checksum[:16]}...")
    print(f"终审判决 approved       : {pub.verdict.approved} (置信度 {pub.verdict.confidence_score})")
    print(f"草稿轮次 revision_of    : {pub.draft.revision_of}")
    print(f"涟漪声明 affects_frozen : {pub.draft.ripple.affects_frozen_zone}")
    print("-" * 60)
    print(f"私有 Dense Track 共 {len(priv.dense_track)} 条（路由从未接触）：")
    for t in priv.dense_track:
        print(f"  [{t.agent}] {t.mark.value} {t.claim}")
