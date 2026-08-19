"""Quintet-Verify · server.py — 前端后端桥（FastAPI）

启动：python3.11 server.py  →  http://localhost:8000

接口：
  POST /api/runs                 发起运行（demo=Stub 秒回 / live=真实 LLM）
  GET  /api/runs/{run_id}        轮询状态（事件流 + 公共快照 + Dense Track）
  POST /api/runs/{run_id}/feedback  用户表态（满意/不满意），恢复中断的图
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import quintet_verify as jq

app = FastAPI(title="Quintet-Verify UI")
RUNS: dict[str, "RunRecord"] = {}

ROLE_NAMES = {"A": "架构师", "B": "执行者", "C": "验证者", "D": "裁判", "E": "陪审员"}


class RunRecord:
    MAX_KEPT = 50  # RUNS 字典上限，超出时清理最旧的已完成记录

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.created_at = time.time()
        self.status = "running"  # running / awaiting_feedback / done / error
        self.error: str | None = None
        self.events: list[dict] = []
        self.public: dict | None = None
        self.dense: list[dict] = []
        self.graph = None
        self.cfg: dict | None = None
        self.custom: tuple | None = None  # (seq, agents_map) 自定义流程
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "error": self.error,
                "events": list(self.events),  # 拷贝，避免迭代时被写线程修改
                "public": self.public,
                "dense": list(self.dense),
            }


def _role_name(role: str) -> str:
    return ROLE_NAMES.get(role, "自定义")


def _summary(role: str, mode: str | None, pub: jq.PublicState) -> str:
    if role == "A" and pub.outline:
        n, f = len(pub.outline.nodes), len(pub.outline.frozen_anchor_ids())
        return f"提纲 {n} 节点，冻结区 {f} 个"
    if role == "B" and pub.draft:
        if pub.draft.revision_of is None:
            return f"初稿完成，{len(pub.draft.sections)} 节"
        rip = "波及冻结区" if (pub.draft.ripple and pub.draft.ripple.affects_frozen_zone) else "不波及冻结区"
        return f"第 {pub.draft.revision_of} 轮返工，涟漪声明：{rip}"
    if role == "C" and pub.critic_report:
        ds = pub.critic_report.defects
        s = sum(1 for d in ds if d.severity == jq.DefectSeverity.S)
        a = sum(1 for d in ds if d.severity == jq.DefectSeverity.A)
        return f"发现缺陷 {len(ds)} 条（S×{s}，A×{a}）"
    if role == "D" and mode == "initial" and pub.verdict:
        v = pub.verdict
        if v.critic_overfit:
            return f"初审：判 C 过拟合，打回重审（{v.overfit_reasoning or ''}）"
        return f"初审：{'通过' if v.approved else '打回 B'}，置信度 {v.confidence_score}"
    if role == "D" and mode == "final" and pub.final_ruling:
        r = pub.final_ruling
        tag = "禁言令！" if r.gag_order else ("维持原判" if r.upheld else "采纳异议改判")
        return f"终审：{tag}"
    if role == "E":
        return "提交异议书" if pub.appeal else "同意判决，不上诉"
    return ""


def _wrap(agent: jq.BaseQuintetAgent, rec: RunRecord, pace: float = 0.0) -> None:
    orig = agent.invoke

    def logged(state: jq.QuintetState, **ctx):
        role, mode = agent.role, ctx.get("mode")
        node = f"{role} {_role_name(role)}" + (f"（{'终审' if mode == 'final' else '初审'}）" if role == "D" else "")
        with rec.lock:
            rec.events.append({"t": time.time(), "type": "start", "node": node})
        if pace:
            time.sleep(pace)  # demo 模式放慢节奏，让前端动画可见
        out = orig(state, **ctx)
        with rec.lock:
            rec.public = out["public"].model_dump()
            rec.dense = [t.model_dump() for t in out["private"].dense_track]
            rec.events.append({
                "t": time.time(), "type": "end", "node": node,
                "summary": _summary(role, mode, out["public"]),
            })
        return out

    agent.invoke = logged  # type: ignore[method-assign]


def _run_graph(rec: RunRecord, agents, init: jq.QuintetState, interactive: bool, pace: float = 0.0) -> None:
    try:
        wrapped = list(rec.custom[1].values()) if rec.custom else list(agents)
        for a in wrapped:
            _wrap(a, rec, pace)
        if interactive:
            rec.graph = jq.build_graph(*agents, checkpointer=MemorySaver())
            rec.cfg = {"configurable": {"thread_id": rec.id}}
            rec.graph.invoke(init, rec.cfg)
            if rec.graph.get_state(rec.cfg).next:  # 停在 e_juror 之前
                with rec.lock:
                    rec.status = "awaiting_feedback"
                return
        else:
            if rec.custom:
                flow, amap = rec.custom
                graph = jq.build_custom_graph(flow, amap)
            elif init.public.pipeline == "fast":
                graph = jq.build_fast_graph(agents[1], agents[2], agents[3])
            else:
                graph = jq.build_graph(*agents, interrupt_before_e=False)
            graph.invoke(init)
        with rec.lock:
            rec.status = "done"
    except Exception as e:  # noqa: BLE001
        with rec.lock:
            rec.status = "error"
            rec.error = f"{type(e).__name__}: {e}"


def _gc_runs() -> None:
    """清理最旧的已完成/出错记录，防止 RUNS 无限增长。"""
    if len(RUNS) < RunRecord.MAX_KEPT:
        return
    finished = sorted(
        (r for r in RUNS.values() if r.status in ("done", "error")),
        key=lambda r: r.created_at,
    )
    for r in finished[: max(1, len(RUNS) - RunRecord.MAX_KEPT + 1)]:
        RUNS.pop(r.id, None)


class RoleCfg(BaseModel):
    """单个角色的 API 配置；字段留空则回落到全局配置。"""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float | None = None


class CustomRoleReq(BaseModel):
    """v1.4 新建角色：提示词 + 权限继承 + 独立 LLM 配置（留空回落全局）。"""

    system_prompt: str = ""
    inherits_from: str | None = None
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float | None = None


class StartReq(BaseModel):
    task: str
    mode: str = "demo"  # demo / live
    pipeline: str = "standard"  # standard=A~E / fast=仅B/C/D / custom=自定义线性流程(Beta)
    pipeline_spec: str = ""    # custom 时的流程定义，如 "A->B->C->F->D->E"
    role_prompts: dict[str, str] = {}    # Beta：内置角色系统提示词覆盖
    custom_roles: dict[str, CustomRoleReq] = {}  # v1.4：新建角色 字母→配置
    intensity: str = "high"  # low / high / max
    max_iterations: int = 5  # 熔断上限（1~10，由 PublicState 校验）
    plugins: dict[str, bool] = {}  # 插件开关，缺省全部启用
    # 全局默认配置
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-pro"
    # 分角色覆盖（键 "A"~"E"，可选；fast 模式只用 B/C/D）
    roles: dict[str, RoleCfg] = {}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "runs": len(RUNS)}


@app.post("/api/runs")
def start_run(req: StartReq) -> dict:
    _gc_runs()
    rec = RunRecord()
    RUNS[rec.id] = rec
    if req.pipeline not in ("standard", "fast", "custom"):
        raise HTTPException(400, "pipeline 必须是 standard / fast / custom")
    custom_flow = None
    if req.pipeline == "custom":
        try:
            custom_flow = jq.parse_flow(req.pipeline_spec)
        except ValueError as e:
            raise HTTPException(400, f"流程定义解析失败：{e}")
        defined = set("ABCDE") | set(req.custom_roles)
        errors = jq.validate_flow(req.pipeline_spec, defined)
        if errors:
            raise HTTPException(400, "流程静态校验未通过：" + "；".join(errors))
    if req.intensity not in ("low", "high", "max"):
        raise HTTPException(400, "intensity 必须是 low / high / max")
    try:
        init_public = jq.PublicState(
            task=req.task,
            max_iterations=req.max_iterations,
            pipeline="fast" if req.pipeline == "custom" else req.pipeline,  # custom 复用 fast 权限语义
            intensity=req.intensity,
            flags={**jq.DEFAULT_FLAGS, **req.plugins},
        )
    except Exception:
        raise HTTPException(400, "max_iterations 必须在 1~10 之间")
    fast = req.pipeline == "fast"
    if req.mode == "live":
        need = "BCD" if fast else "ABCDE"
        if not req.api_key and not any(req.roles.get(r, RoleCfg()).api_key for r in need):
            raise HTTPException(400, "live 模式必须提供 api_key（全局或至少一个在用角色）")
        cfgs = {
            role: jq.LLMConfig(
                api_key=(r := req.roles.get(role, RoleCfg())).api_key or req.api_key,
                base_url=r.base_url or req.base_url,
                model=r.model or req.model,
                temperature=r.temperature if r.temperature is not None else 0.2,
            )
            for role in "ABCDE"
        }
        agents = jq.build_llm_agents(cfgs)
        init = jq.QuintetState(public=init_public)  # 满意与否等中断后表态
        if custom_flow:
            # v1.4：自定义流程——内置角色可覆盖提示词；新角色支持继承 + 独立 LLM 配置
            builtin = dict(zip("ABCDE", agents))
            amap = {}
            for letter in custom_flow.nodes:
                if letter in builtin:
                    a = builtin[letter]
                    if letter in req.role_prompts and req.role_prompts[letter]:
                        a.cfg = a.cfg.model_copy(update={"system_prompt": req.role_prompts[letter]})
                    amap[letter] = a
                else:
                    cr = req.custom_roles.get(letter, CustomRoleReq())
                    amap[letter] = jq.LLMGenericRole(letter, jq.LLMConfig(
                        api_key=cr.api_key or req.api_key,
                        base_url=cr.base_url or req.base_url,
                        model=cr.model or req.model,
                        temperature=cr.temperature if cr.temperature is not None else 0.2,
                        system_prompt=cr.system_prompt
                        or f"你是自定义角色 {letter}，对输入材料进行专业处理并输出结果。",
                        inherits_from=cr.inherits_from if cr.inherits_from in ("A", "B", "C", "D", "E") else None,
                    ))
            rec.custom = (custom_flow, amap)
        threading.Thread(target=_run_graph, args=(rec, agents, init, not fast and not custom_flow), daemon=True).start()
    else:
        agents = jq.build_stub_agents()
        init = jq.QuintetState(public=init_public.model_copy(update={"user_satisfied": True}))
        if custom_flow:
            builtin = dict(zip("ABCDE", agents))
            amap = {l: (builtin[l] if l in builtin else jq.StubGenericRole(
                l, inherits_from=req.custom_roles.get(l, CustomRoleReq()).inherits_from))
                for l in custom_flow.nodes}
            rec.custom = (custom_flow, amap)
        threading.Thread(target=_run_graph, args=(rec, agents, init, False, 1.0), daemon=True).start()
    return {"run_id": rec.id}


class FlowCheckReq(BaseModel):
    definition: str
    defined_roles: list[str]


@app.post("/api/validate-flow")
def validate_flow_api(req: FlowCheckReq) -> dict:
    return {"errors": jq.validate_flow(req.definition, set(req.defined_roles))}


@app.get("/api/plugins")
def plugins_meta() -> dict:
    return {"plugins": jq.PLUGINS, "defaults": jq.DEFAULT_FLAGS}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(404, "run 不存在")
    return rec.snapshot()


class FeedbackReq(BaseModel):
    satisfied: bool


@app.post("/api/runs/{run_id}/feedback")
def feedback(run_id: str, req: FeedbackReq) -> dict:
    rec = RUNS.get(run_id)
    if rec is None or rec.graph is None or rec.cfg is None:
        raise HTTPException(404, "run 不存在或不在等待反馈")
    with rec.lock:
        if rec.status != "awaiting_feedback" or rec.public is None:
            raise HTTPException(409, "当前状态不可反馈")
        rec.status = "running"
        pub = jq.PublicState(**rec.public).model_copy(update={"user_satisfied": req.satisfied})
    rec.graph.update_state(rec.cfg, {"public": pub})

    def resume() -> None:
        try:
            rec.graph.invoke(None, rec.cfg)
            with rec.lock:
                if rec.graph.get_state(rec.cfg).next:
                    rec.status = "awaiting_feedback"  # 新一轮又停在 E 前
                else:
                    rec.status = "done"
        except Exception as e:  # noqa: BLE001
            with rec.lock:
                rec.status = "error"
                rec.error = f"{type(e).__name__}: {e}"

    threading.Thread(target=resume, daemon=True).start()
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="warning")
