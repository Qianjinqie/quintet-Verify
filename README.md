Quintet-Verify (QV)

A multi-agent cognitive control framework based on anchored cognitive architecture – using a five‑role (A/B/C/D/E) tribunal‑style division of labor and private cognitive traces (✓/?/✗) to impose rigid checks and balances on agents at inference time: no fine‑tuning, no weight changes, only constraints on “who speaks when, with what authority, and based on what evidence.”

· Model‑agnostic · Python 3.11+ · LangGraph · Pydantic v2 · FastAPI

🌐 Online Demo (deployed via workbuddy): https://a524b836ec13d5d9f.app.workbuddy.link

---

Table of Contents

· Results First
  · Fable 5 Blind Evaluation Conclusion
· Key Constraints
· Why Not Just Use One Large Model?
· Core Features
· How Is It Different From Other Multi‑Agent Frameworks?
· Quality Explanation
· Real‑Run Data
· Who Should Use It?
· Quick Start
  · Installation
  · Minimal Example (Real LLM)
  · Offline Demo Without LLM
  · Web Console
· Contributing
· Theoretical Origins and Independence Statement
  · Inspirations
  · Fundamental Architectural Differences
  · Core Difference Summary
  · Independence Statement

---

Results First

📊 Fable 5 Blind Evaluation Conclusion

The blind evaluation by Fable 5 gave our project an overall score of 9.17/10, while the contestant (Fable5, unknown to the judges) received 8.0/10 – a margin of +1.17.

All three tasks were won:

· Code Review: 9.2 vs 8.5 (contestant F5 missed subtle issues like rpop message loss, state field overwriting, and predictable task_id leakage)
· Technical Solution Assessment: 9.5 vs 8.0 (F5 miscomputed pairwise as pointwise in the ranking layer – a fatal error)
· Legal Document Review: 8.8 vs 7.5 (F5 did not provide a specific verification plan)

Note: The legal document task used DeepSeek‑V4‑Pro‑0813; code review and technical assessment used DeepSeek‑V4‑Flash‑0731. The framework ran in standard mode – high intensity.

The evaluator (F5) concluded qualitatively:

The evaluated system outperforms the subject in five dimensions: technical depth, computational accuracy, risk identification, data support, and professional rigor. Suitable for high‑stakes scenarios such as investment decisions, technical due diligence, pre‑production deployment reviews, and compliance audits.

Note: In practice, QV makes many API calls, incurring higher costs – it trades money and time for quality.

---

Key Constraints

· A does not write the main text, B does not alter the framework, C does not provide full‑text revisions, D does not ghostwrite on behalf of others, E does not directly overturn rulings.
· Cognitive markers ✓/?/✗ exist as private traces and never enter routing decisions.
· Remands must include anchor coordinates; rework is limited to designated regions (ripple declarations constrain the scope of changes).

---

Why Not Just Use One Large Model?

A single model’s “generation equals finality” is the root cause of hallucinations reaching the endpoint directly: it generates, reviews, and confirms by itself – the supervisor and the supervised share the same standpoint. Quintet‑Verify breaks this closed loop into five roles with mutually locked permissions –

Checks and balances evolve from “self‑discipline” to “other‑discipline”: the supervisor and the supervised do not share a standpoint; every remand is constrained within anchor coordinates, so rework never causes the entire text to drift.

---

Core Features

· Dual‑channel cognitive architecture: public (visible to routing) and private Dense Track (✓/?/✗) are physically isolated, with three layers of protection (structural / type / runtime assertions) ensuring private cognition never enters routing.
· Five‑role rigid checks and balances: A (Architect) · B (Executor) · C (Verifier) · D (Judge) · E (Juror) – each with locked permissions and forbidden zones.
· Three iron rules (enforced in code): ① confidence isolation ② incremental modification (anchor coordinates + ripple declarations) ③ circuit‑breaker gag order (max_iterations=110, D can issue a gag order).
· Intensity tiers: low / high (default) / max; under stringent mode, defects must be accompanied by an impact chain, and D may not pass if confidence is insufficient.
· Fast mode: only the B/C/D triangle, no outline, no appeal branch – suitable for simple tasks and low‑cost scenarios.
· Overfitting governance: C has no mandatory quota (zero defects allowed, but must include a coverage statement); D may deem C overly pedantic and send it back for review (max 2 times).
· Plugin mechanism: PLUGINS registry + flags (all off by default); optional qv_extras.py can be mounted via standardised pre/post_act_hooks.
· Customisable workflow (v1.4): front‑end editing of workflow definitions, creating new roles and inheriting AE permissions, static validation, and per‑role independent LLM configuration.

---

How Is It Different From Other Multi‑Agent Frameworks?

· AutoGen / CrewAI / MetaGPT: all support multi‑agent collaboration and are model‑agnostic, but they lack three capabilities: inference‑time rigid checks and balances, private cognitive trace auditing, and incremental modification (token‑saving).
· Quintet‑Verify: on top of supporting multi‑agent collaboration and model‑agnosticism, it additionally provides:
  · Inference‑time rigid checks and balances (tribunal‑style permission interlocking)
  · Auditable private cognitive traces (Dense Track)
  · Incremental modification (anchor coordinates + ripple declarations, saving on average ~80% context)

Most frameworks do “let AI collaborate”; Quintet‑Verify does “let AI check and balance each other” – the former pursues output efficiency, the latter pursues conclusion trustworthiness.

---

Quality Explanation

Compared with using a single model directly, Quintet‑Verify’s gains come from structural checks and balances, not from larger parameter counts:

· Compared with a pure model: hallucinations no longer reach the endpoint directly – they must first pass C’s review and D’s ruling, and each remand is constrained within anchor coordinates. C’s zero‑defect conclusion and D’s overfitting judgement simultaneously prevent internal friction from “nitpicking for the sake of it”.
· Compared with pure J‑Space: J‑Space’s dense cognitive management operates inside a single model; the ceiling of self‑supervision is “catching oneself”. Quintet‑Verify retains ✓/?/✗ as private cognitive traces while delegating the ruling authority to independent roles – upgrading checks and balances from self‑discipline to other‑discipline.

The above is a qualitative comparison at the mechanism level; practical effectiveness is generally far superior to its base model – see Results First for details.

---

Real‑Run Data

Full‑mode real run (August 2026, DeepSeek‑V4‑Pro‑0813, legal document review task):

· Token consumption: ~130,000
· API calls: 19
· Cost: ~¥2.3
· Final ruling: D approved with 90/100 confidence

The output concluded “conditionally feasible with hybrid architecture”: empirical assertions lacking data support in the initial draft were either removed entirely or downgraded to hypotheses pending verification after two remands; acceptance criteria were rewritten into auditable terms.

---

Who Should Use It?

· Developers building RAG applications but plagued by hallucinations
· Legal, financial, and healthcare scenarios requiring auditable AI decisions
· Researchers wanting to experience multi‑agent checks and balances without writing complex orchestrations
· Teams satisfied with AutoGen/CrewAI collaboration efficiency but demanding higher conclusion trustworthiness
· Programmers using AI to write high‑quality code
· Researchers using AI for mathematical research assistance

Note: Related plugins will be open‑sourced later.

---

Quick Start

Installation

```bash
git clone https://github.com/Qianjinqie/quintet-Verify.git
cd quintet-Verify
pip install -r requirements.txt
```

Minimal Example (Real LLM)

```python
from quintet_verify import (
    DEFAULT_FLAGS,
    LLMConfig,
    PublicState,
    QuintetState,
    build_graph,
    build_llm_agents,
)

# 1. Configure model (OpenAI‑compatible endpoint; shared or per‑role)
cfg = LLMConfig(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
)
agents = build_llm_agents(cfg)  # returns five roles (A, B, C, D, E)

# 2. Build graph and run (interrupt_before_e=False means no pause before E for user feedback)
graph = build_graph(*agents, interrupt_before_e=False)
final = graph.invoke(
    QuintetState(
        public=PublicState(
            task="Evaluate the feasibility of a technical solution and provide a conclusion.",
            user_satisfied=True,   # satisfied → E agrees with the ruling, workflow ends
            intensity="high",      # intensity tiers: low / high / max
            flags=dict(DEFAULT_FLAGS),  # plugin switches, all off by default
        )
    )
)

# 3. Read results
pub = final["public"]
print(pub.verdict.approved, pub.verdict.confidence_score)  # D's ruling and confidence
for section in pub.draft.sections:
    print(section.anchor_id, section.content)
```

Offline Demo Without LLM

```python
from quintet_verify import PublicState, QuintetState, build_graph, build_stub_agents

final = build_graph(*build_stub_agents(), interrupt_before_e=False).invoke(
    QuintetState(public=PublicState(task="Demo task", user_satisfied=True))
)
print(final["public"].verdict)
```

Web Console

```bash
python server.py  # default port 8000, open in browser
```

The console supports: Standard / Fast / Custom pipelines, intensity tiers, plugin toggles, per‑role API configuration, workflow definition editing and real‑time validation, role prompt overrides, and run visualisation (event stream / draft / Dense Track).

---

Contributing

Both PRs and Issues are welcome. Please run python server.py and the offline demo before submitting, and describe your motivation and test results in the PR. Beginners can start with issues labelled good first issue.

---

Theoretical Origins and Independence Statement

Inspirations

Quintet‑Verify draws inspiration from prior work in the following aspects:

· Cognitive marker symbols (✓/?/✗) and the naming of “private cognitive traces” (Dense Track): derived from the J‑Space Cognition Suite V3.6 (doi:10.5281/zenodo.21971181). That work first systematically proposed using compact symbols inside a single model to manage cognitive states, and validated the positive effect of “introspective markers” on reasoning quality.
· The organisational form of multi‑role tribunal‑style checks and balances: inspired by the Quintet Verification multi‑model debate system shown at b23.tv/z3C0vVM, which demonstrated the feasibility of “transforming generation hallucinations into consensus games through role separation and objection procedures”.

We express our respect and gratitude for the original contributions of the above works.

Fundamental Architectural Differences

Although Quintet‑Verify borrows the symbol naming from J‑Space and the organisational form from Quintet Verification, its underlying cognitive architecture differs irreducibly from both. These differences are not incremental improvements, but rather a redefinition of three fundamental assumptions: cognitive subject, information topology, and control nature.

Aspect J‑Space (Global Workspace Theory) Quintet Verification (multi‑model debate) Quintet‑Verify (Anchored Cognitive Theory)
Cognitive Subject Inside a single model Multiple independent models (stateless) Multiple independent agents (each with private state)
Carrier of Cognitive State Global context window Independent outputs of each model Public shared state + private traces (physically isolated)
Information Flow Global broadcast + competitive access Debate‑style full transmission Anchored references (via anchor_id + checksum for shared facts)
Modification Mechanism Full rewrite (relies on introspective correction) Full rewrite (relies on debate correction) Incremental modification (precise coordinates + ripple declarations)
Control Nature Prompt‑guided soft constraints Prompt‑guided soft constraints Compile‑time validation + routing guards (hard constraints)
Recovery Mechanism Textual ledger rollback No standardised recovery Atomic state checkpoints (exact rollback to any round)
Checks and Balances Self‑supervision (same standpoint) Cross‑model debate (independent standpoints) Permission interlocking + tribunal procedure (standpoint separation + procedural rigidity)
Auditability Chain‑of‑thought text (non‑structural rollback) Debate text (non‑structural rollback) Structured Dense Track audit (traceable verification evidence for ✓ and falsification evidence for ✗)

Core Difference Summary

· J‑Space addresses “how a single person thinks better” (deep introspection of a single consciousness);
· Quintet Verification addresses “how a group reaches consensus through debate” (external checks via multiple standpoints);
· Quintet‑Verify addresses “how a group of agents with independent cognitive states converges on verifiable conclusions through rigid rules” (constitutional governance of distributed cognition).

Quintet‑Verify’s unique contribution is the first combination of “private cognitive traces (Dense Track)” with a “rigid permission matrix”, allowing multi‑agent systems to achieve system‑level reliability that is auditable, traceable, and forcibly terminable while maintaining individual reasoning depth.

Independence Statement

Based on the fundamental architectural differences above, Quintet‑Verify hereby declares:

· Not a derivative work: Quintet‑Verify is not a derivative work or fork of J‑Space Cognition Suite V3.6. The two adopt completely different code implementations, cognitive topologies, and control philosophies.
· Independent intellectual property: All code of Quintet‑Verify was independently written by our team; no source code, prompt templates, or protocol texts from J‑Space or Quintet Verification have been copied or translated.
· Citation, not dependency: This system lists J‑Space and Quintet Verification as “theoretical references” rather than “technical dependencies”. Removing these references does not affect the full operation of Quintet‑Verify’s core workflow.
· Academic integrity: This statement aims to accurately trace the intellectual origins while clearly defining the boundaries of our original contributions. We encourage readers to conduct comparative studies between the above works and this system to understand the different evolutionary paths of distributed cognitive architectures.

Recommendation for readers:

· If you are interested in “how a single model improves reasoning through introspective markers”, please refer to J‑Space Cognition Suite V3.6.
· If you are interested in “how multiple models reach consensus through debate”, please refer to the Quintet Verification system.
· If you are interested in “how to make multi‑agent collaboration produce auditable and checked conclusions through rigid rules”, please continue using Quintet‑Verify.