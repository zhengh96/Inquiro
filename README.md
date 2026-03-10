# Inquiro

**Domain-agnostic evidence research & synthesis engine powered by multi-model AI agents.**

[中文版](#中文) | [English](#english)

---

<a id="english"></a>

## What is Inquiro?

Inquiro is a **general-purpose evidence research engine** that autonomously searches, cleans, analyzes, and synthesizes evidence from multiple data sources using multi-model AI agents. It is domain-agnostic by design — the same pipeline works for drug target evaluation, competitive intelligence, literature review, or any scenario that requires systematic evidence-based research.

### The Problem

Asking an LLM a research question gives you a plausible-sounding answer based on stale training data. There's no way to verify it, no systematic evidence collection, and no protection against single-model bias or hallucination. The gap between "asking an AI" and "doing actual research" is enormous.

### The Solution

Inquiro doesn't ask an AI for answers. It uses AI agents to **do the research** — planning search strategies, querying specialized databases, cleaning and deduplicating evidence, having multiple models independently analyze the same evidence set, identifying coverage gaps, and iterating until convergence. Every conclusion traces back to a specific source.

---

## Core Pipeline: DiscoveryLoop

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DiscoveryLoop (per round)                         │
│                                                                      │
│  SearchExp ──→ EvidencePipeline ──→ AnalysisExp ──→ GapAnalysis      │
│  (AI search)   (zero-LLM clean)   (3-model vote)  (coverage check)  │
│                                                                      │
│         Converged? ── No → back to SearchExp (with gap guidance +    │
│                            evolution experience injection)           │
│                       Yes → SynthesisExp (final synthesis)           │
│                                                                      │
│  Evolution: after each round, collect trajectory → extract           │
│             experiences → update fitness → inject into next round    │
└──────────────────────────────────────────────────────────────────────┘
```

Each research task goes through **multi-round iterative discovery**:

1. **Search** — AI agents autonomously query data sources via MCP servers, planning search strategies and following citation chains
2. **Clean** — Deterministic pipeline (zero LLM cost): deduplication, noise filtering, source classification
3. **Analyze** — 3 different LLMs (Claude / GPT / Gemini) independently analyze the same evidence, reaching consensus via weighted voting
4. **Gap Check** — Deterministic convergence: 5 stop conditions, no LLM involved
5. **Synthesize** — Final cross-evidence synthesis with traceable citations

---

## Design Philosophy

### 1. Domain-Agnostic by Constraint, Not by Accident

Inquiro's codebase contains **zero domain-specific terminology**. This is not a soft guideline — it's an architectural constraint enforced across the entire codebase.

Domain knowledge enters through exactly one interface — `EvaluationTask`:

```python
task = EvaluationTask(
    topic="EGFR druggability for Small Molecule in NSCLC",
    rules="## Decision Framework\n- POSITIVE: Target has confirmed...",
    checklist=["Crystal structure available", "Binding pocket identified", ...],
    output_schema={"decision": "enum[POSITIVE,CAUTIOUS,NEGATIVE]", ...},
)
result = await discovery_loop.run(task)  # Inquiro doesn't know what EGFR is
```

This means the same engine can evaluate drug targets, analyze competitive landscapes, review academic literature, or assess investment opportunities — by changing the rules, checklist, and schema. The engine handles the research process; the caller defines what "good research" means.

### 2. Multi-Model Consensus Over Single-Model Trust

Every analysis in Inquiro is performed by **3 independent LLMs** (Claude / GPT / Gemini), with weighted voting:

```yaml
weights:
  claude-sonnet-4-5: 0.40
  gpt-5-2: 0.30
  gemini-3-pro: 0.30
consensus_threshold: 0.70
```

Why not just use the best model? Because we observed systematic bias:

| Model | Observed Bias |
|-------|---------------|
| Claude | Tends conservative (CAUTIOUS), underweights positive signals |
| GPT | Tends optimistic (POSITIVE), underweights risks |
| Gemini | Relatively balanced, but weaker on niche topics |

3-model voting significantly reduces systematic bias. The consensus ratio itself is a natural confidence indicator — unanimous agreement is more reliable than a 2:1 split.

### 3. Zero-LLM Evidence Cleaning

The EvidencePipeline between search and analysis is **purely deterministic** — zero LLM cost:

1. **Multi-item splitting** — One tool response may contain multiple evidence items
2. **Content deduplication** — Hash-based + URL normalization
3. **Noise filtering** — 20+ pattern-matching rules
4. **Source tagging** — URL → ACADEMIC / PATENT / CLINICAL_TRIAL / REGULATORY / NEWS

Why deterministic? Because all 3 analysis models must see the **exact same evidence baseline**. If we used an LLM for cleaning, each run would produce slightly different evidence sets, making cross-model comparison meaningless.

### 4. Deterministic Convergence

The DiscoveryLoop never asks an LLM "is this enough evidence?" Instead, 5 deterministic stop conditions:

| Condition | Description |
|-----------|-------------|
| Coverage met | Checklist coverage ≥ threshold |
| Budget exhausted | Cost limit reached |
| Max rounds | Hard iteration cap |
| Diminishing returns | Coverage gain < δ for N consecutive rounds |
| Search exhaustion | New evidence < threshold in current round |

This prevents the common failure mode of LLM-based stopping: the model says "I've found enough" when it hasn't, or keeps searching indefinitely.

### 5. Strict LLM Boundary

```
API Layer    — no LLM — HTTP routing, SSE streaming
Runner       — no LLM — Resource management, concurrency
Exp Layer    — no LLM — Prompt construction, quality gates, output validation
Agent Layer  — HAS LLM — Reasoning, tool calling
```

Only the Agent layer holds LLM instances. The Exp layer builds prompts but doesn't execute them; the Agent layer executes prompts but doesn't validate outputs. This means the Exp layer is **purely deterministic and fully testable** — given input, verify prompt content and output parsing, no LLM mocking needed.

### 6. Evidence Token Management: 3-Tier Condenser

A single research sub-item can accumulate 400+ evidence items. Feeding all of them to an LLM causes token overflow and attention dilution.

| Tier | Condition | Strategy |
|------|-----------|----------|
| 0 | ≤150 items | Passthrough |
| 1 | 151-400 items | 6-signal weighted scoring, take Top-160 |
| 2 | 401+ items | Tier 1 + grouped LLM summarization |

The 6 scoring signals: keyword relevance, source quality, quality tags, structural completeness, recency, journal impact.

Safety net: at least one item per evidence type (ACADEMIC, PATENT, CLINICAL_TRIAL, etc.) is always preserved.

### 7. Closed-Loop Self-Evolution

Inquiro implements a complete **closed-loop learning system** — agents automatically improve from past executions through four independent mechanisms:

```
Task execution → Trajectory collection → 4 mechanisms extract experiences
                                                    ↓
                                         PostgreSQL persistence
                                                    ↓
Next task ← Prompt injection ← Token budget management ← Fitness-ranked query
                                                    ↓
                                    Before/after metric comparison → EMA fitness update
```

**Four learning mechanisms:**

| Mechanism | What it does | Cost |
|-----------|-------------|------|
| Experience Extraction | LLM analyzes trajectories → discrete insights | ~$0.05/eval |
| Tool Selection Bandit | Thompson Sampling tracks tool success rates | Zero LLM |
| Round Reflection | Reflexion-style inter-round self-critique | ~$0.02/round |
| Action Principle Distiller | Cross-task principle extraction with A/B testing | Amortized |

**Fitness evaluation** compares before/after metrics (evidence count, confidence, cost) via weighted signals, updated with EMA smoothing (α=0.3). Low-fitness experiences decay and are pruned; high-fitness ones get injected more often.

**A/B testing** prevents self-reinforcing loops: 50% of evaluations run without evolution injection (control), ensuring only genuinely helpful experiences survive.

All evolution mechanisms share an 800-token injection budget, allocated by mechanism priority. This caps the prompt overhead while maximizing learning value.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                 │
│  Evaluation endpoints / SSE streaming / REST API     │
├─────────────────────────────────────────────────────┤
│  Core Orchestration                                  │
│  DiscoveryLoop / AggregationEngine / EvidencePool    │
├─────────────────────────────────────────────────────┤
│  Experiments (Exp Layer)                             │
│  SearchExp / AnalysisExp / SynthesisExp              │
├─────────────────────────────────────────────────────┤
│  Agents                                              │
│  SearchAgent / SynthesisAgent (built on EvoMaster)   │
├─────────────────────────────────────────────────────┤
│  Infrastructure                                      │
│  LLMProviderPool / MCPConnectionPool / CostTracker   │
│  CircuitBreaker / QualityGate / SkillService         │
├─────────────────────────────────────────────────────┤
│  Evolution                                           │
│  ExperienceStore / PromptEnricher / FitnessEvaluator │
│  4 Mechanisms / A-B Testing / Fitness Ranking        │
└─────────────────────────────────────────────────────┘
         │
         ▼
   EvoMaster (Agent Framework)
   BaseAgent / BaseExp / BaseTool / ToolRegistry / SkillRegistry
```

### How the Layers Interact

- **TargetMaster** (or any domain application) defines `EvaluationTask` with domain rules, checklist, and output schema
- **DiscoveryLoop** orchestrates multi-round research: for each round, it creates SearchExp → cleans evidence → creates AnalysisExp → checks gaps → decides whether to continue
- **SearchExp / AnalysisExp** each dynamically create Agent instances with tailored prompts, tool subsets, and LLM assignments
- **Agents** (built on EvoMaster's BaseAgent) execute the actual LLM reasoning and tool calling via the `run() → _step() → _execute_tool()` loop
- **MCP tools** (80+ tools across 11 servers) are bridged into EvoMaster's ToolRegistry via MCPToolWrapper, with automatic JSON Schema generation for LLM function calling
- **Evolution** runs after each round: collects trajectory → extracts experiences → updates fitness → injects into next round's prompts

### Relationship with EvoMaster

Inquiro is **not a fork** of EvoMaster, nor does it reimplement EvoMaster's functionality. Instead, it uses EvoMaster's primitives (BaseAgent, BaseTool, BaseLLM, ToolRegistry, SkillRegistry, Trajectory) directly, and builds **production-grade service enhancements** on top via the decorator pattern:

| EvoMaster Primitive | Inquiro Enhancement | Why |
|---------------------|---------------------|-----|
| `BaseAgent.run()/_step()` | + CostTracker + CancellationToken + SSE events | Production observability and control |
| `BaseTool` | + MCPToolWrapper with dynamic Schema generation | 80+ MCP tools, zero hand-written params |
| `MCPConnection` | + Persistent connections + CircuitBreaker + ResponseCache | Long-running service needs fault tolerance |
| `BaseLLM` | + LLMProviderPool (6 providers, parallel analysis) | Multi-model consensus voting |
| `ToolRegistry` | + Server-based routing + per-agent filtering | 80+ tools need coarse-grained filtering |
| `SkillRegistry` | + CompositeSkillRegistry (multi-directory merge) | Layered project (Inquiro + domain app) |
| `BaseExp` | + Quality gates + feedback retries + async | Production reliability |
| `Trajectory` | + DiscoveryRoundRecord + evolution collection | Closed-loop self-improvement |

EvoMaster's MCPToolManager handles batch-oriented MCP setup; Inquiro's MCPConnectionPool adds persistent connections, circuit breakers, and caching for long-running service deployments. EvoMaster's ConfigManager handles single-file agent-centric config; Inquiro's ConfigLoader handles multi-file service-centric config. These are complementary architectural choices, not redundant implementations.

## Module Overview

| Module | Description |
|--------|-------------|
| `agents/` | Research agents (SearchAgent, SynthesisAgent) built on EvoMaster's BaseAgent, with cost tracking, cancellation, and SSE event emission |
| `api/` | FastAPI application with SSE streaming and REST endpoints |
| `core/` | DiscoveryLoop orchestration, aggregation engine (weighted voting), evidence pool, 3-tier condenser, gap analysis, data models |
| `exps/` | Experiment implementations with quality gates, feedback-driven retries, and parallel multi-model analysis |
| `infrastructure/` | LLM provider pool (6 providers), MCP connection pool (11 servers), cost tracking with budget enforcement, circuit breaker, tool routing |
| `evolution/` | 4 learning mechanisms, experience store (PostgreSQL), fitness evaluation (EMA), prompt enrichment, A/B testing framework |
| `prompts/` | Jinja2 prompt templates and dynamic section builders |
| `skills/` | 7 reusable skills (alias expansion, query templates, evidence grading, convergence rules, search reflection, etc.) |
| `tools/` | MCP tool bridge (auto Schema generation from MCP JSON Schema), schema-validating FinishTool |
| `configs/` | YAML configurations for LLM providers, MCP servers, evaluation modes, ensemble voting |
| `tests/` | 78 test files with comprehensive coverage |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) framework

### Installation

```bash
git clone --recurse-submodules https://github.com/zhengh96/Inquiro.git
cd Inquiro
pip install -e ".[llm,dev]"
```

### Configuration

```bash
cp .env.example .env
# Set your API keys:
#   ANTHROPIC_API_KEY=...
#   OPENAI_API_KEY=...
#   GOOGLE_API_KEY=...
```

### Usage Example

```python
from inquiro.core.types import EvaluationTask
from inquiro.core.runner import EvalTaskRunner

task = EvaluationTask(
    topic="Effectiveness of remote work on productivity",
    rules="## Evaluation Criteria\n- POSITIVE: Strong evidence supports...",
    checklist=["Peer-reviewed studies", "Meta-analyses", "Industry reports"],
    output_schema={"decision": "enum[POSITIVE,CAUTIOUS,NEGATIVE]", "confidence": "float"},
)

runner = EvalTaskRunner(config=your_config)
result = await runner.run(task)
```

### Run Tests

```bash
pytest inquiro/tests/ -v
```

---

## Origin

Inquiro was originally developed as the core evidence research engine within [TargetMaster](https://github.com/zhengh96/TargetMaster), an AI-driven drug target evaluation platform. The relationship is clean:

```
TargetMaster = Domain knowledge (AZ 5R framework, 17 sub-items, expert panel)
    │  injects rules + checklist + output schema
    ▼
Inquiro = Domain-agnostic research engine (this repo)
    │  inherits BaseAgent / BaseExp / BaseTool / Skill / Trajectory
    ▼
EvoMaster = Agent framework
```

TargetMaster evaluates drug targets across 5 dimensions x 17 sub-items, running 50-100 Inquiro agent instances per evaluation, completing in ~30 minutes what traditionally takes a BD team 2-4 weeks.

## License

[Apache License 2.0](LICENSE)

---

<a id="中文"></a>

# Inquiro — 领域无关的证据研究与综合引擎

**基于多模型 AI Agent 的通用证据研究引擎。**

## Inquiro 是什么？

Inquiro 是一个**通用证据研究引擎**，能够自主地从多个数据源搜索、清洗、分析和综合证据，采用多模型 AI Agent 协作。它在设计上是领域无关的——同一套管道可用于药物靶点评估、竞争情报分析、文献综述，或任何需要系统性证据研究的场景。

### 问题

直接问 LLM 一个研究问题，你得到的是基于过时训练数据的"看似合理"的回答。无法验证、没有系统性证据收集、不能防范单模型偏见和幻觉。"问 AI"和"做研究"之间有巨大的鸿沟。

### 解决方案

Inquiro 不问 AI 要答案。它用 AI Agent **做研究**——规划搜索策略、查询专业数据库、清洗去重证据、让多个模型独立分析同一组证据、识别覆盖缺口、迭代直到收敛。每个结论都可追溯到具体来源。

---

## 核心管道：DiscoveryLoop

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DiscoveryLoop（每轮）                               │
│                                                                      │
│  SearchExp ──→ EvidencePipeline ──→ AnalysisExp ──→ GapAnalysis      │
│  (AI 搜索)     (零 LLM 清洗)       (3 模型投票)     (覆盖判断)        │
│                                                                      │
│         收敛？── 否 → 回到 SearchExp（带缺口引导 + 进化经验注入）      │
│                  是 → SynthesisExp（最终综合）                         │
│                                                                      │
│  进化：每轮结束后采集轨迹 → 提取经验 → 更新 fitness → 注入下轮        │
└──────────────────────────────────────────────────────────────────────┘
```

每个研究任务经历**多轮迭代发现循环**：

1. **搜索** — AI Agent 通过 MCP 服务器自主查询数据源，规划搜索策略、追踪引用链
2. **清洗** — 确定性管道（零 LLM 成本）：去重、噪声过滤、来源分类
3. **分析** — 3 个不同 LLM（Claude / GPT / Gemini）独立分析同组证据，加权投票达成共识
4. **缺口检查** — 确定性收敛：5 种停止条件，无 LLM 参与
5. **综合** — 基于所有证据的最终综合，每个结论可追溯到原始来源

---

## 设计思路

### 1. 领域无关是架构约束，不是偶然

Inquiro 代码中**不含任何领域术语**。这不是一个软性指导，而是贯穿整个代码库的架构约束。

领域知识通过唯一的接口进入——`EvaluationTask`：

```python
task = EvaluationTask(
    topic="EGFR 小分子在 NSCLC 中的成药性",
    rules="## 决策框架\n- POSITIVE: 靶点有确认的...",
    checklist=["晶体结构已有", "结合口袋已鉴定", ...],
    output_schema={"decision": "enum[POSITIVE,CAUTIOUS,NEGATIVE]", ...},
)
result = await discovery_loop.run(task)  # Inquiro 不知道 EGFR 是什么
```

这意味着同一个引擎可以评估药物靶点、分析竞争格局、综述学术文献、评估投资机会——只需更换规则、检查清单和输出格式。引擎负责研究过程；调用者定义"好的研究"的标准。

### 2. 多模型共识，而非单模型信任

Inquiro 的每个分析都由 **3 个独立 LLM**（Claude / GPT / Gemini）完成，加权投票：

```yaml
weights:
  claude-sonnet-4-5: 0.40
  gpt-5-2: 0.30
  gemini-3-pro: 0.30
consensus_threshold: 0.70
```

为什么不只用最好的模型？因为我们观察到系统性偏差：

| 模型 | 观察到的偏差 |
|------|------------|
| Claude | 倾向保守（CAUTIOUS），低估正面信号 |
| GPT | 倾向乐观（POSITIVE），低估风险 |
| Gemini | 相对均衡，但对小众主题证据覆盖较弱 |

3 模型投票显著减少了系统性偏差。共识比本身就是天然的置信度指标——全票一致比 2:1 分裂可靠得多。

### 3. 零 LLM 证据清洗

搜索和分析之间的 EvidencePipeline 是**纯确定性的**——零 LLM 成本：

1. **多项拆分** — 一个工具响应可能包含多条证据
2. **内容去重** — 哈希 + URL 规范化
3. **噪声过滤** — 20+ 模式匹配规则
4. **来源标签** — URL → ACADEMIC / PATENT / CLINICAL_TRIAL / REGULATORY / NEWS

为什么必须确定性？因为 3 个分析模型必须看到**完全相同的证据基线**。如果用 LLM 清洗，每次运行会产生微妙不同的证据集，跨模型比较就失去了意义。

### 4. 确定性收敛

DiscoveryLoop 从不问 LLM "证据够不够了？" 而是用 5 个确定性停止条件：

| 条件 | 说明 |
|------|------|
| 覆盖达标 | 检查清单覆盖率 ≥ 阈值 |
| 预算用尽 | 成本上限到达 |
| 最大轮次 | 硬性迭代上限 |
| 收益递减 | 连续 N 轮覆盖率提升 < δ |
| 搜索枯竭 | 当轮新证据 < 阈值 |

这避免了 LLM 停止判断的常见失败模式：模型说"够了"但实际上没够，或者无止境地继续搜索。

### 5. 严格的 LLM 边界

```
API 层      — 无 LLM — HTTP 路由、SSE 推送
Runner      — 无 LLM — 资源管理、并发调度
Exp 层      — 无 LLM — Prompt 构建、质量门、输出验证
Agent 层    — 持有 LLM — 推理、工具调用
```

只有 Agent 层持有 LLM。Exp 层构建 Prompt 但不执行；Agent 层执行 Prompt 但不验证输出。这意味着 Exp 层**完全确定性、完全可测试**——给定输入，验证 Prompt 内容和输出解析，无需 mock LLM。

### 6. 证据 Token 管理：三级冷凝器

单个研究子项可能积累 400+ 条证据。全部喂给 LLM 会 token 溢出并稀释注意力。

| 层级 | 条件 | 策略 |
|------|------|------|
| Tier 0 | ≤150 条 | 直通 |
| Tier 1 | 151-400 条 | 6 信号加权评分，取 Top-160 |
| Tier 2 | 401+ 条 | Tier 1 + 按标签分组 LLM 摘要 |

6 个评分信号：关键词相关性、来源质量、质量标签、结构完整性、新鲜度、期刊影响力。

安全网：保证每种证据类型（ACADEMIC、PATENT、CLINICAL_TRIAL 等）至少保留一条。

### 7. 闭环自进化

Inquiro 实现了完整的**闭环学习系统**——Agent 从历史执行中自动学习改进，通过四个独立机制：

```
任务执行 → 轨迹采集 → 4 个机制提取经验 → PostgreSQL 持久化
                                                ↓
下次任务 ← Prompt 注入 ← Token 预算管理 ← 按 fitness 排序查询
                                                ↓
                              对比前后指标 → EMA 平滑更新 fitness
```

**四个学习机制：**

| 机制 | 做什么 | 成本 |
|------|--------|------|
| Experience Extraction | LLM 分析轨迹 → 离散洞察 | ~$0.05/次 |
| Tool Selection Bandit | Thompson Sampling 追踪工具成功率 | 零 LLM |
| Round Reflection | Reflexion 风格轮间自我反思 | ~$0.02/轮 |
| Action Principle Distiller | 跨任务原则蒸馏 + A/B 测试验证 | 分摊极低 |

**Fitness 评估**对比注入前后的指标（证据数量、置信度、成本），用加权信号和 EMA 平滑（α=0.3）更新。低 fitness 经验衰减并被淘汰，高 fitness 经验被更频繁地注入。

**A/B 测试**防止自我强化的正反馈回路：50% 的评估不注入进化经验（control 组），确保只有真正有益的经验存活。

所有机制共享 800 token 注入预算，按机制优先级分配。控制 Prompt 开销的同时最大化学习价值。

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│  API 层 (FastAPI)                                    │
│  评估接口 / SSE 流式推送 / REST API                    │
├─────────────────────────────────────────────────────┤
│  核心编排                                             │
│  DiscoveryLoop / AggregationEngine / EvidencePool    │
├─────────────────────────────────────────────────────┤
│  实验层 (Exp Layer)                                   │
│  SearchExp / AnalysisExp / SynthesisExp              │
├─────────────────────────────────────────────────────┤
│  Agent 层                                            │
│  SearchAgent / SynthesisAgent（基于 EvoMaster）       │
├─────────────────────────────────────────────────────┤
│  基础设施                                             │
│  LLMProviderPool / MCPConnectionPool / CostTracker   │
│  CircuitBreaker / QualityGate / SkillService         │
├─────────────────────────────────────────────────────┤
│  自进化                                               │
│  ExperienceStore / PromptEnricher / FitnessEvaluator │
│  4 个学习机制 / A-B 测试 / Fitness 排序               │
└─────────────────────────────────────────────────────┘
         │
         ▼
   EvoMaster (Agent 框架)
   BaseAgent / BaseExp / BaseTool / ToolRegistry / SkillRegistry
```

### 层间交互

- **TargetMaster**（或任何领域应用）定义 `EvaluationTask`，注入领域规则、检查清单和输出格式
- **DiscoveryLoop** 编排多轮研究：每轮创建 SearchExp → 清洗证据 → 创建 AnalysisExp → 检查缺口 → 决定是否继续
- **SearchExp / AnalysisExp** 动态创建 Agent 实例，配置定制的 Prompt、工具子集和 LLM
- **Agent**（基于 EvoMaster 的 BaseAgent）执行 LLM 推理和工具调用，通过 `run() → _step() → _execute_tool()` 循环
- **MCP 工具**（11 个服务器 80+ 工具）通过 MCPToolWrapper 桥接到 ToolRegistry，自动生成 JSON Schema 供 LLM function calling
- **Evolution** 在每轮后运行：采集轨迹 → 提取经验 → 更新 fitness → 注入下轮 Prompt

## 模块概览

| 模块 | 说明 |
|------|------|
| `agents/` | 研究 Agent（SearchAgent、SynthesisAgent），基于 EvoMaster 的 BaseAgent，支持成本追踪、取消和 SSE 事件 |
| `api/` | FastAPI 应用，SSE 流式推送和 REST 接口 |
| `core/` | DiscoveryLoop 编排、聚合引擎（加权投票）、证据池、三级冷凝器、缺口分析、数据模型 |
| `exps/` | 实验实现，支持质量门、反馈驱动重试、多模型并行分析 |
| `infrastructure/` | LLM 连接池（6 Provider）、MCP 连接池（11 服务器）、成本追踪与预算执行、熔断器、工具路由 |
| `evolution/` | 4 个学习机制、经验存储（PostgreSQL）、fitness 评估（EMA）、Prompt 增强、A/B 测试框架 |
| `prompts/` | Jinja2 Prompt 模板和动态段落构建器 |
| `skills/` | 7 个可复用 Skill（别名扩展、查询模板、证据评级、收敛规则、搜索反思等） |
| `tools/` | MCP 工具桥接（自动从 MCP JSON Schema 生成参数定义）、带 Schema 验证的 FinishTool |
| `configs/` | YAML 配置：LLM Provider、MCP 服务器、评估模式、集成投票 |
| `tests/` | 78 个测试文件，全面覆盖 |

---

## 快速开始

### 前置条件

- Python 3.10+
- [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) 框架

### 安装

```bash
git clone --recurse-submodules https://github.com/zhengh96/Inquiro.git
cd Inquiro
pip install -e ".[llm,dev]"
```

### 配置

```bash
cp .env.example .env
# 设置 API Key：
#   ANTHROPIC_API_KEY=...
#   OPENAI_API_KEY=...
#   GOOGLE_API_KEY=...
```

### 使用示例

```python
from inquiro.core.types import EvaluationTask
from inquiro.core.runner import EvalTaskRunner

task = EvaluationTask(
    topic="远程办公对生产力的影响",
    rules="## 评估标准\n- POSITIVE: 有强有力的证据支持...",
    checklist=["同行评审研究", "荟萃分析", "行业报告"],
    output_schema={"decision": "enum[POSITIVE,CAUTIOUS,NEGATIVE]", "confidence": "float"},
)

runner = EvalTaskRunner(config=your_config)
result = await runner.run(task)
```

### 运行测试

```bash
pytest inquiro/tests/ -v
```

---

## 项目起源

Inquiro 最初作为 [TargetMaster](https://github.com/zhengh96/TargetMaster)（AI 驱动的药物靶点评估平台）的核心证据研究引擎开发。两者的关系：

```
TargetMaster = 领域知识（AZ 5R 框架、17 子项、专家圆桌）
    │  注入规则 + 检查清单 + 输出格式
    ▼
Inquiro = 领域无关的研究引擎（本仓库）
    │  继承 BaseAgent / BaseExp / BaseTool / Skill / Trajectory
    ▼
EvoMaster = Agent 框架
```

TargetMaster 沿 5 维度 × 17 子项评估药物靶点，每次评估运行 50-100 个 Inquiro Agent 实例，在约 30 分钟内完成传统 BD 团队 2-4 周的工作。

## 开源协议

[Apache License 2.0](LICENSE)
