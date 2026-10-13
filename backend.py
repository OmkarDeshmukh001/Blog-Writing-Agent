from __future__ import annotations

import logging
import operator
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "generated_blogs"

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup",
                       "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(...,
                          description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # optional planner hints from the API (do not replace Plan schema)
    requested_audience: Optional[str]
    requested_tone: Optional[str]
    requested_blog_kind: Optional[str]
    enable_research: Optional[bool]
    generate_images: bool

    # artifact location
    blog_id: str

    # workers
    sections: Annotated[List[tuple[int, str]],
                        operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    image_assets: List[dict]
    image_errors: List[str]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""


def router_node(state: State) -> dict:
    # Optional API control: skip Tavily entirely. When omitted/True, the
    # existing router still decides closed_book vs hybrid vs open_book.
    if state.get("enable_research") is False:
        return {
            "needs_research": False,
            "mode": "closed_book",
            "queries": [],
            "recency_days": 3650,
        }

    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (
            d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}


# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None
    requested_kind = state.get("requested_blog_kind")
    requested_audience = state.get("requested_audience")
    requested_tone = state.get("requested_tone")

    hint_lines = []
    if requested_audience:
        hint_lines.append(f"Requested audience: {requested_audience}")
    if requested_tone:
        hint_lines.append(f"Requested tone: {requested_tone}")
    if requested_kind and not forced_kind:
        hint_lines.append(f"Requested blog_kind: {requested_kind}")
    hints = ("\n".join(hint_lines) + "\n") if hint_lines else ""

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{hints}"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"
    elif requested_kind:
        plan.blog_kind = requested_kind  # type: ignore[assignment]
    if requested_audience:
        plan.audience = requested_audience
    if requested_tone:
        plan.tone = requested_tone

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================


def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(
        state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    if state.get("generate_images") is False:
        return {
            "md_with_placeholders": merged_md,
            "image_specs": [],
        }

    planner = llm.with_structured_output(GlobalImagePlan)
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError(
            "No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _safe_image_filename(name: str) -> str:
    base = Path(str(name or "")).name
    base = re.sub(r"[^a-zA-Z0-9._-]", "_", base).strip("._")
    if not base:
        base = "image.png"
    if Path(base).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        base = f"{Path(base).stem or 'image'}.png"
    return base


def _blog_dir(blog_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", blog_id)
    if not safe_id:
        raise ValueError("Invalid blog_id")
    path = (OUTPUT_ROOT / safe_id).resolve()
    if path.parent != OUTPUT_ROOT.resolve():
        raise ValueError("Invalid blog_id")
    return path


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    image_assets: List[dict] = []
    image_errors: List[str] = []

    blog_id = state.get("blog_id") or uuid.uuid4().hex[:12]
    out_dir = _blog_dir(blog_id)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (md or "").strip():
        logger.warning("Skipping file write: markdown is empty for blog_id=%s", blog_id)
        return {
            "final": "",
            "image_assets": [],
            "image_errors": ["Generated markdown was empty; no file was written."],
        }

    for spec in image_specs:
        placeholder = spec.get("placeholder") or ""
        filename = _safe_image_filename(spec.get("filename") or "image.png")
        out_path = images_dir / filename

        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                img_bytes = _gemini_generate_image_bytes(spec.get("prompt") or "")
                out_path.write_bytes(img_bytes)
            img_md = f"![{spec.get('alt', '')}](images/{filename})\n*{spec.get('caption', '')}*"
            md = md.replace(placeholder, img_md)
            image_assets.append(
                {
                    "filename": filename,
                    "alt": spec.get("alt") or "",
                    "caption": spec.get("caption") or "",
                    "placeholder": placeholder,
                }
            )
        except Exception as e:
            logger.exception("Image generation failed for %s", filename)
            image_errors.append(f"{filename}: {e}")
            prompt_block = (
                f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                f"> **Alt:** {spec.get('alt', '')}\n>\n"
                f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                f"> **Error:** {e}\n"
            )
            md = md.replace(placeholder, prompt_block)

    leftover = re.findall(r"\[\[IMAGE_\d+\]\]", md)
    for token in leftover:
        md = md.replace(token, "> **[IMAGE PLACEHOLDER NOT REPLACED]**\n")
        image_errors.append(f"Unreplaced placeholder: {token}")

    if not md.strip():
        logger.warning("Refusing to write empty markdown for blog_id=%s", blog_id)
        return {
            "final": "",
            "image_assets": image_assets,
            "image_errors": image_errors + ["Final markdown was empty."],
        }

    (out_dir / "blog.md").write_text(md, encoding="utf-8")
    return {
        "final": md,
        "image_assets": image_assets,
        "image_errors": image_errors,
    }


# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {
                        "research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
graph = app


def new_blog_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def run_blog_generation(
    topic: str,
    *,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
    blog_kind: Optional[str] = None,
    enable_research: Optional[bool] = None,
    generate_images: bool = True,
    blog_id: Optional[str] = None,
) -> dict:
    """Invoke the existing LangGraph workflow and return a serializable result."""
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic is required")

    bid = blog_id or new_blog_id()
    initial: State = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": date.today().isoformat(),
        "recency_days": 3650,
        "requested_audience": audience,
        "requested_tone": tone,
        "requested_blog_kind": blog_kind,
        "enable_research": enable_research,
        "generate_images": generate_images,
        "blog_id": bid,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "image_assets": [],
        "image_errors": [],
        "final": "",
    }

    result = graph.invoke(initial)
    plan: Optional[Plan] = result.get("plan")
    final_md = (result.get("final") or "").strip()
    if not final_md:
        final_md = (result.get("merged_md") or "").strip()
    if not final_md:
        raise RuntimeError("Workflow finished but produced empty markdown.")

    out_dir = _blog_dir(bid)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "blog.md"
    if not md_path.exists() or not md_path.read_text(encoding="utf-8").strip():
        md_path.write_text(final_md, encoding="utf-8")

    evidence = result.get("evidence") or []
    evidence_dump = [
        e.model_dump() if hasattr(e, "model_dump") else e for e in evidence
    ]

    return {
        "id": bid,
        "title": plan.blog_title if plan else topic,
        "markdown": final_md,
        "status": "completed",
        "mode": result.get("mode") or "",
        "needs_research": bool(result.get("needs_research")),
        "queries": result.get("queries") or [],
        "evidence": evidence_dump,
        "blog_kind": plan.blog_kind if plan else blog_kind,
        "audience": plan.audience if plan else audience,
        "tone": plan.tone if plan else tone,
        "images": result.get("image_assets") or [],
        "image_errors": result.get("image_errors") or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
    }
