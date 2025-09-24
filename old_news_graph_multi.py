# news_graph_multi.py
import os, asyncio, datetime, json, math
from typing import List, Optional, TypedDict, Dict, Any

import feedparser, httpx, trafilatura
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph

# -----------------------------------------------------------------------------
# Setup (same as your original, kept stable)
# -----------------------------------------------------------------------------
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env")

oai = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "keyword_embs.json")
EMBED_SIM_THRESHOLD = float(os.getenv("EMBED_SIM_THRESHOLD", "0.75"))
SCORING_MODE = os.getenv("SCORING_MODE", "embedding")  # kept for future use

FEEDS: Dict[str, List[str]] = {
    "AI": [
        "https://www.reddit.com/r/MachineLearning/.rss",
        "https://www.artificialintelligence-news.com/feed/",
        "https://www.theguardian.com/technology/artificialintelligenceai/rss",
    ],
    "Tech": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
    ],
    "Finance": [
        "https://www.marketwatch.com/feeds/topstories",
        "https://www.ft.com/?format=rss",
        "https://www.google.com/finance/"
    ],
}

KEYWORDS = [
    "AI", "NVIDIA", "OPENAI", "APPLE", "GOOGLE", "TESLA", "FACEBOOK", "AMAZON", "SAMSUNG",
    "chip", "FED", "ECB", "inflation", "etf", "Bitcoin", "crypto", "BTC", "GPT"
    "TRUMP", "POWELL", "Warren Buffet", "MUSK", "Dimon", "Solomon",
    "Soros", "Dalio", "Ackman"
]
MAX_TOTAL = 18
TOP_N = 9

def keyword_score(title: str, text: str) -> float:
    base = (title + " " + text[:800]).lower()
    return sum(2.0 if kw.isupper() else 1.0 for kw in KEYWORDS if kw.lower() in base)

# -----------------------------------------------------------------------------
# Minimal embedding cache support (re-using your approach)
# -----------------------------------------------------------------------------
def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def _load_keyword_cache() -> Dict[str, List[float]]:
    if os.path.exists(EMBED_CACHE_PATH):
        try:
            with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: v for k, v in data.items() if k in KEYWORDS}
        except Exception:
            return {}
    return {}

def _save_keyword_cache(cache: Dict[str, List[float]]):
    try:
        with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def _embed_text(txt: str) -> List[float]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=txt)
    return resp.data[0].embedding

_KEYWORD_EMBS: Dict[str, List[float]] = _load_keyword_cache()
_missing = [kw for kw in KEYWORDS if kw not in _KEYWORD_EMBS]
if _missing:
    for kw in _missing:
        _KEYWORD_EMBS[kw] = _embed_text(kw)
    _save_keyword_cache(_KEYWORD_EMBS)

# -----------------------------------------------------------------------------
# State schema
# -----------------------------------------------------------------------------
class Article(TypedDict, total=False):
    title: str
    link: str
    category: str
    text: str
    summary: Optional[str]
    score: float
    risk: Optional[str]         # added: critic’s risk note
    grounded: Optional[bool]    # added: critic’s grounding verdict
    entities: Optional[Dict[str, Any]]  # added: tagger’s JSON (companies/people/tickers/topics)

class AgentState(TypedDict):
    articles: List[Article]
    top_n: int
    digest_md: Optional[str]

# -----------------------------------------------------------------------------
# Agents / Nodes
# -----------------------------------------------------------------------------
async def fetcher_node(state: AgentState) -> AgentState:
    items: List[Article] = []
    for cat, feeds in FEEDS.items():
        for url in feeds:
            f = feedparser.parse(url)
            for e in f.entries[:6]:
                link = getattr(e, "link", None)
                title = getattr(e, "title", "Untitled")
                if not link:
                    continue
                items.append({"title": title, "link": link, "category": cat, "text": "", "summary": None, "score": 0.0})

    # Deduplicate/trim
    seen, uniq = set(), []
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)
    items = uniq[:MAX_TOTAL]

    # Pull HTML + extract text
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        async def extract(it: Article):
            try:
                r = await client.get(it["link"])
                html = r.text if r.status_code == 200 else ""
                text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
                it["text"] = text
            except Exception:
                it["text"] = ""
            return it
        items = await asyncio.gather(*(extract(it) for it in items))

    state["articles"] = items
    return state

async def summarizer_node(state: AgentState) -> AgentState:
    async def do_sum(a: Article):
        if not a["text"]:
            a["summary"] = "(no extractable content)"
            a["score"] = keyword_score(a["title"], "")
            return a
        prompt = f"""You are a financial/tech news summarizer.
Summarize the article in 3–5 crisp, extractive bullets (no speculation).
Add sentiment tag in square brackets at the end of the last bullet: [Bullish|Neutral|Cautious].
Use ONLY facts present in the text.

Title: {a['title']}
Text:
{a['text'][:8000]}"""
        rsp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        a["summary"] = rsp.choices[0].message.content.strip()
        a["score"] = keyword_score(a["title"], a["text"])
        return a
    state["articles"] = await asyncio.gather(*(do_sum(a) for a in state["articles"]))
    return state

async def critic_node(state: AgentState) -> AgentState:
    """Risk & Grounding Critic: flags rumor/risk and checks extractiveness."""
    async def critique(a: Article):
        if not a.get("summary"):
            a["risk"] = "no-summary"
            a["grounded"] = False
            return a
        prompt = f"""You are a Risk & Grounding Critic for financial/tech news.
Given the article text and its 3–5 bullet summary, do two things:

1) Is each bullet extractive (appears in the text) and free of speculation? Reply "YES" only if all bullets are extractive; else "NO".
2) Provide a short risk note: one of ["rumor-risk","low-evidence","balanced","well-sourced"].

Return strict JSON with keys: grounded (true/false), risk (string), issues (array of bullet indices with problems).

TITLE: {a['title']}
TEXT:
{a['text'][:6000]}
SUMMARY:
{a['summary']}"""
        rsp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )
        data = json.loads(rsp.choices[0].message.content)
        a["grounded"] = bool(data.get("grounded", False))
        a["risk"] = str(data.get("risk", "balanced"))
        # Optionally: you could redact problematic bullets here
        return a
    state["articles"] = await asyncio.gather(*(critique(a) for a in state["articles"]))
    return state

async def tagger_node(state: AgentState) -> AgentState:
    """Entity & Topic Tagger: companies, people, tickers, topics."""
    async def tag(a: Article):
        if not a.get("text"):
            return a
        prompt = f"""Extract entities and topics as JSON with keys:
companies (array), people (array), tickers (array), topics (array: e.g., 'chips','LLMs','rates','ETF').
Base only on the TEXT below. No guessing.

TEXT:
{a['text'][:6000]}"""
        rsp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )
        a["entities"] = json.loads(rsp.choices[0].message.content)
        return a
    state["articles"] = await asyncio.gather(*(tag(a) for a in state["articles"]))
    return state

def curator_node(state: AgentState) -> AgentState:
    # Prioritise grounded items; then keyword score; keep category balance.
    state["articles"].sort(key=lambda a: (1 if a.get("grounded") else 0, a.get("score", 0.0)), reverse=True)
    per_cat_limit = max(1, state["top_n"] // 3)
    kept, counts = [], {"AI":0, "Tech":0, "Finance":0}
    for a in state["articles"]:
        if counts[a["category"]] < per_cat_limit:
            kept.append(a)
            counts[a["category"]] += 1
        if len(kept) >= state["top_n"]:
            break
    if len(kept) < state["top_n"]:
        chosen = set(id(x) for x in kept)
        for a in state["articles"]:
            if id(a) in chosen:
                continue
            kept.append(a)
            if len(kept) >= state["top_n"]:
                break
    state["articles"] = kept
    return state

def formatter_node(state: AgentState) -> AgentState:
    today = datetime.date.today().isoformat()
    lines = [f"# Daily AI • Tech • Finance — {today}", ""]
    for cat in ["AI", "Tech", "Finance"]:
        section = [a for a in state["articles"] if a["category"] == cat]
        if not section:
            continue
        lines.append(f"## {cat}")
        for a in section:
            risk = a.get("risk", "balanced")
            grounded = "Grounded✅" if a.get("grounded") else "Needs-check⚠️"
            lines.append(f"### [{a['title']}]({a['link']})  \n*{grounded} • risk={risk}*")
            lines.append(a.get("summary") or "(no summary)")
            ents = a.get("entities") or {}
            if ents:
                lines.append(f"_Entities:_ {ents.get('companies', [])}  \n_Tickers:_ {ents.get('tickers', [])}  \n_Topics:_ {ents.get('topics', [])}")
            lines.append("")
    state["digest_md"] = "\n".join(lines)
    return state

# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("fetcher", fetcher_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("critic", critic_node)
    g.add_node("curator", curator_node)
    g.add_node("tagger", tagger_node)
    g.add_node("formatter", formatter_node)

    g.set_entry_point("fetcher")
    g.add_edge("fetcher", "summarizer")
    g.add_edge("summarizer", "critic")
    g.add_edge("critic", "curator")
    g.add_edge("curator", "tagger")
    g.add_edge("tagger", "formatter")
    g.set_finish_point("formatter")
    return g.compile()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()
    result: AgentState = asyncio.run(app.ainvoke({"articles": [], "top_n": TOP_N, "digest_md": None}))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"daily_digest_{stamp}.md"

    header = f"_(Generated {stamp})_\n\n"
    content = header + (result.get("digest_md") or "(No content)")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Wrote {out_path}")
