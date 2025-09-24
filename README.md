# multiagent-news-fetcher


# multiagent-news-fetcher

Async news fetcher with text extraction and category balancing. Works standalone or as part of a LangGraph multi-agent pipeline.

## Quickstart

```bash
git clone https://github.com/wik2020wik/multiagent-news-fetcher.git
cd multiagent-news-fetcher
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your OpenAI key
python news_graph_multi.py
```

## Environment

- `OPENAI_API_KEY`: your key (required)
- `OPENAI_MODEL`: default `gpt-4o-mini`
- `OPENAI_EMBED_MODEL`: default `text-embedding-3-small`
- `EMBED_CACHE_PATH`: cache file for keyword embeddings
- `EMBED_SIM_THRESHOLD`: similarity cutoff (0–1)
- `SCORING_MODE`: `embedding`

## Notes
- `.env` is ignored; **never** commit secrets.
- `keyword_embs_1.json` is a sample cache; delete to rebuild if needed.

