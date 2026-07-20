"""Alcove — Web search tool via Tavily.

Gives both personas real-time information from the live web. Persona A
uses it for local events, news, current prices, and general knowledge
questions; Persona B uses it for domain-specific lookups
and cooking technique lookups.

Uses Tavily (https://tavily.com) which returns a pre-synthesized answer
plus source URLs — cleaner for a voice interface than raw Google-style
result pages. Requires TAVILY_API_KEY in the environment (set via
runtime.env). Falls back gracefully if the key is missing or the API is
unreachable so a search failure never breaks the persona's voice loop.

Simple HTTP-in-thread pattern so we don't block the
asyncio event loop while waiting on Tavily.
"""

import os
import asyncio
import logging
from typing import Any, Dict, List

import requests

from alcove.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_S = 15.0
MAX_RESULTS = 3


class WebSearch(Tool):
    """Search the live web for real-time information."""

    name = "web_search"
    description = (
        "Search the live web for anything you don't already know from your training or "
        "recent context. Use this for current events, local activities (e.g. 'what's "
        "happening in Colorado Springs on July 4?'), news, business hours, prices, "
        "recipe research, ingredient substitutions, or cooking techniques. Returns a "
        "short synthesized answer plus a few source titles you can mention. Prefer "
        "specific queries with location or date when relevant."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language search query. Be specific — include location, "
                    "date, or ingredient name if relevant."
                ),
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": (
                    "'basic' for quick lookups (default, faster). 'advanced' for "
                    "thorough research when the user asks for details or comparisons."
                ),
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Query Tavily and return a synthesized answer + top sources."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "Please provide a search query."}

        api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
        if not api_key:
            logger.error("web_search: TAVILY_API_KEY is not set in the environment")
            return {"error": "Web search is unavailable — the search API key is not configured."}

        search_depth = kwargs.get("search_depth") or "basic"
        if search_depth not in ("basic", "advanced"):
            search_depth = "basic"

        logger.info("Tool call: web_search query=%r depth=%s", query, search_depth)

        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": True,
            "max_results": MAX_RESULTS,
        }

        def _do_request() -> Dict[str, Any]:
            resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=DEFAULT_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await asyncio.to_thread(_do_request)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "?")
            logger.warning("web_search: Tavily returned HTTP %s for %r", status, query)
            return {"error": f"Web search failed (HTTP {status}). Try rephrasing the question."}
        except requests.RequestException as e:
            logger.warning("web_search: network error contacting Tavily: %s", e)
            return {"error": "Web search failed — couldn't reach the search service."}
        except Exception as e:
            logger.exception("web_search: unexpected error")
            return {"error": f"Web search failed: {e}"}

        answer = (data.get("answer") or "").strip()
        raw_results = data.get("results") or []
        sources: List[Dict[str, str]] = []
        for r in raw_results[:MAX_RESULTS]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            if title and url:
                sources.append({"title": title, "url": url})

        logger.info(
            "web_search: returned %d-char answer + %d sources for %r",
            len(answer),
            len(sources),
            query,
        )

        return {
            "answer": answer or "No synthesized answer available for that query.",
            "sources": sources,
            "query": query,
        }
