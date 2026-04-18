#!/usr/bin/env python3
"""Weekly leak report v2 — EV-ranked clusters + batched LLM coach narratives.

Replaces the old top-level-spot-rate report with the leak_miner clustering
pipeline. Each cluster gets:
  - LLM-written headline + explanation (validated against allowed hand IDs)
  - Deterministic direction label, top hand IDs, EV loss, GTOW practice URL

Public entry points (preserved for PTB JobQueue compatibility):
  - generate_weekly_report(pool, chat_id, period_end=None) -> str | None
  - send_weekly_reports(pool, bot) -> int
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import asyncpg

logger = logging.getLogger("poker_bot")

# ── Display dictionaries ──

_DIRECTION_ZH = {
    "too_passive":    "太 passive",
    "too_aggressive": "太 aggressive",
    "mixed":          "混合方向",
    "aligned":        "頻率大致正確但 EV 有落差",
}

_POT_TYPE_ZH = {
    "SRP":      "SRP",
    "srp":      "SRP",
    "3bet":     "3bet pot",
    "3bp":      "3bet pot",
    "4bet":     "4bet pot",
    "4bp":      "4bet pot",
    "squeezed": "squeezed pot",
    "squeeze":  "squeezed pot",
    "limp":     "limp pot",
    "iso":      "iso pot",
}

_STREET_ZH = {
    "preflop": "翻牌前",
    "flop":    "翻牌",
    "turn":    "轉牌",
    "river":   "河牌",
}

_BOARD_TEXTURE_ZH = {
    "dry":      "乾燥面",
    "wet":      "潮濕面",
    "monotone": "同花面",
    "paired":   "公對面",
}


# ── Hand ID validator ──

# Strict: require leading H prefix to avoid matching dates/percentages/EV figures.
_HAND_ID_PATTERN = re.compile(r'\bH(\d{3,6})\b')


@dataclass
class ClusterNarrative:
    cluster_id:   str
    headline:     str
    explanation:  str
    practice_hint: str
    is_fallback:  bool = False


def _validate_narrative_hand_ids(
    narrative: ClusterNarrative,
    allowed_hand_ids: set[int],
) -> bool:
    """True iff every H<id> mentioned in headline+explanation is in allowed set.

    Empty mentions are vacuously valid. Bare numerics (no `H` prefix) are
    ignored — the LLM is instructed to use H-prefixed IDs.
    """
    text = f"{narrative.headline} {narrative.explanation}"
    mentioned = {int(m) for m in _HAND_ID_PATTERN.findall(text)}
    return mentioned.issubset(allowed_hand_ids)


# ── Templated fallback ──

def _templated_narrative(cluster, cluster_id: str) -> ClusterNarrative:
    """Deterministic narrative when LLM is unavailable or hallucinates.

    No prose claims, no invented hand IDs — just facts from the cluster.
    """
    desc = _cluster_descriptor(cluster)
    direction_zh = _DIRECTION_ZH.get(cluster.aggression_label, cluster.aggression_label)
    n = cluster.sample_count
    headline = f"{desc} 偏離（{direction_zh}）"
    explanation = (
        f"本週共 {n} 次決策落在這個分類，總 EV 落差 "
        f"{cluster.total_ev_loss_bb:.2f} bb。"
        f"方向標籤：{direction_zh}。"
    )
    practice_hint = "到 GTO Wizard 對應的 spot 練習，先比對自己最常選的線。"
    return ClusterNarrative(
        cluster_id=cluster_id,
        headline=headline[:60],
        explanation=explanation,
        practice_hint=practice_hint,
        is_fallback=True,
    )


# ── Cluster descriptor (used in both render + fallback) ──

def _cluster_descriptor(cluster) -> str:
    """Build the parenthetical descriptor for a cluster.

    Preflop: "{pot_type_zh}, {hero_pos}"
    Postflop: "{pot_type_zh} {street_zh}, {board_texture_zh}"
    """
    key = cluster.key
    pot_zh = _POT_TYPE_ZH.get(key.pot_type or "", key.pot_type or "")
    if key.street == "preflop":
        if pot_zh:
            return f"{pot_zh}, {key.hero_pos}"
        return f"{key.hero_pos} 翻牌前"
    street_zh = _STREET_ZH.get(key.street, key.street)
    texture_zh = _BOARD_TEXTURE_ZH.get(key.board_texture or "", key.board_texture or "")
    base = f"{pot_zh} {street_zh}".strip()
    if texture_zh:
        return f"{base}, {texture_zh}"
    return base


# ── Markdown renderer ──

def _render_cluster_line(
    cluster,
    narrative: ClusterNarrative,
    url: str | None,
    rank: int,
) -> str:
    """One cluster → markdown block."""
    desc = _cluster_descriptor(cluster)
    direction_zh = _DIRECTION_ZH.get(cluster.aggression_label, cluster.aggression_label)
    n = cluster.sample_count
    total_loss = f"-{cluster.total_ev_loss_bb:.2f}bb"

    top_hands = cluster.top_hand_ids[:3]
    # Best-effort per-hand EV loss display: we don't get individual losses
    # back from mine_clusters (only top_hand_ids) — show ID only if losses
    # not surfaced. The miner currently doesn't return per-hand losses.
    hand_strs = [f"H{hid}" for hid in top_hands]
    hands_line = " · ".join(hand_strs) if hand_strs else "(無)"

    lines = [
        f"**{rank}. {narrative.headline}**（{desc}, n={n}, {total_loss}）",
        f"   方向：{direction_zh}",
        f"   最貴決策：{hands_line}",
    ]
    if url:
        lines.append(f"   → [到 GTO Wizard 練這個 spot]({url})")
    if narrative.explanation:
        lines.append(f"   {narrative.explanation}")
    if narrative.practice_hint:
        lines.append(f"   {narrative.practice_hint}")
    return "\n".join(lines)


# ── LLM coach layer ──

CLUSTER_NARRATIVES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["cluster_id", "headline", "explanation", "practice_hint"],
        "properties": {
            "cluster_id":    {"type": "string"},
            "headline":      {"type": "string"},
            "explanation":   {"type": "string"},
            "practice_hint": {"type": "string"},
        },
    },
}

_COACH_PROMPT_TEMPLATE = """你是一位 MTT 撲克教練。我會給你本週 hero 最重要的幾個 leak cluster（按 EV 損失排序），每個 cluster 包含它的結構化資料和最貴決策的 hand_id。

你的任務：為每個 cluster 產生一個簡短的教練訊息（繁體中文），包含：
- headline: ≤40 字，命名這個 pattern，例如「LJ 開牌 + HJ flat 後乾板 cbet 過頻」
- explanation: 2-3 句話，解釋為什麼這是 leak，**必須引用我給你的 hand_id**（格式 H1234），不可以發明任何 hand_id。不可以引用我沒給你的數字。
- practice_hint: 1 句話，告訴 hero 練習時該注意什麼

**嚴格禁止：**
- 發明 hand_id
- 引用我沒給你的 EV 數字或頻率百分比
- 說「GTO 建議 XX%」這種我沒給你的數字

嚴格遵循 response schema。只輸出 cluster narrative 陣列，不要其他內容。

Clusters:
{clusters_json}
"""


def _build_clusters_payload(clusters: list) -> list[dict]:
    """Compact JSON payload for the LLM (no internal-only fields)."""
    out = []
    for i, c in enumerate(clusters):
        out.append({
            "cluster_id":         str(i),
            "descriptor":         _cluster_descriptor(c),
            "spot_category":      c.key.spot_category,
            "street":             c.key.street,
            "pot_type":           c.key.pot_type,
            "hero_pos":           c.key.hero_pos,
            "villain_pos":        c.key.villain_pos,
            "board_texture":      c.key.board_texture,
            "sample_count":       c.sample_count,
            "total_ev_loss_bb":   round(c.total_ev_loss_bb, 2),
            "aggression_label":   c.aggression_label,
            "top_hand_ids":       [f"H{h}" for h in c.top_hand_ids[:3]],
            "effective_bb_median": round(c.effective_bb_median, 1),
        })
    return out


async def _call_llm_for_narratives(
    model_client,
    prompt: str,
) -> list[dict] | None:
    """Invoke Gemini Pro structured output. Returns parsed array or None on error.

    Uses the same google-genai client pattern as gemini_session.py.
    """
    try:
        from google.genai import types as genai_types  # type: ignore
    except Exception:
        genai_types = None  # type: ignore

    try:
        if genai_types is not None:
            config = genai_types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=CLUSTER_NARRATIVES_SCHEMA,
            )
        else:
            config = {
                "temperature": 0.3,
                "response_mime_type": "application/json",
                "response_schema": CLUSTER_NARRATIVES_SCHEMA,
            }

        # Model name: use Pro by convention, but if the client wraps it we
        # accept whatever model the caller injected. We expect a google-genai
        # AsyncClient interface (client.aio.models.generate_content) since
        # that's what gemini_session.py uses.
        response = await asyncio.wait_for(
            model_client.aio.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config,
            ),
            timeout=120,
        )
    except Exception as e:
        logger.warning(f"weekly_report: LLM call failed: {e}")
        return None

    text = getattr(response, "text", None) or ""
    if not text.strip():
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to peel a fenced code block
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if not m:
            logger.warning("weekly_report: LLM response was not valid JSON")
            return None
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            logger.warning("weekly_report: LLM response not parseable as JSON even after stripping fences")
            return None

    if not isinstance(data, list):
        return None
    return data


def _parse_narratives_response(
    raw: list[dict],
    clusters: list,
) -> list[ClusterNarrative]:
    """Convert raw LLM dicts → ClusterNarrative objects, indexed by cluster_id.

    Missing entries fall back to None (will be templated by caller).
    """
    by_id: dict[str, ClusterNarrative] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cluster_id", ""))
        if not cid:
            continue
        by_id[cid] = ClusterNarrative(
            cluster_id=cid,
            headline=str(item.get("headline", "")).strip(),
            explanation=str(item.get("explanation", "")).strip(),
            practice_hint=str(item.get("practice_hint", "")).strip(),
            is_fallback=False,
        )

    result: list[ClusterNarrative] = []
    for i, _c in enumerate(clusters):
        cid = str(i)
        if cid in by_id:
            result.append(by_id[cid])
        else:
            result.append(ClusterNarrative(
                cluster_id=cid, headline="", explanation="",
                practice_hint="", is_fallback=True,
            ))
    return result


async def generate_cluster_narratives(
    clusters: list,
    model_client=None,
    max_retries: int = 1,
) -> list[ClusterNarrative]:
    """Batched LLM coach layer.

    Single Gemini call with array schema. Validates that every hand_id
    mentioned in each narrative appears in the corresponding cluster's
    top_hand_ids. Retries once on validation failure. Falls back to
    templated narrative for any cluster whose narrative is invalid after
    retries (or if model_client is None).
    """
    if not clusters:
        return []

    if model_client is None:
        return [_templated_narrative(c, str(i)) for i, c in enumerate(clusters)]

    payload = _build_clusters_payload(clusters)
    prompt = _COACH_PROMPT_TEMPLATE.format(
        clusters_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )

    allowed_per_cluster: list[set[int]] = [
        set(c.top_hand_ids) for c in clusters
    ]

    narratives: list[ClusterNarrative] | None = None

    for attempt in range(max_retries + 1):
        raw = await _call_llm_for_narratives(model_client, prompt)
        if raw is None:
            continue
        candidate = _parse_narratives_response(raw, clusters)

        # Validate per-cluster
        all_valid = True
        for nar, allowed in zip(candidate, allowed_per_cluster):
            if nar.is_fallback:
                # Missing from response — count as invalid, will retry.
                all_valid = False
                break
            if not _validate_narrative_hand_ids(nar, allowed):
                all_valid = False
                break

        if all_valid:
            narratives = candidate
            break
        else:
            logger.info(
                f"weekly_report: narrative validation failed on attempt "
                f"{attempt + 1}/{max_retries + 1}"
            )
            narratives = candidate  # remember last attempt for partial fallback

    if narratives is None:
        # All attempts failed entirely → full templated fallback.
        return [_templated_narrative(c, str(i)) for i, c in enumerate(clusters)]

    # Per-cluster validation: replace any invalid narrative with templated.
    final: list[ClusterNarrative] = []
    for i, (nar, c) in enumerate(zip(narratives, clusters)):
        if (
            nar.is_fallback
            or not nar.headline
            or not _validate_narrative_hand_ids(nar, allowed_per_cluster[i])
        ):
            final.append(_templated_narrative(c, str(i)))
        else:
            final.append(nar)
    return final


# ── URL builder wrapper ──

def _build_url_for_cluster(cluster) -> str | None:
    """Wrap build_trainer_url with safe error handling."""
    try:
        from gtow_trainer_url import build_trainer_url, SpotNotSupportedError
    except Exception as e:
        logger.warning(f"weekly_report: gtow_trainer_url import failed: {e}")
        return None
    try:
        return build_trainer_url(
            spot_category=cluster.key.spot_category,
            street=cluster.key.street,
            effective_bb=cluster.effective_bb_median or 30.0,
            pot_type=cluster.key.pot_type,
        )
    except SpotNotSupportedError as e:
        logger.info(f"weekly_report: skipping URL for cluster ({e})")
        return None
    except Exception as e:
        logger.warning(f"weekly_report: build_trainer_url failed: {e}")
        return None


# ── Top-level renderers ──

def _empty_state_message() -> str:
    return (
        "📊 本週沒有足夠的手牌來產生報告"
        "（至少需要 5 個同類型決策才會標記為 leak）"
    )


def _render_report(
    clusters: list,
    narratives: list[ClusterNarrative],
    period_start: datetime,
    period_end: datetime,
    total_hands: int | None = None,
    total_decisions: int | None = None,
) -> str:
    """Assemble the full markdown report."""
    start_str = period_start.strftime("%m/%d")
    end_str = period_end.strftime("%m/%d")

    header_bits = [f"📊 週報（{start_str} – {end_str}）"]
    if total_hands is not None:
        header_bits.append(f"{total_hands} 手")
    if total_decisions is not None:
        header_bits.append(f"{total_decisions} 決策")
    header = " · ".join(header_bits)

    lines = [header, "", "💸 本週 EV 落差 Top 5：", ""]
    total_loss = 0.0
    for i, (cluster, narrative) in enumerate(zip(clusters, narratives), start=1):
        url = _build_url_for_cluster(cluster)
        lines.append(_render_cluster_line(cluster, narrative, url, i))
        lines.append("")
        total_loss += cluster.total_ev_loss_bb

    lines.append(f"🎯 本週累積 EV 落差：-{total_loss:.2f}bb")
    return "\n".join(lines)


# ── Public entry points ──

async def generate_weekly_report(
    pool: asyncpg.Pool,
    chat_id: int,
    period_end: datetime | None = None,
    model_client=None,
) -> str | None:
    """Generate the weekly leak report for a user.

    Returns:
        Formatted markdown string, or None if there's no data to report
        (preserves the existing JobQueue contract: None → skip sending).
    """
    if period_end is None:
        period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=7)

    try:
        from leak_miner import mine_clusters
    except Exception as e:
        logger.error(f"weekly_report: leak_miner import failed: {e}")
        return None

    try:
        clusters = await mine_clusters(
            pool, chat_id, period_start, period_end,
            min_sample=5, top_k=5,
        )
    except Exception as e:
        logger.warning(f"[chat={chat_id}] weekly_report: mine_clusters failed: {e}")
        return None

    if not clusters:
        # Preserve existing JobQueue behavior — return None so the sender
        # skips this user instead of spamming an empty-state message.
        return None

    # Optional totals for header (best-effort, not fatal if it fails)
    totals = await _fetch_period_totals(pool, chat_id, period_start, period_end)

    narratives = await generate_cluster_narratives(
        clusters=clusters,
        model_client=model_client,
        max_retries=1,
    )

    return _render_report(
        clusters=clusters,
        narratives=narratives,
        period_start=period_start,
        period_end=period_end,
        total_hands=totals.get("total_hands") if totals else None,
        total_decisions=totals.get("total_decisions") if totals else None,
    )


async def _fetch_period_totals(
    pool: asyncpg.Pool,
    chat_id: int,
    start: datetime,
    end: datetime,
) -> dict | None:
    """Best-effort fetch of total hands + decisions for the header."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS total_decisions,
                    COUNT(DISTINCT hand_history_id) AS total_hands
                FROM deviations
                WHERE chat_id = $1
                  AND created_at >= $2
                  AND created_at <  $3
                """,
                chat_id, start, end,
            )
        if not row:
            return None
        return {
            "total_decisions": int(row["total_decisions"] or 0),
            "total_hands":     int(row["total_hands"] or 0),
        }
    except Exception as e:
        logger.debug(f"weekly_report: totals fetch failed: {e}")
        return None


async def send_weekly_reports(
    pool: asyncpg.Pool,
    bot: Any,
    model_client=None,
) -> int:
    """Generate and send weekly reports to all active users with deviations.

    Returns the count of reports sent.
    """
    async with pool.acquire() as conn:
        users = await conn.fetch(
            """
            SELECT DISTINCT d.chat_id
            FROM deviations d
            JOIN users u ON u.user_id = d.chat_id
            WHERE u.is_active = TRUE
              AND d.created_at >= NOW() - INTERVAL '7 days'
            """
        )

    sent = 0
    for user_row in users:
        chat_id = user_row["chat_id"]
        try:
            report = await generate_weekly_report(
                pool, chat_id, model_client=model_client,
            )
            if report:
                await bot.send_message(
                    chat_id=chat_id,
                    text=report,
                    parse_mode="Markdown",
                )
                sent += 1
                logger.info(f"Sent weekly report to chat_id={chat_id}")
        except Exception as e:
            logger.warning(
                f"Failed to send weekly report to chat_id={chat_id}: {e}"
            )
            continue

    logger.info(f"Weekly report: sent {sent}/{len(users)} reports")
    return sent
