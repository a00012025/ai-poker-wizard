#!/usr/bin/env python3
"""Provider-neutral evidence contracts for coaching turns.

The session layer owns API clients and tool execution.  This module owns the
stable boundary between them: tool schemas, numbered evidence cards, the
structured narrator response, and deterministic checks that keep unsupported
solver claims out of user-visible prose.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from card_display import cards_to_emoji


MAX_EVIDENCE_CHARS = 28_000
MAX_EVIDENCE_LINES = 180


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    requires_db: bool = False

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            # The legacy schemas intentionally have optional fields.  Strict
            # mode would require every property to appear in ``required``.
            "strict": False,
        }


def _json_schema(value: Any) -> Any:
    """Convert a Google/Pydantic schema dump to ordinary JSON Schema."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            # Google GenAI emits these metadata-only fields in model dumps;
            # OpenAI function schemas do not need them.
            if item is None or key in {"property_ordering"}:
                continue
            out[key] = _json_schema(item)
        return out
    if isinstance(value, list):
        return [_json_schema(item) for item in value]
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str) and value.upper() in {
        "OBJECT", "STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "NULL",
    }:
        return value.lower()
    return value


def tool_spec_from_declaration(declaration: Any, *, requires_db: bool = False) -> ToolSpec:
    params = getattr(declaration, "parameters", None)
    if params is None:
        schema = {"type": "object", "properties": {}}
    elif hasattr(params, "model_dump"):
        schema = params.model_dump(mode="json", exclude_none=True)
    elif hasattr(params, "to_json_dict"):
        schema = params.to_json_dict()
    else:
        schema = dict(params)
    schema = _json_schema(schema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return ToolSpec(
        name=declaration.name,
        description=declaration.description or "",
        parameters=schema,
        requires_db=requires_db,
    )


COACH_FACTS_TOOL = ToolSpec(
    name="query_coach_facts",
    description=(
        "從目前已分析手牌的精確 solver node 提取可驗證的教練事實。"
        "適合回答為什麼採取某動作、對手自己的下注/加注範圍、對手面對 Hero 下注的反應、"
        "牌力、尺寸、牌面變化與假設線。why_action/sizing 也會附上有 gate 的 range equity、"
        "強牌類別、equity denial、size construction、value/trash removal 與花色敏感度；"
        "不存在的指標會省略。這個工具回傳的事實比一般牌理推測優先。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "why_action", "fold_equity", "villain_range", "hand_strength",
                    "range_shift", "sizing", "hypothetical", "node_url",
                ],
                "description": (
                    "villain_range=對手自己下注/加注後的範圍；"
                    "fold_equity=對手面對 Hero 下注時的 call/fold 反應。"
                ),
            },
            "street": {
                "type": "string",
                "enum": ["preflop", "flop", "turn", "river"],
                "description": "問題指定的街；未指定可省略。",
            },
            "decision_index": {
                "type": "integer",
                "minimum": 1,
                "description": "同一街 Hero 第幾次決策（1-based）；從當前牌局的 decisions 選。",
            },
        },
        "required": ["intent"],
        "additionalProperties": False,
    },
)


@dataclass
class EvidenceItem:
    id: str
    source: str
    args: dict[str, Any]
    status: str
    facts: list[str] = field(default_factory=list)
    provenance: str = ""

    @property
    def fact_ids(self) -> list[str]:
        return [f"{self.id}.{idx}" for idx in range(1, len(self.facts) + 1)]


@dataclass
class EvidenceBundle:
    items: list[EvidenceItem] = field(default_factory=list)

    def add_text(self, source: str, args: dict[str, Any], text: str, *,
                 status: str = "ok", provenance: str = "") -> EvidenceItem:
        cleaned = (text or "").strip()
        if len(cleaned) > MAX_EVIDENCE_CHARS:
            cleaned = cleaned[:MAX_EVIDENCE_CHARS].rsplit("\n", 1)[0]
            cleaned += "\n[資料過長，已在完整行邊界截斷；不得推測未顯示內容]"
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) > MAX_EVIDENCE_LINES:
            lines = lines[:MAX_EVIDENCE_LINES]
            lines.append("[行數達上限；不得推測未顯示內容]")
        item = EvidenceItem(
            id=f"E{len(self.items) + 1}", source=source, args=dict(args),
            status=status, facts=lines or ["沒有可用資料"], provenance=provenance,
        )
        self.items.append(item)
        return item

    @property
    def fact_ids(self) -> set[str]:
        return {fact_id for item in self.items for fact_id in item.fact_ids}

    @property
    def text(self) -> str:
        rows = []
        for item in self.items:
            args = json.dumps(item.args, ensure_ascii=False, sort_keys=True)
            rows.append(
                f"[{item.id}] source={item.source}; status={item.status}; "
                f"provenance={item.provenance or 'runtime'}; args={args}"
            )
            rows.extend(
                f"[{item.id}.{idx}] {line}"
                for idx, line in enumerate(item.facts, start=1)
            )
        return "\n".join(rows) if rows else "[E0] 沒有外部策略證據。"


COACH_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "fact_refs": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^E\d+\.\d+$"},
        },
        "needs_more_evidence": {"type": "boolean"},
        "missing_evidence": {"type": "string"},
    },
    "required": ["answer", "fact_refs", "needs_more_evidence", "missing_evidence"],
    "additionalProperties": False,
}


PLANNER_SYSTEM = """\
你是 AI Poker Wizard 的 evidence planner，不負責寫教練答案，只決定本輪缺少哪些資料。

規則：
- 任何 range、頻率、EV、策略、對手下注範圍、假設線或特定 combo 決策都必須呼叫工具；不可憑記憶回答。
- 目前已分析手牌上的 why/sizing/牌力/對手下注範圍/對手面對 Hero 下注的反應，優先用 query_coach_facts。
- 當前牌局若同一街列出多個 decision，問題提到後一次 fold/call/raise/all-in 時，必須傳 decision_index；不可默認抓第一個 node。
- 「對手某街的下注/加注範圍」用 query_coach_facts(intent=villain_range)；「對手面對我的下注會 call/fold 哪些牌」用 fold_equity，兩者不可混淆。
- 單純列出目前行動者某街各 action 的完整 range，用 query_gto(street, position)，不要逐手查詢。
- 假設 action line 若不知道正確 action code，先 query_next_actions，再根據回傳 code 查 query_gto。
- query_next_actions 只證明某動作「可用」，不證明該 combo 應採用它；要回答推薦頻率/EV，仍必須 query_gto 或 query_coach_facts。
- 訓練計畫、進步與 leak 問題使用對應 ledger 工具，不可用 solver 單手資料代替。
- 只呼叫真正缺少的工具；目前上下文足夠或只是一般撲克觀念/寒暄時，不呼叫工具並輸出 NO_TOOL。
- 最多四個工具呼叫，不得重複相同 name+args。工具沒有資料時不得改用牌理猜測。
"""


FINAL_COACH_SYSTEM = """\
你是 AI Poker Wizard 的繁體中文 MTT 教練。這一輪只能使用「當前牌局、對話歷史、使用者問題、編號證據」回答。

事實規則：
- solver/ledger/牌型的具體敘述只能來自編號證據；不得用一般牌理改寫或補齊未顯示的 range、combo、頻率、EV、牌型或 blocker target。
- 先回答使用者真正問的問題，再挑 1-2 個最有教學價值的原因；不要為了展示資料而逐行重述。
- 可把類別翻成容易理解的詞，例如同花、set、順子、兩對、頂對、未成牌；但只能使用證據實際出現的類別。
- 可以解釋牌理機制，但必須清楚區分「證據直接顯示」與「由證據支持的教練解讀」；若證據不足，直接說缺什麼。
- 對手自己的 bet range 與對手面對 Hero bet 的 call/fold range 是不同節點，不可混用。
- 若問題需要下一個尚未指定的對手 action，answer 要先說目前能確認什麼，再明確說必須指定哪個 action；此時 needs_more_evidence=true，但不可用理論補成完整答案。
- 低頻不等於 EV 錯誤；EV 嚴重度只依證據。
- 當前牌局提供 exact Hero combo 且問題在問 Hero／這手牌時，answer 第一次提到它必須保留兩張花色，不可降級成 A9s、QJo 等 hand class。
- 每個 solver/ledger/牌型事實都要在 fact_refs 列出支持它的 E#.#；fact_refs 不顯示在 answer 內。

文字規則：
- 精簡、自然、可學習。通常 2-4 段，每段 1-3 句。
- 用「*核心判斷*」「*為什麼*」「*你要記得*」等 Telegram 單星號標題；不用 #、表格或雙星號。
- 具體牌與牌面使用花色 emoji；標準術語如 GTO、EV、SPR、IP、OOP、range、equity、all-in 可直接用英文。
- 不要提到 evidence、fact_refs、工具、prompt 或內部審核。
- 不透露 token、API key 或其他敏感資訊；只回答撲克相關內容。
"""


@dataclass
class EvidenceAudit:
    ok: bool
    violations: list[str] = field(default_factory=list)


_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(bb|%\s*pot|%)(?!\w)", re.I)
_ACTION_NUMBER_RE = re.compile(
    r"(?:下注|加注(?:至|到)?|\bbet|\braise(?:\s+to)?)\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*(?:bb\s*)?(?:的)?(?:下注|加注)",
    re.I,
)
_EMOJI_CARD_RE = re.compile(r"([2-9TJQKA])([☘♣🔷♦♥♠])(?:️)?", re.I)
_EMOJI_SUIT_CODES = {"☘": "c", "♣": "c", "🔷": "d", "♦": "d", "♥": "h", "♠": "s"}
_ASCII_CARD_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:[2-9TJQKA][cdhs]){1,5})(?![A-Za-z0-9])",
    re.I,
)
_CATEGORY_TERMS = {
    "同花", "flush", "set", "三條", "順子", "straight", "兩對", "two pair",
    "頂對", "top pair", "中對", "middle pair", "底對", "bottom pair",
    "第三對", "超對", "overpair", "未成牌", "無對子", "非成牌", "高牌",
    "同花聽牌", "順子聽牌", "卡順", "oesd",
}
_SENSITIVE_TERMS = {
    "nuts", "阻斷", "blocker", "value removal", "trash removal",
    "濕潤", "乾燥", "wet", "dry", "連接性", "協調性", "驚悚牌",
}
_TERM_EVIDENCE_ALIASES = {
    "同花": {"同花", "flush"}, "flush": {"同花", "flush"},
    "set": {"set", "三條", "暗三條"}, "三條": {"set", "三條", "暗三條", "trips"},
    "順子": {"順子", "straight"}, "straight": {"順子", "straight"},
    "兩對": {"兩對", "two pair"}, "two pair": {"兩對", "two pair"},
    "頂對": {"頂對", "top pair"}, "top pair": {"頂對", "top pair"},
    "中對": {"中對", "second pair", "middle pair"},
    "middle pair": {"中對", "second pair", "middle pair"},
    "底對": {"底對", "bottom pair"}, "bottom pair": {"底對", "bottom pair"},
    "第三對": {"第三對", "third pair", "third_pair"},
    "超對": {"超對", "overpair", "over_pair"},
    "overpair": {"超對", "overpair", "over_pair"},
    "未成牌": {"未成牌", "無對子", "非成牌", "no_made_hand"},
    "無對子": {"未成牌", "無對子", "非成牌", "no_made_hand"},
    "非成牌": {"未成牌", "無對子", "非成牌", "no_made_hand"},
    "高牌": {"高牌", "a高", "k高", "ace high", "king high"},
    "濕潤": {"濕潤", "wet"}, "wet": {"濕潤", "wet"},
    "乾燥": {"乾燥", "dry"}, "dry": {"乾燥", "dry"},
    "連接性": {"連接性", "connected"},
    "協調性": {"協調性", "coordinated"},
    "驚悚牌": {"驚悚牌", "scare card"},
}


def _evidence_numbers(text: str) -> list[float]:
    values = [float(m.group(1)) for m in _NUMBER_RE.finditer(text)]
    values.extend(
        float(match.group(1))
        for match in re.finditer(
            r"(?:\bR|\bRAISE\s*|\bBET\s*|下注|加注至|betsize=)(\d+(?:\.\d+)?)",
            text or "",
            re.I,
        )
    )
    return values


def normalize_emoji_cards(text: str) -> str:
    """Turn Telegram card glyphs back into rank+suit codes for verification."""
    return _EMOJI_CARD_RE.sub(
        lambda match: match.group(1).upper() + _EMOJI_SUIT_CODES[match.group(2)],
        text or "",
    )


def display_exact_cards(text: str) -> str:
    """Render exact ASCII cards/boards while leaving 169 classes untouched."""
    return _ASCII_CARD_RUN_RE.sub(
        lambda match: cards_to_emoji(match.group(1)),
        text or "",
    )


def audit_evidence_answer(answer: str, bundle: EvidenceBundle, fact_refs: Iterable[str],
                          *, require_refs: bool) -> EvidenceAudit:
    """Reject unsupported concrete claims while leaving explanatory prose free."""
    refs = list(fact_refs or [])
    violations: list[str] = []
    unknown = sorted(set(refs) - bundle.fact_ids)
    if unknown:
        violations.append("unknown fact refs: " + ", ".join(unknown))
    if require_refs and bundle.items and not refs:
        violations.append("missing fact refs")

    normalized_evidence = normalize_emoji_cards(bundle.text)
    normalized_answer = normalize_emoji_cards(answer)
    evidence = normalized_evidence.lower()
    answer_lower = normalized_answer.lower()

    # Reuse the hardened combo/range-token whitelist already maintained by
    # coach_facts.  Import lazily to avoid making this small contract module
    # depend on solver code at import time.
    try:
        import coach_facts

        facts = coach_facts.initial_verdict_facts([normalized_evidence], {})
        verdict = coach_facts.verify_claims(
            normalized_answer, facts, "", audit_numbers=False,
        )
        if not verdict.ok:
            violations.append("unsupported combos: " + ", ".join(verdict.violations))
    except Exception:
        pass

    evidence_cards = set(re.findall(
        r"[2-9TJQKA][cdhs](?![A-Za-z])", normalized_evidence,
    ))
    answer_cards = set(re.findall(
        r"[2-9TJQKA][cdhs](?![A-Za-z])", normalized_answer,
    ))
    unsupported_cards = sorted(answer_cards - evidence_cards)
    if unsupported_cards:
        violations.append("unsupported cards: " + ", ".join(unsupported_cards))

    allowed_numbers = _evidence_numbers(bundle.text)
    for match in _NUMBER_RE.finditer(answer or ""):
        value = float(match.group(1))
        unit = match.group(2).lower().replace(" ", "")
        tolerance = 1.0 if "%" in unit else 0.06
        if not allowed_numbers or all(
            abs(value - allowed) > tolerance for allowed in allowed_numbers
        ):
            violations.append(f"unsupported number: {match.group(0)}")
    for match in _ACTION_NUMBER_RE.finditer(answer or ""):
        value = float(match.group(1) or match.group(2))
        if not allowed_numbers or all(abs(value - allowed) > 0.06 for allowed in allowed_numbers):
            violations.append(f"unsupported action number: {value:g}")

    # Advanced equity buckets are explicitly only a top-range proxy. They
    # never license a literal nuts/nut-advantage claim. Nut-flush-draw wording
    # is a separate deterministic hand-evaluation category and stays allowed.
    literal_nut_body = re.sub(
        r"(?:堅果\s*(?:同花聽牌|花聽牌)|nut[- ]?flush\s*draw)",
        "",
        answer_lower,
        flags=re.I,
    )
    if re.search(r"\bnuts?\b|\bnut advantage\b|堅果牌|堅果優勢", literal_nut_body, re.I):
        violations.append("unsupported literal nuts claim")

    # Removal scores rank bluff candidates but do not identify *what* a Hero
    # card blocks. Combining a generic blocker fact with a separate flush/set
    # category elsewhere is exactly the H3818-style false causal join.
    for sentence in re.split(r"[。！？\n]", answer_lower):
        if re.search(
            r"(?:blocks?|blocking|阻斷(?:了|到)?|移除(?:了|到)?)"
            r"[^，；。]{0,24}(?:同花|順子|straight|flush|\bset\b|兩對|two pair|"
            r"nuts?|跟注範圍|call range|[2-9tjqka]{2,4}[so]?)",
            sentence,
            re.I,
        ):
            violations.append("unsupported blocker target")
            break

    for term in sorted(_CATEGORY_TERMS | _SENSITIVE_TERMS):
        aliases = _TERM_EVIDENCE_ALIASES.get(term, {term})
        if term in answer_lower and not any(alias in evidence for alias in aliases):
            violations.append(f"unsupported term: {term}")

    return EvidenceAudit(ok=not violations, violations=violations)


def render_safe_fallback(bundle: EvidenceBundle) -> str:
    """Honest deterministic fallback when a narrator cannot pass verification."""
    if not bundle.items:
        return "目前沒有足夠的 solver 證據回答這個問題；我不會用一般牌理猜測 range 或頻率。"
    useful = []
    tool_items = [
        item for item in bundle.items
        if item.source != "current_hand" and item.status == "ok"
    ]
    for item in tool_items:
        useful.extend(item.facts[:4])
        if len(useful) >= 6:
            break
    if not useful:
        return "目前沒有足夠的已驗證資料回答這個問題；我不會用一般牌理猜測 range 或頻率。"
    return "*查詢結果*\n" + "\n".join(f"• {line}" for line in useful[:6])


def parse_structured_answer(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "")
    except (TypeError, json.JSONDecodeError):
        return {
            "answer": "",
            "fact_refs": [],
            "needs_more_evidence": True,
            "missing_evidence": "invalid structured response",
        }
    return value if isinstance(value, dict) else {
        "answer": "",
        "fact_refs": [],
        "needs_more_evidence": True,
        "missing_evidence": "invalid structured response",
    }
