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
- 單純列出目前行動者某街各 action 的完整 range，用 query_gto(street, position, include_range=true)，不要逐手查詢。
- 若完整 range 問題明確問 Hero／我的範圍，且當前牌局有 exact Hero combo，query_gto 同時傳 hand 並保留 include_range=true；工具會回完整 range 並額外提供該花色 combo，避免把 169 class 平均套到 exact suit。
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
- 動作頻率只證明 solver 如何 mix；除非證據明列 action EV、regret 或 EV 差距，不得把 100%／高頻動作改寫成「EV 最高」「EV 更高」。
- Mixed strategy 是 solver 的隨機頻率，不是依玩家當下「想不想、敢不敢、能不能承受」來選；只能說按頻率 randomize，或描述主要／次要分支。
- Hand-class 平均（例如 A9s）不是 exact combo（例如 A♦9♦）策略；若工具沒回 exact combo 的 action 分配，就不要替該花色宣告 call／fold／raise 頻率或建議。
- 若 exact combo 在該 node 的 range／strategy 不可用或 reach 近 0，不得把「整體 range 的 action summary」或 hand-class 平均當成該 combo 的答案；要直接說無法可靠判定，類別數字只能標成參考。
- Node 最上方的 Fold／Call／Bet 百分比是整個行動者 range 的頻率，不是 A8s、32s 等 hand class 平均；只有該 class 自己的局部證據列出 action 時，才能把頻率或「主要 call」歸給它。
- Raw equity、percentile 或「range 頂端」不能單獨決定 bet／raise／all-in；應引用 exact combo mix 與已量化的 causal mechanism。高 equity 只描述牌力，不能直接接「因此下注」。
- 「某方有較多 set／同花／順子」本身只是未按 action 條件化的 range ownership。只有因果優先序明列 strong-end 厚度，或 size/range construction 直接支持時，才能說它支撐下注／加注 range；否則只可描述，不可拿來替另一個 combo 編造下注理由。
- 不得自行補「call 是為了保留／誘導對手下注或 bluff」「控制底池」等未量化機制；solver action frequency 本身不證明這些動機。
- 不得把普通 bet／raise 自行命名成 check-raise、c-bet、donk bet 或 probe；只有 evidence 明列該 action-line label 才能使用。
- 對手面對 Hero 下注後的 R 動作是「加注至某尺寸」，不是再次「下注某尺寸」；描述 response range 時要保留這個 actor/action 語義。
- 當前牌局若有 facing_villain_action=bet_to_X／raise_to_X，它是該節點的確定 action 語義；不得把 bet 寫成 raise，或把 raise 寫成 bet。
- 若證據同時列「實戰尺寸」與「solver 近似節點」，回答要保留這個 mapping；不能只把 solver bucket 當成玩家實際下注尺寸。
- 未成牌、頂對、set 等是牌型類別，不等同於 bluff/value 標籤；只有證據明列 bluff/value construction 時才可使用後兩者，否則寫成「未成牌部分／成牌部分」。
- 當前牌局提供 exact Hero combo 且問題在問 Hero／這手牌時，answer 第一次提到它必須保留兩張花色，不可降級成 A9s、QJo 等 hand class。
- 每個 solver/ledger/牌型事實都要在 fact_refs 列出支持它的 E#.#；fact_refs 不顯示在 answer 內。

文字規則：
- 精簡、自然、可學習。通常 2-4 段，每段 1-3 句。
- 用「*核心判斷*」「*為什麼*」「*你要記得*」等 Telegram 單星號標題；不用 #、表格或雙星號。
- 若證據已寫 range 頂端／中段／底端，優先用這個易懂標籤，不要再重複 percentile 數字。
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
_MALFORMED_EMOJI_CLASS_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<rank1>[2-9TJQKA])(?:[cdhs]|[☘♣🔷♦♥♠](?:️)?)"
    r"(?P<rank2>[2-9TJQKA])(?:[cdhs]|[☘♣🔷♦♥♠](?:️)?)"
    r"(?P<kind>[so])(?![A-Za-z0-9])",
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
    """Render exact cards and collapse impossible emoji-decorated classes."""
    repaired = _MALFORMED_EMOJI_CLASS_RE.sub(
        lambda match: (
            match.group("rank1").upper()
            + match.group("rank2").upper()
            + match.group("kind").lower()
        ),
        text or "",
    )
    return _ASCII_CARD_RUN_RE.sub(
        lambda match: cards_to_emoji(match.group(1)),
        repaired,
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
    strategy_evidence = "\n".join(
        "\n".join(item.facts)
        for item in bundle.items
        if item.source in {"query_coach_facts", "query_gto"}
        and item.status == "ok"
    )
    normalized_strategy_evidence = normalize_emoji_cards(strategy_evidence)

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

    # Solver frequency is not an action-EV comparison.  A pure action is safe
    # to call the recommendation, but not "highest EV" unless the evidence
    # explicitly contains a comparative EV/regret statement.
    ev_comparison_claim = re.search(
        r"(?:\bEV\b[^。；\n]{0,24}(?:最高|更高|較高|最好|優於)|"
        r"(?:最高|更高|較高|最好)[^。；\n]{0,16}\bEV\b)",
        normalized_answer,
        re.I,
    )
    supports_ev_comparison = re.search(
        r"(?:\bEV\b[^。；\n]{0,24}(?:最高|更高|較高|差距|loss|regret|優於)|"
        r"(?:最高|更高|較高|差距|loss|regret)[^。；\n]{0,16}\bEV\b)",
        normalized_evidence,
        re.I,
    )
    if ev_comparison_claim and not supports_ev_comparison:
        violations.append("unsupported EV ranking from action frequency")

    current_hand_text = "\n".join(
        "\n".join(item.facts)
        for item in bundle.items if item.source == "current_hand"
    )
    hero_match = re.search(
        r"\bhero=[A-Z0-9+]+\s+([2-9TJQKA][cdhs][2-9TJQKA][cdhs])\b",
        current_hand_text,
        re.I,
    )
    if hero_match:
        hero_combo = hero_match.group(1)
        exact_strategy_claim = False
        for sentence in re.split(r"[。！？\n]", normalized_answer):
            if hero_combo.lower() not in sentence.lower():
                continue
            # "We cannot determine call/fold" is the required honest answer
            # when the exact combo is absent.  It is an epistemic boundary,
            # not a strategy recommendation.  A sentence that also assigns a
            # percentage remains auditable and cannot hide behind a negation.
            denial = re.search(
                r"(?:無法|不能|不可|不足以|沒有足夠|資料不足|無可靠)"
                r"[^。；\n]{0,36}(?:判定|宣稱|下結論|套用|替代|策略|strategy|"
                r"call|fold|跟注|棄牌|下注|加注)",
                sentence,
                re.I,
            )
            if denial and not re.search(r"\d+(?:\.\d+)?%", sentence):
                continue
            if re.search(
                r"(?:solver|GTO|策略|應該|應以|不該|主要|高頻|低頻|純|\d+(?:\.\d+)?%)"
                r"[^。；\n]{0,36}(?:跟注|棄牌|下注|加注|call|fold|raise|bet|"
                r"all[- ]?in|shove)|"
                r"(?:跟注|棄牌|下注|加注|call|fold|raise|bet|all[- ]?in|shove)"
                r"[^。；\n]{0,24}(?:\d+(?:\.\d+)?%|主要|高頻|低頻|純)",
                sentence,
                re.I,
            ):
                exact_strategy_claim = True
                break
        combo_in_strategy = re.search(
            re.escape(hero_combo) +
            r"(?:[^\n]{0,320}(?:solver 動作|exact-combo 分配|策略\s*:)|"
            r"[^\n]{0,320}\n[^\n]{0,180}(?:solver 動作|exact-combo 分配|策略\s*:))",
            normalized_strategy_evidence,
            re.I,
        )
        if exact_strategy_claim and not combo_in_strategy:
            violations.append("exact combo action inferred from hand-class or non-strategy evidence")

        exact_unavailable_pattern = (
            re.escape(hero_combo)
            + r"[^\n]{0,180}(?:頻率近 0|此線極少出現)|"
            + re.escape(hero_combo)
            + r"[^\n]{0,180}\n[^\n]{0,160}Exact combo"
              r"[^\n]{0,120}(?:沒有可用|不可用)"
        )
        exact_unavailable = re.search(
            exact_unavailable_pattern,
            normalized_strategy_evidence,
            re.I,
        )
        if exact_unavailable:
            for sentence in re.split(r"[。！？\n]", normalized_answer):
                if not re.search(
                    r"(?:實戰|你的|你選擇|Hero)?[^。；\n]{0,24}"
                    r"(?:call|fold|raise|bet|all[- ]?in|跟注|棄牌|加注|下注|全下)"
                    r"[^。；\n]{0,24}(?:合理|正確|可接受|符合|偏離|錯誤)|"
                    r"(?:合理|正確|可接受|符合|偏離|錯誤)[^。；\n]{0,24}"
                    r"(?:call|fold|raise|bet|all[- ]?in|跟注|棄牌|加注|下注|全下)",
                    sentence,
                    re.I,
                ):
                    continue
                if re.search(
                    r"(?:無法|不能|不可|不足以|無可靠)[^。；\n]{0,36}"
                    r"(?:判定|宣稱|當成|視為)",
                    sentence,
                    re.I,
                ):
                    continue
                violations.append("unsupported action evaluation at unavailable exact combo")
                break

    # Bind a hand-class action claim to evidence local to that class.  Global
    # node totals are numerically valid, so a flat number whitelist cannot
    # catch a narrator relabelling "whole range Call 91.6%" as "A8s Call
    # 91.6%".  Search only from the class line forward inside the SAME tool
    # item; this permits ordinary class-detail output and fold-equity examples
    # while preventing cross-section joins.
    action_aliases = {
        "call": (r"\bcall\b", r"跟注"),
        "fold": (r"\bfold\b", r"棄牌"),
        "raise": (r"\braise\b", r"加注", r"all[- ]?in", r"全下"),
        "bet": (r"\bbet\b", r"下注"),
        "check": (r"\bcheck\b", r"過牌"),
    }
    try:
        import coach_facts

        for sentence in re.split(r"[。！？\n]", normalized_answer):
            claimed_actions = {
                name for name, aliases in action_aliases.items()
                if any(re.search(alias, sentence, re.I) for alias in aliases)
            }
            if not claimed_actions:
                continue
            class_tokens = {
                token for token in coach_facts.extract_combo_tokens(sentence)
                if re.fullmatch(r"[2-9TJQKA]{2}[so]?", token, re.I)
            }
            if not class_tokens:
                continue
            pure_denial = re.search(
                r"(?:無法|不能|不可|不足以|沒有足夠|資料不足|無可靠)",
                sentence,
                re.I,
            ) and not re.search(
                r"(?:摘要|平均|顯示|多數|主要|高頻|低頻|應該|應以|"
                r"solver[^。；\n]{0,20}(?:call|fold|raise|bet|check|跟注|棄牌|加注|下注|過牌))",
                sentence,
                re.I,
            )
            if pure_denial and not re.search(r"\d+(?:\.\d+)?%", sentence):
                continue
            claimed_pcts = [
                float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%", sentence)
            ]
            for token in class_tokens:
                local_windows = []
                for item in bundle.items:
                    if item.source not in {"query_coach_facts", "query_gto"}:
                        continue
                    normalized_lines = [normalize_emoji_cards(line) for line in item.facts]
                    for index, line in enumerate(normalized_lines):
                        if token.lower() in line.lower():
                            local_windows.append("\n".join(normalized_lines[index:index + 9]))
                supported = False
                for window in local_windows:
                    if not all(
                        any(re.search(alias, window, re.I) for alias in action_aliases[action])
                        for action in claimed_actions
                    ):
                        continue
                    local_numbers = _evidence_numbers(window)
                    if claimed_pcts and any(
                        all(abs(value - allowed) > 1.0 for allowed in local_numbers)
                        for value in claimed_pcts
                    ):
                        continue
                    supported = True
                    break
                if not supported:
                    violations.append(
                        f"hand-class action attributed from node totals: {token}"
                    )
    except Exception:
        pass

    if any(
        re.search(
            r"(?:equity|percentile|勝率|range\s*(?:頂端|頂部)|範圍(?:頂端|頂部))"
            r"[^。\n]{0,96}(?:因此|所以|故|代表|意味)[^。\n]{0,36}"
            r"(?:下注|加注|bet|raise|all[- ]?in|shove)|"
            r"(?:因為|由於)[^。；\n]{0,24}"
            r"(?:equity|percentile|勝率|range\s*(?:頂端|頂部)|範圍(?:頂端|頂部))"
            r"[^。；\n]{0,28}(?:下注|加注|bet|raise|all[- ]?in|shove)",
            sentence,
            re.I,
        )
        for sentence in re.split(r"[。！？\n]", normalized_answer)
    ):
        violations.append("raw equity used to choose aggressive action")

    # Unconditional category ownership cannot silently become an
    # action-conditioned range-construction claim.  The deterministic causal
    # card explicitly opts in when strong-end thickness or size construction
    # has actually been measured for this explanation.
    category_action_causal = re.search(
        r"(?:較多|更多)[^。；\n]{0,24}"
        r"(?:set|三條|同花|順子|兩對|強牌)[^。；\n]{0,32}"
        r"(?:因此|所以|讓|支撐|使得)[^。；\n]{0,32}"
        r"(?:下注|加注|bet|raise|bluff|詐唬)",
        normalized_answer,
        re.I,
    )
    supports_category_causality = (
        "次要機制：range 頂端與強端的厚度" in normalized_evidence
        or "size construction" in evidence
        or "策略分佈" in normalized_evidence
    )
    if category_action_causal and not supports_category_causality:
        violations.append("unconditioned range category used as action cause")

    if re.search(
        r"(?:保留|誘導|引誘)[^。；\n]{0,24}(?:對手|下注|bet|bluff|詐唬|range|範圍)|"
        r"(?:控制底池|pot control)",
        normalized_answer,
        re.I,
    ):
        violations.append("unsupported induced-action explanation")

    if re.search(
        r"(?:不想|不願|不敢|不能承受|怕)[^。；\n]{0,24}"
        r"(?:下注|加注|bet|raise|跟注|call)[^。；\n]{0,24}(?:棄牌|fold)|"
        r"(?:棄牌|fold)[^。；\n]{0,24}(?:不想|不願|不敢|不能承受|怕)",
        normalized_answer,
        re.I,
    ):
        violations.append("mixed strategy personalized as player preference")

    action_line_labels = {
        "check-raise": r"check[- ]?raise|過牌加注",
        "c-bet": r"\bc[- ]?bet\b|持續下注",
        "donk bet": r"\bdonk(?:\s+bet)?\b|領打",
        "probe": r"\bprobe(?:\s+bet)?\b|探測下注",
    }
    for label, pattern in action_line_labels.items():
        if re.search(pattern, normalized_answer, re.I) and not re.search(
            pattern, normalized_evidence, re.I,
        ):
            violations.append(f"unsupported action-line label: {label}")

    for match in re.finditer(
        r"facing_villain_action=(bet|raise)_to_(\d+(?:\.\d+)?)bb",
        current_hand_text,
        re.I,
    ):
        kind, size = match.group(1).lower(), match.group(2)
        if kind == "bet":
            wrong = rf"(?:加注(?:至|到)?|raise(?:\s+to)?)\s*{re.escape(size)}\s*(?:bb)?"
        else:
            wrong = rf"(?:下注|bet)\s*{re.escape(size)}\s*(?:bb)?"
        if re.search(wrong, normalized_answer, re.I):
            violations.append(
                f"facing action semantics mismatch: expected {kind} to {size}bb"
            )

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
    exact_boundary = []
    exact_context = []
    for item in tool_items:
        for line in item.facts:
            stripped = line.strip(" •")
            if re.search(
                r"Exact combo[^\n]{0,100}(?:沒有可用|不可用)|"
                r"(?:此線極少出現|頻率近 0)",
                stripped,
                re.I,
            ):
                exact_boundary.append(stripped)
            elif item.source == "evaluate_hand" or re.search(
                r"數據參考性低|僅供參考", stripped, re.I,
            ):
                exact_context.append(stripped)
    if exact_boundary:
        # Node-level Fold/Call summaries and class averages are intentionally
        # omitted here: after an exact-combo audit failure they are precisely
        # the facts most likely to be misread as the requested suit's verdict.
        rows = []
        seen = set()
        for line in exact_boundary + exact_context:
            key = re.sub(r"\s+", " ", line).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(line)
        return "*核心資料*\n" + "\n".join(f"• {line}" for line in rows[:5])

    full_range_items = [
        item for item in tool_items
        if item.source == "query_gto" and item.args.get("include_range")
    ]
    if full_range_items:
        # The user explicitly requested the range artifact.  A narrator audit
        # failure must not degrade that deterministic output to the first six
        # summary lines and silently discard the combo list (H3874).
        rows = []
        for item in full_range_items:
            rows.extend(item.facts)
        return "*完整 solver 範圍*\n" + "\n".join(rows)

    candidates = []
    for item in tool_items:
        for line in item.facts:
            stripped = line.strip(" •")
            if not stripped or re.search(r"(?:決策數據|下注尺寸選擇)：$", stripped):
                continue
            priority = 2
            if re.search(
                r"solver 動作|exact-combo|Hero range 角色|因果優先序|"
                r"Range equity gate|Range 強端|可辨認強牌|Pot-odds gate|"
                r"Size construction|GTOW removal",
                stripped,
                re.I,
            ):
                priority = 0
            elif re.search(r"equity|percentile|跟注|棄牌|加注|下注", stripped, re.I):
                priority = 1
            candidates.append((priority, len(candidates), stripped))
    seen = set()
    for _, _, line in sorted(candidates):
        key = re.sub(r"\s+", " ", line).lower()
        if key in seen:
            continue
        seen.add(key)
        useful.append(line)
        if len(useful) >= 6:
            break
    if not useful:
        return "目前沒有足夠的已驗證資料回答這個問題；我不會用一般牌理猜測 range 或頻率。"
    return "*核心資料*\n" + "\n".join(f"• {line}" for line in useful[:6])


def repair_guidance_for_violations(violations: Iterable[str]) -> str:
    """Give the narrator concrete, extensible repairs for semantic failures."""
    joined = " | ".join(violations or [])
    guidance = []
    if "EV ranking" in joined:
        guidance.append(
            "不要使用『EV 最高／更高／較高』。頻率不是 EV 排名；改成逐字引用 solver 的 "
            "mix，並用已列出的 range 角色／因果優先序解釋。若問題比較 raise 與 call，"
            "可明說高頻 raise 不代表它的 EV 高於低頻 call。"
        )
    if "unconditioned range category" in joined:
        guidance.append(
            "刪除『更多 set／同花所以支撐下注 range』的因果句；除非證據有 strong-end "
            "因果 gate 或 size construction，否則只把牌型 ownership 當描述。"
        )
    if "blocker target" in joined:
        guidance.append(
            "刪除 blocker 擋住哪類牌／combo 的說法；removal direction 只能說有利或不利。"
        )
    if "literal nuts" in joined:
        guidance.append(
            "把 nuts／nut advantage 改成證據實際提供的強端 proxy 或具名牌型。"
        )
    if "induced-action" in joined:
        guidance.append(
            "刪除『保留／誘導對手下注或 bluff、控制底池』；只說 solver 如何分配這個 "
            "combo，以及證據明列的 range 角色或 size construction。"
        )
    if "personalized as player preference" in joined:
        guidance.append(
            "不要寫『不想／不敢／不能承受某動作就 fold』；mixed strategy 應描述為按頻率 "
            "randomize，並列出主要與次要分支。"
        )
    if "exact combo action" in joined:
        guidance.append(
            "刪除 exact 花色 combo 的 action 建議／頻率；hand-class 平均不可套到該 combo。"
            "若 exact combo 在此 node 不可用或 reach 近 0，就明說無法可靠判定；"
            "整體 range 或 hand-class 的數字只能標成參考，不能評論 Hero 該 call 或 fold。"
        )
    if "hand-class action attributed" in joined:
        guidance.append(
            "不要把 node 最上方的整體 Fold／Call／Bet 百分比稱作某個 hand class 的平均。"
            "若該 class 沒有自己的 action 列，只能說整體 range 如何行動，並明確與 exact "
            "combo 分開；不能寫 A8s／32s 主要 call 或 fold。"
        )
    if "raw equity used" in joined:
        guidance.append(
            "刪除『equity／percentile／range 頂端，因此 bet／raise』。改成先引用 exact "
            "combo 的 solver mix，再用證據明列的牌型角色或 causal mechanism 解釋。"
        )
    if "action-line label" in joined:
        guidance.append(
            "刪除 check-raise／c-bet／donk／probe 等未被 evidence 命名的標籤；"
            "只按 action line 寫『下注、面對加注、跟注』。"
        )
    if "facing action semantics" in joined:
        guidance.append(
            "依 current-hand 的 facing_villain_action 改正動作：bet_to_X 必須寫『下注 X』，"
            "raise_to_X 才能寫『加注至 X』；不要只照 raw R code 猜名稱。"
        )
    if "unsupported action evaluation at unavailable exact combo" in joined:
        guidance.append(
            "exact combo 在此 node 不可用時，不得稱實戰 call／fold／raise『合理、正確、"
            "可接受、錯誤或偏離』；只能記錄實戰動作，並說現有資料無法評價。"
        )
    if not guidance:
        guidance.append("刪除所有未被編號證據逐字或直接支持的具體事實。")
    return "\n".join(f"- {line}" for line in guidance)


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
