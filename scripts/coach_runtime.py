#!/usr/bin/env python3
"""Evidence-first GPT coaching orchestration.

The session manager still owns conversation state, credentials, and concrete
tool execution.  This module owns the provider workflow: plan evidence, run a
bounded number of tools, produce a structured answer, and reject unsupported
claims before they can enter user-visible history.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from coach_evidence import (
    COACH_ANSWER_SCHEMA,
    FINAL_COACH_SYSTEM,
    PLANNER_SYSTEM,
    EvidenceBundle,
    audit_evidence_answer,
    display_exact_cards,
    normalize_emoji_cards,
    parse_structured_answer,
    repair_guidance_for_violations,
    render_safe_fallback,
)
from coach_prompts import FOLLOWUP_REQUEST, _needs_solver_grounding, _normalize_terms

_FULL_RANGE_REQUEST_RE = re.compile(
    r"(?:範圍|range|哪些(?:手)?牌|哪些\s*combo|combo\s*(?:範圍|range))",
    re.I,
)


@dataclass(frozen=True)
class WorkflowDeps:
    get_hand_context: Callable[[int], dict | None]
    clear_followup_node_street: Callable[[int], None]
    build_evidence_context: Callable[[int], str]
    history_for_evidence: Callable[[int], str]
    evaluate_hand: Callable[[int, dict], str]
    execute_tool: Callable[..., Any]
    model_response: Callable[..., Any]
    accept_history: Callable[[int, str, str], None]
    tool_status: Callable[[str], str]
    model: str
    max_tool_calls: int
    max_evidence_rounds: int
    reasoning: str
    max_output_tokens: int
    logger: logging.Logger
    record_tool_call: Callable[..., Any] | None = None
    analyze_hand: Callable[..., Any] | None = None
    build_teaching_block: Callable[[dict], str] | None = None
    generate_initial: Callable[..., Any] | None = None
    extract_followups: Callable[[str], tuple[str, list[str]]] | None = None


class ChatWorkflow:
    """Provider-free evidence workflow built from explicit callables."""

    def __init__(self, deps: WorkflowDeps):
        self.deps = deps

    async def run(self, chat_id: int, user_text: str, **kwargs) -> str:
        return await run_evidence_chat(self.deps, chat_id, user_text, **kwargs)

    async def prepare_hand(self, chat_id: int, hand: dict, **kwargs) -> dict:
        if self.deps.analyze_hand is None:
            raise RuntimeError("hand analysis is not configured")
        result = self.deps.analyze_hand(chat_id, hand, **kwargs)
        return await result if inspect.isawaitable(result) else result

    async def coach_hand(
        self,
        chat_id: int,
        context: dict,
        *,
        hand_description: str,
        user_text: str,
        source_instruction: str,
        on_status=None,
        user_id: int | None = None,
        refresh_token: str | None = None,
        usage_acc: dict | None = None,
        disable_tools: bool = False,
    ) -> str:
        if not all(
            (
                self.deps.build_teaching_block,
                self.deps.generate_initial,
                self.deps.extract_followups,
            )
        ):
            raise RuntimeError("initial coaching is not configured")
        teaching_block = self.deps.build_teaching_block(context)
        solver_data = (
            "逐街 solver 結果已另行顯示；以下 deterministic 教學骨架是本段唯一事實來源。"
            if teaching_block
            else context.get("text", "")
        )
        prompt = (
            f"{source_instruction}\n\n"
            f"手牌摘要：\n{hand_description}\n\n"
            f"用戶要求：\n{user_text}\n\n"
            f"已驗證 solver 事實：\n{solver_data}\n\n"
            "逐街 summary 已由系統顯示；請總評整手，並從教學骨架挑 1–2 個"
            "最有價值的 range／action 邏輯自然解釋，不要重新解析或改寫動作。\n\n"
            f"{teaching_block}\n\n{FOLLOWUP_REQUEST}"
        )
        result = self.deps.generate_initial(
            chat_id,
            prompt,
            context,
            user_text,
            on_status=on_status,
            user_id=user_id,
            refresh_token=refresh_token,
            usage_acc=usage_acc,
            disable_tools=disable_tools,
        )
        response = await result if inspect.isawaitable(result) else result
        answer, followups = self.deps.extract_followups(response)
        context["followup_questions"] = followups
        return answer


async def _record_tool_call(deps: WorkflowDeps, **payload) -> None:
    if deps.record_tool_call is None:
        return
    try:
        result = deps.record_tool_call(**payload)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        deps.logger.debug(
            "[chat=%s] save_tool_call dispatch failed: %s",
            payload.get("chat_id"),
            exc,
        )


async def run_evidence_chat(
    deps: WorkflowDeps,
    chat_id: int,
    user_text: str,
    *,
    on_status=None,
    user_id: int | None = None,
    refresh_token: str | None = None,
    usage_acc: dict | None = None,
    request_id: str = "-",
    tool_specs: list[Any] | None = None,
) -> str:
    """Plan evidence, execute bounded tools, narrate, then fact-audit."""
    ctx = deps.get_hand_context(chat_id)
    deps.clear_followup_node_street(chat_id)

    hand_context = deps.build_evidence_context(chat_id)
    history_text = deps.history_for_evidence(chat_id)
    bundle = EvidenceBundle()
    bundle.add_text(
        "current_hand",
        {},
        hand_context,
        provenance="analyzed_hand_cache" if ctx else "conversation",
    )

    # Hand type is deterministic and local. Preload it for causal/action
    # questions so the narrator never has to count outs or infer draws, even
    # if the evidence planner forgets to request evaluate_hand.
    raw_hand = (ctx or {}).get("hand") or {}
    raw_hero = raw_hand.get("hero_hand") or (ctx or {}).get("hero_hand", "")
    hero_position = (
        (ctx or {}).get("hero_position") or raw_hand.get("hero_position") or ""
    )
    if re.fullmatch(r"[2-9TJQKA][cdhs][2-9TJQKA][cdhs]", raw_hero, re.I) and re.search(
        r"(為什麼|为何|why|牌型|牌力|聽牌|听牌|draw|blocker|阻斷|all[- ]?in|全下)",
        user_text,
        re.I,
    ):
        requested_street = next(
            (
                street
                for street, words in (
                    ("river", ("river", "河牌")),
                    ("turn", ("turn", "轉牌", "转牌")),
                    ("flop", ("flop", "翻牌")),
                )
                if any(word in user_text.lower() for word in words)
            ),
            None,
        )
        states = (ctx or {}).get("street_states") or {}
        state = states.get(requested_street) if requested_street else None
        if state is None:
            state = next(
                (
                    states.get(street)
                    for street in ("river", "turn", "flop")
                    if states.get(street)
                ),
                None,
            )
        board = (state or {}).get("board", "")
        if board:
            evaluated = deps.evaluate_hand(
                chat_id,
                {"hand": raw_hero, "board": board},
            )
            bundle.add_text(
                "evaluate_hand",
                {"hand": raw_hero, "board": board},
                evaluated,
                provenance="deterministic_local",
            )

    specs = tool_specs or []
    openai_tools = [spec.as_openai_tool() for spec in specs]
    planner_input = (
        f"當前牌局：\n{hand_context}\n\n"
        f"對話歷史：\n{history_text}\n\n"
        f"使用者問題：\n{user_text}\n\n"
        f"已預載證據（不要重複查）：\n{bundle.text}"
    )
    grounding_required = _needs_solver_grounding(user_text)
    seen_calls: set[str] = set()
    tools_called = 0
    response = await deps.model_response(
        model=deps.model,
        instructions=PLANNER_SYSTEM,
        input=planner_input,
        tools=openai_tools,
        tool_choice="auto",
        max_tool_calls=deps.max_tool_calls,
        parallel_tool_calls=True,
        reasoning={"effort": deps.reasoning},
        text={"verbosity": "low"},
        max_output_tokens=700,
        usage_acc=usage_acc,
    )

    for round_index in range(deps.max_evidence_rounds):
        calls = [
            item
            for item in (getattr(response, "output", None) or [])
            if getattr(item, "type", None) == "function_call"
        ]
        if not calls:
            has_strategy_evidence = any(
                item.source in {"query_coach_facts", "query_gto"}
                and item.status == "ok"
                for item in bundle.items
            )
            if (
                grounding_required
                and not has_strategy_evidence
                and round_index + 1 < deps.max_evidence_rounds
            ):
                grounded_names = {"query_coach_facts", "query_gto"}
                forced_tools = [
                    spec.as_openai_tool()
                    for spec in specs
                    if spec.name in grounded_names
                ]
                response = await deps.model_response(
                    model=deps.model,
                    instructions=PLANNER_SYSTEM,
                    input=planner_input
                    + (
                        "\n\n這是 solver 策略問題，但目前仍沒有 range／"
                        "exact-combo 策略證據。query_next_actions 只可發現"
                        "動作代碼，不能支持策略結論；本輪必須呼叫 "
                        "query_coach_facts 或 query_gto，不得直接回答。"
                    ),
                    tools=forced_tools,
                    tool_choice="required",
                    max_tool_calls=deps.max_tool_calls,
                    parallel_tool_calls=True,
                    reasoning={"effort": deps.reasoning},
                    text={"verbosity": "low"},
                    max_output_tokens=700,
                    usage_acc=usage_acc,
                )
                continue
            break

        function_outputs = []
        remaining = max(0, deps.max_tool_calls - tools_called)
        for call in calls[:remaining]:
            name = call.name
            try:
                args = json.loads(call.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except Exception as exc:
                args = {}
                result = f"工具參數無法解析：{exc}"
                status = "error"
            else:
                if (
                    name == "query_gto"
                    and args.get("position")
                    and _FULL_RANGE_REQUEST_RE.search(user_text or "")
                ):
                    # Preserve the user's range intent even when the planner
                    # also attaches ``hand`` for exact-combo grounding.  The
                    # formatter and image queue need an explicit signal to
                    # return both artifacts instead of choosing one branch.
                    args["include_range"] = True
                # A Hero range query can include ``hand`` without losing the
                # whole-range breakdown. Enrich it deterministically so the
                # narrator never applies a 169-class average to the exact suit
                # combo merely because the planner omitted one optional arg.
                if (
                    name == "query_gto"
                    and not args.get("hand")
                    and re.fullmatch(
                        r"[2-9TJQKA][cdhs][2-9TJQKA][cdhs]",
                        raw_hero,
                        re.I,
                    )
                    and str(args.get("position") or "").upper()
                    == str(hero_position).upper()
                    and re.search(
                        r"(?:\bHero\b|我的|我在|我方|這手|这手)",
                        user_text,
                        re.I,
                    )
                ):
                    args["hand"] = raw_hero
                signature = (
                    f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                )
                if signature in seen_calls:
                    result = "相同工具與參數已執行，本次重複呼叫未執行。"
                    status = "duplicate"
                else:
                    seen_calls.add(signature)
                    started = time.time()
                    try:
                        result = await deps.execute_tool(
                            chat_id,
                            user_text,
                            name,
                            args,
                            on_status=on_status,
                            user_id=user_id,
                            refresh_token=refresh_token,
                        )
                        status = deps.tool_status(result)
                    except Exception as exc:
                        deps.logger.warning(
                            "[chat=%s] Evidence tool %s failed: %s",
                            chat_id,
                            name,
                            exc,
                        )
                        result = f"工具查詢失敗：{exc}"
                        status = "error"
                    elapsed = time.time() - started
                    tools_called += 1
                    await _record_tool_call(
                        deps,
                        chat_id=chat_id,
                        request_id=request_id,
                        tool_name=name,
                        tool_args=args,
                        tool_result=result,
                        latency_ms=int(elapsed * 1000),
                    )
            item = bundle.add_text(
                name,
                args,
                result,
                status=status,
                provenance="deterministic_runtime",
            )
            function_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(
                        {
                            "evidence_id": item.id,
                            "status": status,
                            "facts": item.facts,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        if round_index + 1 >= deps.max_evidence_rounds or not function_outputs:
            break
        response = await deps.model_response(
            model=deps.model,
            instructions=PLANNER_SYSTEM,
            input=function_outputs,
            previous_response_id=response.id,
            tools=openai_tools,
            tool_choice="auto",
            max_tool_calls=max(1, deps.max_tool_calls - tools_called),
            parallel_tool_calls=True,
            reasoning={"effort": deps.reasoning},
            text={"verbosity": "low"},
            max_output_tokens=700,
            usage_acc=usage_acc,
        )

    solver_items = [
        item
        for item in bundle.items
        if item.source in {"query_coach_facts", "query_gto"} and item.status == "ok"
    ]
    # ``query_next_actions`` only discovers legal action codes/sizes; it does
    # not contain a range or exact-combo strategy.  A generated one-street
    # hypothetical commonly needs that discovery first.  If the planner stops
    # there, deterministically resolve the full hypothetical fact card instead
    # of returning a false no-data answer or narrating availability as strategy.
    hypothetical = bool(
        re.search(r"(如果|假設|假如|what\s*if|\bif\b)", user_text, re.I)
    )
    has_fact_tool = any(spec.name == "query_coach_facts" for spec in specs)
    if (
        grounding_required
        and hypothetical
        and not solver_items
        and has_fact_tool
        and tools_called < deps.max_tool_calls
    ):
        street = next(
            (
                name
                for name, words in (
                    ("river", ("river", "河牌")),
                    ("turn", ("turn", "轉牌", "转牌")),
                    ("flop", ("flop", "翻牌")),
                )
                if any(word in user_text.lower() for word in words)
            ),
            None,
        )
        args = {"intent": "hypothetical"}
        if street:
            args["street"] = street
        signature = (
            f"query_coach_facts:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        )
        if signature not in seen_calls:
            seen_calls.add(signature)
            started = time.time()
            try:
                result = await deps.execute_tool(
                    chat_id,
                    user_text,
                    "query_coach_facts",
                    args,
                    on_status=on_status,
                    user_id=user_id,
                    refresh_token=refresh_token,
                )
                status = deps.tool_status(result)
            except Exception as exc:
                deps.logger.warning(
                    "[chat=%s] Deterministic hypothetical fetch failed: %s",
                    chat_id,
                    exc,
                )
                result = f"工具查詢失敗：{exc}"
                status = "error"
            elapsed = time.time() - started
            tools_called += 1
            bundle.add_text(
                "query_coach_facts",
                args,
                result,
                status=status,
                provenance="deterministic_runtime",
            )
            await _record_tool_call(
                deps,
                chat_id=chat_id,
                request_id=request_id,
                tool_name="query_coach_facts",
                tool_args=args,
                tool_result=result,
                latency_ms=int(elapsed * 1000),
            )
        solver_items = [
            item
            for item in bundle.items
            if item.source in {"query_coach_facts", "query_gto"} and item.status == "ok"
        ]
    if grounding_required and not solver_items:
        answer = (
            "目前這個 action line 沒有取得可驗證的 solver 資料；"
            "我不會用一般牌理猜測 range、頻率或 EV。"
        )
        deps.accept_history(chat_id, user_text, answer)
        return answer

    final_input = (
        f"當前牌局：\n{hand_context}\n\n"
        f"對話歷史：\n{history_text}\n\n"
        f"使用者問題：\n{user_text}\n\n"
        f"編號證據：\n{bundle.text}"
    )
    accepted = ""
    last_violations: list[str] = []
    for attempt in range(3):
        prompt = final_input
        if last_violations:
            prompt += (
                "\n\n上一版未通過事實檢查："
                + "；".join(last_violations)
                + "。請重寫且只保留有編號證據支持的具體事實。\n"
                + repair_guidance_for_violations(last_violations)
            )
        final_response = await deps.model_response(
            model=deps.model,
            instructions=FINAL_COACH_SYSTEM,
            input=prompt,
            reasoning={"effort": deps.reasoning},
            text={
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "coach_answer",
                    "description": "Verified poker coaching answer and internal fact references",
                    "strict": True,
                    "schema": COACH_ANSWER_SCHEMA,
                },
            },
            max_output_tokens=deps.max_output_tokens,
            usage_acc=usage_acc,
        )
        parsed = parse_structured_answer(final_response.output_text)
        draft = display_exact_cards(
            _normalize_terms(str(parsed.get("answer") or "").strip())
        )
        refs = parsed.get("fact_refs") or []
        audit = audit_evidence_answer(
            draft,
            bundle,
            refs,
            require_refs=grounding_required,
        )
        violations = list(audit.violations)
        if grounding_required:
            solver_fact_ids = {
                fact_id for item in solver_items for fact_id in item.fact_ids
            }
            if not (set(refs) & solver_fact_ids):
                violations.append("solver answer does not reference solver evidence")
        needs_exact_hero = bool(
            re.search(
                r"(為什麼|为何|why|這手|这手|我的|hero)",
                user_text,
                re.I,
            )
        )
        if needs_exact_hero and re.fullmatch(
            r"[2-9TJQKA][cdhs][2-9TJQKA][cdhs]", raw_hero, re.I
        ):
            normalized_draft = normalize_emoji_cards(draft)
            c1, c2 = raw_hero[:2], raw_hero[2:]
            named_cards = set(
                re.findall(
                    r"[2-9TJQKA][cdhs]",
                    normalized_draft,
                    re.I,
                )
            )
            if not {c1, c2}.issubset(named_cards):
                violations.append("missing exact Hero combo")
        # ``needs_more_evidence`` is an honest partial-answer state, not a
        # failure. The visible answer must still pass every fact audit.
        if draft and not violations:
            accepted = draft
            break
        last_violations = violations or ["empty answer"]
        deps.logger.warning(
            "[chat=%s] Evidence answer audit attempt %s failed: %s",
            chat_id,
            attempt + 1,
            ", ".join(last_violations),
        )

    if not accepted:
        accepted = display_exact_cards(render_safe_fallback(bundle))
    deps.accept_history(chat_id, user_text, accepted)
    return accepted
