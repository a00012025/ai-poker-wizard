#!/usr/bin/env python3
"""Evidence-first GPT coaching orchestration.

The session manager still owns conversation state, credentials, and concrete
tool execution.  This module owns the provider workflow: plan evidence, run a
bounded number of tools, produce a structured answer, and reject unsupported
claims before they can enter user-visible history.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from coach_evidence import (
    COACH_ANSWER_SCHEMA,
    FINAL_COACH_SYSTEM,
    PLANNER_SYSTEM,
    EvidenceBundle,
    audit_evidence_answer,
    display_exact_cards,
    normalize_emoji_cards,
    parse_structured_answer,
    render_safe_fallback,
)
from coach_prompts import _needs_solver_grounding, _normalize_terms


async def run_evidence_chat(
    session: Any,
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
    ctx = session.hand_contexts.get(chat_id)
    if ctx is not None:
        ctx.pop("_followup_node_street", None)

    hand_context = session._build_compact_evidence_context(chat_id)
    history_text = session._history_for_evidence(chat_id)
    bundle = EvidenceBundle()
    bundle.add_text(
        "current_hand", {}, hand_context,
        provenance="analyzed_hand_cache" if ctx else "conversation",
    )

    # Hand type is deterministic and local. Preload it for causal/action
    # questions so the narrator never has to count outs or infer draws, even
    # if the evidence planner forgets to request evaluate_hand.
    raw_hero = ((ctx or {}).get("hand") or {}).get("hero_hand", "")
    if (re.fullmatch(r"[2-9TJQKA][cdhs][2-9TJQKA][cdhs]", raw_hero, re.I)
            and re.search(
                r"(為什麼|为何|why|牌型|牌力|聽牌|听牌|draw|blocker|阻斷|all[- ]?in|全下)",
                user_text,
                re.I,
            )):
        requested_street = next(
            (street for street, words in (
                ("river", ("river", "河牌")),
                ("turn", ("turn", "轉牌", "转牌")),
                ("flop", ("flop", "翻牌")),
            ) if any(word in user_text.lower() for word in words)),
            None,
        )
        states = (ctx or {}).get("street_states") or {}
        state = states.get(requested_street) if requested_street else None
        if state is None:
            state = next(
                (states.get(street) for street in ("river", "turn", "flop")
                 if states.get(street)),
                None,
            )
        board = (state or {}).get("board", "")
        if board:
            evaluated = session._execute_evaluate_hand(
                chat_id, {"hand": raw_hero, "board": board},
            )
            bundle.add_text(
                "evaluate_hand", {"hand": raw_hero, "board": board},
                evaluated, provenance="deterministic_local",
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
    response = await session._openai_response(
        model=session.coach_narrator_model,
        instructions=PLANNER_SYSTEM,
        input=planner_input,
        tools=openai_tools,
        tool_choice="auto",
        max_tool_calls=session.coach_max_tool_calls,
        parallel_tool_calls=True,
        reasoning={"effort": session.coach_narrator_reasoning},
        text={"verbosity": "low"},
        max_output_tokens=700,
        usage_acc=usage_acc,
    )

    for round_index in range(session.coach_max_evidence_rounds):
        calls = [
            item for item in (getattr(response, "output", None) or [])
            if getattr(item, "type", None) == "function_call"
        ]
        if not calls:
            if (grounding_required and tools_called == 0
                    and round_index + 1 < session.coach_max_evidence_rounds):
                grounded_names = {
                    "query_coach_facts", "query_gto", "query_next_actions",
                }
                forced_tools = [
                    spec.as_openai_tool() for spec in specs
                    if spec.name in grounded_names
                ]
                response = await session._openai_response(
                    model=session.coach_narrator_model,
                    instructions=PLANNER_SYSTEM,
                    input=planner_input + (
                        "\n\n這是 solver 策略問題；上一輪沒有查資料。"
                        "本輪必須選一個工具，不得直接回答。"
                    ),
                    tools=forced_tools,
                    tool_choice="required",
                    max_tool_calls=session.coach_max_tool_calls,
                    parallel_tool_calls=True,
                    reasoning={"effort": session.coach_narrator_reasoning},
                    text={"verbosity": "low"},
                    max_output_tokens=700,
                    usage_acc=usage_acc,
                )
                continue
            break

        function_outputs = []
        remaining = max(0, session.coach_max_tool_calls - tools_called)
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
                signature = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                if signature in seen_calls:
                    result = "相同工具與參數已執行，本次重複呼叫未執行。"
                    status = "duplicate"
                else:
                    seen_calls.add(signature)
                    started = time.time()
                    try:
                        result = await session._execute_coach_tool(
                            chat_id, user_text, name, args,
                            on_status=on_status, user_id=user_id,
                            refresh_token=refresh_token,
                        )
                        status = session._tool_status(result)
                    except Exception as exc:
                        session._logger.warning(
                            "[chat=%s] Evidence tool %s failed: %s",
                            chat_id, name, exc,
                        )
                        result = f"工具查詢失敗：{exc}"
                        status = "error"
                    elapsed = time.time() - started
                    tools_called += 1
                    if session.db:
                        try:
                            asyncio.create_task(session.db.save_tool_call(
                                chat_id=chat_id,
                                request_id=request_id,
                                hand_id=session.last_hand_ids.get(chat_id),
                                tool_name=name,
                                tool_args=args,
                                tool_result=result,
                                latency_ms=int(elapsed * 1000),
                            ))
                        except Exception as exc:
                            session._logger.debug(
                                "[chat=%s] save_tool_call dispatch failed: %s",
                                chat_id, exc,
                            )
            item = bundle.add_text(
                name, args, result, status=status,
                provenance="deterministic_runtime",
            )
            function_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps({
                    "evidence_id": item.id,
                    "status": status,
                    "facts": item.facts,
                }, ensure_ascii=False),
            })

        if round_index + 1 >= session.coach_max_evidence_rounds or not function_outputs:
            break
        response = await session._openai_response(
            model=session.coach_narrator_model,
            instructions=PLANNER_SYSTEM,
            input=function_outputs,
            previous_response_id=response.id,
            tools=openai_tools,
            tool_choice="auto",
            max_tool_calls=max(1, session.coach_max_tool_calls - tools_called),
            parallel_tool_calls=True,
            reasoning={"effort": session.coach_narrator_reasoning},
            text={"verbosity": "low"},
            max_output_tokens=700,
            usage_acc=usage_acc,
        )

    solver_items = [
        item for item in bundle.items
        if item.source in {"query_coach_facts", "query_gto"}
        and item.status == "ok"
    ]
    if grounding_required and not solver_items:
        answer = (
            "目前這個 action line 沒有取得可驗證的 solver 資料；"
            "我不會用一般牌理猜測 range、頻率或 EV。"
        )
        session._append_accepted_history(chat_id, user_text, answer)
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
                + "。請重寫且只保留有編號證據支持的具體事實。"
            )
        final_response = await session._openai_response(
            model=session.coach_narrator_model,
            instructions=FINAL_COACH_SYSTEM,
            input=prompt,
            reasoning={"effort": session.coach_narrator_reasoning},
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
            max_output_tokens=session.coach_narrator_max_output_tokens,
            usage_acc=usage_acc,
        )
        parsed = parse_structured_answer(final_response.output_text)
        draft = display_exact_cards(
            _normalize_terms(str(parsed.get("answer") or "").strip())
        )
        refs = parsed.get("fact_refs") or []
        audit = audit_evidence_answer(
            draft, bundle, refs, require_refs=grounding_required,
        )
        violations = list(audit.violations)
        if grounding_required:
            solver_fact_ids = {
                fact_id for item in solver_items for fact_id in item.fact_ids
            }
            if not (set(refs) & solver_fact_ids):
                violations.append("solver answer does not reference solver evidence")
        needs_exact_hero = bool(re.search(
            r"(為什麼|为何|why|這手|这手|我的|hero|all[- ]?in|全下)",
            user_text,
            re.I,
        ))
        if (needs_exact_hero
                and re.fullmatch(r"[2-9TJQKA][cdhs][2-9TJQKA][cdhs]", raw_hero, re.I)):
            normalized_draft = normalize_emoji_cards(draft)
            c1, c2 = raw_hero[:2], raw_hero[2:]
            named_cards = set(re.findall(
                r"[2-9TJQKA][cdhs]", normalized_draft, re.I,
            ))
            if not {c1, c2}.issubset(named_cards):
                violations.append("missing exact Hero combo")
        # ``needs_more_evidence`` is an honest partial-answer state, not a
        # failure. The visible answer must still pass every fact audit.
        if draft and not violations:
            accepted = draft
            break
        last_violations = violations or ["empty answer"]
        session._logger.warning(
            "[chat=%s] Evidence answer audit attempt %s failed: %s",
            chat_id, attempt + 1, ", ".join(last_violations),
        )

    if not accepted:
        accepted = display_exact_cards(render_safe_fallback(bundle))
    session._append_accepted_history(chat_id, user_text, accepted)
    return accepted
