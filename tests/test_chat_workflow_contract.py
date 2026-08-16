import asyncio
import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coach_evidence import ToolSpec, normalize_emoji_cards
from coach_runtime import ChatWorkflow, WorkflowDeps


def _response(*, calls=(), text="", response_id="response"):
    return SimpleNamespace(output=list(calls), output_text=text, id=response_id)


def _call(name, arguments, call_id="call-1", *, encode=True):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments) if encode else arguments,
        call_id=call_id,
    )


def _answer(text, refs=(), *, needs_more_evidence=False):
    return _response(
        text=json.dumps(
            {
                "answer": text,
                "fact_refs": list(refs),
                "needs_more_evidence": needs_more_evidence,
                "missing_evidence": "more data needed" if needs_more_evidence else "",
            },
            ensure_ascii=False,
        )
    )


def _tool(name="query_gto"):
    return ToolSpec(
        name=name,
        description="test tool",
        parameters={"type": "object", "properties": {}},
    )


class Harness:
    def __init__(
        self,
        responses=(),
        *,
        contexts=None,
        tool_result="solver fact",
        max_tool_calls=4,
        max_evidence_rounds=1,
    ):
        self.responses = list(responses)
        self.contexts = contexts or {}
        self.tool_result = tool_result
        self.accepted = []
        self.executed = []
        self.recorded = []
        self.model_calls = []
        self.evaluated = []

        async def model_response(**kwargs):
            self.model_calls.append(kwargs)
            if callable(self.responses):
                return await self.responses(**kwargs)
            return self.responses.pop(0)

        async def execute_tool(chat_id, user_text, name, args, **kwargs):
            self.executed.append((chat_id, name, dict(args)))
            if isinstance(self.tool_result, Exception):
                raise self.tool_result
            if callable(self.tool_result):
                return self.tool_result(name, args)
            return self.tool_result

        async def record_tool_call(**kwargs):
            self.recorded.append(kwargs)

        def evaluate_hand(chat_id, args):
            self.evaluated.append((chat_id, dict(args)))
            return "QdJs：高牌"

        def context_text(chat_id):
            context = self.contexts.get(chat_id) or {}
            hand = context.get("hand") or {}
            hero = hand.get("hero_hand") or ""
            return f"hero={hand.get('hero_position', '')} {hero}".strip()

        self.deps = WorkflowDeps(
            get_hand_context=self.contexts.get,
            clear_followup_node_street=lambda chat_id: None,
            build_evidence_context=context_text,
            history_for_evidence=lambda chat_id: f"history-{chat_id}",
            evaluate_hand=evaluate_hand,
            execute_tool=execute_tool,
            model_response=model_response,
            accept_history=lambda chat_id, question, answer: self.accepted.append(
                (chat_id, question, answer)
            ),
            record_tool_call=record_tool_call,
            tool_status=lambda result: "ok",
            model="coach-model",
            max_tool_calls=max_tool_calls,
            max_evidence_rounds=max_evidence_rounds,
            reasoning="low",
            max_output_tokens=900,
            logger=logging.getLogger("test_chat_workflow_contract"),
        )
        self.workflow = ChatWorkflow(self.deps)


def test_prepare_hand_supports_sync_and_async_analyzers():
    sync_calls = []

    def sync_analyze(chat_id, hand, **kwargs):
        sync_calls.append((chat_id, hand, kwargs))
        return {"hand": hand, "kind": "sync"}

    async def async_analyze(chat_id, hand, **kwargs):
        return {"hand": hand, "kind": "async", "kwargs": kwargs}

    harness = Harness()
    sync_workflow = ChatWorkflow(replace(harness.deps, analyze_hand=sync_analyze))
    async_workflow = ChatWorkflow(replace(harness.deps, analyze_hand=async_analyze))

    sync_result = asyncio.run(
        sync_workflow.prepare_hand(1, {"hero_hand": "AsKs"}, user_id=2)
    )
    async_result = asyncio.run(
        async_workflow.prepare_hand(3, {"hero_hand": "QhQd"}, refresh_token="r")
    )

    assert sync_result["kind"] == "sync"
    assert sync_calls == [(1, {"hero_hand": "AsKs"}, {"user_id": 2})]
    assert async_result == {
        "hand": {"hero_hand": "QhQd"},
        "kind": "async",
        "kwargs": {"refresh_token": "r"},
    }


def test_prepare_hand_requires_analyzer():
    with pytest.raises(RuntimeError, match="hand analysis is not configured"):
        asyncio.run(Harness().workflow.prepare_hand(1, {}))


def test_coach_hand_requires_all_initial_coaching_ports():
    with pytest.raises(RuntimeError, match="initial coaching is not configured"):
        asyncio.run(
            Harness().workflow.coach_hand(
                1,
                {},
                hand_description="hand",
                user_text="analyze",
                source_instruction="source",
            )
        )


def test_coach_hand_uses_verified_context_and_replaces_stale_followups():
    prompts = []

    async def generate_initial(chat_id, prompt, context, user_text, **kwargs):
        prompts.append((chat_id, prompt, context, user_text, kwargs))
        return "verified answer"

    context = {
        "text": "raw solver facts",
        "followup_questions": ["stale question"],
    }
    harness = Harness()
    workflow = ChatWorkflow(
        replace(
            harness.deps,
            build_teaching_block=lambda value: "teaching digest",
            generate_initial=generate_initial,
            extract_followups=lambda text: (text, []),
        )
    )

    answer = asyncio.run(
        workflow.coach_hand(
            4,
            context,
            hand_description="Hero AsKs",
            user_text="請分析",
            source_instruction="使用已解析手牌",
            user_id=9,
        )
    )

    assert answer == "verified answer"
    assert context["followup_questions"] == []
    assert "Hero AsKs" in prompts[0][1]
    assert "teaching digest" in prompts[0][1]
    assert "raw solver facts" not in prompts[0][1]
    assert prompts[0][4]["user_id"] == 9


def test_coach_hand_without_teaching_block_uses_solver_text():
    prompts = []
    harness = Harness()
    workflow = ChatWorkflow(
        replace(
            harness.deps,
            build_teaching_block=lambda context: "",
            generate_initial=lambda chat_id, prompt, context, user_text, **kwargs: (
                prompts.append(prompt) or "answer\nFOLLOWUP: next"
            ),
            extract_followups=lambda text: ("answer", ["next"]),
        )
    )
    context = {"text": "solver facts"}

    answer = asyncio.run(
        workflow.coach_hand(
            5,
            context,
            hand_description="hand",
            user_text="question",
            source_instruction="source",
        )
    )

    assert answer == "answer"
    assert "solver facts" in prompts[0]
    assert context["followup_questions"] == ["next"]


def test_plain_followup_never_calls_tools_and_commits_only_final_answer():
    harness = Harness([_response(), _answer("請指定決策點。")])

    answer = asyncio.run(harness.workflow.run(7, "你好"))

    assert answer == "請指定決策點。"
    assert harness.executed == []
    assert harness.recorded == []
    assert harness.accepted == [(7, "你好", answer)]


def test_range_query_preserves_whole_range_and_exact_hero_combo():
    context = {"hand": {"hero_position": "BTN", "hero_hand": "AhKh"}}
    harness = Harness(
        [
            _response(
                calls=[
                    _call(
                        "query_gto",
                        {"street": "preflop", "position": "BTN"},
                    )
                ]
            ),
            _answer("Hero AhKh Call 61%。", ["E2.1"]),
        ],
        contexts={8: context},
        tool_result="AhKh\nsolver 動作：Call 61%",
    )

    answer = asyncio.run(
        harness.workflow.run(8, "Hero 的 range 有哪些牌？", tool_specs=[_tool()])
    )

    assert "61%" in answer
    assert harness.executed == [
        (
            8,
            "query_gto",
            {
                "street": "preflop",
                "position": "BTN",
                "include_range": True,
                "hand": "AhKh",
            },
        )
    ]


def test_exact_combo_is_not_injected_for_opponent_position():
    context = {"hand": {"hero_position": "BTN", "hero_hand": "AhKh"}}
    harness = Harness(
        [
            _response(
                calls=[
                    _call(
                        "query_gto",
                        {"street": "turn", "position": "BB"},
                    )
                ]
            ),
            _answer("BB turn range 已取得。", ["E2.1"]),
        ],
        contexts={8: context},
        tool_result="BB turn range 已取得",
    )

    asyncio.run(harness.workflow.run(8, "對手 BB turn range？", tool_specs=[_tool()]))

    assert harness.executed[0][2] == {
        "street": "turn",
        "position": "BB",
        "include_range": True,
    }


def test_hand_strength_is_preloaded_from_local_evaluator():
    context = {
        "hand": {"hero_position": "HJ", "hero_hand": "QdJs"},
        "street_states": {"turn": {"board": "6hAc5d2c"}},
    }
    harness = Harness(
        [
            _response(calls=[_call("query_coach_facts", {"intent": "why_action"})]),
            _answer("Hero QdJs 是高牌；solver 建議過牌。", ["E2.1", "E3.1"]),
        ],
        contexts={9: context},
        tool_result="QdJs solver 建議過牌",
    )

    answer = asyncio.run(
        harness.workflow.run(
            9,
            "為什麼 Hero turn 要 check？",
            tool_specs=[_tool("query_coach_facts")],
        )
    )

    assert "高牌" in answer
    assert harness.evaluated == [(9, {"hand": "QdJs", "board": "6hAc5d2c"})]


def test_hand_strength_uses_latest_board_when_question_omits_street():
    context = {
        "hand": {"hero_position": "HJ", "hero_hand": "QdJs"},
        "street_states": {
            "flop": {"board": "6hAc5d"},
            "river": {"board": "6hAc5d2c9s"},
        },
    }
    harness = Harness(
        [_response(), _answer("Hero QdJs 是高牌。", ["E2.1"])],
        contexts={9: context},
    )

    asyncio.run(harness.workflow.run(9, "Hero 為什麼是高牌？"))

    assert harness.evaluated == [(9, {"hand": "QdJs", "board": "6hAc5d2c9s"})]


@pytest.mark.parametrize("arguments", ["{bad json", json.dumps(["not", "object"])])
def test_malformed_tool_arguments_are_evidence_not_exceptions(arguments):
    harness = Harness(
        [
            _response(calls=[_call("local_tool", arguments, encode=False)]),
            _answer("工具參數無法解析。", ["E2.1"]),
        ]
    )

    answer = asyncio.run(
        harness.workflow.run(10, "整理工具結果", tool_specs=[_tool("local_tool")])
    )

    assert "無法解析" in answer
    assert harness.executed == []
    assert harness.recorded == []


def test_tool_exception_is_recorded_and_does_not_escape():
    harness = Harness(
        [
            _response(calls=[_call("local_tool", {})]),
            _answer("工具查詢失敗。", ["E2.1"]),
        ],
        tool_result=RuntimeError("boom"),
    )

    answer = asyncio.run(
        harness.workflow.run(11, "整理工具結果", tool_specs=[_tool("local_tool")])
    )

    assert answer == "工具查詢失敗。"
    assert harness.recorded[0]["tool_result"] == "工具查詢失敗：boom"


def test_tool_recording_failure_is_non_fatal():
    harness = Harness(
        [
            _response(calls=[_call("local_tool", {})]),
            _answer("已取得資料。", ["E2.1"]),
        ]
    )

    async def broken_recorder(**kwargs):
        raise RuntimeError("database unavailable")

    workflow = ChatWorkflow(replace(harness.deps, record_tool_call=broken_recorder))

    assert (
        asyncio.run(workflow.run(12, "整理工具結果", tool_specs=[_tool("local_tool")]))
        == "已取得資料。"
    )


def test_duplicate_tool_calls_execute_only_once():
    duplicate = {"street": "turn", "position": "HJ"}
    harness = Harness(
        [
            _response(
                calls=[
                    _call("query_gto", duplicate, "call-1"),
                    _call("query_gto", duplicate, "call-2"),
                ]
            ),
            _answer("HJ turn 下注 62%。", ["E2.1"]),
        ],
        tool_result="HJ turn 下注 62%",
    )

    asyncio.run(harness.workflow.run(13, "HJ turn 下注頻率？", tool_specs=[_tool()]))

    assert len(harness.executed) == 1
    assert len(harness.recorded) == 1


def test_parallel_calls_respect_total_tool_budget():
    harness = Harness(
        [
            _response(
                calls=[
                    _call("query_gto", {"street": "flop"}, "call-1"),
                    _call("query_gto", {"street": "turn"}, "call-2"),
                    _call("query_gto", {"street": "river"}, "call-3"),
                ]
            ),
            _answer("已取得 flop 資料。", ["E2.1"]),
        ],
        max_tool_calls=2,
    )

    asyncio.run(harness.workflow.run(14, "flop 怎麼打？", tool_specs=[_tool()]))

    assert [args["street"] for _, _, args in harness.executed] == ["flop", "turn"]


def test_strategy_question_forces_solver_tool_on_second_round():
    harness = Harness(
        [
            _response(response_id="planner-skipped"),
            _response(
                calls=[_call("query_gto", {"street": "turn"})],
                response_id="planner-forced",
            ),
            _answer("HJ turn 下注 62%。", ["E2.1"]),
        ],
        tool_result="HJ turn 下注 62%",
        max_evidence_rounds=2,
    )

    asyncio.run(
        harness.workflow.run(
            15,
            "HJ turn 應該用哪些牌下注？",
            tool_specs=[_tool(), _tool("local_tool")],
        )
    )

    forced = harness.model_calls[1]
    assert forced["tool_choice"] == "required"
    assert {tool["name"] for tool in forced["tools"]} == {"query_gto"}


def test_planner_can_chain_two_distinct_tool_rounds():
    harness = Harness(
        [
            _response(
                calls=[_call("query_next_actions", {"street": "turn"})],
                response_id="discover",
            ),
            _response(
                calls=[_call("query_gto", {"street": "turn"})],
                response_id="strategy",
            ),
            _answer("Hero turn 跟注 61%。", ["E3.1"]),
        ],
        tool_result=lambda name, args: (
            "turn 可用動作：下注 4bb"
            if name == "query_next_actions"
            else "Hero turn 跟注 61%"
        ),
        max_evidence_rounds=2,
    )

    answer = asyncio.run(
        harness.workflow.run(
            15,
            "Hero turn range 怎麼打？",
            tool_specs=[_tool("query_next_actions"), _tool("query_gto")],
        )
    )

    assert "61%" in answer
    assert [name for _, name, _ in harness.executed] == [
        "query_next_actions",
        "query_gto",
    ]


def test_hypothetical_discovery_is_completed_with_strategy_evidence():
    def tool_result(name, args):
        if name == "query_next_actions":
            return "turn 可用動作：下注 4bb"
        return "Hero turn 面對下注 4bb：跟注 61%"

    harness = Harness(
        [
            _response(calls=[_call("query_next_actions", {"street": "turn"})]),
            _answer("Hero turn 面對下注 4bb，跟注 61%。", ["E3.1"]),
        ],
        tool_result=tool_result,
    )

    answer = asyncio.run(
        harness.workflow.run(
            16,
            "如果 turn 對手下注 4bb，我該怎麼打？",
            tool_specs=[_tool("query_next_actions"), _tool("query_coach_facts")],
        )
    )

    assert "61%" in answer
    assert [name for _, name, _ in harness.executed] == [
        "query_next_actions",
        "query_coach_facts",
    ]
    assert harness.executed[1][2] == {"intent": "hypothetical", "street": "turn"}


def test_missing_solver_evidence_fails_closed_without_narrator():
    harness = Harness(
        [_response(calls=[_call("query_gto", {"street": "turn"})])],
        tool_result="API 查詢失敗：timeout",
    )
    workflow = ChatWorkflow(replace(harness.deps, tool_status=lambda result: "error"))

    answer = asyncio.run(workflow.run(17, "turn range 頻率？", tool_specs=[_tool()]))

    assert "不會用一般牌理猜測" in answer
    assert len(harness.model_calls) == 1
    assert harness.accepted == [(17, "turn range 頻率？", answer)]


def test_unsupported_number_is_repaired_before_history_commit():
    harness = Harness(
        [
            _response(calls=[_call("query_gto", {"street": "turn"})]),
            _answer("HJ turn 下注 99%。", ["E2.1"]),
            _answer("HJ turn 下注 62%。", ["E2.1"]),
        ],
        tool_result="HJ turn 下注 62%",
    )

    answer = asyncio.run(
        harness.workflow.run(18, "HJ turn 下注頻率？", tool_specs=[_tool()])
    )

    assert "62%" in answer and "99%" not in answer
    assert harness.accepted == [(18, "HJ turn 下注頻率？", answer)]


def test_missing_exact_combo_is_repaired_before_history_commit():
    context = {"hand": {"hero_position": "HJ", "hero_hand": "QdJs"}}
    harness = Harness(
        [
            _response(
                calls=[
                    _call(
                        "query_gto",
                        {"street": "turn", "position": "HJ", "hand": "QdJs"},
                    )
                ]
            ),
            _answer("Hero 主要跟注 61%。", ["E2.1"]),
            _answer("Hero QdJs 跟注 61%。", ["E2.1"]),
        ],
        contexts={19: context},
        tool_result="QdJs\nsolver 動作：跟注 61%",
    )

    answer = asyncio.run(
        harness.workflow.run(19, "Hero 這手 turn 怎麼打？", tool_specs=[_tool()])
    )

    assert "QdJs" in normalize_emoji_cards(answer)
    assert harness.accepted == [(19, "Hero 這手 turn 怎麼打？", answer)]


def test_three_invalid_narrations_fall_back_to_deterministic_evidence():
    harness = Harness(
        [
            _response(calls=[_call("query_gto", {"street": "turn"})]),
            _response(text="not json"),
            _response(text="still not json"),
            _response(text="invalid again"),
        ],
        tool_result="HJ turn 下注 62%",
    )

    answer = asyncio.run(
        harness.workflow.run(20, "HJ turn 下注頻率？", tool_specs=[_tool()])
    )

    assert answer == "*核心資料*\n• HJ turn 下注 62%"
    assert harness.accepted == [(20, "HJ turn 下注頻率？", answer)]


def test_honest_partial_answer_can_be_accepted():
    harness = Harness(
        [
            _response(calls=[_call("query_gto", {"street": "turn"})]),
            _answer(
                "目前只能確認 HJ turn 下注 62%。",
                ["E2.1"],
                needs_more_evidence=True,
            ),
        ],
        tool_result="HJ turn 下注 62%",
    )

    answer = asyncio.run(
        harness.workflow.run(21, "HJ turn 下注頻率？", tool_specs=[_tool()])
    )

    assert "只能確認" in answer
    assert harness.accepted == [(21, "HJ turn 下注頻率？", answer)]


def test_concurrent_chats_keep_history_and_answers_isolated():
    accepted = []

    async def model_response(**kwargs):
        await asyncio.sleep(0)
        if kwargs["instructions"].startswith("You are the evidence planner"):
            return _response()
        chat = "chat-a" if "question-a" in str(kwargs["input"]) else "chat-b"
        return _answer(f"answer-{chat}")

    harness = Harness()
    workflow = ChatWorkflow(
        replace(
            harness.deps,
            model_response=model_response,
            accept_history=lambda chat_id, question, answer: accepted.append(
                (chat_id, question, answer)
            ),
        )
    )

    async def run_both():
        return await asyncio.gather(
            workflow.run(101, "question-a"),
            workflow.run(202, "question-b"),
        )

    answers = asyncio.run(run_both())

    assert answers == ["answer-chat-a", "answer-chat-b"]
    assert sorted(accepted) == [
        (101, "question-a", "answer-chat-a"),
        (202, "question-b", "answer-chat-b"),
    ]
