import asyncio
import json
import logging
from dataclasses import replace
from types import SimpleNamespace

from coach_evidence import ToolSpec
from coach_runtime import ChatWorkflow, WorkflowDeps


def _response(*, calls=(), text="", response_id="response"):
    return SimpleNamespace(output=list(calls), output_text=text, id=response_id)


def _call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
    )


def _workflow(model_responses, *, context=None, tool_result="solver fact"):
    accepted = []
    tool_calls = []
    recorded = []

    async def model_response(**kwargs):
        return model_responses.pop(0)

    async def execute_tool(chat_id, user_text, name, args, **kwargs):
        tool_calls.append((name, dict(args)))
        return tool_result

    async def record_tool_call(**kwargs):
        recorded.append(kwargs)

    deps = WorkflowDeps(
        get_hand_context=lambda chat_id: context,
        clear_followup_node_street=lambda chat_id: None,
        build_evidence_context=lambda chat_id: "cached hand" if context else "no hand",
        history_for_evidence=lambda chat_id: "previous turn",
        evaluate_hand=lambda chat_id, args: "top pair",
        execute_tool=execute_tool,
        model_response=model_response,
        accept_history=lambda chat_id, question, answer: accepted.append(
            (chat_id, question, answer)
        ),
        record_tool_call=record_tool_call,
        tool_status=lambda result: "ok",
        model="coach-model",
        max_tool_calls=4,
        max_evidence_rounds=1,
        reasoning="low",
        max_output_tokens=900,
        logger=logging.getLogger("test_chat_workflow"),
    )
    return ChatWorkflow(deps), accepted, tool_calls, recorded


def test_plain_followup_uses_model_and_accepts_only_final_answer():
    workflow, accepted, tool_calls, recorded = _workflow(
        [
            _response(),
            _response(
                text=json.dumps(
                    {
                        "answer": "先確認你想討論哪個決策點。",
                        "fact_refs": [],
                        "needs_more_evidence": False,
                    }
                )
            ),
        ]
    )

    answer = asyncio.run(workflow.run(7, "你好"))

    assert answer == "先確認你想討論哪個決策點。"
    assert accepted == [(7, "你好", answer)]
    assert tool_calls == []
    assert recorded == []


def test_range_followup_preserves_full_range_and_exact_hero_combo():
    context = {
        "hero_position": "BTN",
        "hand": {"hero_position": "BTN", "hero_hand": "AhKh"},
    }
    workflow, accepted, tool_calls, recorded = _workflow(
        [
            _response(
                calls=[_call("query_gto", {"street": "preflop", "position": "BTN"})]
            ),
            _response(
                text=json.dumps(
                    {
                        "answer": "Hero AhKh 在這個 range 中應採用已查得的策略。",
                        "fact_refs": ["E2.1"],
                        "needs_more_evidence": False,
                    }
                )
            ),
        ],
        context=context,
    )
    tool = ToolSpec(
        name="query_gto",
        description="solver",
        parameters={"type": "object", "properties": {}},
    )

    answer = asyncio.run(workflow.run(8, "我的 range 有哪些牌？", tool_specs=[tool]))

    assert answer.startswith("Hero A") and "K" in answer
    assert tool_calls == [
        (
            "query_gto",
            {
                "street": "preflop",
                "position": "BTN",
                "include_range": True,
                "hand": "AhKh",
            },
        )
    ]
    assert recorded and recorded[0]["tool_name"] == "query_gto"
    assert accepted == [(8, "我的 range 有哪些牌？", answer)]


def test_solver_question_without_solver_evidence_fails_closed():
    workflow, accepted, tool_calls, _ = _workflow(
        [
            _response(calls=[_call("query_gto", {"street": "turn"})]),
        ],
        tool_result="API 查詢失敗：timeout",
    )
    workflow = ChatWorkflow(
        replace(
            workflow.deps,
            tool_status=lambda result: "error",
        )
    )
    tool = ToolSpec(
        name="query_gto",
        description="solver",
        parameters={"type": "object", "properties": {}},
    )

    answer = asyncio.run(
        workflow.run(
            9,
            "turn 的 range 頻率是什麼？",
            tool_specs=[tool],
        )
    )

    assert "沒有取得可驗證的 solver 資料" in answer
    assert tool_calls == [("query_gto", {"street": "turn"})]
    assert accepted == [(9, "turn 的 range 頻率是什麼？", answer)]


def test_parsed_hand_pipeline_uses_injected_solver_and_verified_coach():
    workflow, _, _, _ = _workflow([])
    prompts = []

    async def analyze_hand(chat_id, hand, **kwargs):
        return {"text": "solver facts", "hand": hand}

    async def generate_initial(chat_id, prompt, context, user_text, **kwargs):
        prompts.append(prompt)
        return "verified answer\nFOLLOWUP: 下一題"

    workflow = ChatWorkflow(
        replace(
            workflow.deps,
            analyze_hand=analyze_hand,
            build_teaching_block=lambda context: "teaching digest",
            generate_initial=generate_initial,
            extract_followups=lambda text: ("verified answer", ["下一題"]),
        )
    )

    context = asyncio.run(workflow.prepare_hand(10, {"hero_hand": "AsKs"}))
    answer = asyncio.run(
        workflow.coach_hand(
            10,
            context,
            hand_description="Hero AsKs",
            user_text="請分析",
            source_instruction="使用已解析手牌",
        )
    )

    assert answer == "verified answer"
    assert context["followup_questions"] == ["下一題"]
    assert "Hero AsKs" in prompts[0]
    assert "teaching digest" in prompts[0]


def test_bot_run_uses_surviving_parser_and_coach_model_names():
    from src.telegram_bot.bot import PokerWizardBot

    messages = []

    class Log:
        def info(self, message, *args):
            messages.append(message % args)

    app = SimpleNamespace(
        run_polling=lambda **kwargs: messages.append(kwargs),
    )
    bot = object.__new__(PokerWizardBot)
    bot.session_manager = SimpleNamespace(
        parse_model="gemini-parser",
        coach_narrator_model="gpt-coach",
        max_turns="N/A",
    )
    bot.log = Log()
    bot.setup_handlers = lambda **kwargs: app

    bot.run()

    assert messages == [
        "Bot starting — parser=gemini-parser, coach=gpt-coach, max_turns=N/A",
        {"drop_pending_updates": False},
    ]


def test_main_entrypoint_reports_surviving_models(monkeypatch, capsys):
    import src.main_gemini as entrypoint

    runs = []

    class Session:
        parse_model = "gemini-parser"
        coach_narrator_model = "gpt-coach"

        def __init__(self, db):
            self.db = db

    class Bot:
        def __init__(self, **kwargs):
            runs.append(kwargs)

        def run(self, **kwargs):
            runs.append(kwargs)

    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("POKER_BOT_PROCESS", "0")
    monkeypatch.setattr(entrypoint, "GeminiSessionManager", Session)
    monkeypatch.setattr(entrypoint, "PokerWizardBot", Bot)

    entrypoint.main()

    output = capsys.readouterr().out
    assert "Parser: gemini-parser; Coach: gpt-coach" in output
    assert runs[0]["session_manager"].parse_model == "gemini-parser"
    assert runs[1] == {
        "post_init": entrypoint.post_init,
        "post_shutdown": entrypoint.post_shutdown,
    }
