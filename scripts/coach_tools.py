#!/usr/bin/env python3
"""Provider-neutral schemas for the evidence coach tools."""

from coach_evidence import COACH_FACTS_TOOL, ToolSpec

_BASE_TOOLS = (
    ToolSpec(
        name="query_next_actions",
        description="查詢某個決策點的所有可用動作及其 code。在建構假設情境（override actions）之前必須先呼叫此工具，以獲取正確的 action code（如 R3.6 而非猜測的 R1.2）。回傳每個可用動作的 code、betsize 和 betsize_by_pot。",
        parameters={
            "properties": {
                "street": {
                    "description": "要查詢哪條街的可用動作",
                    "enum": ["preflop", "flop", "turn", "river"],
                    "type": "string",
                },
                "effective_bb": {
                    "description": "有效籌碼深度（bb 數）。不同深度的 solver sizing "
                    "不同。不指定則使用目前手牌的深度。",
                    "type": "number",
                },
                "actions_so_far": {
                    "description": "這條街到目前為止的動作序列（如果要查詢街中某個後續決策點）。例如查詢 flop 上 "
                    "SB bet 後 BB 的選項，傳入 "
                    "'R3.6'。留空表示查詢該街第一個行動者的選項。",
                    "type": "string",
                },
                "preflop_actions_override": {
                    "description": "覆蓋 preflop 動作序列（同 query_gto "
                    "的格式）。用於查詢不同 preflop 路線下的可用動作。",
                    "type": "string",
                },
                "board_override": {
                    "description": "假設不同的 board。",
                    "type": "string",
                },
                "flop_actions_override": {
                    "description": "假設不同的翻牌動作（查詢 turn/river 時使用）。",
                    "type": "string",
                },
                "turn_actions_override": {
                    "description": "假設不同的轉牌動作（查詢 river 時使用）。",
                    "type": "string",
                },
                "num_players": {
                    "description": "桌上人數（6-9）。ICM 查詢時必須指定。",
                    "type": "integer",
                },
                "icm_phase": {
                    "description": "ICM 錦標賽階段。指定後會使用 ICM solver 而非 Chip "
                    "EV。常見階段：START=初期, PCT25=剩25%人, BUBBLEMID=泡沫期, "
                    "FT=決賽桌。",
                    "enum": [
                        "START",
                        "PCT75",
                        "PCT50",
                        "PCT25",
                        "PCT10",
                        "PCT5",
                        "BUBBLEEARLY",
                        "BUBBLEMID",
                        "BUBBLELATE",
                        "FT",
                        "T2",
                        "T3",
                    ],
                    "type": "string",
                },
                "player_stacks": {
                    "description": "ICM 各位置籌碼（bb），用逗號分隔，按座位順序（UTG 到 BB）。例如 8 "
                    "人桌全部 20bb: "
                    "'20,20,20,20,20,20,20,20'。不指定則預設所有人相同籌碼（= "
                    "effective_bb）。",
                    "type": "string",
                },
            },
            "required": ["street"],
            "type": "object",
        },
        requires_db=False,
    ),
    ToolSpec(
        name="query_gto",
        description="查詢 GTO solver 策略數據。可以查詢目前手牌中任何位置在任何街的完整範圍或特定手牌策略。也可以修改 board 或 actions 來查詢假設情境。重要：使用 override actions 時，必須先用 query_next_actions 取得正確的 action code。查詢不同位置的 preflop 策略時，用 preflop_actions_override 指定到該位置行動前的動作序列。Raise size 不需要精確，系統會自動校正到最接近的 solver sizing（例如 R2 會自動校正為 R2.1）。\n\n用戶描述獨立情境（不基於已有手牌）時，必須同時提供：effective_bb、preflop_actions_override、board_override，以及 flop/turn/river_actions_override。Board 必須帶花色（例如 QhTd3c），如果用戶沒指定花色就用 rainbow（不同花色）。Action 格式：X=check, C=call, F=fold, R{pot%}=bet/raise（如 R1.15 = ~33% pot bet）。查詢 turn 時，board_override 必須包含 turn 牌（4 張牌，例如 QhTd3c3s）。",
        parameters={
            "properties": {
                "street": {
                    "description": "要查詢哪條街的策略",
                    "enum": ["preflop", "flop", "turn", "river"],
                    "type": "string",
                },
                "decision_index": {
                    "description": "同一條街 Hero 的第幾次決策（1-based）。例如 Hero "
                    "check-raise 後再面對 3bet，後一個決策是 2。查 played "
                    "line 時優先用此欄位避免抓到第一個 node。",
                    "type": "integer",
                },
                "position": {
                    "description": "要查詢哪個位置的範圍或策略（例如 BB, CO, BTN）。不指定則回傳當前行動者的整體策略。",
                    "type": "string",
                },
                "hand": {
                    "description": "查詢特定手牌的策略。不指定則回傳該位置的完整範圍概覽。\n"
                    "Postflop 查詢時，如果用戶指定了花色（如 Ah8h），必須傳入完整花色（如 Ah8h 而非 "
                    "A8s），因為不同花色在有同花/同花聽牌的牌面上策略差異極大。\n"
                    "例如 board Jc4d3s5d: Ad8d（方塊花聽）96% bet vs Ah8h（無聽牌）97% "
                    "check。\n"
                    "Preflop 查詢用簡化格式即可：66, AKs, QTo。",
                    "type": "string",
                },
                "include_range": {
                    "description": "問題在問完整 range、哪些牌或哪些 combos 時設為 true。即使同時傳 "
                    "hand 以保留 Hero exact combo，也會一併回傳完整 "
                    "action-by-action range 並產生 13x13 range 圖。",
                    "type": "boolean",
                },
                "effective_bb": {
                    "description": "有效籌碼深度（bb 數）。當用戶問的情境深度與目前手牌不同時必須指定。例如用戶問 "
                    "'30bb effective' 就傳 30。系統會自動選擇最近的 solver "
                    "深度。不指定則使用目前手牌的深度。",
                    "type": "number",
                },
                "preflop_actions_override": {
                    "description": "覆蓋 preflop 動作序列。格式：每個位置一個動作，按 "
                    "UTG-UTG+1-LJ-HJ-CO-BTN-SB-BB 順序，用 "
                    "- 分隔。F=Fold, C=Call, RX=Raise to "
                    "X, AI=All-in。Raise size "
                    "不用精確，系統會自動校正。例如查詢 BB 面對 UTG+1 "
                    "open 的策略：傳入 F-R2-F-F-F-F-F。例如查詢 "
                    "UTG+1 open 後 BB 3bet 後 UTG+1 "
                    "的決策：傳入 F-R2-F-F-F-F-F-AI。",
                    "type": "string",
                },
                "board_override": {
                    "description": "指定 board 牌面（帶花色）。獨立情境查詢時必須提供。Flop 查詢傳 3 張（如 "
                    "QhTd3c），turn 查詢傳 4 張（如 QhTd3c3s），river 查詢傳 "
                    "5 張。也可用於覆蓋已有手牌的 board。",
                    "type": "string",
                },
                "flop_actions_override": {
                    "description": "翻牌動作序列。格式：X=check, C=call, F=fold, "
                    "R{size}=bet/raise。\n"
                    "size 可以是絕對 bb 數（如 R3.7）或底池百分比（如 "
                    "R50%）。系統會自動轉換百分比為正確的 bb 數。\n"
                    "推薦使用百分比格式，避免因 ante 導致底池計算錯誤。\n"
                    "例如 LJ bet 50% pot, BTN call = "
                    "R50%-C。\n"
                    "查詢 flop 時：填到要查詢的決策點之前的動作。\n"
                    "查詢 turn 時：填完整的 flop 動作。",
                    "type": "string",
                },
                "turn_actions_override": {
                    "description": "轉牌動作序列。格式同上（支援 R50% 百分比格式）。查詢 turn "
                    "某位置策略時，填到該位置行動前。",
                    "type": "string",
                },
                "river_actions_override": {
                    "description": "假設不同的河牌動作序列。格式同上（支援 R50% 百分比格式）。",
                    "type": "string",
                },
                "num_players": {
                    "description": "桌上人數（6-9）。ICM 查詢時必須指定。",
                    "type": "integer",
                },
                "icm_phase": {
                    "description": "ICM 錦標賽階段。指定後會使用 ICM solver 而非 Chip "
                    "EV。常見階段：START=初期, PCT25=剩25%人, BUBBLEMID=泡沫期, "
                    "FT=決賽桌。",
                    "enum": [
                        "START",
                        "PCT75",
                        "PCT50",
                        "PCT25",
                        "PCT10",
                        "PCT5",
                        "BUBBLEEARLY",
                        "BUBBLEMID",
                        "BUBBLELATE",
                        "FT",
                        "T2",
                        "T3",
                    ],
                    "type": "string",
                },
                "player_stacks": {
                    "description": "ICM 各位置籌碼（bb），用逗號分隔，按座位順序（UTG 到 BB）。例如 8 "
                    "人桌全部 20bb: "
                    "'20,20,20,20,20,20,20,20'。不指定則預設所有人相同籌碼（= "
                    "effective_bb）。",
                    "type": "string",
                },
            },
            "required": ["street"],
            "type": "object",
        },
        requires_db=False,
    ),
    ToolSpec(
        name="evaluate_hand",
        description="判斷手牌在牌面上的確切牌型（成手牌 + 聽牌）。牌型判斷是 100% 確定性的，必須用此工具驗證，絕對不要自行推算。board 可省略，會自動使用當前最新牌面。",
        parameters={
            "properties": {
                "hand": {
                    "description": "手牌 (如 KQo, AhKh, T7s, 66)",
                    "type": "string",
                },
                "board": {
                    "description": "牌面 (如 8hTc2sAc)，省略則用當前最新牌面",
                    "type": "string",
                },
            },
            "required": ["hand"],
            "type": "object",
        },
        requires_db=False,
    ),
)

_DB_TOOLS = (
    ToolSpec(
        name="lookup_hand",
        description="根據 Hand ID 從用戶的手牌歷史中查詢手牌資料。用戶提到某個 Hand ID（如 H42 或 TM5600279272）時，使用此工具撈取手牌 JSON。可用於跨對話引用之前分析過的手牌。",
        parameters={
            "properties": {
                "hand_id": {
                    "description": "手牌 ID（如 H42 或 TM5600279272）",
                    "type": "string",
                }
            },
            "required": ["hand_id"],
            "type": "object",
        },
        requires_db=True,
    ),
    ToolSpec(
        name="get_training_plan",
        description="取得本週訓練計畫（週日 21:00 自動生成的記分卡）：焦點 spot（EV loss 排序）+ GTOW Trainer drill 連結 + 上週焦點回讀 + 現場手牌練習佇列。當用戶問「我該練什麼」「給我訓練計畫」「本週計畫」時使用。",
        parameters={"properties": {}, "required": [], "type": "object"},
        requires_db=True,
    ),
    ToolSpec(
        name="get_progress",
        description="查詢每週 EV loss 趨勢（bb/100 決策，帶樣本數 n）。可選按 spot 大類（RFI/vsOpen/vs3bet/…/flop/turn/river）或精確 spot_leaf 過濾。當用戶問「我有進步嗎」「XX 有改善嗎」時使用。注意：技能趨勢是月尺度，單週波動不是訊號。",
        parameters={
            "properties": {
                "category": {
                    "description": "spot 大類，如 vs3bet、flop（可選）",
                    "type": "string",
                },
                "spot_leaf": {
                    "description": "精確 action-line spot leaf（可選）",
                    "type": "string",
                },
                "weeks": {"description": "查詢最近幾週（預設 8）", "type": "integer"},
            },
            "required": [],
            "type": "object",
        },
        requires_db=True,
    ),
    ToolSpec(
        name="query_ledger_summary",
        description="查詢全量帳本（GTOW Analyzer 評分的線上 MTT 決策，action-line 分類）的 EV loss 聚合。可按 spot 大類（RFI/vsOpen/vsRaiseCall/vs3bet/vs4bet/vsSqueeze/flop/turn/river）或 hero 位置類（EP/MP/LP/SB/BB）或天數過濾。回傳 EV loss/100 決策、總損失、樣本數 n、excluded 數與 top spot。使用者問『我最大的弱點 / 什麼地方打最差 / 我的統計 / 我哪裡漏 EV / 某類 spot 表現如何 / 我 3bet pot 打得怎樣』時用這個。",
        parameters={
            "properties": {
                "category": {
                    "description": "spot 大類，如 vs3bet、flop、vsRaiseCall",
                    "type": "string",
                },
                "hero_cat": {
                    "description": "hero 位置類：EP/MP/LP/SB/BB",
                    "type": "string",
                },
                "days": {"description": "回看天數，省略=全期", "type": "integer"},
            },
            "required": [],
            "type": "object",
        },
        requires_db=True,
    ),
    ToolSpec(
        name="query_ledger_hands",
        description="列出帳本中符合條件的具體手牌（EV loss 排序），附 GTOW Analyze 復盤連結。使用者要看『哪幾手 / 最貴的手 / 某類 spot 的實例』時用這個。",
        parameters={
            "properties": {
                "category": {"description": "spot 大類", "type": "string"},
                "min_ev_loss": {"description": "bb 門檻，預設 0.5", "type": "number"},
                "days": {"description": "預設 90", "type": "integer"},
                "limit": {"description": "預設 5，最大 10", "type": "integer"},
            },
            "required": [],
            "type": "object",
        },
        requires_db=True,
    ),
)


def coach_tool_specs(db_enabled: bool) -> list[ToolSpec]:
    specs = [COACH_FACTS_TOOL, *_BASE_TOOLS]
    if db_enabled:
        specs.extend(_DB_TOOLS)
    return specs
