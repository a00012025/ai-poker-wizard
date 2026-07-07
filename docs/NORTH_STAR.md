# AI Poker Wizard 北極星文件（North Star）

> **Status**: Living document（本專案的最高對齊文件）
> **Version**: 1.0 — 2026-07-07
> **Owner**: Harry
> **讀者**: Harry 本人，以及未來接手規劃/實作的任何 AI 模型與協作者

## 本文件是什麼、怎麼使用

這份文件定義本專案的**終態（endgame）願景**：我們要建成一個什麼樣的系統、為什麼、以及所有設計必須遵守的原則。它不是實作規格 — 它是所有未來實作規格的**上游憲法**。

**給未來接手的 AI 模型的指示：**

1. 開始任何新 feature 的規劃前，先讀完本文件（至少 §2、§3、§7、§8、§13）。
2. 你的提案必須通過 §13 的對齊檢查清單。過不了就修改提案，不要修改本文件來遷就提案。
3. 本文件刻意停留在 high-level（目的、邊界、不變量、驗收訊號），把實作自由度留給你 — 但**不變量（§7）沒有商量空間**。
4. 修改本文件需要 Harry 明確同意，且任何修改要在 §15 版本紀錄留痕。

---

## 0. 選手檔案與訓練現況（2026-07 snapshot）

任何訓練系統都是為特定選手設計的。本節描述目前的事實基準，隨時間更新。

- **選手**: Harry。長期目標：成為世界頂尖的德州撲克錦標賽（MTT）牌手。
- **主戰場**: GGPoker 線上 MTT，買入 **$1–$10**，**PKO 與一般 MTT 都打**；近期開始打**線下 MTT**。
- **典型週時間分配**: 打牌 ~15h、bot 復盤 ~2h、其他 study（課程/書/GTOW Trainer）~6h、開發 ~1h。
- **已有工作流**（系統必須銜接，不是取代）:
  - 手牌 HH 批次上傳 **GTO Wizard Analyzer** → 已能拿到全量 EV loss、frequency diff、effective bb 等分析。
  - 常態使用 **GTO Wizard Trainer/Drill**，練過的手牌（practiced hands）GTOW 全部有紀錄。
  - 線下 MTT 會**選擇性**記錄手牌：只記「不確定打得對不對」或「想更了解該類策略」的手。
  - 手上有 GTO Wizard《**Daily Dose of GTO**》千頁級 PDF（基礎到進階理論 + 大量情境選擇題）。
- **級別含義**（設計權重的依據）:
  - $1–10 的 pool 有巨大且系統性的偏離 → **剝削層的錢比 GTO 精修多**（bencb 級別建議：30–50% 學基準、50–70% 學剝削）。
  - PKO 佔比高 → bounty 數學是這個級別最便宜的大 edge（pool 普遍把 bounty 當彩蛋而不是即時改變 calling range 的 pot equity）。
  - 微級別 rake 高 → 紀律、選場、量的管理直接影響 ROI。
  - 線下低買入 MTT：深籌碼慢結構、multiway limp pot 極多（恰好是 solver 覆蓋最弱的區域）、對手偏離更極端 → heuristic 與剝削規則的價值高於頻率精度。
- **升級階梯**: 世界頂尖不是一步到位，路徑是 stakes ladder（$1-10 → $10-50 → $50-200 → …）。每一級的「畢業條件」由元迴圈（§3）管理：證明的決策品質 + bankroll 門檻，而不是感覺。

---

## 1. 使命與定位：一人職業戰隊

**一句話終態：從「你使用它」變成「它訓練你」。**

工具回應請求；系統追求目標。本專案的終態不是一個更好的分析 bot，而是一支圍繞單一選手的職業戰隊，AI 扮演六個角色：

| 角色 | 職責 | 對應子系統 |
|---|---|---|
| 數據分析師 | 把你打的每一手變成乾淨、誠實的帳 | Flight Recorder、Decision Ledger、診斷引擎 |
| 總教練 | 決定你這週該練什麼、管理課表與週期 | Head Coach |
| 研究員 | 聚合研究 spot family、蒸餾可帶上桌的規則 | 研究工作台、Playbook |
| 陪練員 | 出題、唱反調、逼你先作答再看答案 | Dojo、Endgame Lab |
| 心理教練 | 儀式、tilt 管理、狀態與量的管理 | Performance Layer |
| 經理人 | 賽程、bankroll、game selection、升級決策 | 元迴圈 / 經濟層 |

**與 GTO Wizard 的關係（核心定位）**：GTO Wizard 是健身房與器材（solver、Analyzer、Trainer、題庫）；我們的系統是**教練與訓練計畫** — 疊加在器材之上的「反饋、訓練、優化、評估」的迴圈管理層。器材能 reuse 的一律 reuse（§4），我們只建「你的資料才能做」的個人層。

載體（Telegram bot、web、桌面）只是殼；**Ledger + Playbook + Head Coach 是魂**（§9）。

---

## 2. 北極星指標：先定義「變強」，再設計系統

MTT 結果的噪音大到不能導航（ROI 需要數千場才有統計意義），所以系統以決策品質為主軸。

### 2.1 主指標

> **真實對局的 EV loss / 100 決策**（EV 加權、信心過濾、按 spot family 分解）— 持續往 0 推。

三個修飾詞都是硬要求：

- **EV 加權**：混合策略節點上「頻率不同」≠「有損失」；只有算得出的 EV 損失才計入。禁止以 deviation 次數/頻率差當主排序（歷史教訓：H3510，0.02bb 被稱「嚴重錯誤」）。
- **信心過濾**：低信心判定（parse 存疑、深度不確定、近似過重）不進統計（§5.2）。
- **按 family 分解**：整體數字會被 spot 組成變化混淆，family 分解才能看到真實變化。

### 2.2 歸因機制（系統的防自欺條款）

單純的下降趨勢不夠 — 可能是這週場次軟、spot 組成變了、或純變異數。**「可歸因的下降」才算數**：

- 練過的 family（treated）應比沒練過的 family（untreated）下降得快 — 對自己做 difference-in-differences。
- 三種讀數：treated 降 / untreated 平 → 訓練有效；全部齊降 → 存疑（環境因素）；都不動 → 方法有問題，回爐。
- 這是外迴圈（§3）的裁決標準，也是本系統與市面所有訓練工具的最大差異：**別人只能告訴你練了多少，我們能告訴你練的東西有沒有轉化成實戰**。

### 2.3 指標金字塔（由快到慢、由可控到滯後）

1. **知識層**（天級回饋）：drill 準確率（90% 精通門檻）、playbook 條目精通度、**信心校準**（你申報 80% 確定的題，實際對 80% 嗎 — 校準誤差本身是頂尖技能）。
2. **決策層**（週級回饋）：主指標 EV loss/100，整體 + 分 family + 分賽段（早期/中期/泡沫/FT）。
3. **過程層**（天級，完全可控）：課表依從率、warmup/cooldown 完成率、tilt 事件數、volume。
4. **結果層**（月/年級，永不導航）：ROI、ITM、深入率 — 呈現時必附變異數感知的信賴區間（例：「400 場、觀測 ROI 15%、95% CI ±40% — 尚無訊號」），防止被短期結果帶著走。

### 2.4 季度標準化測驗（幫自己寫的 regression test）

- 固定的標準化題庫（覆蓋各 family × 深度 × 賽段的代表性決策），每季重考，同一把尺量整年。
- **同構變體（isomorph）機制**：每季生成「同構但換面」的變體（花色置換、等價 board、鏡像位置），防背題（anti-Goodhart）。
- 產出：分 family 的 EV loss/決策 曲線 + 校準曲線。這是元迴圈升級決策的輸入之一。

---

## 3. 時間結構：四層迴圈

系統的一切功能都必須服務於這四層迴圈之一（§13 檢查清單第一條）。

| 迴圈 | 週期 | 內容 | 關鍵產出 |
|---|---|---|---|
| **內圈** | 分鐘 | 一題 drill → 作答（動作+頻率+信心）→ 立即回饋 | 學習的原子單位 |
| **中圈** | 週 | 診斷 → 挑 1–2 個焦點 family → 聚合研究 → 壓縮 heuristic 入 playbook → drill 到 90% → 隔週重測 | 新的 playbook 規則 + 精通度 |
| **外圈** | 月 | 實戰驗證：練過的 family 在真實對局的 EV loss 有沒有降（§2.2 歸因） | 「學會了」的最終裁決；沒降就回爐 |
| **元圈** | 季 | 標準化測驗 + 宏觀檢討：級別升降、format 配比、bankroll、年度目標 | 賽季計畫 |

中圈就是選手指定的六步 loop：**打牌 → 標記偏差 → 按 spot family 聚合研究（不是單手）→ 壓縮成可帶上桌的 heuristic → drill 到 ~90% → 間隔重測**。內圈是它的引擎，外圈是它的品管，元圈是它的方向盤。

---

## 4. 資料底座策略：GTO Wizard 作為 substrate

**原則：先 reuse，後自建；自建只做 GTOW 沒有的。** 這不是妥協，是槓桿 — 省下的重活全部投入 GTOW 永遠不會做的個人層。

### 4.1 分工表

| 能力 | 提供者 | 說明 |
|---|---|---|
| 線上 HH 全量評分（EV loss、freq diff、effective bb） | **GTOW Analyzer**（reuse） | 選手已有的工作流；系統吃它的分析結果進 Ledger，不重新逐手解析 |
| Drill 執行與紀錄 | **GTOW Trainer**（reuse） | 教練開處方（spot 配置）→ 選手去練 → 系統回收 practiced hands 紀錄 |
| Solver 查詢、聚合報告、deep link | **GTOW 解庫**（reuse） | 研究工作台的查詢後端；deep link 已建成 |
| 理論概念 + 情境選擇題 | **Daily Dose of GTO PDF**（reuse） | 攝取為概念圖譜 + 題庫（§5.11） |
| 線下手牌評分（無 HH 檔） | **自建 pipeline** | 現有 text/截圖 → 解析 → solver 管線的主要存續理由之一 |
| 截圖快速捕獲 | **自建 pipeline** | 線上臨場快問、無 HH 的 app |
| GTOW 標記「non-existing spot」的手（multiway、奇異樹） | **自建 pipeline** | 現有 multiway 簡化 + 近似標注 |
| ICM/PKO 補充判斷（Analyzer 覆蓋不到的 nuance） | **自建 + GTOW ICM 解庫/HRC 級工具** | §5.7 |
| **Villain 側資料（pool 挖礦）** | **自建 pipeline** | Analyzer 只評 hero；對手決策的聚合是我們獨有的價值（§5.8），必須自己解析 HH |

### 4.2 架構不變量：Ledger 與評分者解耦（grader-agnostic）

- 不管誰評的分（GTOW Analyzer / 自建 pipeline / 未來本地 solver），入帳格式**相同**，且都必須帶信心與近似標注（§5.2）。GTOW 的判定也有近似（最近預解庫、chipEV vs ICM 語境、樹外 sizing snap）— 誠實層對所有 grader 一視同仁。
- **Spot family 分類法（taxonomy）必須是我們自己的**：所有來源（Analyzer、Trainer、線下手、DD 題目）都映射到同一套 family 分類，聚合、診斷、歸因才會跨來源成立。現有 `spot_categorizer` 的 ~15 桶是這套分類法的起點。

### 4.3 整合機制與風險

- GTOW 無公開 API；「自動吃資料」的機制（匯出檔 / 瀏覽器自動化 / 其他）屬實作細節，於各 phase 的實作計畫再定。設計上只承諾：**攝取是批次的、可重跑的、失敗可補征的**。
- 風險：ToS 灰色地帶、介面變動、訂閱價格、功能收回。對沖：(a) grader-agnostic Ledger 讓底座可替換；(b) 自建 pipeline 永遠保持可用作 fallback；(c) 個人解庫長期降依賴（§9）。

---

## 5. 子系統設計

每個子系統：**目的 → 終態描述 → 邊界 → 驗收訊號**。

### 5.1 Flight Recorder（全量捕獲）

**目的**：leak 資料從「你注意到的手」變成「你打的每一手」，selection bias 歸零。

**終態 — 四條捕獲流**：

1. **線上全量流（主幹）**：PokerCraft HH → GTOW Analyzer → 分析結果批次入帳。節奏至少每週，理想每個 session 後。這一流讓 Ledger 樣本從每週幾十手（手動投遞）變成上千個決策。
2. **訓練流**：GTOW Trainer practiced hands → 入帳，**標記為訓練資料**（`source=train`），與實戰資料嚴格分開統計 — drill 準確率屬知識層指標，實戰 EV loss 屬決策層指標，混在一起兩者都會失真。
3. **線下流（意圖標記流）**：線下 MTT 的選擇性記錄。捕獲 UX 必須極輕（休息時間的一段文字 / 語音轉文字 / 筆記照片），系統以對話補問缺失資訊（stack、位置、動作序列、賽段/剩餘人數），再走自建 pipeline 評分入帳。**每筆帶意圖標籤**：`uncertain`（不確定打得對不對）/ `curious`（想學這類 spot）。意圖標籤是一級公民 — 它是選手自己的 metacognition 訊號，直接餵給診斷引擎的盲點矩陣（§5.3）與 Head Coach 的好奇心隊列（§5.10）。
4. **截圖流（輔助）**：現有 OCR 管線，降級為便利捕獲工具（線上臨場快問）。**HH-first 原則**：凡是 HH 拿得到的資料，一律以 HH 為準（HH 精確度 100%，截圖 ~74–78%）。

**Session 元資料**：每個 session 記錄時段、時長、桌數、format、買入、自評狀態（1–5），供 Performance Layer 找相關性。

**邊界**：捕獲不做即時（對局中）策略處理 — 見紅線（§7 不變量 8）。

**驗收訊號**：連續 4 週，線上實際手數與入帳手數一致（100% 捕獲）；線下每場賽事至少完成一筆意圖標記捕獲且 24h 內完成評分。

### 5.2 Decision Ledger（誠實帳本）

**目的**：單一事實來源。上層一切（診斷、課表、歸因、測驗）都建立在這本帳上 — **帳本髒了，樓上全是假的**。

**終態**：每個決策一筆帳，概念欄位見 §6。三個硬性設計：

1. **信心與近似標注（誠實層）**：每筆判定必帶 (a) 解析信心（parse/OCR/補問完整度）；(b) 近似等級（ICM postflop 是否 chipEV 替代、multiway 是否 recast、sizing 是否樹外 snap、深度是否敏感）；(c) **敏感度旗標** — 鄰近深度/stack 重算一次，結論會翻轉的標記為 `fragile`。低信心不進統計，但保留在帳上（可查、可日後重評）。
2. **Grader 欄位**：judgment 來自誰（gtow_analyzer / own_pipeline / local_solver），可重評、可比對。
3. **來源隔離**：`source ∈ {online, live, train, exam}`，統計永不跨源混算。

**驗收訊號**：任何一筆帳可以回答「這個判定可信嗎、為什麼」；抽查 20 筆 fragile 標記，人工複核一致率 > 80%。

### 5.3 診斷引擎

**目的**：把帳本變成「這週該修什麼」的答案。錢漏在哪，不是頻率差在哪。

**終態**：

- **EV 加權聚合**：按 family × texture 類 × 深度帶 × 賽段聚合 EV loss，輸出「金額排序」的 leak 榜與趨勢。
- **三型 leak 分類**（治療完全不同）：
  - **知識型**：同一 family 穩定地錯 → 送研究工作台（不會）。
  - **紀律型**：平時會，tilt 窗口 / 深夜 / 長 session / 高桌數時才錯 → 送 Performance Layer（會但做不到）。
  - **邊界型**：只在特定深度/texture/賽段錯 → playbook 規則太粗，送回細化（會一半）。
- **盲點矩陣（2×2）**：選手意圖標籤 × 系統評分結果交叉：

  | | 評分：錯 | 評分：對 |
  |---|---|---|
  | **標記不確定** | 已知弱點（正常學習隊列） | 理解未鞏固（送 Feynman 驗證理由） |
  | **未標記** | **盲點 — 最危險，最高優先** | 健康 |

  線上全量流的最大價值就是右上到左下的偵測能力：你「毫無感覺卻一直在漏錢」的地方。線下選擇性記錄天然只覆蓋左列 — 這正是兩條流互補的原因。
- **校準診斷**：申報信心 vs 實際正確率的校準曲線，長期過度自信/過度懷疑都是可訓練的 leak。

**驗收訊號**：每週自動產出 top-N leak（EV 排序）+ 每個 leak 的三型判定 + 盲點矩陣更新；診斷結論可追溯到具體手牌清單。

### 5.4 研究工作台（Study Bench）

**目的**：把「單手復盤」升級為「spot family 聚合研究」。單手是入口，family 才是學習單位。

**終態**：

- 對焦點 family 一鍵展開**聚合視圖**：多個 texture × 選手常用深度帶的 solver 策略對照（後端 = GTOW 聚合報告/解庫 + deep link 跳轉）。
- **變因隔離**：改一個變數（深度、sizing、位置、賽段）看策略怎麼動 — 頂尖研究法的核心（「很多最有價值的研究 spot，你實戰永遠不會遇到」）。
- **Why-engine**：解釋用 range 形態語言（range morphology、nut advantage、blocker、incentive、SPR），不是頻率背誦。所有解釋走現有 coach_facts 硬驗證管線 — 反幻覺是本專案的皇冠資產，永不放鬆。
- 研究的**唯一合格出口**是 playbook 候選規則（§5.5）— 看完就結束的研究等於沒研究。

**驗收訊號**：每次中圈研究 session 產出 ≥1 條候選規則；規則陳述能通過 solver 抽查驗證。

### 5.5 Playbook（核心資產）

**目的**：一切學習沉澱到這裡。**如果哪天整個系統丟掉，playbook 是你要帶走的那個檔案** — 它是「你的策略」的 version control。

**終態**：

- 一條規則 = **可測試的主張**：陳述（帶上桌的語言）、適用範圍（family/深度/賽段/pool 條件）、solver 證據連結、已知例外、精通狀態、實戰驗證狀態。
- **生命週期**：`候選`（來自研究/課程/DD/影片）→ `已驗證`（過 solver 抽查）→ `已訓練`（drill 90%）→ `實戰證明`（外圈確認該 family EV loss 下降）→ `維護中`（SRS 抽查防遺忘）→ `淘汰`（meta 變化 / 升級後失效）。
- **兩類規則明確分開**：GTO 基準規則 vs **剝削規則**（pool-conditional，必附「面對強手/未知對手退回基準」的條件）。
- 課程、書、影片筆記全走同一管線：內容 → 抽出主張 → solver 驗證 → 候選入庫（§5.11）。

**驗收訊號**：playbook 條目數穩定成長且各生命週期階段都有流動；任抽一條 `已驗證` 規則，solver 抽查通過；`實戰證明` 條目對應的 family 在 Ledger 上確實改善。

### 5.6 Dojo（訓練場）

**目的**：把系統從答案機變訓練機。核心不變量：**沒有作答就沒有答案**（retrieval-first）。

**終態 — 執行分兩層**：

- **GTOW Trainer 編排層（reuse）**：Head Coach 每週把焦點 family 轉成 **drill 處方**（GTOW Trainer 的 spot 篩選配置 + 目標題量/時間），選手去 GTOW 練，系統回收 practiced hands 紀錄評估精通度。GTOW Trainer 沒有的是跨 session 課表、SRS 排程、與你的 leak 掛鉤的出題 — 那就是我們這層的全部工作。
- **自建題型層（GTOW 做不到的）**：
  - **自己錯誤的 SRS**：實戰錯的手（含 GTOW Trainer 練錯的手）以同構變體（花色置換、等價 board）morph 成新題，按間隔複習排程回歸（例：3 天 → 7 天 → 21 天），連續答對出 deck，答錯重置。全市場沒有人做「你自己的錯誤」的 SRS — 因為沒有人有這本帳。
  - **Daily Dose 題庫**：DD 的情境選擇題按 family/概念 tag 後混入（§5.11），用於 warmup 與概念診斷。
  - **Feynman 模式**：你用嘴巴（語音/文字）解釋這個 spot 的策略邏輯，AI 對照 solver 事實批改你的**理由**，不只動作 — 動作對但理由錯是未來的 leak。
  - **對抗模式**：教練對你的答案唱反調逼你 defend；或 GTO 純理派 vs 剝削派兩個 persona 辯論同一手，暴露你沒想過的面向。
  - **頻率 + 信心作答**：每題要求動作、頻率預測、信心申報三件套，餵校準診斷。
- **練習參數**（採用業界已驗證的學習科學參數）：單 family 每次 ≤20 分鐘；90% 精通門檻才推進；有意義地 interleave 不同 family（相近 family 不連續排）；RNG 模式練混合策略執行。

**驗收訊號**：SRS 到期題每日清空率；90% 門檻的 family 在隔週重測仍 ≥85%（防短期記憶假象）；Feynman 批改與 solver 事實的一致性抽查。

### 5.7 Endgame Lab（ICM / PKO / FT 專門實驗室）

**目的**：MTT 的錢集中在終局（獎金結構非線性），但真實 FT 手數天生稀缺 — 等實戰餵資料會餓死，**必須靠模擬與專門訓練補課**。這也是現有系統最弱的地方（ICM postflop 靜默 fallback chipEV、PKO 無 bounty 數學）。

**終態**：

- **內容軸**：泡沫/ladder 情境、FT 各 stack 角色（尤其中等 stack 2nd–4th — risk premium 最高的位置）、PKO bounty-EV 內化（把 bounty 換算成 pot equity 的即時直覺）、satellite 專門規則、**risk premium 直覺**（看 stack 配置能心算風險溢價帶）。
- **反字面主義內建**：教材明確教「什麼時候 ICM 調整是假的」（例：50% 場付 50% 的早期 ICM 調整；chipEV sizing 直接搬進 ICM 語境）。工具的字面輸出會讓人變差，這個警告寫進每個 ICM 教學輸出。
- **後端**：GTOW ICM/PKO 解庫（已大幅擴充）優先 reuse；不足處以 HRC 級工具補；我們的 icm_modes 近似全部帶誠實標注。
- **與診斷聯動**：賽段（早/中/泡沫/FT）是 Ledger 的一級維度，終局 leak 單獨成榜。

**驗收訊號**：終局 family 的 drill 覆蓋率與精通度；線下/線上進入泡沫與 FT 的實戰決策，於 48h 內完成復盤。

### 5.8 Exploit Intelligence（族群剝削情報局）

**目的**：從「學不輸」跨到「學贏」。在 $1–10 的 pool，這層的期望值高於 GTO 精修 — 但它必須建立在基準之上，不是替代基準。

**終態**：

- **Pool 挖礦**：從自有 HH 全量 corpus 聚合 villain 決策（GG 匿名無妨 — 族群分析不需要身分）：你的級別 pool 在哪些節點族偏離基準最遠（例：面對 river raise 的 overfold、3bet 頻率、limp 行為）。小樣本以收縮估計貼回 GTO 先驗，避免拿噪音當讀牌。這需要自建 HH 解析（Analyzer 只評 hero）— §4.1 分工表的自建項。
- **Max-exploit 研究**：對 pool 偏離最大的節點做 node-lock 重解（GTOW Nodelocking 或本地 solver），產出**剝削規則**入 playbook（帶 pool 條件與退回條件）。
- **線上/線下雙 pool**：線下 pool 單獨建檔（偏離模式完全不同：limp 文化、被動、sizing tell）。線下樣本小 → 以「已發表的 population 研究 + 選手觀察」為先驗，自有資料修正。
- **Session 內讀牌輔助**（僅賽後）：cooldown 時回顧當桌對手的可觀察偏離，訓練「下次遇到同型對手怎麼調」。
- **邊界**：剝削規則永不覆蓋基準規則的學習順序 — 先會走（GTO 基準）再會跑（偏離）。每條剝削規則必附「這樣偏離讓我暴露什麼反剝削風險」。

**驗收訊號**：pool 傾向報告的每個主張都有樣本數與信賴度；至少一條剝削規則走完 `候選 → 實戰證明` 全生命週期。

### 5.9 Performance Layer（心智與狀態）

**目的**：技術天花板之外的另一半 — A-game 出現率。把 Tendler 的 inchworm 資料化：目標是砍掉 C-game 左尾，不只拉高 A-game 右尾。

**終態**：

- **A/B/C-game 分佈**：從 Ledger 算 session 級 EV loss 百分位分佈，量化你的左尾；月報追蹤左尾收窄。
- **Tilt 簽名**：挖掘你個人的觸發模式（bad beat 後 N 分鐘窗口？session 第幾小時？特定時段？多桌數？）。現有 `detect_tilt` 是雛形 — 終態它接上推播與 cooldown 復盤，不再只寫 DB。
- **儀式（bot 主持）**：
  - **Warmup**（開桌前 ~10 分鐘）：SRS 到期題 + 最弱 family 快問 + 意圖設定（今天的一個過程目標）。
  - **Cooldown**（收桌後 ~10 分鐘）：本場最貴 3 手一鍵標記（隔天復盤）、tilt 窗口回看、情緒 tag、自評。
- **量與狀態管理**：EV loss vs session 時長曲線 → 個人化的 session 長度上限建議；週 volume 與依從率追蹤；（可選）睡眠/精力手動 tag 找相關性。
- **紅線（不變量 8）**：對局進行中零策略輸入。局中支援僅限時間型提醒與自評 check-in（例：每 90 分鐘「狀態 1–5？」）。這是誠信與帳號安全的雙重紅線，永不跨越。

**驗收訊號**：warmup/cooldown 完成率；tilt 事件月趨勢；左尾（worst 20% session 的 EV loss）收窄。

### 5.10 Head Coach（總教練 / 編排器）

**目的**：讓十個子系統從工具箱變成系統的那一塊。今天的系統等你來問；終態的系統知道你今天該練什麼。**注意力分配是最稀缺的資源，也是目前唯一沒被軟體輔助的環節。**

**終態**：

- **週課表生成**：按 `期望收益 = EV loss 金額 × 出現頻率 × 可學性` 挑 1–2 個焦點 family，把中圈的五個環節（研究→壓縮→drill→重測）排進你**真實可用的時間預算**（目前 ~8h/週）。排不完自動降載 — 課表塞爆是依從性殺手。
- **好奇心隊列**：線下流的 `curious` 標記與你主動提的問題進入隊列，教練在課表中保留固定比例給它（動機是複利的燃料，不全然服從 EV 排序）。
- **週期化（periodization）**：大賽/賽節前 → 只鞏固不學新（taper）；賽後 → 恢復週；平時 → 累積週。借自運動科學,防 burnout。
- **配比治理**：GTO 基準 / 剝削 / 終局 / 心智的學習配比隨級別與診斷動態調整（目前級別的預設傾向：基準與終局為主、剝削快速跟上、心智貫穿）。
- **人的否決權**：課表是預設議程，選手可一鍵改；教練記錄偏好並適應。
- **週報 = 記分卡 + 下週課表**（不再只是報告）；月報 = 外圈歸因報告；季報 = 標準考 + 元圈檢討。

**驗收訊號**：連續 4 週課表自動生成且依從率 >60%；焦點 family 的選擇能以數據解釋（可追溯）。

### 5.11 Knowledge Ingestion（內容攝取）

**目的**：你每週 ~6h 的課程/書/影片學習目前在系統外流失。終態：一切內容學習匯入同一條複利管線。

**終態**：

- **Daily Dose of GTO（優先攝取源）**：千頁 PDF 解析為 (a) **概念單元** → 概念圖譜（range morphology、blocker、SPR、MDF、risk premium…），每個概念鏈到相關 family 與 playbook 規則；(b) **題庫** → 每題 tag（概念 × family × 難度），進 Dojo 的 warmup/概念診斷輪替。
- **概念 × 執行雙軸診斷**：DD 題測「理解」，Trainer/實戰測「執行」。交叉找出兩種人格分裂：「懂但不會做」→ 加 drill；「會做但不懂」→ 加概念學習（後者是 tree-overfitting 高危群 — 只記住了這棵樹的答案）。
- **課程/書/影片**：筆記或轉錄 → 抽出可測試主張 → solver 抽查驗證（在你的深度帶）→ playbook 候選。內容來源的主張與 solver 衝突時，以 solver + 註記呈報，由你裁決。
- **邊界**：攝取不追求全文照搬，追求「主張 → 驗證 → 入庫 → drill」的轉化率。

**驗收訊號**：DD 題庫入庫並可按 family 抽題；每月至少 4 條外部內容主張走完驗證入庫流程。

---

## 6. 概念資料模型

實作自由，但概念上這五個實體是系統的骨架，欄位語義不可縮水：

```yaml
DecisionRecord:            # Ledger 的一筆帳
  hand_ref: ...            # 手牌參照（HH 檔/截圖/線下捕獲記錄）
  source: online|live|train|exam
  grader: gtow_analyzer|own_pipeline|local_solver
  context: {family, texture_class, depth_band, phase, positions, format: pko|vanilla|satellite|cash, venue: gg|live}
  verdict: {ev_loss_bb, ev_loss_pct_pot, freq_diff, taken, best}
  honesty: {parse_confidence, approximation_flags[], fragile: bool}   # 不變量 2
  intent_tags: [uncertain|curious]           # 線下流的一級公民
  session_ref: ...
  timestamps: ...

PlaybookRule:
  statement: ...           # 帶上桌的語言（中文）
  kind: gto_baseline|exploit
  scope: {families[], depth_bands[], phases[], pool_conditions[]}
  evidence: [solver_refs...]
  exceptions: [...]
  fallback: ...            # exploit 規則必填：退回基準的條件
  lifecycle: candidate|verified|drilled|field_proven|maintained|deprecated
  mastery: {drill_acc, last_tested, srs_due}

QuizItem:
  origin: own_mistake|dd_pdf|synthesized|exam_bank
  family: ...              # 我們的 taxonomy（§4.2）
  isomorph_seed: ...       # 同構變體生成的種子
  answer_model: {actions, freqs, rationale_facts}   # 批改依據，須 solver-grounded
  difficulty: ...
  srs: {interval, ease, due}

SessionRecord:
  {venue, format, buyin, start/end, tables, self_rating_pre/post, tilt_windows[], notes}

MetricSnapshot:            # 指標金字塔的定期快照
  {period, ev_loss_per_100_by_family, treated_vs_untreated, drill_mastery, calibration_error, adherence, volume, outcome_ci}
```

---

## 7. 設計原則與不變量（12 條，無商量空間）

1. **沒有作答就沒有答案** — 一切訓練互動 retrieval-first；分析輸出前先要求選手的答案/理由（復盤情境亦然，至少一句「你當時的思路？」）。
2. **每個判定帶信心與近似標注**；低信心不進統計。誠實層對所有 grader 一視同仁（含 GTOW）。
3. **一切 EV 加權**，不做頻率計數排序。
4. **單手是入口，spot family 是學習單位** — 任何復盤功能必須有「升級到 family 聚合」的出口。
5. **「學會」的唯一定義：該 family 的實戰 EV loss 下降**（外圈裁決）。drill 分數只是中間指標 — 防 Goodhart 的最終條款。
6. **一切知識沉澱進 playbook**；每條規則可測試、有證據、有生命週期。看完即忘的輸出是浪費。
7. **系統主動排課，人保留否決權。**
8. **對局中零策略輸入**（RTA 紅線 — 誠信與帳號安全，永不跨越）。
9. **資料主權**：手牌、判定、規則、solve 結果全部自有、可匯出、可重算。底座（GTOW）可替換。
10. **不重造市場已有的通用功能** — solver browser、generic trainer、逐手評分引擎一律 reuse；只建「你的資料才能做」的個人層。
11. **回饋延遲預算**：能即時的不過夜（drill 批改）、能過夜的不過週（session 復盤）、能過週的不過月（歸因）。
12. **中文為教學語言**，術語一致（沿用現有 terminology 規範）；英文術語自然混用但不翻譯硬凹。

---

## 8. 非目標（Non-goals）

- ❌ **實時輔助 / RTA** — 永遠不做，包括任何「局中看牌給建議」的變形。
- ❌ **以結果為主 KPI 的儀表板** — ROI/獎金曲線只在結果層帶信賴區間呈現，永不導航。
- ❌ **重造 GTO Wizard 的通用功能** — 包括自建全量逐手評分（Analyzer 已做）、generic solver UI、通用 trainer（先 reuse，自建題型只補它做不到的個人化部分）。
- ❌ **取代人類 study group** — 反而輸出「本週最值得跟人討論的 3 手」議程，幫你經營它。頂尖選手最一致的成功歸因是回饋密度，AI 補不滿這一塊。
- ❌ **（現階段）多人產品化** — 先把 N=1 做到極致；架構不排除未來開放，但任何「為了未來用戶」的通用化都是現在的 scope creep。
- ❌ **追求 solver 頻率的背誦精度** — 系統教的是 heuristic、range 邏輯與敏感度，不是小數點頻率（tree-overfitting 是明確要防的病）。

---

## 9. 載體與基礎設施演化

**載體是殼，Ledger + Playbook + Head Coach 是魂。** 魂的資料結構設計好，換殼是低成本的。

- **演化順序**：Telegram bot（保留為脊椎：捕獲入口、快問快答、推播、儀式主持）→ **web dashboard**（drill/研究/趨勢需要富 UI：range grid、樹視圖、曲線 — TG 做不出好的訓練體驗）→ 桌面捕獲 agent（HH 自動入庫）→ 手機碎片時間 SRS。
- **Solver 分層**：GTOW 解庫（能用多久用多久）→ 本地 postflop solver（開源系 fork，做 node-lock/自訂樹）→ ICM 引擎（GTOW ICM 庫優先，HRC 級工具補）→ **個人解庫**（所有算過的 spot 永久快取、按 family 索引，逐年長大 → 對外依賴逐年變小）。
- **AI 層**：模型可替換；所有教學輸出經過事實驗證管線（現有模式延續）；模型每升級一代，Feynman 批改、why-engine、Head Coach 免費變強 — 介面設計要吃得到這個紅利。
- **資料層**：單一 DB of record（現 Supabase 延續即可），全量可匯出。

---

## 10. 分階段路線圖

原則：**先讓迴圈轉起來（即使醜），再優化保真度** — 與過去「先把保真度磨到極致」相反的排序。理由：迴圈的複利 > 邊際精度。每個 phase 都以「某一層迴圈開始自轉」為完成定義。

### Phase 1 —「帳本」（讓資料全量、誠實）
- GTOW Analyzer 結果攝取 → Ledger（線上全量流）；GTOW Trainer 紀錄攝取（訓練流）。
- 線下流 v1：輕量捕獲對話 + 意圖標籤 + 自建 pipeline 評分。
- EV 加權診斷 + 三型 leak + 盲點矩陣 v1；週記分卡（取代現行週報的排序邏輯）。
- **驗收**：連續 4 週 100% 線上手入帳；週報以 EV 排序且分 family；每筆帳有信心標注；盲點矩陣有第一批「未標記+錯」案例浮出。

### Phase 2 —「迴圈」（中圈自轉）
- Head Coach 週課表 v1（焦點 family 選擇 + 時間預算適配）。
- Dojo v1：GTOW Trainer 處方/回收 + 自建 SRS（自己錯誤的同構變體）+ 頻率/信心作答。
- Playbook v1（生命週期到 `已訓練`）；研究工作台 v1（聚合視圖 + 候選規則出口）。
- Daily Dose 攝取 v1（題庫先行，概念圖譜其次）；warmup/cooldown 儀式上線。
- **驗收**：連續 4 週中圈完整轉動（診斷→研究→入庫→drill→重測）；依從率可量測；playbook ≥10 條 `已驗證`。

### Phase 3 —「裁決與專精」（外圈 + 終局）
- 外圈歸因報告（treated vs untreated）；「實戰證明」生命週期打通。
- 季度標準化測驗 v1（題庫 + 同構變體生成）。
- Endgame Lab v1（ICM/PKO 教學帶反字面主義標注、賽段一級維度、終局 drill 處方）。
- Tilt 簽名 + A/B/C-game 分佈；tilt 推播接通。
- **驗收**：第一份歸因報告產出且結論可信；第一次季考完成；至少一條規則達 `實戰證明`。

### Phase 4 —「邊際與主權」（剝削 + 平台）
- Exploit Intelligence：villain 側 HH 解析 + pool 傾向報告 + node-lock 研究流 + 剝削規則生命週期。
- Web dashboard；本地 solver 分層與個人解庫；線下 pool 檔案。
- **驗收**：pool 報告主張帶樣本數；一條剝削規則走完全生命週期；GTOW 斷供演練（fallback 可用）。

Phase 之間允許重疊；但**任何 phase 內的功能提案都要先過 §13 清單**。

---

## 11. 風險與對沖

| 風險 | 對沖 |
|---|---|
| GTOW 依賴（無 API、ToS 灰色、價格/功能變動、token 脆弱） | grader-agnostic Ledger（§4.2）；自建 pipeline 永久 fallback；個人解庫累積（§9）；攝取機制批次可重跑 |
| Goodhart / 背題刷分 | 同構變體；隔週重測門檻；不變量 5（實戰裁決） |
| 過度自動化 → 回到被動消費 | 不變量 1（retrieval-first）；研究出口綁 playbook（不變量 6） |
| 課表塞爆 → 依從崩潰 | 時間預算適配 + 自動降載（§5.10）；過程層指標盯依從率 |
| 資料品質（截圖/解析錯誤污染帳本） | 誠實層 + validator（已建）+ HH-first 原則 |
| 單人系統迴音室（沒人跟你吵架） | 對抗模式 persona（§5.6）；study group 議程輸出（§8） |
| RTA 越線（漸進式功能蠕變） | 不變量 8 為紅線；任何「局中」功能提案自動否決 |
| 心理面：solver 完美主義 → 控制強迫與 burnout | 週期化與恢復週（§5.10）；結果層信賴區間呈現（variance 教育）；Performance Layer 盯量 |
| 微級別高 rake 侵蝕邊際 | 元圈管理 game selection 與升級節奏 |

---

## 12. 現有資產對照（2026-07）

| 資產 | 處置 | 理由 |
|---|---|---|
| coach_facts 硬驗證 / 反幻覺管線 | **保留，皇冠資產** | 領先商業實踐；所有教學輸出的地基 |
| deviations DB + spot_categorizer | **升級為 Ledger + 官方 taxonomy** | 加 EV/信心/來源/意圖欄位；分類法成為跨源標準 |
| 自建解析→solver pipeline（text/截圖/HH） | **保留，降級為特化 grader** | 線下手、截圖、multiway、villain 挖礦的唯一路徑；全量評分讓位給 Analyzer 攝取 |
| OCR 管線（CardCNN 等） | **凍結精度投資，維持可用** | 保真度已進遞減區；HH-first 之後它是便利工具 |
| 週報 / leak 排序 | **重做** | 頻率排序 → EV 加權；報告 → 記分卡+課表 |
| 教練 persona / COACH_SYSTEM | **重做方向** | 解說員 → 教練（先要求作答、會反問、批改理由） |
| detect_tilt | **接通** | 已實作未使用 → 接推播與 cooldown |
| GTOW deep link | **保留** | 研究跳轉的膠水 |
| 快照/regression 測試文化 | **保留並延伸** | 同樣的紀律用到訓練系統的資料品質上 |
| Telegram bot | **保留為脊椎** | 捕獲+推播+快問；訓練 UI 另建 |

**結論：不需要砍掉重練。** 已建成的是「評分服務與帳本地基」，新系統是在上面蓋的訓練樓層；舊系統成為新系統的一個器官，不是違建。

---

## 13. 未來提案的對齊檢查清單（gate）

任何新 feature / 計畫 / PR，先回答：

1. **服務哪一層迴圈**（內/中/外/元）？說不出來 → 不做。
2. **產出沉澱到哪個資產**（Ledger / Playbook / 題庫 / 指標）？不沉澱 → 重想。
3. **違反哪條不變量**（§7）？尤其 1（作答先於答案）、2（信心標注）、5（實戰裁決）、8（RTA 紅線）。有違反 → 改提案，不改憲法。
4. **GTOW 或市場已有嗎**？有 → reuse 或整合，不重造（不變量 10）。
5. **對北極星指標的影響路徑**是什麼？（哪個 family 的 EV loss、透過哪個機制、多快能觀測）
6. **會不會增加被動消費**？（多一份沒人讀的報告 = 負分；多一次被迫作答 = 加分）
7. **維護成本落在哪**？（脆弱的外部整合要有 fallback 與退場設計）

---

## 14. 術語表

| 術語 | 定義 |
|---|---|
| **Spot family** | 我們官方分類法下的決策情境類別（如 facing_3bet、cbet_oop），學習與統計的基本單位 |
| **Ledger** | Decision Ledger，全量決策帳本（§5.2） |
| **Playbook** | 個人策略規則資產庫（§5.5） |
| **內/中/外/元圈** | 四層訓練迴圈（§3） |
| **盲點矩陣** | 意圖標籤 × 評分結果的 2×2 診斷（§5.3） |
| **三型 leak** | 知識型 / 紀律型 / 邊界型（§5.3） |
| **同構變體（isomorph）** | 花色置換/等價 board 生成的等價新題，防背題 |
| **Treated / Untreated** | 練過 / 沒練過的 family，歸因對照用（§2.2） |
| **誠實層** | 信心 + 近似 + 敏感度標注體系（§5.2） |
| **Fragile** | 鄰近深度/stack 重算會翻轉結論的判定 |
| **DD** | Daily Dose of GTO（GTOW 千頁教材 PDF） |
| **Risk premium** | ICM 下相對 chipEV 需要的額外 equity 溢價 |
| **Node-lock** | 鎖定對手某節點策略後重解，產生剝削解 |
| **RTA** | Real-Time Assistance，對局中即時輔助 — 本專案紅線 |
| **Inchworm** | Tendler 的技能分佈觀：進步 = 同時推進 A-game 與砍掉 C-game 左尾 |

---

## 附錄 A：終態的一週

- **週一（打牌日）**：19:40 手機跳出今日 SRS 到期 14 題（8 題 BB 防守、4 題 FT 泡沫 jam/fold、2 題上週實戰錯誤的變體），12 分鐘清完。Warmup 完成、意圖設定「今晚 3bet pot OOP 不下意識 cbet」。收桌後 cooldown：287 個決策、最貴 3 手標記明日復盤、22:05–22:20 偵測到 tilt 窗口（在一次 bad beat 之後）、情緒 tag、自評 3/5。
- **週二**：Analyzer 攝取昨晚全量 → Ledger 更新；診斷引擎把一手「未標記+評分錯」的 BTN vs BB turn 決策丟進盲點隊列。SRS 8 題。
- **週四（study 日）**：Head Coach 已備課：本週焦點 = 3bet pot OOP cbet（上週 EV loss 的 34%）。研究台展開 12 個 texture 的聚合視圖，做兩輪變因隔離，Feynman 口述被批改一處理由錯誤，2 條 heuristic 入 playbook（`已驗證`）。教練開出 GTOW Trainer 處方（該 family × 25–40bb，40 題）。
- **週六（線下賽）**：休息時間語音記了兩手 `uncertain` 的手。回家後系統補問三個細節，評分入帳：一手正確（信心不足型 → 排 Feynman）、一手 −1.8bb（知識型 → 進下週課表候選）。
- **週日晚**：週記分卡：EV loss/100 四週趨勢 5.1 → 4.2（treated family 9.8 → 5.5，untreated 持平 → 歸因成立）；playbook 2 條升級 `已訓練`；tilt 事件 3 → 1；依從率 78%。下週課表已生成，焦點輪到第二大 leak。
- **季末**：100 題標準考（同構變體），分 family 成績與校準曲線入年度曲線；元圈檢討：bankroll 與決策品質達標 → 買入配比上移一級。

## 附錄 B：本文件的維護規則

- 事實快照（§0）過時就更新，不需審批。
- 子系統（§5）的**終態描述**可隨學習演進，但改動要對齊 §2/§3/§7。
- 不變量（§7）與非目標（§8）的任何修改：需 Harry 明確同意 + 在 §15 留痕。
- 每完成一個 Phase（§10），回頭核對驗收訊號並更新狀態。

## 15. 版本紀錄

- **1.0（2026-07-07）**：初版。由 Harry 與 Claude（Fable 5）在深度討論後共同定稿：定位（一人職業戰隊、GTOW 為底座的教練層）、北極星指標與歸因、四層迴圈、11 個子系統、12 條不變量、四階段路線圖。
