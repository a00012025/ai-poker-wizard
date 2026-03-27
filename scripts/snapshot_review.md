# Snapshot Image Review

Review of OCR parse results vs ground truth from screenshots.
Used to build expected_json for regression tests.

## Summary: OCR vs Gemini Accuracy (15 images)

| Field | OCR Correct | Gemini Correct | Notes |
|-------|------------|----------------|-------|
| Hero Position | 12/15 (80%) | 9/15 (60%) | OCR wins. Errors: #5(SB→BB), #9(BTN→SB), #10(BB→CO) |
| Hero Hand | 15/15 (100%) | 12/15 (80%) | OCR perfect |
| Effective BB | ~6/15 (40%) | ~4/15 (27%) | Both weak. User accepts tolerance |
| Players Count | 11/15 (73%) | 10/15 (67%) | Close |
| Preflop Actions | 12/15 (80%) | 12/15 (80%) | Tie |
| Board Card Suits | 7/15 (47%) | 12/15 (80%) | **Gemini wins big**. OCR systematic ♥→♦ |
| Street Actions | 13/15 (87%) | 13/15 (87%) | Tie |

### Key OCR bugs to fix:
1. **Suit recognition ♥→♦** — 8/15 images affected, systematic
2. **Hero position** — 3 images wrong (#5, #9, #10)
3. **Missing actions** — #10 (time bank interference), #11 (missed UTG open)
4. **Player count** — derived from position mapping, cascades from #2

## Image 1: photo_2026-03-22 13.53.03.jpeg (H2491, OCR conf=0.85)

**OCR Result:**
- Hero: CO AdTd (17.3bb eff)
- Players: 8
- Preflop: F-F-F-F-R2-F-C-F
- Flop: 6cQs8d → SB:X CO:R2 SB:C
- Turn: 6d → SB:X CO:X
- River: 2d → SB:R4 CO:F

**Issues:**
- River card: 2d should be **2h** (red heart visible in screenshot)

**Expected corrections:**
- River card: 2h (street card fix in OCR)
- Everything else correct

---

## Image 2: photo_2026-03-22 13.53.04.jpeg (H2492, OCR conf=0.75)

**OCR Result:**
- Hero: UTG Ac6c (23.9bb eff)
- Players: 8
- Preflop: R2-F-F-C-F-F-F-F
- Flop: 2cTdJd → UTG:X LJ:X
- Turn: 4s → UTG:R4.9 LJ:C
- River: 8s → UTG:R27.6 LJ:F

**Issues:**
- Flop board: Td should be **Th** (red heart)

**Expected corrections:**
- Flop board: 2cThJd

---

## Image 3: photo_2026-03-22 13.53.05.jpeg (H2493, OCR conf=0.99)

**OCR Result:**
- Hero: BTN AdJh (28.4bb eff)
- Players: 6
- Preflop: F-F-F-R2.2-F-C
- Flop: Td5dAd → BB:X BTN:R1.1 BB:C
- Turn: 4c → BB:X BTN:R6 BB:C
- River: 9d → BB:X BTN:R89.9 BB:C(19.1)

**Issues:**
- Flop board: Td5dAd should be **Th5hAh** (hearts not diamonds)
- Hero hand: AdJh is correct (OCR right, Gemini wrong saying AhJh)

**Expected corrections:**
- Flop board: Th5hAh
- Hero hand stays AdJh

---

## Image 4: photo_2026-03-22 13.53.06.jpeg (H2494, OCR conf=1.00)

**OCR Result:**
- Hero: HJ AcTc (48.6bb eff)
- Players: 6
- Preflop: F-R2.2-F-F-C-C
- Flop: Kc9d3d → SB:X BB:X HJ:R2.5 SB:C BB:F
- Turn: Js → SB:X HJ:X
- River: 5d → SB:X HJ:R6.2 SB:C

**Issues:**
- Effective BB: 48.6 should be **41.1**
- River card: 5d should be **5h**

**Expected corrections:**
- effective_bb: 41.1
- River card: 5h

---

## Image 5: photo_2026-03-22 13.53.07.jpeg (H2495, OCR conf=1.00)

**OCR Result:**
- Hero: SB 8s8d (62.4bb eff)
- Players: 7
- Preflop: F-R2-F-F-F-R9-C
- Flop: 9dAcTd → SB:R5 LJ:R14.7 SB:F

**Issues:**
- Hero position: SB should be **BB**
- Players: 7 should be **6**
- Flop board: 9dAcTd should be **9hAcTh**
- Note: preflop has 7 actions (F-R2-F-F-F-R9-C) but table is 6 players — need to fix to 6 actions

**Expected corrections:**
- hero_position: BB
- players_at_table: 6
- Flop board: 9hAcTh
- preflop_actions: adjust to 6-player format

---

## Image 6: photo_2026-03-22 13.53.08.jpeg (H2496, OCR conf=0.88)

**OCR Result:**
- Hero: BTN 5s5c (10.2bb eff)
- Players: 7
- Preflop: F-F-R2-F-C-C-F
- Flop: 9d9s4d → SB:X LJ:X BTN:R2.5 SB:C LJ:F
- Turn: 3d → SB:X BTN:R19.8 SB:C(5.7)

**Issues:**
- Turn card: 3d should be **3h**
- OCR missing river street entirely — should be **3s**
- Flop opponent: LJ vs HJ uncertain (OCR says LJ, Gemini says HJ)

**Expected corrections:**
- Turn card: 3h
- Add river: 3s (need to check actions from screenshot)

---

## Image 7: photo_2026-03-22 13.53.09.jpeg (H2497, OCR conf=0.99)

**OCR Result:**
- Hero: BB Qc7d (16.2bb eff)
- Players: 7
- Preflop: R2.1-F-F-F-F-F-C
- Flop: 9sQsJc → BB:X UTG:R2.3 BB:R5 UTG:C
- Turn: As → BB:X UTG:X
- River: 3s → BB:X UTG:R2.5 BB:C

**Issues:** None — OCR and Gemini agree, user confirmed.

---

## Image 8: photo_2026-03-22 13.53.10.jpeg (H2498, OCR conf=1.00)

**OCR Result:**
- Hero: LJ AsQs (17.4bb eff)
- Players: 8
- Preflop: F-F-R2.2-F-F-F-F-C
- Flop: KcJc8d → BB:X LJ:R1.9 BB:C
- Turn: 6s → BB:X LJ:R7.1 BB:R13.3 LJ:F

**Issues:** None — correct as-is.

---

## Image 9: photo_2026-03-22 13.53.11.jpeg (H2499, OCR conf=0.94)

**OCR Result:**
- Hero: BTN AsJh (35.5bb eff)
- Players: 9
- Preflop: F-F-F-F-R2-F-R7-F-C
- Flop: KdAc4h → BTN:R4 BB:C
- Turn: 8d → BTN:R24.5 CO:F

**Issues:**
- Hero position: BTN should be **SB**
- Players: 9 should be **8**
- Preflop has 9 actions but table is 8 players — need to fix

**Expected corrections:**
- hero_position: SB
- players_at_table: 8
- preflop_actions: adjust to 8-player format

---

## Image 10: photo_2026-03-22 13.53.12.jpeg (H2500, OCR conf=0.85)

**OCR Result:**
- Hero: BB KhQd (None eff)
- Players: 8
- Preflop: F-F-F-F-F-F-R10-C
- Flop: Ks2dQc → BB:C(7.5)
- Turn: 4h → BB:X
- River: 4c → BB:R22.4 SB:F

**Issues (major):**
- Hero position: BB should be **CO**
- Effective BB: None should be **62.3**
- Preflop: F-F-F-F-F-F-R10-C should be **F-F-F-F-R2.2-F-F-R10-C** (CO open 2.2, SB 3bet 10, CO call). OCR missed the R2.2 open — possibly confused by a "BB time bank 10s" display in the panel.
- Flop: OCR only shows BB:C(7.5) but misses that **BB bet 7.5** first and **hero (CO) called**. Should be: BB:R7.5 CO:C
- Turn: OCR shows BB:X but missing hero action. Should be: **BB:X CO:X**
- River: OCR shows BB:R22.4 SB:F but hero is CO not BB. Should be: **BB:X CO:R22.4 BB:F** (BB check, hero all-in 22.4, BB fold)
- Both OCR and Gemini got hero position wrong (OCR=BB, Gemini=BB)

**Expected corrections:**
- hero_position: CO
- effective_bb: 62.3
- preflop_actions: F-F-F-F-R2.2-F-F-R10-C
- Flop: Ks2dQc → BB:R7.5 CO:C
- Turn: 4h → BB:X CO:X
- River: 4c → BB:X CO:R22.4 BB:F

---

## Image 11: photo_2026-03-22 13.53.13.jpeg (H2501, OCR conf=0.70)

**OCR Result:**
- Hero: BB Qs9h (42.7bb eff)
- Players: 8
- Preflop: F-F-F-F-F-F-F-C
- Flop: 4s8d6s → UTG:X
- Turn: Ts → BB:X UTG:R2.8 BB:C
- River: Jc → BB:R8.4 UTG:C

**Issues:**
- Players: 8 should be **9**
- Preflop: F-F-F-F-F-F-F-C should be **R2-F-F-F-F-F-F-F-C** (UTG open R2, OCR missed it)
- Flop: UTG:X missing BB:X before it. Should be **BB:X UTG:X**

**Expected corrections:**
- players_at_table: 9
- preflop_actions: R2-F-F-F-F-F-F-F-C
- Flop actions: BB:X UTG:X

---

## Image 12: photo_2026-03-22 13.53.14.jpeg (H2502, OCR conf=0.72)

**OCR Result:**
- Hero: CO Ah4h (50.3bb eff)
- Players: 6
- Preflop: F-R2-C-F-F-C
- Flop: KdAsAd → BB:X LJ:R3.7 CO:C HJ:F
- Turn: 2d → LJ:R5.5 CO:C
- River: 3h → BB:R6 CO:C(5.9)

**Issues:**
- Effective BB: 50.3 should be **17.1**
- Flop: opponent is **HJ** not LJ. Action should be BB:X HJ:R3.7 CO:C BB:F
- Turn: should be **HJ:R5.5** CO:C (not LJ)
- River: should be **HJ:R6** CO:C (not BB). Hero call is all-in 5.9bb.

**Expected corrections:**
- effective_bb: 17.1
- Flop actions: BB:X HJ:R3.7 CO:C BB:F
- Turn actions: HJ:R5.5 CO:C
- River actions: HJ:R6 CO:C(5.9)

---

## Image 13: photo_2026-03-22 13.53.15.jpeg (H2503, OCR conf=1.00)

**OCR Result:**
- Hero: CO 8s8d (37.8bb eff)
- Players: 6
- Preflop: F-F-R2.2-F-F-C
- Flop: 4c9d7c → BB:X CO:X
- Turn: As → BB:X CO:X
- River: Td → BB:X CO:X

**Issues:**
- Effective BB: 37.8 should be **31.6**
- River card: Td should be **Th**

**Expected corrections:**
- effective_bb: 31.6
- River card: Th

---

## Image 14: photo_2026-03-22 13.53.17.jpeg (H2504, OCR conf=1.00)

**OCR Result:**
- Hero: BB Ad9d (11.4bb eff)
- Players: 7
- Preflop: F-F-R2-F-F-F-C
- Flop: Qs3dAd → BB:X HJ:R1 BB:C
- Turn: Jd → BB:X HJ:R2 BB:C
- River: 3c → BB:X HJ:R7 BB:C(6.4)

**Issues:** None — OCR and Gemini agree, user confirmed.

---

## Image 15: photo_2026-03-22 13.53.18.jpeg (H2505, OCR conf=0.85)

**OCR Result:**
- Hero: LJ 2d2c (23.4bb eff)
- Players: 7
- Preflop: F-R2.2-F-F-F-F-C
- Flop: 9dQs3d → BB:X LJ:R1.3 BB:C
- Turn: 6d → BB:X LJ:X
- River: 9c → BB:R6.3 LJ:F

**Issues:**
- Flop 3rd card: 3d should be **3h**

**Expected corrections:**
- Flop board: 9dQs3h

---
