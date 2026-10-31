# GTOW browser verification — 2026-10-31

## Implemented training flow

`/drills` → `建立 RFI drills` provisions 12 saved drills: EP/MP/LP/SB × three training bands. These bands are independent of historical ledger/queue classifications:

- ≤25bb: 8, 9, 10, 12, 14, 17, 20, 25.
- 30–60bb: 30, 35, 40, 50, 60.
- >60bb: 80, 100.

Each drill loads every selected position × depth node, then unions continuation hands and boundary folds from the existing solver-grounded selector. An unavailable node fails that drill visibly; retries reuse saved settings. No representative-node approximation is used. SB RFI retains with-limps; other shallow RFI uses no-limps. All drills use Street mode.

All drill entry points share the new training boundaries. Historical diagnostic classifications remain unchanged; queue training scope is derived from numeric source depths. When a prescription spans solution families, its children are exposed separately with lifetime results and manual parent completion; they do not claim a combined attempt score. Custom action lines are replayed from source evidence in the selected family. Postflop preserves source depth and does not dynamically restrict hand ranges.

This integrates GTOW's existing trainer into the training loop; drill scores remain intermediate feedback, not evidence of improved real-game EV loss.

## Browser evidence

Authenticated Chrome CDP 9222, live app asset MX-_FbVC.js.

- New Drill > Modes > Street sets `fh_trainer_mode=stop_end_of_street`; Vue temporary router confirmed. Current app maps this to API training scope `street`.
- Multi Size MTT ChipEV No limps presents 8 depths: 8,9,10,12,14,17,20,25. Live gameformats metadata confirms `gametype=MTTMRonlyGeneral`, `variant=no_limps`, 8 players, ante 0.125, depths encoded e.g. `25.125`. 8/9 preflop only; 10–25 all spots.
- With limps metadata/UI includes 30,35,40,50,60,80,100 plus shallow depths. Live study 30.125 MTTGeneral history displays each UTG/UTG1/LJ/HJ/CO/BTN Fold/Raise2.1/Allin30; no Call. SB displays Fold/Call/Raise3.5/Allin30. Thus deep with-limps solution has no non-SB open limp at the observed 30bb node.
- Router has `depth_list` and `gmff_variant`, no direct `gametype_list` field. App derives API gametype_list from selected gameformat rows. Variant Any would include both families at shallow depths, so it does not express shallow no-limp plus deep with-limp; splitting is safest.
- Opened representative trainer URL with MTTMRonlyGeneral, no_limps, depths10.125,12.125,14.125,17.125,20.125, hero BTN, RFI, manual groups Q6o,Q7o,AA, stop_end_of_street. Config displayed 5 solutions. START TRAINING succeeded and dealt Qd7s to BTN at14.125 with preflop_actions=F-F-F-F-F. All filter params retained; realized fh_action_tmp=RFI, fh_hero_tmp=BTN.
- Live API game.players confirms literal seat order UTG,UTG+1,LJ,HJ,CO,BTN,SB,BB (UI renders UTG1).
- Study solution schema: action_solutions contains strategy and evs arrays length169; players_info has player and range length169. Trainer API has per-hand solutions and players_info range169.
- Manual hand subset is encoded `fh_groups_selection=manual&fh_groups=Q6o,Q7o,AA`; actual dealt Qd7s verifies it is applied. This subset is a parameter smoke test, not a suggested training range.

Saved drill roundtrip: created one temporary `Parameter verification — BTN RFI street` through SPA's own `api_call` dispatch, POST returned201. GET list readback retained exactly all11 supplied settings including street/no-limp/depths/manual range. Second GET and canonical settings comparison found exactly1 matching drill with the same id (`c4bcb307-1889-409c-b4de-cc9b5cb74fa7`), verifying settings identity supports reuse. Deleted only this temporary smoke-test drill after verification (204), avoiding cluttering the user's drill library. No preexisting drills changed.

Follow-up metadata: MTTGeneral with_limps also has subscribed 8-player8.125 and9.125, marked preflop_only. Thus RFI-only batch can include8/9 for all positions including SB. With-limps additionally has2–7bb preflop_only, but no-limp starts8bb, so8bb is common minimum coverage.
