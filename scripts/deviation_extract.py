#!/usr/bin/env python3
"""Deviation capture for the live coaching flow (fire-and-forget after
each analysis). Extracted from src/gemini_session.py (god-file split).
This is also the module where honesty fields (is_icm / validation /
multiway notes) get threaded into deviations rows in a later phase."""
from __future__ import annotations


async def extract_deviations(db, logger, chat_id: int, hand_id: str | None,
                             hand_json: dict, context: dict):
    """Fire-and-forget: extract deviations from analysis and store in DB.

    Reads hero_spots and solutions from the analysis context, categorizes
    each hero decision point, compares to GTO, and inserts into deviations table.
    """
    if not db or not db.pool:
        return
    try:
        from spot_categorizer import (
            categorize_spot,
            classify_board_texture,
            compute_preflop_line_key,
            compute_pot_type_from_preflop,
            identify_primary_villain,
            map_spot_to_gtow,
            _identify_preflop_aggressor,
        )
        from gto_formatter import combo_index_for_hand, _COMBO_INDEX, _get_board_cards, _combo_to_hand_name
        from hh_deviation_check import (
            _get_action_evs_preflop,
            _get_action_evs_postflop,
        )
        from leak_service import (
            insert_deviation,
            DeviationMeta,
            compute_ev_loss,
            pick_best_ev_action,
            classify_aggression_direction,
        )

        hero_spots = context.get("hero_spots", [])
        solutions = context.get("solutions", [])
        hero_pos = context.get("hero_position", "")
        hero_hand = context.get("hero_hand", "")
        hero_hand_raw = hand_json.get("hero_hand", "")
        effective_bb = hand_json.get("effective_bb")
        combo_idx = combo_index_for_hand(hero_hand_raw)

        # Parse hand_history_id from hand_id (e.g. "H1234" → 1234)
        hh_id = None
        if hand_id and hand_id.startswith("H"):
            try:
                hh_id = int(hand_id[1:])
            except ValueError:
                pass

        preflop_action_index = 0
        for i, (spot, sol) in enumerate(zip(hero_spots, solutions)):
            if not sol or "action_solutions" not in sol:
                continue

            street = spot.get("street", "")
            is_preflop = (street == "preflop")

            # Determine action_index for this street
            if is_preflop:
                action_idx = preflop_action_index
                preflop_action_index += 1
            else:
                # Count previous hero spots on the same postflop street
                action_idx = sum(
                    1 for j in range(i)
                    if hero_spots[j].get("street") == street and hero_spots[j].get("street") != "preflop"
                )

            # Build street_actions_before_hero from the spot context
            # This is tricky — we reconstruct from what we know
            street_actions_before = spot.get("street_actions_before_hero", [])

            cat, texture = categorize_spot(
                hand_json, street, action_index=action_idx if is_preflop else 0,
                street_actions_before_hero=street_actions_before if not is_preflop else None,
            )

            # Get board texture for postflop
            if not is_preflop and not texture:
                board = spot.get("params", {}).get("board", "")
                texture = classify_board_texture(board)

            # Extract hero's action and GTO recommendation
            taken_code = spot.get("taken_code")
            if not taken_code:
                # For preflop open spots, hero's action is in the preflop string
                continue

            # Get hero's action frequency from solution
            hero_freq = None
            gto_action = ""
            gto_freq = None

            action_solutions = sol.get("action_solutions", [])
            player_info = None
            for pi in sol.get("players_info", []):
                if pi["player"]["position"] == hero_pos:
                    player_info = pi
                    break

            if player_info and "range" in player_info:
                range_arr = player_info["range"]

                if is_preflop and len(range_arr) == 169:
                    # Preflop 169-element lookup
                    from hh_deviation_check import HAND_TO_169
                    idx_169 = HAND_TO_169.get(hero_hand)
                    if idx_169 is not None and range_arr[idx_169] >= 0.005:
                        action_freqs = {}
                        for asol in action_solutions:
                            strat = asol.get("strategy", [])
                            if len(strat) == 169:
                                action_freqs[asol["action"]["code"]] = strat[idx_169]
                        hero_freq = action_freqs.get(taken_code)
                        if action_freqs:
                            best_code = max(action_freqs, key=action_freqs.get)
                            gto_action = best_code
                            gto_freq = action_freqs[best_code]
                elif not is_preflop and len(range_arr) == 1326:
                    # Postflop 1326-element lookup
                    use_idx = combo_idx
                    if use_idx is not None and use_idx < len(range_arr) and range_arr[use_idx] >= 0.005:
                        action_freqs = {}
                        for asol in action_solutions:
                            strat = asol.get("strategy", [])
                            if len(strat) == 1326:
                                freq = strat[use_idx]
                                if freq > 0.005:
                                    action_freqs[asol["action"]["code"]] = freq
                        hero_freq = action_freqs.get(taken_code, 0)
                        if action_freqs:
                            best_code = max(action_freqs, key=action_freqs.get)
                            gto_action = best_code
                            gto_freq = action_freqs[best_code]

            if hero_freq is None:
                # Fallback: use total_frequency from action_solutions
                for asol in action_solutions:
                    if asol["action"]["code"] == taken_code:
                        hero_freq = asol.get("total_frequency")
                        break
                if not gto_action:
                    best_asol = max(action_solutions,
                                   key=lambda a: a.get("total_frequency", 0),
                                   default=None)
                    if best_asol:
                        gto_action = best_asol["action"]["code"]
                        gto_freq = best_asol.get("total_frequency")

            # Convert frequencies to percentages (0-100)
            hero_freq_pct = hero_freq * 100 if hero_freq is not None else None
            gto_freq_pct = gto_freq * 100 if gto_freq is not None else None

            is_deviation = (hero_freq is not None and hero_freq < 0.10)

            # ── Per-action EVs → true EV loss + best-EV action ──
            # Note: "gto_action" / best_code above is MAX FREQUENCY; we
            # track it as dominant_action and compute the true best-EV
            # action separately for the URL builder / ev_loss formula.
            try:
                if is_preflop:
                    action_evs = _get_action_evs_preflop(sol, hero_hand, hero_pos)
                else:
                    action_evs = _get_action_evs_postflop(
                        sol, hero_hand, hero_pos, combo_idx=combo_idx
                    )
            except Exception:
                action_evs = None

            ev_loss = compute_ev_loss(action_evs, taken_code)
            gto_best_ev_code = pick_best_ev_action(action_evs)
            gto_dominant_code = gto_action or None

            # ── Meta fields ──
            preflop_actions_str = hand_json.get("preflop_actions", "") or ""
            num_players = hand_json.get("players_at_table", 8)
            try:
                line_key = compute_preflop_line_key(
                    preflop_actions_str,
                    hero_pos,
                    num_players=num_players,
                    # Preflop: key captures up to hero's current decision.
                    # Postflop: consume the full preflop sequence so the
                    # pot_type reflects the line going into the flop.
                    action_index=(action_idx if is_preflop else None),
                )
            except Exception:
                line_key = None
            try:
                pot_type = compute_pot_type_from_preflop(
                    preflop_actions_str, num_players=num_players,
                )
            except Exception:
                pot_type = None

            try:
                villain_pos = identify_primary_villain(
                    hand_json,
                    hero_pos,
                    street,
                    street_actions_before if not is_preflop else None,
                )
            except Exception:
                villain_pos = None

            try:
                pf_agg = _identify_preflop_aggressor(preflop_actions_str, num_players)
            except Exception:
                pf_agg = None
            hero_is_pf_aggressor = (pf_agg == hero_pos)

            gtow_type, gtow_hero_role = map_spot_to_gtow(
                cat, pot_type, street, hero_is_pf_aggressor
            )

            aggression_direction = classify_aggression_direction(
                taken_code, gto_best_ev_code
            )

            dm = DeviationMeta(
                villain_pos=villain_pos,
                preflop_line_key=line_key or None,
                pot_type=pot_type,
                aggression_direction=aggression_direction,
                gtow_type=gtow_type,
                gtow_hero_role=gtow_hero_role,
                gto_dominant_action=gto_dominant_code,
                gto_best_ev_action=gto_best_ev_code,
            )

            await insert_deviation(
                pool=db.pool,
                chat_id=chat_id,
                hand_history_id=hh_id,
                street=street,
                action_index=action_idx,
                spot_category=cat,
                position=hero_pos,
                hero_action=taken_code,
                gto_action=gto_action or taken_code,
                hero_freq=hero_freq_pct,
                gto_freq=gto_freq_pct,
                ev_loss_estimate=ev_loss,
                board_texture=texture,
                effective_bb=effective_bb,
                is_deviation=is_deviation,
                meta=dm.to_jsonb() or None,
            )

    except Exception as e:
        logger.warning(f"[chat={chat_id}] Failed to extract deviations: {e}")
