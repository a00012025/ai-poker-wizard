-- A possible-squeeze prescription changes materially with the opener's range.
-- Keep the existing broad parent for diagnosis, but make the training leaf
-- match GTOW's opponent-position filter: hero category × opener category ×
-- IP/OOP.  Existing ledger rows already carry every required dimension.
UPDATE ledger_decisions
SET spot_leaf = hero_cat || '_vsRaiseCall_v' || villain_cat || '_' || ip_oop,
    spot_keys = jsonb_build_array(
      'vsRaiseCall',
      hero_cat || '_vsRaiseCall',
      hero_cat || '_vsRaiseCall_v' || villain_cat || '_' || ip_oop
    )
WHERE spot_category = 'vsRaiseCall'
  AND hero_cat IS NOT NULL
  AND villain_cat IS NOT NULL
  AND ip_oop IN ('IP', 'OOP')
  AND spot_leaf IS DISTINCT FROM
      hero_cat || '_vsRaiseCall_v' || villain_cat || '_' || ip_oop;
