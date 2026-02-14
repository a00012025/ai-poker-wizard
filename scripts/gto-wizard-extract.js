// ============================================================
// GTO Wizard Data Extraction Scripts
// Usage: agent-browser eval "$(cat scripts/gto-wizard-extract.js)"
// Or copy individual functions into agent-browser eval "..."
// ============================================================

// -----------------------------------------------------------
// 1. Discover available actions per position
//    Returns: [{pos: 0, actions: ["Fold","Raise 2.3","Allin 50"]}, ...]
// -----------------------------------------------------------
// agent-browser eval "JSON.stringify(Array.from(document.querySelectorAll('.hspotcrd_actions')).map(function(el,i){return {pos:i, actions:el.innerText.split('\n')}}))"

// -----------------------------------------------------------
// 2. Get action summary (overall frequencies + combos)
//    Returns: ["Allin 50\n0.9%\n12.3\ncombos", "Raise 16.1\n2.9%\n39.08\ncombos", ...]
// -----------------------------------------------------------
// agent-browser eval "JSON.stringify(Array.from(document.querySelectorAll('[class*=sab_item]')).filter(function(e){return e.innerText.match(/\\d+\\.?\\d*%/)}).map(function(e){return e.innerText}))"

// -----------------------------------------------------------
// 3. Get non-fold hands (quick check)
//    Returns: ["AA", "AKs", "AQs", ...]
// -----------------------------------------------------------
// agent-browser eval "var cells = document.querySelectorAll('.rtc.ra_table_cell'); var h = []; for (var i = 0; i < 169 && i < cells.length; i++) { var c = cells[i]; var s = c.getAttribute('style') || ''; if (s.indexOf('240, 60, 60') > -1 || s.indexOf('125, 31, 31') > -1) h.push(c.innerText.trim()); } JSON.stringify(h);"

// -----------------------------------------------------------
// 4. FULL per-hand mixed strategy extraction (THE MAIN SCRIPT)
//    Parses background-size CSS to get exact frequency ratios
//    Returns: [{hand: "AA", strategy: [{action: "raise", pct: 100}]}, ...]
//
//    Color mapping (verified):
//      rgb(240, 60, 60)  = raise (bright red)
//      rgb(125, 31, 31)  = allin (dark red/maroon)
//      rgb(61, 124, 184) = fold (blue)
//      rgb(76, 175, 80)  = call (green)
// -----------------------------------------------------------
// Usage: js_code=$(cat scripts/gto-wizard-extract.js | sed -n '/^\/\/ BEGIN_RANGE_SCRIPT/,/^\/\/ END_RANGE_SCRIPT/p' | grep -v '^//' | tr '\n' ' ') && agent-browser eval "$js_code"

// BEGIN_RANGE_SCRIPT
var COLOR_MAP = {"240, 60, 60": "raise", "125, 31, 31": "allin", "61, 124, 184": "fold", "76, 175, 80": "call", "90, 185, 102": "call"};
function parseColor(rgb) { for (var key in COLOR_MAP) { if (rgb.indexOf(key) > -1) return COLOR_MAP[key]; } return "unknown"; }
var cells = document.querySelectorAll(".rtc.ra_table_cell");
var result = [];
for (var i = 0; i < 169 && i < cells.length; i++) {
  var c = cells[i];
  var hand = c.innerText.trim();
  var s = c.getAttribute("style") || "";
  var colorsMatch = s.match(/rgb\([\d, ]+\)/g) || [];
  var sizesMatch = s.match(/background-size:\s*([^;]+)/);
  var sizesStr = sizesMatch ? sizesMatch[1].trim() : "";
  if (colorsMatch.length === 0) continue;
  var uniqueColors = [];
  for (var j = 0; j < colorsMatch.length; j += 2) {
    var action = parseColor(colorsMatch[j]);
    if (uniqueColors.indexOf(action) === -1) uniqueColors.push(action);
  }
  if (uniqueColors.length === 1 && uniqueColors[0] === "fold") continue;
  var sizesParts = sizesStr.split(",").map(function(p) { var m = p.trim().match(/([\d.]+)%/); return m ? parseFloat(m[1]) : 100; });
  var actions = [];
  if (uniqueColors.length === 1) {
    actions.push({action: uniqueColors[0], pct: 100});
  } else if (uniqueColors.length === 2) {
    var first_pct = sizesParts[0] || 50;
    actions.push({action: uniqueColors[0], pct: Math.round(first_pct * 10) / 10});
    actions.push({action: uniqueColors[1], pct: Math.round((100 - first_pct) * 10) / 10});
  } else if (uniqueColors.length === 3) {
    var p1 = sizesParts[0] || 33;
    var p2 = (sizesParts[1] || 66) - p1;
    actions.push({action: uniqueColors[0], pct: Math.round(p1 * 10) / 10});
    actions.push({action: uniqueColors[1], pct: Math.round(p2 * 10) / 10});
    actions.push({action: uniqueColors[2], pct: Math.round((100 - p1 - p2) * 10) / 10});
  }
  result.push({hand: hand, strategy: actions});
}
JSON.stringify(result);
// END_RANGE_SCRIPT

// -----------------------------------------------------------
// 5. Click on a specific hand in the grid to see detailed info
//    Replace "AA" with the desired hand
// -----------------------------------------------------------
// agent-browser eval "var cells = document.querySelectorAll('.rtc.ra_table_cell'); for (var i = 0; i < cells.length; i++) { if (cells[i].innerText.trim() === 'AA') { cells[i].click(); break; } }"

// -----------------------------------------------------------
// 6. Get depth selector options (available stack sizes)
// -----------------------------------------------------------
// First click to open: agent-browser eval "document.querySelector('.gmfover_title_text_depth').click()"
// Then extract rows: agent-browser eval "JSON.stringify(Array.from(document.querySelectorAll('.gmfstckstbl_table_td_effective_stack')).map(function(e){return e.innerText.trim()}))"

// -----------------------------------------------------------
// 7. Select a depth from the solutions library
//    Replace '17' with desired depth
// -----------------------------------------------------------
// agent-browser eval "var rows = document.querySelectorAll('.inftable_row.inftable_body_row'); for (var i = 0; i < rows.length; i++) { var td = rows[i].querySelector('.gmfstckstbl_table_td_effective_stack'); if (td && td.innerText.trim() === '17') { rows[i].click(); break; } }"

// -----------------------------------------------------------
// 8. Click a postflop action on seat card (e.g., "Bet 22.25")
//    Replace target text as needed
// -----------------------------------------------------------
// agent-browser eval "var btns = document.querySelectorAll('[class*=hspotcrd_action]'); for (var i = 0; i < btns.length; i++) { if (btns[i].innerText.trim() === 'Bet 22.25') { btns[i].click(); break; } }"

// -----------------------------------------------------------
// 9. Select a turn/river card from the board card picker modal
//    Suit classes: .poker-card.spades, .hearts, .diamonds, .clubs
//    Card selector rows (top to bottom): spades, hearts, diamonds, clubs
// -----------------------------------------------------------
// Click 6 of clubs:
// agent-browser eval "var cards = document.querySelectorAll('.poker-card.clubs .card-value'); for (var i = 0; i < cards.length; i++) { if (cards[i].innerText.trim() === '6') { cards[i].parentElement.click(); break; } }"
// Then confirm:
// agent-browser eval "var btns = document.querySelectorAll('button'); for (var i = 0; i < btns.length; i++) { if (btns[i].innerText.trim() === 'Confirm') { btns[i].click(); break; } }"

// -----------------------------------------------------------
// 10. Get action summary with frequencies and combos (postflop)
// -----------------------------------------------------------
// agent-browser eval "var items = document.querySelectorAll('[class*=sab_btn]'); var r = []; for (var i = 0; i < items.length; i++) { var t = items[i].innerText.trim(); if (t && t.indexOf('%') > -1 && t.length < 60) r.push(t); } JSON.stringify(r);"

// -----------------------------------------------------------
// 11. DYNAMIC color map from summary action bar (POSTFLOP)
//     Reads computed background-color of sab_btn_back elements
//     to build a color→action mapping for ANY bet sizes
//     Returns: {colorMap: [{color: "rgb(202,50,50)", action: "Allin 25.4"}, ...]}
//
//     Postflop has multiple shades of red for different bet sizes:
//       Darkest red → largest bet/allin
//       Medium red → medium bet
//       Brightest red → smallest bet
//       Green → Check
//       Blue → Fold (if present)
// -----------------------------------------------------------
// Usage: js_code=$(cat scripts/gto-wizard-extract.js | sed -n '/^\/\/ BEGIN_COLORMAP_SCRIPT/,/^\/\/ END_COLORMAP_SCRIPT/p' | grep -v '^//' | tr '\n' ' ') && agent-browser eval "$js_code"

// BEGIN_COLORMAP_SCRIPT
var sabItems = document.querySelectorAll("[class*=sab_btn]");
var dynamicMap = {};
var actionOrder = [];
for (var i = 0; i < sabItems.length; i++) {
  var text = sabItems[i].innerText.trim();
  var back = sabItems[i].querySelector("[class*=sab_btn_back]");
  if (back && text && text.indexOf("%") > -1) {
    var actionName = text.split("\n")[0];
    var bg = window.getComputedStyle(back).backgroundColor;
    if (!dynamicMap[bg]) {
      dynamicMap[bg] = actionName;
      actionOrder.push({color: bg, action: actionName});
    }
  }
}
JSON.stringify({colorMap: actionOrder});
// END_COLORMAP_SCRIPT

// -----------------------------------------------------------
// 12. DYNAMIC per-hand strategy extraction (UNIVERSAL VERSION)
//     Works for both preflop and postflop!
//     First builds color map from summary bar, then parses grid
//     Uses nearest-color matching (tolerance < 30 RGB distance)
//     Returns: {colorMap: [...], hands: [{hand: "AA", strategy: [...]}]}
// -----------------------------------------------------------
// Usage: js_code=$(cat scripts/gto-wizard-extract.js | sed -n '/^\/\/ BEGIN_DYNAMIC_RANGE/,/^\/\/ END_DYNAMIC_RANGE/p' | grep -v '^//' | tr '\n' ' ') && agent-browser eval "$js_code"

// BEGIN_DYNAMIC_RANGE
var sabItems2 = document.querySelectorAll("[class*=sab_btn]");
var dMap = {};
var aOrder = [];
for (var si = 0; si < sabItems2.length; si++) {
  var st = sabItems2[si].innerText.trim();
  var sb = sabItems2[si].querySelector("[class*=sab_btn_back]");
  if (sb && st && st.indexOf("%") > -1) {
    var an = st.split("\n")[0];
    var abg = window.getComputedStyle(sb).backgroundColor;
    if (!dMap[abg]) { dMap[abg] = an; aOrder.push({color: abg, action: an}); }
  }
}
function mColor(rgb) {
  var m = rgb.match(/(\d+),\s*(\d+),\s*(\d+)/);
  if (!m) return "unknown";
  var r = parseInt(m[1]), g = parseInt(m[2]), b = parseInt(m[3]);
  var best = "unknown"; var bestD = 9999;
  for (var key in dMap) {
    var km = key.match(/(\d+),\s*(\d+),\s*(\d+)/);
    if (!km) continue;
    var d = Math.abs(r-parseInt(km[1])) + Math.abs(g-parseInt(km[2])) + Math.abs(b-parseInt(km[3]));
    if (d < bestD) { bestD = d; best = dMap[key]; }
  }
  return bestD < 30 ? best : "unknown";
}
var dcells = document.querySelectorAll(".rtc.ra_table_cell");
var dresult = [];
for (var di = 0; di < 169 && di < dcells.length; di++) {
  var dc = dcells[di];
  var dhand = dc.innerText.trim();
  var ds = dc.getAttribute("style") || "";
  var dcm = ds.match(/rgb\([\d, ]+\)/g) || [];
  var dsm = ds.match(/background-size:\s*([^;]+)/);
  var dss = dsm ? dsm[1].trim() : "";
  if (dcm.length === 0) continue;
  var ua = [];
  for (var dj = 0; dj < dcm.length; dj += 2) {
    var da = mColor(dcm[dj]);
    if (ua.length === 0 || ua[ua.length-1] !== da) ua.push(da);
  }
  var skip = ua.length === 1 && (ua[0].indexOf("Check") > -1 || ua[0].indexOf("Fold") > -1);
  if (skip) continue;
  var dsp = dss.split(",").map(function(p) { var pm = p.trim().match(/([\d.]+)%/); return pm ? parseFloat(pm[1]) : 100; });
  var dacts = [];
  if (ua.length === 1) { dacts.push({action: ua[0], pct: 100}); }
  else if (ua.length === 2) { var fp = dsp[0] || 50; dacts.push({action: ua[0], pct: Math.round(fp*10)/10}); dacts.push({action: ua[1], pct: Math.round((100-fp)*10)/10}); }
  else if (ua.length === 3) { var p1 = dsp[0]||33; var p2 = (dsp[1]||66)-p1; dacts.push({action: ua[0], pct: Math.round(p1*10)/10}); dacts.push({action: ua[1], pct: Math.round(p2*10)/10}); dacts.push({action: ua[2], pct: Math.round((100-p1-p2)*10)/10}); }
  else { for (var dk = 0; dk < ua.length; dk++) { var prev = dk > 0 ? dsp[dk-1] : 0; var cur = dk === ua.length-1 ? 100-prev : (dsp[dk]||0)-prev; dacts.push({action: ua[dk], pct: Math.round(cur*10)/10}); } }
  dresult.push({hand: dhand, strategy: dacts});
}
JSON.stringify({colorMap: aOrder, hands: dresult});
// END_DYNAMIC_RANGE

// -----------------------------------------------------------
// 13. Click a specific hand cell and extract per-combo detail
//     from the legend panel (after clicking the cell)
//     Replace "KK" with desired hand
//     Returns: ["Allin 25.4\n0", "Bet 12.9\n93.1", ...]
// -----------------------------------------------------------
// Step 1: Click the hand cell
// agent-browser eval "var cells = document.querySelectorAll('.rtc.ra_table_cell'); for (var i = 0; i < cells.length; i++) { if (cells[i].innerText.trim() === 'KK') { cells[i].click(); break; } }"
// Step 2: Extract legend details (wait ~1s after click)
// agent-browser eval "var legends = document.querySelectorAll('[class*=htc_graph_legend_item]'); var r = []; for (var i = 0; i < legends.length; i++) { r.push(legends[i].innerText); } JSON.stringify(r);"

// -----------------------------------------------------------
// 14. Click a seat card action (generic, clicks LAST match)
//     Works for preflop and postflop actions
//     Clicks last matching button (most recent street's action)
//     Replace 'Call' with desired action text
// -----------------------------------------------------------
// agent-browser eval "var btns = document.querySelectorAll('[class*=hspotcrd_action]'); for (var i = btns.length - 1; i >= 0; i--) { if (btns[i].innerText.trim() === 'Call') { btns[i].click(); break; } }"

// -----------------------------------------------------------
// 15. List all buttons on page (debugging helper)
// -----------------------------------------------------------
// agent-browser eval "var btns = document.querySelectorAll('button'); var r = []; for (var i = 0; i < btns.length; i++) { var t = btns[i].innerText.trim(); if (t.length > 0 && t.length < 50) r.push(t); } JSON.stringify(r);"

// -----------------------------------------------------------
// 16. Select turn/river card (NO Confirm needed)
//     Card picker auto-confirms on click
//     Suit classes: .poker-card.spades/.hearts/.diamonds/.clubs
//     Replace suit and rank as needed
// -----------------------------------------------------------
// agent-browser eval "var cards = document.querySelectorAll('.poker-card.clubs .card-value'); for (var i = 0; i < cards.length; i++) { if (cards[i].innerText.trim() === '6') { cards[i].parentElement.click(); break; } }"
