#!/usr/bin/env python3
# FPL — XI-Weighted Optimiser with Proactive Transfers + Robust GW detection
# - Proactively suggests transfers if XI points improve by ≥ 2 after hits
# - One free transfer; -4 per additional transfer
# - Uses fixtures (home/away correct) across next 3 GWs and availability
# - XI weighting so premiums are prioritised
# - Robust current GW detection (cache-busted + deadline window + buffer)
# - Transfer UI shows only the ADOPTED moves (post-threshold), avoiding confusion

import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pulp import (
    LpProblem, LpVariable, LpMaximize,
    lpSum, LpBinary, LpInteger, PULP_CBC_CMD
)

# ── Config ─────────────────────────────────────────────────────────────── #
BUDGET       = 1000   # £100.0m in tenths
SQUAD15_SIZE = 15
START11_SIZE = 11
BENCH_WEIGHT = 0.35          # bench points worth 35% of starter points in objective
NET_GAIN_THRESHOLD = 2.0     # only recommend transfers if net XI gain ≥ 2.0 pts

# Blend weights for base expected points (ideally sum to 1.0)
W_EB      = 0.40  # Empirical Bayes from past performance
W_EP_NEXT = 0.30  # FPL's ep_next
W_STAT90  = 0.30  # Our xG/xA/CS/Saves model per 90

# Availability factor weights
PLAY_PROB_WT       = 0.60  # FPL chance flags
MINUTES_FACTOR_WT  = 0.40  # recent minutes

# Fixture weights (next three GWs)
FX_WT_1, FX_WT_2, FX_WT_3 = 0.6, 0.3, 0.1
NEUTRAL_DIFF = 3  # if fixtures missing, assume neutral difficulty

# FPL points mapping by position (1=GK,2=DEF,3=MID,4=FWD)
GOAL_POINTS = {1:6, 2:6, 3:5, 4:4}
ASSIST_POINTS = 3
CS_POINTS_DEF_GK = 4
SAVES_PER_POINT = 3  # GK: 1pt per 3 saves

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# ── Data & Model Pipeline ──────────────────────────────────────────────── #

def fetch_bootstrap():
    js = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    elements = js["elements"]
    teams    = js["teams"]
    teams_df = pd.DataFrame([{"team_id": t["id"], "club": t["short_name"]} for t in teams])

    players_df = pd.DataFrame([{
        "id":           e["id"],
        "name":         f"{e['first_name']} {e['second_name']}",
        "cost":         e["now_cost"],
        "position_id":  e["element_type"],    # 1 GK, 2 DEF, 3 MID, 4 FWD
        "team_id":      e["team"],
        "status":       e.get("status"),      # 'a','d','i','s','n'
        "chance_this":  e.get("chance_of_playing_this_round"),
        "chance_next":  e.get("chance_of_playing_next_round"),
        "minutes":      e.get("minutes", 0),
        "goals_scored": e.get("goals_scored", 0),
        "assists":      e.get("assists", 0),
        "clean_sheets": e.get("clean_sheets", 0),
        "saves":        e.get("saves", 0),
        "bonus":        e.get("bonus", 0),
        "bps":          e.get("bps", 0),
        "xg":           e.get("expected_goals", 0.0) or 0.0,
        "xa":           e.get("expected_assists", 0.0) or 0.0,
        "xgi":          e.get("expected_goal_involvements", 0.0) or 0.0,
        "xgc":          e.get("expected_goals_conceded", 0.0) or 0.0,
        "ep_next":      e.get("ep_next", 0.0) or 0.0,  # FPL expected points next GW
    } for e in elements])

    players_df = players_df.merge(teams_df, on="team_id", how="left")
    return players_df

def fetch_current_gw():
    """Robust: cache-bust, then use deadline windows with a buffer to pick current GW."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/?_=%d" % int(datetime.now().timestamp())
    js = requests.get(url, headers={"Cache-Control":"no-cache"}).json()
    evs = js["events"]

    # 1) Prefer explicit is_current
    cur = next((e for e in evs if e.get("is_current")), None)
    if cur:
        return cur["id"]

    # 2) Window-based with buffer
    now = datetime.now(timezone.utc)
    buf = timedelta(minutes=10)
    dts = [datetime.fromisoformat(e["deadline_time"].replace("Z","+00:00")) for e in evs]
    for i, e in enumerate(evs):
        start = dts[i] - buf
        end = dts[i+1] + buf if i+1 < len(evs) else None
        if (now >= start) and (end is None or now < end):
            return e["id"]

    # 3) Fallbacks
    nxt = next((e for e in evs if e.get("is_next")), None)
    if nxt:
        return nxt["id"]
    latest_idx = max(range(len(evs)), key=lambda i: dts[i])
    return evs[latest_idx]["id"]

def fetch_fixtures_gw(gw):
    df = pd.DataFrame(requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}").json())
    if df.empty:
        return pd.DataFrame(columns=["team_h","team_h_difficulty","team_a","team_a_difficulty"])
    return df[["team_h","team_h_difficulty","team_a","team_a_difficulty"]]

def build_summary_df(players_df):
    # Aggregate history to get EB baseline + recent minutes (for availability)
    recs = []
    for pid in players_df["id"]:
        resp = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{pid}/").json()
        past = resp.get("history_past", [])
        if past:
            last = past[-1]
            pp, pg = last.get("total_points",0), last.get("appearances",38)
        else:
            pp, pg = 0, 0
        hist = resp.get("history", [])
        cp, cg = sum(h.get("total_points",0) for h in hist), len(hist)

        mins_last5 = 0.0
        if hist:
            last5 = hist[-5:]
            mins_last5 = sum(h.get("minutes",0) for h in last5) / max(1, len(last5))

        recs.append({"id":pid, "n_games":pg+cg, "total_pts":pp+cp, "mins5":mins_last5})
    return pd.DataFrame(recs)

def compute_empirical_bayes(summary_df):
    df = summary_df.copy()
    mask = df["n_games"] > 0
    if mask.sum() > 1:
        rates = df.loc[mask, "total_pts"] / df.loc[mask, "n_games"]
        lam, var = rates.mean(), rates.var(ddof=0)
        phi = (lam*lam)/(var-lam) if var > lam else lam
    else:
        lam, phi = 2.0, 38.0
    a0, b0 = lam*phi, phi
    df["exp_pts_eb"] = (a0 + df["total_pts"]) / (b0 + df["n_games"])
    return df[["id","exp_pts_eb","mins5"]]

def merge_and_scale(players_df, exp_df, fixtures1, fixtures2, fixtures3):
    df = players_df.merge(exp_df, on="id")

    # Ensure numeric types
    num_cols = [
        "minutes", "xg", "xa", "xgi", "xgc", "saves",
        "clean_sheets", "goals_scored", "assists", "bonus", "bps",
        "ep_next", "mins5", "cost", "position_id", "team_id"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fixture difficulties (home/away correct)
    def build_diff(fx, outcol):
        if fx.empty:
            return pd.DataFrame(columns=["team_id", outcol])
        h = fx[["team_h", "team_h_difficulty"]].rename(
            columns={"team_h": "team_id", "team_h_difficulty": outcol}
        )
        a = fx[["team_a", "team_a_difficulty"]].rename(
            columns={"team_a": "team_id", "team_a_difficulty": outcol}
        )
        return pd.concat([h, a], ignore_index=True)

    diff1 = build_diff(fixtures1, "diff1")
    diff2 = build_diff(fixtures2, "diff2")
    diff3 = build_diff(fixtures3, "diff3")

    df = df.merge(diff1, on="team_id", how="left") \
           .merge(diff2, on="team_id", how="left") \
           .merge(diff3, on="team_id", how="left")

    df["diff1"] = df["diff1"].fillna(NEUTRAL_DIFF)
    df["diff2"] = df["diff2"].fillna(NEUTRAL_DIFF)
    df["diff3"] = df["diff3"].fillna(NEUTRAL_DIFF)

    df["fixture_mult"] = (
        FX_WT_1 * (6 - df["diff1"]) / 5.0 +
        FX_WT_2 * (6 - df["diff2"]) / 5.0 +
        FX_WT_3 * (6 - df["diff3"]) / 5.0
    )

    # Availability factor
    raw_chance = df["chance_this"].fillna(df["chance_next"]).fillna(100)
    play_prob = pd.to_numeric(raw_chance, errors="coerce").fillna(100.0) / 100.0
    play_prob = play_prob.mask(df["status"].isin(["i","s","n"]), 0.0)  # out = 0
    minutes_factor = (pd.to_numeric(df["mins5"], errors="coerce") / 90.0).clip(0.0, 1.0).fillna(0.0)
    df["availability_factor"] = PLAY_PROB_WT * play_prob + MINUTES_FACTOR_WT * minutes_factor

    # Per90 stat model
    mins = pd.to_numeric(df["minutes"], errors="coerce").astype("Float64").replace(0, pd.NA)
    xg90     = (pd.to_numeric(df["xg"], errors="coerce")    / mins * 90).astype("Float64").fillna(0.0)
    xa90     = (pd.to_numeric(df["xa"], errors="coerce")    / mins * 90).astype("Float64").fillna(0.0)
    saves90  = (pd.to_numeric(df["saves"], errors="coerce") / mins * 90).astype("Float64").fillna(0.0)
    matches_approx = (mins / 90.0)
    cs_per90 = (pd.to_numeric(df["clean_sheets"], errors="coerce") / matches_approx).astype("Float64").fillna(0.0)

    goal_pts = df["position_id"].map(GOAL_POINTS).fillna(4)
    att_pts90 = xg90 * goal_pts + xa90 * ASSIST_POINTS
    is_def_or_gk = df["position_id"].isin([1,2])
    cs_pts90 = (cs_per90 * CS_POINTS_DEF_GK).where(is_def_or_gk, 0.0)
    is_gk = df["position_id"].eq(1)
    save_pts90 = (saves90 / SAVES_PER_POINT).where(is_gk, 0.0)

    df["xg90"]    = xg90
    df["xa90"]    = xa90
    df["cs90"]    = cs_per90.where(is_def_or_gk, 0.0)
    df["saves90"] = saves90.where(is_gk, 0.0)
    df["stat_pts90"] = att_pts90 + cs_pts90 + save_pts90

    ep_next = pd.to_numeric(df["ep_next"], errors="coerce").fillna(0.0)
    df["base_ep"] = (W_EB * df["exp_pts_eb"] +
                     W_EP_NEXT * ep_next +
                     W_STAT90 * df["stat_pts90"])

    df["position_id"] = pd.to_numeric(df["position_id"], errors="coerce").fillna(0).astype(int)
    df["pos_name"] = df["position_id"].map(POS_MAP).fillna("?")

    df["adj_pts"] = df["base_ep"] * df["fixture_mult"] * df["availability_factor"]
    return df

# ── Optimisation helpers ───────────────────────────────────────────────── #

def select_best_11(sub_df):
    prob = LpProblem("FPL_11", LpMaximize)
    ids = sub_df["id"].tolist()
    pos = sub_df.set_index("id")["position_id"]
    adj = sub_df.set_index("id")["adj_pts"]

    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in ids}
    prob += lpSum(adj[i]*y[i] for i in ids)
    prob += lpSum(y[i] for i in ids) == START11_SIZE

    for pid,(lo,hi) in {1:(1,1),2:(3,5),3:(3,5),4:(1,3)}.items():
        prob += lpSum(y[i] for i in ids if pos[i]==pid) >= lo
        prob += lpSum(y[i] for i in ids if pos[i]==pid) <= hi

    prob.solve(PULP_CBC_CMD(msg=False))
    chosen = [i for i in ids if y[i].value()==1]
    df11 = sub_df[sub_df["id"].isin(chosen)].copy()
    return df11.sort_values(["position_id","adj_pts"], ascending=[True,False])

def xi_points(sub_df):
    xi = select_best_11(sub_df)
    return xi["adj_pts"].sum(), xi

def xi_weighted_15(df, bench_weight=BENCH_WEIGHT):
    prob = LpProblem("FPL_15_XI_weighted", LpMaximize)
    ids = df["id"].tolist()
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in ids}  # in 15
    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in ids}  # in XI

    adj  = df.set_index("id")["adj_pts"]
    cost = df.set_index("id")["cost"]
    pos  = df.set_index("id")["position_id"]
    team = df.set_index("id")["team_id"]

    prob += lpSum(adj[i]*y[i] + bench_weight*adj[i]*(x[i]-y[i]) for i in ids)

    prob += lpSum(x[i] for i in ids) == SQUAD15_SIZE
    prob += lpSum(y[i] for i in ids) == START11_SIZE
    for i in ids: prob += y[i] <= x[i]
    prob += lpSum(cost[i]*x[i] for i in ids) <= BUDGET

    for pid,(lo,hi) in {1:(2,2),2:(5,5),3:(5,5),4:(3,3)}.items():
        prob += lpSum(x[i] for i in ids if pos[i]==pid) >= lo
        prob += lpSum(x[i] for i in ids if pos[i]==pid) <= hi
    for pid,(lo,hi) in {1:(1,1),2:(3,5),3:(3,5),4:(1,3)}.items():
        prob += lpSum(y[i] for i in ids if pos[i]==pid) >= lo
        prob += lpSum(y[i] for i in ids if pos[i]==pid) <= hi

    for tm in df["team_id"].unique():
        prob += lpSum(x[i] for i in ids if team[i]==tm) <= 3

    prob.solve(PULP_CBC_CMD(msg=False))
    chosen15 = [i for i in ids if x[i].value()==1]
    return df[df["id"].isin(chosen15)].copy()

def reoptimize_with_transfers(curr15, df, bench_weight=BENCH_WEIGHT):
    """
    Proactive transfer engine:
    - chooses any transfers (0,1,2,...) that maximise XI-weighted points
    - penalises -4 * (transfers - 1)
    """
    old = curr15[:]
    prob = LpProblem("FPL_15_reopt_with_transfers", LpMaximize)
    ids = df["id"].tolist()
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in ids}  # in 15 after transfers
    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in ids}  # in XI

    # Transfer count relative to old squad
    t = {i: LpVariable(f"t_{i}", cat=LpBinary) for i in old if i in x}
    T_exc = LpVariable("T_exc", lowBound=0, cat=LpInteger)
    for i in t:
        prob += t[i] >= 1 - x[i]
        prob += t[i] <= 1 - x[i]
    prob += T_exc >= lpSum(t[i] for i in t) - 1  # 1 FT, extras count
    prob += T_exc >= 0

    adj  = df.set_index("id")["adj_pts"]
    cost = df.set_index("id")["cost"]
    pos  = df.set_index("id")["position_id"]
    team = df.set_index("id")["team_id"]

    prob += (lpSum(adj[i]*y[i] + bench_weight*adj[i]*(x[i]-y[i]) for i in ids) - 4 * T_exc)

    prob += lpSum(x[i] for i in ids) == SQUAD15_SIZE
    prob += lpSum(y[i] for i in ids) == START11_SIZE
    for i in ids: prob += y[i] <= x[i]
    prob += lpSum(cost[i]*x[i] for i in ids) <= BUDGET

    for pid,(lo,hi) in {1:(2,2),2:(5,5),3:(5,5),4:(3,3)}.items():
        prob += lpSum(x[i] for i in ids if pos[i]==pid) >= lo
        prob += lpSum(x[i] for i in ids if pos[i]==pid) <= hi
    for pid,(lo,hi) in {1:(1,1),2:(3,5),3:(3,5),4:(1,3)}.items():
        prob += lpSum(y[i] for i in ids if pos[i]==pid) >= lo
        prob += lpSum(y[i] for i in ids if pos[i]==pid) <= hi

    for tm in df["team_id"].unique():
        prob += lpSum(x[i] for i in ids if team[i]==tm) <= 3

    prob.solve(PULP_CBC_CMD(msg=False))
    chosen = [i for i in ids if x[i].value()==1]
    penalty = int(T_exc.value() * 4)
    return df[df["id"].isin(chosen)].copy(), penalty

def weekly_points_from_baseep(ids, df, triple_captain=False):
    base_sum = df.set_index("id").loc[ids,"base_ep"].sum()
    cap      = df.set_index("id").loc[ids,"base_ep"].max()
    bonus    = (2.0 if triple_captain else 1.0) * cap
    return base_sum + bonus

def suggest_transfers(old_ids, new_ids, players_df):
    old_set = set(old_ids or [])
    new_set = set(new_ids or [])
    sells = old_set - new_set
    buys  = new_set - old_set
    sell_names = players_df.loc[players_df["id"].isin(sells), "name"].tolist()
    buy_names  = players_df.loc[players_df["id"].isin(buys),  "name"].tolist()
    return sell_names, buy_names

# ── GUI ────────────────────────────────────────────────────────────────── #

class FPLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FPL — XI-Weighted Optimiser with Proactive Transfers")
        self.geometry("1180x960")
        self.build_ui()

    def build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill=tk.X, pady=5)
        ttk.Label(frm, text="Starting 11 IDs (blank for GW1):").grid(row=0, column=0, sticky=tk.W)
        self.start_ent = ttk.Entry(frm, width=60); self.start_ent.grid(row=0, column=1)
        ttk.Label(frm, text="Subs (4 IDs, blank for GW1):").grid(row=1, column=0, sticky=tk.W)
        self.subs_ent = ttk.Entry(frm, width=60); self.subs_ent.grid(row=1, column=1)
        ttk.Label(frm, text="(Optional) Manually mark injured IDs (comma-separated):").grid(row=2, column=0, sticky=tk.W)
        self.inj_ent  = ttk.Entry(frm, width=60); self.inj_ent.grid(row=2, column=1)

        # Manual GW override (optional)
        ttk.Label(frm, text="Override GW (optional):").grid(row=3, column=0, sticky=tk.W)
        self.gw_override = ttk.Entry(frm, width=10); self.gw_override.grid(row=3, column=1, sticky=tk.W)

        self.tc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Triple Captain this GW", variable=self.tc_var)\
            .grid(row=4, column=1, sticky=tk.W, pady=(2,0))

        ttk.Button(frm, text="Run Model", command=self.run)\
            .grid(row=5, column=1, pady=8, sticky=tk.W)

        self.out_var = tk.StringVar()
        ttk.Label(self, textvariable=self.out_var, font=("Arial",12)).pack(pady=4)

        tf = ttk.LabelFrame(self, text="Transfer Suggestions")
        tf.pack(fill=tk.X, padx=10, pady=5)
        self.trans_text = tk.Text(tf, height=8, width=140)
        self.trans_text.pack(fill=tk.X)

        self.make_tree("Previous 15-Squad","tree_old15")
        self.make_tree("New Optimized 15-Squad (ADOPTED)","tree15")
        self.make_tree("Optimized Starting 11","tree11")

    def make_tree(self,title,attr):
        ttk.Label(self, text=title).pack()
        frame = ttk.Frame(self); frame.pack(fill=tk.BOTH,expand=True, padx=10,pady=3)
        cols = ["ID","Name","Pos","Club","Cost","BaseEP","xG90","xA90","CS90","Saves90","Avail","Fix","Adj"]
        widths = [70,220,60,70,70,80,70,70,70,80,70,60,70]
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        for c,w in zip(cols,widths):
            tree.heading(c, text=c); tree.column(c, width=w, anchor=tk.CENTER)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        setattr(self, attr, tree)

    def run(self):
        s_txt = self.start_ent.get().strip()
        b_txt = self.subs_ent.get().strip()
        i_txt = self.inj_ent.get().strip()

        if not s_txt and not b_txt:
            curr_start, curr_subs, injured = [],[],[]
        else:
            try:
                curr_start = [int(x) for x in s_txt.split(",") if x.strip()]
                curr_subs  = [int(x) for x in b_txt.split(",") if x.strip()]
                injured    = [int(x) for x in i_txt.split(",") if x.strip()]
            except ValueError:
                messagebox.showerror("Error","IDs must be integers"); return
            if len(curr_start)!=11 or len(curr_subs)!=4:
                messagebox.showerror("Error","Enter 11 start and 4 subs or leave blank"); return

        curr15 = curr_start + curr_subs

        # Determine GW (override optional)
        gw_txt = (getattr(self, "gw_override", None).get() or "").strip()
        if gw_txt.isdigit():
            gw = int(gw_txt)
        else:
            gw = fetch_current_gw()

        # fetch & prepare
        players = fetch_bootstrap()
        f1 = fetch_fixtures_gw(gw+1)
        f2 = fetch_fixtures_gw(gw+2)
        f3 = fetch_fixtures_gw(gw+3)
        summary = build_summary_df(players)
        exp_df  = compute_empirical_bayes(summary)
        df      = merge_and_scale(players, exp_df, f1, f2, f3)

        # Optional manual injury hard-zero
        if curr15 and i_txt:
            df.loc[df["id"].isin(injured), "availability_factor"] = 0.0
            df["adj_pts"] = df["base_ep"] * df["fixture_mult"] * df["availability_factor"]

        df_pool = df.copy()
        old15_df = df[df["id"].isin(curr15)].copy() if curr15 else pd.DataFrame(columns=df.columns)

        if not curr15:
            # No inputs: just build best 15 (no transfer report)
            adopted15_df = xi_weighted_15(df_pool, bench_weight=BENCH_WEIGHT)
            penalty = 0
            before_pts = 0.0
            after_pts  = xi_points(adopted15_df)[0]
        else:
            # Score current XI
            before_pts, _ = xi_points(old15_df)

            # Candidate via proactive optimiser
            candidate15_df, candidate_penalty = reoptimize_with_transfers(curr15, df_pool, bench_weight=BENCH_WEIGHT)
            if gw == 1:
                candidate_penalty = 0  # GW1 unlimited

            # Score candidate XI
            after_pts, _ = xi_points(candidate15_df)
            net_gain = after_pts - before_pts - candidate_penalty

            # Adopt only if it clears the threshold
            if net_gain >= NET_GAIN_THRESHOLD:
                adopted15_df = candidate15_df
                penalty = candidate_penalty
            else:
                adopted15_df = old15_df.copy()
                penalty = 0
                after_pts = before_pts  # unchanged

        # Build outputs from ADOPTED squad only (to avoid confusion)
        new15_df  = adopted15_df
        new15_ids = new15_df["id"].tolist()
        start11_df = select_best_11(new15_df)
        sells, buys = suggest_transfers(curr15, new15_ids, players)

        # Header diagnostics
        cap_name = start11_df.loc[start11_df["base_ep"].idxmax(),"name"] if not start11_df.empty else "—"
        weekly  = weekly_points_from_baseep(start11_df["id"].tolist(), new15_df, triple_captain=bool(self.tc_var.get()))
        fixtures_info = f"(fixtures weighted: {gw+1}, {gw+2}, {gw+3}; 1 FT, -4/extra)"
        self.out_var.set(
            f"GW{gw} — Captain: {cap_name}    Weekly pts (BaseEP): {weekly:.1f}"
            f"{'  (GW1: no hit cost)' if gw==1 else ''}    |  {fixtures_info}"
        )

        # Transfer text
        self.trans_text.delete("1.0", tk.END)
        if curr15:
            net = after_pts - (xi_points(old15_df)[0] if not old15_df.empty else 0.0) - penalty
            self.trans_text.insert(tk.END,
                f"Sell:    {', '.join(sells) or 'None'}\n"
                f"Buy:     {', '.join(buys)  or 'None'}\n"
                f"Penalty: {penalty} pts{' (GW1 unlimited)' if gw == 1 else ''}\n"
                f"XI gain (adj_pts): before {before_pts:.2f} → after {after_pts:.2f}  Net: {net:+.2f}\n"
                f"(Transfers only applied if net ≥ {NET_GAIN_THRESHOLD:.1f})"
            )

        # Fill tables
        for tree, tbl in ((self.tree_old15, old15_df),
                          (self.tree15,    new15_df),
                          (self.tree11,    start11_df)):
            self.fill_tree(tree, tbl)

        # console export for next week
        main11 = start11_df["id"].tolist()
        subs   = [i for i in new15_ids if i not in main11]
        print("\n=== NEXT WEEK INPUTS ===")
        print("Starting XI IDs:", ",".join(map(str,main11)))
        print("Subs (4 IDs):   ", ",".join(map(str,subs)))
        print("========================")

    def fill_tree(self, tree, df):
        for iid in tree.get_children(): tree.delete(iid)
        if df is None or df.empty: return
        view = df.sort_values(["position_id","base_ep"], ascending=[True,False])
        for _,r in view.iterrows():
            tree.insert("",tk.END,values=(
                r["id"], r["name"], r.get("pos_name","?"),
                r.get("club",""),
                f"£{r['cost']/10:.1f}",
                f"{r['base_ep']:.2f}",
                f"{r['xg90']:.2f}",
                f"{r['xa90']:.2f}",
                f"{r['cs90']:.2f}",
                f"{r['saves90']:.2f}",
                f"{r['availability_factor']:.2f}",
                f"{r['fixture_mult']:.2f}",
                f"{r['adj_pts']:.2f}",
            ))

if __name__=="__main__":
    FPLApp().mainloop()
