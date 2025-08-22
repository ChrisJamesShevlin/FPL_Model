#!/usr/bin/env python3
# FPL — XI-Weighted Optimiser with Proactive Transfers (mins_season gating, hard startability)
# - Auto GW detection (uses NEXT GW after current deadline passes)
# - Optional manual GW override (leave blank for auto)
# - XI-weighted objective (bench discounted)
# - Proactive transfers: 1 free, -4 per extra; adopt only if net XI gain ≥ 2.0
# - Forced-out logic: Injured box acts as force-sell; auto-fix >3-per-club
# - Correct home/away fixture difficulty over next 3 GWs (60/30/10)
# - Captain chosen by adj_pts; captain bonus applied to base_ep
# - Prevents 0-minute players from starting (HARD GATE: mins_season > 0 or mins5 ≥ 45)

import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
from datetime import datetime, timezone
from collections import Counter
from pulp import (
    LpProblem, LpVariable, LpMaximize,
    lpSum, LpBinary, LpInteger, PULP_CBC_CMD
)

# ── Config ─────────────────────────────────────────────────────────────── #
BUDGET       = 1000   # £100.0m in tenths
SQUAD15_SIZE = 15
START11_SIZE = 11
BENCH_WEIGHT = 0.25
NET_GAIN_THRESHOLD = 2.0

# Blend weights for base expected points (ideally sum to 1.0)
W_EB      = 0.40
W_EP_NEXT = 0.30
W_STAT90  = 0.30

# Availability factor weights
PLAY_PROB_WT       = 0.60
MINUTES_FACTOR_WT  = 0.40

# Fixture weights (next three GWs)
FX_WT_1, FX_WT_2, FX_WT_3 = 0.6, 0.3, 0.1
NEUTRAL_DIFF = 3

# FPL points mapping by position (1=GK,2=DEF,3=MID,4=FWD)
GOAL_POINTS = {1:6, 2:6, 3:5, 4:4}
ASSIST_POINTS = 3
CS_POINTS_DEF_GK = 4
SAVES_PER_POINT = 3

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# ── Effective GW detection ─────────────────────────────────────────────── #
def fetch_effective_gw():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    js = requests.get(url, headers={"Cache-Control": "no-cache"}).json()
    events = js["events"]

    cur = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)
    now = datetime.now(timezone.utc)

    if cur:
        deadline = datetime.fromisoformat(cur["deadline_time"].replace("Z", "+00:00"))
        if now >= deadline and nxt:
            return nxt["id"]
        return cur["id"]

    if nxt:
        return nxt["id"]
    return events[-1]["id"]

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
        "position_id":  e["element_type"],
        "team_id":      e["team"],
        "status":       e.get("status"),
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
        "ep_next":      e.get("ep_next", 0.0) or 0.0,
    } for e in elements])

    players_df = players_df.merge(teams_df, on="team_id", how="left")
    return players_df

def fetch_fixtures_gw(gw):
    df = pd.DataFrame(requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}").json())
    if df.empty:
        return pd.DataFrame(columns=["team_h","team_h_difficulty","team_a","team_a_difficulty"])
    return df[["team_h","team_h_difficulty","team_a","team_a_difficulty"]]

def build_summary_df(players_df):
    recs = []
    for pid in players_df["id"]:
        resp = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{pid}/").json()

        past = resp.get("history_past", [])
        if past:
            last = past[-1]
            pp, pg = last.get("total_points", 0), last.get("appearances", 38)
        else:
            pp, pg = 0, 0

        hist = resp.get("history", [])
        cp   = sum(h.get("total_points", 0) for h in hist)
        cg   = len(hist)

        mins_last5   = 0.0
        starts_last5 = 0
        mins_season  = 0
        if hist:
            last5 = hist[-5:]
            mins_last5   = sum(h.get("minutes", 0) for h in last5) / max(1, len(last5))
            starts_last5 = sum(1 for h in last5 if h.get("minutes", 0) >= 60)
            mins_season  = sum(h.get("minutes", 0) for h in hist)

        recs.append({
            "id": pid,
            "n_games": pg + cg,
            "total_pts": pp + cp,
            "mins5": mins_last5,
            "starts5": starts_last5,
            "mins_season": mins_season,
        })
    return pd.DataFrame(recs)

def compute_empirical_bayes(summary_df):
    df = summary_df.copy()
    mask = df["n_games"] > 0
    if mask.sum() > 1:
        rates = df.loc[mask, "total_pts"] / df.loc[mask, "n_games"]
        lam, var = rates.mean(), rates.var(ddof=0)
        phi = (lam * lam) / (var - lam) if var > lam else lam
    else:
        lam, phi = 2.0, 38.0
    a0, b0 = lam * phi, phi
    df["exp_pts_eb"] = (a0 + df["total_pts"]) / (b0 + df["n_games"])
    return df[["id", "exp_pts_eb", "mins5", "starts5", "mins_season"]]

def merge_and_scale(players_df, exp_df, fixtures1, fixtures2, fixtures3):
    df = players_df.merge(exp_df, on="id")

    # Ensure numeric types
    num_cols = [
        "minutes", "xg", "xa", "xgi", "xgc", "saves",
        "clean_sheets", "goals_scored", "assists", "bonus", "bps",
        "ep_next", "mins5", "starts5", "mins_season", "cost", "position_id", "team_id"
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
    play_prob = play_prob.mask(df["status"].isin(["i","s","n"]), 0.0)
    minutes_factor = (pd.to_numeric(df["mins5"], errors="coerce") / 90.0).clip(0.0, 1.0).fillna(0.0)
    df["availability_factor"] = PLAY_PROB_WT * play_prob + MINUTES_FACTOR_WT * minutes_factor

    # Extra downweight using true this-season minutes
    df["mins_season"] = pd.to_numeric(df.get("mins_season", 0), errors="coerce").fillna(0)
    df.loc[df["mins_season"] == 0, "availability_factor"] *= 0.10

    # Per90 stat model (role-aware)
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

# ── Forced-out helpers ─────────────────────────────────────────────────── #
def choose_forced_out_ids(curr15, df, user_forced_ids=None):
    user_forced_ids = set(user_forced_ids or [])
    forced = set(user_forced_ids)
    if not curr15:
        return forced

    sub = df[df["id"].isin(curr15)].copy()
    counts = Counter(sub["team_id"])
    excess = {tm: c - 3 for tm, c in counts.items() if c > 3}
    if not excess:
        return forced

    for tm, k in excess.items():
        pool = sub[sub["team_id"] == tm].copy()
        pool["prio"] = pool["id"].isin(user_forced_ids).map(lambda b: 0 if b else 1)
        pool = pool.sort_values(["prio", "adj_pts"])  # lowest adj_pts first
        forced.update(pool["id"].head(k).tolist())
    return forced

# ── Shared startability gate (HARD) ─────────────────────────────────────── #
def startable_dict(sub_df, af_min=0.45):
    """
    HARD gate for starting:
      - must have mins_season > 0 OR mins5 ≥ 45
      - AND availability_factor ≥ af_min
    """
    af    = sub_df.set_index("id")["availability_factor"]
    mins5 = sub_df.set_index("id")["mins5"] if "mins5" in sub_df.columns else pd.Series(0, index=af.index)
    mseas = sub_df.set_index("id")["mins_season"] if "mins_season" in sub_df.columns else pd.Series(0, index=af.index)
    ok = (((mseas > 0) | (mins5 >= 45)) & (af >= af_min)).astype(int)
    return ok.to_dict()

# ── Optimisation helpers ───────────────────────────────────────────────── #
def select_best_11(sub_df):
    prob = LpProblem("FPL_11", LpMaximize)
    ids = sub_df["id"].tolist()

    pos   = sub_df.set_index("id")["position_id"]
    adj   = sub_df.set_index("id")["adj_pts"]

    # HARD startability
    startable = startable_dict(sub_df, af_min=0.45)

    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in ids}
    prob += lpSum(adj[i] * y[i] for i in ids)
    prob += lpSum(y[i] for i in ids) == START11_SIZE

    # cannot start if not startable
    for i in ids:
        prob += y[i] <= startable.get(i, 0)

    # formation bounds
    for pid, (lo, hi) in {1:(1,1), 2:(3,5), 3:(3,5), 4:(1,3)}.items():
        prob += lpSum(y[i] for i in ids if pos[i] == pid) >= lo
        prob += lpSum(y[i] for i in ids if pos[i] == pid) <= hi

    prob.solve(PULP_CBC_CMD(msg=False))
    chosen = [i for i in ids if y[i].value() == 1]
    df11 = sub_df[sub_df["id"].isin(chosen)].copy()
    return df11.sort_values(["position_id", "adj_pts"], ascending=[True, False])

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

    # HARD startability for XI inside 15-optimiser as well
    startable = startable_dict(df, af_min=0.45)

    prob += lpSum(adj[i]*y[i] + bench_weight*adj[i]*(x[i]-y[i]) for i in ids)

    prob += lpSum(x[i] for i in ids) == SQUAD15_SIZE
    prob += lpSum(y[i] for i in ids) == START11_SIZE
    for i in ids:
        prob += y[i] <= x[i]
        prob += y[i] <= startable.get(i, 0)
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

def reoptimize_with_transfers(curr15, df, bench_weight=BENCH_WEIGHT, forced_out_ids=None):
    forced_out_ids = set(forced_out_ids or [])
    old = curr15[:]
    prob = LpProblem("FPL_15_reopt_with_transfers", LpMaximize)
    ids = df["id"].tolist()

    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in ids}  # in 15 after transfers
    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in ids}  # in XI

    # HARD startability
    startable = startable_dict(df, af_min=0.45)

    # force-sell constraint
    for i in forced_out_ids:
        if i in x:
            prob += x[i] == 0

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
    for i in ids:
        prob += y[i] <= x[i]
        prob += y[i] <= startable.get(i, 0)
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

def weekly_points_from_baseep(ids, df, cap_id=None, triple_captain=False):
    base = df.set_index("id")
    base_sum = base.loc[ids, "base_ep"].sum()
    if cap_id is None or cap_id not in base.index:
        cap = base.loc[ids, "base_ep"].max()
    else:
        cap = base.loc[cap_id, "base_ep"]
    bonus = (2.0 if triple_captain else 1.0) * cap
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
        self.geometry("1180x1020")
        self.build_ui()

    def build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill=tk.X, pady=5)
        ttk.Label(frm, text="Starting 11 IDs (blank for GW1):").grid(row=0, column=0, sticky=tk.W)
        self.start_ent = ttk.Entry(frm, width=60); self.start_ent.grid(row=0, column=1)
        ttk.Label(frm, text="Subs (4 IDs, blank for GW1):").grid(row=1, column=0, sticky=tk.W)
        self.subs_ent = ttk.Entry(frm, width=60); self.subs_ent.grid(row=1, column=1)
        ttk.Label(frm, text="Manual Transfer/Bench").grid(row=2, column=0, sticky=tk.W)
        self.inj_ent  = ttk.Entry(frm, width=60); self.inj_ent.grid(row=2, column=1)

        ttk.Label(frm, text="Override GW (optional):").grid(row=3, column=0, sticky=tk.W)
        self.gw_override = ttk.Entry(frm, width=10); self.gw_override.grid(row=3, column=1, sticky=tk.W)

        self.tc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Triple Captain this GW", variable=self.tc_var)\
            .grid(row=4, column=1, sticky=tk.W, pady=(2,0))

        ttk.Button(frm, text="Run Model", command=self.run)\
            .grid(row=5, column=1, pady=8, sticky=tk.W)

        self.out_var = tk.StringVar()
        ttk.Label(self, textvariable=self.out_var, font=("Arial",12)).pack(pady=4)

        tf = ttk.LabelFrame(self, text="Transfer Suggestions (Adopted & Candidate)")
        tf.pack(fill=tk.X, padx=10, pady=5)
        self.trans_text = tk.Text(tf, height=12, width=140)
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
            curr_start, curr_subs, injured_ids = [],[],[]
        else:
            try:
                curr_start = [int(x) for x in s_txt.split(",") if x.strip()]
                curr_subs  = [int(x) for x in b_txt.split(",") if x.strip()]
                injured_ids= [int(x) for x in i_txt.split(",") if x.strip()]
            except ValueError:
                messagebox.showerror("Error","IDs must be integers"); return
            if len(curr_start)!=11 or len(curr_subs)!=4:
                messagebox.showerror("Error","Enter 11 start and 4 subs or leave blank"); return

        curr15 = curr_start + curr_subs

        # Determine GW (manual override takes precedence)
        gw_txt = (getattr(self, "gw_override", None).get() or "").strip()
        if gw_txt.isdigit():
            gw = int(gw_txt); gw_src = "manual override"
        else:
            gw = fetch_effective_gw(); gw_src = "auto-detected"

        # fetch & prepare
        players = fetch_bootstrap()
        f1 = fetch_fixtures_gw(gw+1)
        f2 = fetch_fixtures_gw(gw+2)
        f3 = fetch_fixtures_gw(gw+3)
        summary = build_summary_df(players)
        exp_df  = compute_empirical_bayes(summary)
        df      = merge_and_scale(players, exp_df, f1, f2, f3)

        # Injured IDs: AF=0 for retained; also force-sell list
        if curr15 and injured_ids:
            df.loc[df["id"].isin(injured_ids), "availability_factor"] = 0.0
            df["adj_pts"] = df["base_ep"] * df["fixture_mult"] * df["availability_factor"]

        df_pool = df.copy()
        old15_df = df[df["id"].isin(curr15)].copy() if curr15 else pd.DataFrame(columns=df.columns)

        # Build forced outs: (a) injured list, (b) auto-repair >3-per-club
        forced_out = choose_forced_out_ids(curr15, df_pool, user_forced_ids=injured_ids)

        # Decide adopted squad
        if not curr15:
            adopted15_df = xi_weighted_15(df_pool, bench_weight=BENCH_WEIGHT)
            penalty = 0
            before_pts = 0.0
            after_pts  = xi_points(adopted15_df)[0]
            candidate15_df, candidate_penalty = adopted15_df, 0
            must_repair = False
        else:
            before_pts, _ = xi_points(old15_df)
            candidate15_df, candidate_penalty = reoptimize_with_transfers(
                curr15, df_pool, bench_weight=BENCH_WEIGHT, forced_out_ids=forced_out
            )
            after_pts, _ = xi_points(candidate15_df)
            net_gain = after_pts - before_pts - candidate_penalty
            must_repair = len(forced_out) > 0

            if must_repair:
                adopted15_df = candidate15_df
                penalty = candidate_penalty
            else:
                if net_gain >= NET_GAIN_THRESHOLD:
                    adopted15_df = candidate15_df
                    penalty = candidate_penalty
                else:
                    adopted15_df = old15_df.copy()
                    penalty = 0
                    after_pts = before_pts

        # Build outputs (ADOPTED squad)
        new15_df  = adopted15_df
        new15_ids = new15_df["id"].tolist()
        start11_df = select_best_11(new15_df)

        # Captain: choose by adj_pts (eligible only), bonus to base_ep
        if not start11_df.empty:
            elig = start11_df[
                (start11_df["availability_factor"] >= 0.80) &
                ((start11_df["mins_season"] > 0) | (start11_df["mins5"] >= 45))
            ]
            pool = elig if not elig.empty else start11_df
            cap_row = pool.sort_values("adj_pts", ascending=False).iloc[0]
            cap_id   = int(cap_row["id"])
            cap_name = cap_row["name"]
        else:
            cap_id, cap_name = None, "—"

        weekly = weekly_points_from_baseep(
            start11_df["id"].tolist(), new15_df, cap_id=cap_id, triple_captain=bool(self.tc_var.get())
        )

        # Transfers (adopted & candidate for diagnostics)
        sells, buys = suggest_transfers(curr15, new15_ids, players)
        cand_sells, cand_buys = suggest_transfers(curr15, candidate15_df["id"].tolist() if curr15 else [], players)

        # Header diagnostics
        fixtures_info = f"(GW source: {gw_src}; fixtures weighted: {gw+1}, {gw+2}, {gw+3}; 1 FT, -4/extra)"
        self.out_var.set(
            f"GW{gw} — Captain: {cap_name}    Weekly pts (BaseEP): {weekly:.1f}    |  {fixtures_info}"
        )

        # Transfer text
        self.trans_text.delete("1.0", tk.END)
        if curr15:
            before_pts_now = xi_points(old15_df)[0] if not old15_df.empty else 0.0
            after_pts_now  = xi_points(new15_df)[0]   if not new15_df.empty  else 0.0
            net_now        = after_pts_now - before_pts_now - (penalty or 0)

            cand_after     = xi_points(candidate15_df)[0] if curr15 else 0.0
            cand_net       = cand_after - before_pts - (candidate_penalty or 0)

            if must_repair:
                self.trans_text.insert(tk.END, "⚠️ Repair mode: forced to sell players (injured/4-from-one-club) to meet rules.\n")

            self.trans_text.insert(tk.END,
                "ADOPTED (after hits & rules):\n"
                f"  Sell: {', '.join(sells) or 'None'}\n"
                f"  Buy:  {', '.join(buys)  or 'None'}\n"
                f"  Penalty applied: {penalty} pts\n"
                f"  XI adj_pts: before {before_pts_now:.2f} → after {after_pts_now:.2f}  Net: {net_now:+.2f}\n"
                f"  Threshold (if not repairing): {NET_GAIN_THRESHOLD:.1f}\n"
                "\nCANDIDATE (best before adoption check):\n"
                f"  Sell: {', '.join(cand_sells) or 'None'}\n"
                f"  Buy:  {', '.join(cand_buys)  or 'None'}\n"
                f"  Candidate net (after hits): {cand_net:+.2f}\n"
            )
        else:
            self.trans_text.insert(tk.END, "No current squad entered — built best XI-weighted 15 from scratch.\n")

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
