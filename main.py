import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
from pulp import (
    LpProblem, LpVariable, LpMaximize,
    lpSum, LpBinary, LpInteger, PULP_CBC_CMD
)

# --- Config ---
BUDGET       = 1000   # £100.0m in tenths
SQUAD15_SIZE = 15
START11_SIZE = 11

# --- Data & Model Pipeline ---

def fetch_bootstrap():
    data = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/"
    ).json()["elements"]
    return pd.DataFrame([{
        "id":          e["id"],
        "name":        f"{e['first_name']} {e['second_name']}",
        "cost":        e["now_cost"],
        "position_id": e["element_type"],
        "team_id":     e["team"],
    } for e in data])

def fetch_current_gw():
    evs = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/"
    ).json()["events"]
    for ev in evs:
        if ev.get("is_current"): return ev["id"]
    for ev in evs:
        if ev.get("is_next"):    return ev["id"]
    return 1

def fetch_fixtures_gw(gw):
    df = pd.DataFrame(requests.get(
        f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
    ).json())
    return df[["team_h","team_h_difficulty","team_a","team_a_difficulty"]]

def build_summary_df(players_df):
    recs = []
    for pid in players_df["id"]:
        resp = requests.get(
            f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
        ).json()
        past = resp.get("history_past", [])
        if past:
            last = past[-1]
            pp, pg = last.get("total_points",0), last.get("appearances",38)
        else:
            pp, pg = 0, 0
        hist = resp.get("history", [])
        cp, cg = sum(h.get("total_points",0) for h in hist), len(hist)
        recs.append({"id":pid, "n_games":pg+cg, "total_pts":pp+cp})
    return pd.DataFrame(recs)

def compute_empirical_bayes(summary_df):
    df = summary_df.copy()
    mask = df["n_games"]>0
    if mask.sum()>1:
        rates = df.loc[mask,"total_pts"]/df.loc[mask,"n_games"]
        lam, var = rates.mean(), rates.var(ddof=0)
        phi = (lam*lam)/(var-lam) if var>lam else lam
    else:
        lam, phi = 2.0, 38.0
    a0, b0 = lam*phi, phi
    df["exp_pts_eb"] = (a0+df["total_pts"])/(b0+df["n_games"])
    return df[["id","exp_pts_eb"]]

def merge_and_scale(players_df, exp_df, fixtures1, fixtures2, fixtures3):
    df = players_df.merge(exp_df, on="id")
    def build_diff(fx, colname, outcol):
        h = fx[["team_h",colname]].rename(
            columns={"team_h":"team_id", colname:outcol})
        a = fx[["team_a",colname]].rename(
            columns={"team_a":"team_id", colname:outcol})
        return pd.concat([h,a], ignore_index=True)
    diff1 = build_diff(fixtures1, "team_h_difficulty", "diff1")
    diff2 = build_diff(fixtures2, "team_h_difficulty", "diff2")
    diff3 = build_diff(fixtures3, "team_h_difficulty", "diff3")
    df = df.merge(diff1,on="team_id",how="left")\
           .merge(diff2,on="team_id",how="left")\
           .merge(diff3,on="team_id",how="left")
    df["multiplier"] = (
        0.6*(6-df["diff1"])/5.0 +
        0.3*(6-df["diff2"])/5.0 +
        0.1*(6-df["diff3"])/5.0
    )
    df["adj_pts"] = df["exp_pts_eb"] * df["multiplier"]
    return df

# --- Optimization Routines ---

def select_best_15(df):
    prob = LpProblem("FPL_15", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in df["id"]}
    prob += lpSum(df.loc[df["id"]==i,"adj_pts"].item()*x[i] for i in x)
    prob += lpSum(df.loc[df["id"]==i,"cost"].item()*x[i] for i in x) <= BUDGET
    prob += lpSum(x.values()) == SQUAD15_SIZE
    for pid,(lo,hi) in {1:(2,2),2:(5,5),3:(5,5),4:(3,3)}.items():
        prob += lpSum(x[i] for i in x
                      if df.loc[df["id"]==i,"position_id"].iloc[0]==pid) >= lo
        prob += lpSum(x[i] for i in x
                      if df.loc[df["id"]==i,"position_id"].iloc[0]==pid) <= hi
    for tm in df["team_id"].unique():
        prob += lpSum(x[i] for i in x
                      if df.loc[df["id"]==i,"team_id"].iloc[0]==tm) <= 3
    prob.solve(PULP_CBC_CMD(msg=False))
    chosen = [i for i in x if x[i].value()==1]
    return df[df["id"].isin(chosen)].copy()

def select_best_11(sub_df):
    prob = LpProblem("FPL_11", LpMaximize)
    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in sub_df["id"]}
    prob += lpSum(sub_df.loc[sub_df["id"]==i,"adj_pts"].item()*y[i] for i in y)
    prob += lpSum(y.values()) == START11_SIZE
    for pid,(lo,hi) in {1:(1,1),2:(3,5),3:(3,5),4:(1,3)}.items():
        prob += lpSum(y[i] for i in y
                      if sub_df.loc[sub_df["id"]==i,"position_id"].iloc[0]==pid) >= lo
        prob += lpSum(y[i] for i in y
                      if sub_df.loc[sub_df["id"]==i,"position_id"].iloc[0]==pid) <= hi
    prob.solve(PULP_CBC_CMD(msg=False))
    chosen = [i for i in y if y[i].value()==1]
    df11 = sub_df[sub_df["id"].isin(chosen)].copy()
    return df11.sort_values(["position_id","adj_pts"], ascending=[True,False])

def reoptimize_squad(curr15, injured, df):
    old = curr15[:]  # full list
    prob = LpProblem("FPL_15_reopt", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in df["id"]}
    # lock healthy
    for i in set(curr15) - set(injured):
        prob += x[i] == 1
    # transfer flags for all originally in squad
    t = {i: LpVariable(f"t_{i}", cat=LpBinary) for i in old}
    # excess-transfers variable
    T_exc = LpVariable("T_exc", lowBound=0, cat=LpInteger)
    # t_i = 1 - x_i  for each old player
    for i in old:
        prob += t[i] >= 1 - x[i]
        prob += t[i] <= 1 - x[i]
    # penalize beyond 1 free move
    prob += T_exc >= lpSum(t[i] for i in old) - 1
    prob += T_exc >= 0

    # objective: adj pts minus 4 * excess transfers
    prob += (
        lpSum(df.loc[df["id"]==i,"adj_pts"].item()*x[i] for i in x)
        - 4 * T_exc
    )

    # squad constraints
    prob += lpSum(x.values()) == SQUAD15_SIZE
    prob += lpSum(df.loc[df["id"]==i,"cost"].item()*x[i] for i in x) <= BUDGET
    for pid,(lo,hi) in {1:(2,2),2:(5,5),3:(5,5),4:(3,3)}.items():
        prob += lpSum(x[i] for i in x
                      if df.loc[df["id"]==i,"position_id"].iloc[0]==pid) >= lo
        prob += lpSum(x[i] for i in x
                      if df.loc[df["id"]==i,"position_id"].iloc[0]==pid) <= hi
    for tm in df["team_id"].unique():
        prob += lpSum(x[i] for i in x
                      if df.loc[df["id"]==i,"team_id"].iloc[0]==tm) <= 3

    prob.solve(PULP_CBC_CMD(msg=False))
    chosen = [i for i in x if x[i].value()==1]
    penalty = int(T_exc.value() * 4)
    return df[df["id"].isin(chosen)].copy(), penalty

def team_score(ids, df):
    base = df.set_index("id").loc[ids,"adj_pts"].sum()
    cap  = df.set_index("id").loc[ids,"adj_pts"].max()
    return base + cap

def suggest_transfers(curr15, new15, injured, players_df):
    sells = sorted(set(curr15) - set(new15) | set(injured))
    buys  = sorted(set(new15) - set(curr15))
    sell_names = [players_df.loc[players_df["id"]==i,"name"].iloc[0] for i in sells]
    buy_names  = [players_df.loc[players_df["id"]==i,"name"].iloc[0] for i in buys]
    return sell_names, buy_names

# --- GUI ---

class FPLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FPL 15+Subs Selector")
        self.geometry("920x780")
        self.build_ui()

    def build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill=tk.X, pady=5)
        ttk.Label(frm, text="Starting 11 IDs (blank for GW1):")\
            .grid(row=0, column=0, sticky=tk.W)
        self.start_ent = ttk.Entry(frm, width=55); self.start_ent.grid(row=0, column=1)
        ttk.Label(frm, text="Subs (4 IDs, blank for GW1):")\
            .grid(row=1, column=0, sticky=tk.W)
        self.subs_ent = ttk.Entry(frm, width=55); self.subs_ent.grid(row=1, column=1)
        ttk.Label(frm, text="Injured IDs:")\
            .grid(row=2, column=0, sticky=tk.W)
        self.inj_ent  = ttk.Entry(frm, width=55); self.inj_ent.grid(row=2, column=1)
        ttk.Button(frm, text="Run Model", command=self.run)\
            .grid(row=3, column=1, pady=8)

        self.out_var = tk.StringVar()
        ttk.Label(self, textvariable=self.out_var,
                  font=("Arial",12)).pack(pady=4)

        tf = ttk.LabelFrame(self, text="Transfers & Moves")
        tf.pack(fill=tk.X, padx=10, pady=5)
        self.trans_text = tk.Text(tf, height=6, width=100)
        self.trans_text.pack(fill=tk.X)

        self.make_tree("Previous 15-Squad","tree_old15")
        self.make_tree("New Optimized 15-Squad","tree15")
        self.make_tree("Optimized Starting 11","tree11")

    def make_tree(self,title,attr):
        ttk.Label(self, text=title).pack()
        frame = ttk.Frame(self); frame.pack(fill=tk.BOTH,expand=True, padx=10,pady=3)
        cols = ["ID","Name","Pos","Team","Cost","ExpPts"]
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=5)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        for c in cols:
            tree.heading(c, text=c); tree.column(c, width=100, anchor=tk.CENTER)
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

        # fetch & prepare
        gw      = fetch_current_gw()
        players = fetch_bootstrap()
        f1 = fetch_fixtures_gw(gw+1)
        f2 = fetch_fixtures_gw(gw+2)
        f3 = fetch_fixtures_gw(gw+3)
        summary = build_summary_df(players)
        exp_df  = compute_empirical_bayes(summary)
        df      = merge_and_scale(players, exp_df, f1, f2, f3)
        df_pool = df[~df["id"].isin(injured)]

        old15 = df[df["id"].isin(curr15)].copy() \
               if curr15 else pd.DataFrame(columns=df.columns)

        if curr15:
            new15_df, penalty = reoptimize_squad(curr15, injured, df_pool)
        else:
            new15_df = select_best_15(df_pool)
            penalty = 0

        new15_ids   = new15_df["id"].tolist()
        start11_df  = select_best_11(new15_df)

        sells, buys = suggest_transfers(curr15, new15_ids, injured, players)

        cap_row = start11_df.loc[start11_df["adj_pts"].idxmax()]
        cap_name= cap_row["name"]
        gross   = team_score(start11_df["id"].tolist(), new15_df)
        if curr15:
            net = gross - penalty
            self.out_var.set(f"Captain: {cap_name}    Gross pts: {gross:.1f}    Net pts: {net:.1f}")
        else:
            self.out_var.set(f"Captain: {cap_name}    Expected pts: {gross:.1f}")

        # show transfers
        self.trans_text.delete("1.0",tk.END)
        if sells or buys:
            self.trans_text.insert(tk.END,
                f"Sell:    {', '.join(sells) or 'None'}\n"
                f"Buy:     {', '.join(buys)  or 'None'}\n"
                f"Penalty: {penalty} pts"
            )

        # populate tables
        for tree, tbl in ((self.tree_old15, old15),
                          (self.tree15,    new15_df),
                          (self.tree11,    start11_df)):
            for iid in tree.get_children(): tree.delete(iid)
            for _,r in tbl.iterrows():
                tree.insert("",tk.END,values=(
                    r["id"], r["name"], r["position_id"],
                    r["team_id"], f"£{r['cost']/10:.1f}",
                    f"{r['adj_pts']:.2f}"
                ))

        # console export
        main11 = start11_df["id"].tolist()
        subs    = [i for i in new15_ids if i not in main11]
        print("\n=== NEXT WEEK INPUTS ===")
        print("Starting XI IDs:", ",".join(map(str,main11)))
        print("Subs (4 IDs):   ", ",".join(map(str,subs)))
        print("========================")

if __name__=="__main__":
    FPLApp().mainloop()
