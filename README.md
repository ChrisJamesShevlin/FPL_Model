Here’s the updated README with the **entire screenshot section removed** and everything flowing cleanly.

---

# **FPL — XI-Weighted Optimiser with Proactive Transfers**

*A fully automated, fixture-aware, startability-gated FPL optimiser with transfer intelligence and a Tkinter GUI.*

---

## ⭐ Overview

This project is a **complete FPL decision engine** designed to:

* Build the **optimal 15-man squad** using XI-weighted expected points
* Choose the **optimal Starting XI** using a hard startability gate
* Perform **proactive transfers**, only adopting hits where the predicted XI improvement ≥ **2.0 expected points**
* Auto-detect the **current effective Gameweek** based on FPL deadlines
* Correctly model **fixture difficulty** over the next 3 gameweeks (60/30/10 blend)
* Apply an empirical-Bayes smoothing for past points
* Use **per-90 expected stats** (xG, xA, CS, saves) as part of the EP computation
* Enforce all FPL rules:

  * 15-man squad
  * Formation constraints
  * ≤ 3 per club
  * Injury / force-sell logic
  * Startability gating using mins_season > 0 or mins5 ≥ 45 mins
* Provide **captaincy selection**
* Provide **transfer suggestions, repair-mode flags, candidate vs adopted squads**

All results are presented inside a desktop GUI (Tkinter).

---

## ⚙️ Key Features

### 🔍 **1. Auto Gameweek Detection**

Automatically selects the “effective” GW:

* If the current GW deadline has passed → uses **next** GW
* Otherwise → uses **current** GW
  You can optionally override this manually.

---

### 🧠 **2. XI-Weighted Objective Function**

When optimising a full 15-man squad:

* **Starters** contribute 100% of adj_pts
* **Bench players** contribute `25%` (configurable)

This avoids bench-inflated optimisation.

---

### 🚑 **3. Proactive Transfers**

* 1 free transfer
* Extra transfers cost: **-4 FP each**
* Transfers only adopted when:

```
Net gain ≥ 2.0 expected XI points (after hits)
```

Exception: repair-mode (injuries or >3-per-club violations) **forces transfers**.

---

### 🔒 **4. HARD Startability Gate**

A player can start only if:

```
(minutes_season > 0) OR (mins_last_5_games ≥ 45)
AND availability_factor ≥ 0.45
```

This prevents non-playing cheap players from sneaking into the XI.

---

### 📊 **5. Weighted Fixture Difficulty**

Weighted over the next three GWs:

| GW  | Weight |
| --- | ------ |
| t+1 | 0.6    |
| t+2 | 0.3    |
| t+3 | 0.1    |

Correctly handles home vs away difficulty.

---

### 📈 **6. EP Model Components**

`base_ep` is built from:

| Component            | Description               | Weight |
| -------------------- | ------------------------- | ------ |
| Empirical Bayes (EB) | smoothed historic points  | 0.40   |
| FPL ep_next          | official EP               | 0.30   |
| Per-90 Stats Model   | xG90, xA90, CS90, Saves90 | 0.30   |

Adjusted by:

* Fixture multiplier
* Availability factor
* Hard startability rules

---

### 🧮 **7. Linear Programming Engine**

Uses PuLP + CBC integer solver for both:

* Best 15
* Best XI

With full FPL rule constraints:

* Formation
* Team limits
* Budget
* Startability
* Forced-outs
* Transfer penalties

---

### 🧑‍💻 **8. Tkinter GUI**

Provides:

* Squad inputs
* Injury / force-out input
* Gameweek override
* Triple Captain toggle
* Three tables:

  * Old 15
  * New 15 (adopted)
  * Starting XI
* Transfer rationales

---

## 📦 Requirements

```
Python 3.9+
requests
pandas
pulp
tkinter  (usually built-in)
```

On Linux:

```bash
sudo apt install python3-tk
pip install pandas pulp requests
```

---

## ▶️ Running

```
python3 fpl_optimizer.py
```

On first run (no squad entered), it builds a fully optimised 15.

---

## 📤 Next-Week Export

After running, the console prints:

```
=== NEXT WEEK INPUTS ===
Starting XI IDs: ...
Subs (4 IDs):   ...
========================
```

Use these to feed the model next week.

---

## 🧠 Transfer Logic Summary

### Candidate Squad

The optimiser finds the mathematically best squad, even if it requires hits.

### Adoption

If not in repair mode:

```
(adjusted XI pts after) - (before) - (hits) >= 2.0
```

If false → **keeps current squad**.

### Repair Mode

Automatically activated if:

* Injured / forced-out players
* > 3 players from same club
* Player fails hard startability check

In repair mode:

```
Transfers always adopted.
```

---

## 🏆 Captain Selection

From eligible XI players:

* Must pass startability
* Must have AF ≥ 0.80
* Highest adj_pts preferred

Triple Captain applies 3× base_ep for the captain.

---

## 📚 Files

```
fpl_optimizer.py
README.md
```

---

If you want this in **a more compact**, **more visual**, or **more formal** GitHub style — or want badges added — just say.
