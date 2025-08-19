#!/usr/bin/env python3
"""
Stockfish FEN analyser + Rating-Aware Style Coach

Build notes (this version):
- FIX: race on self.target_elo -> set early, safe getter, and worker try/except.
- Rating-targeted search depth (shallower at low Elo to feel less enginey).
- Softer 1000-Elo profile (more human-like choices).
- Looser adherence caps so styles diverge (but still safe).
- Serialized engine access + thread throttle; fullscreen toggle; promotion dialog.

Install:
    python -m pip install python-chess pillow cairosvg
    sudo apt install stockfish
"""

import io, os, math, shutil, random, threading, tkinter as tk
from typing import Optional, List, Tuple, Dict
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg

# ---------- Base Config (upper bounds; per-position depths are chosen by depth_profile) ----------
DEPTH = 16          # max main analysis depth
MULTIPV = 8         # max PVs

# Per-move probe defaults (overridden by depth_profile at runtime)
PER_MOVE_DEPTH = 7
MAX_PROBE_MOVES = 6

BOARD_SIZE = 360
SQUARE = BOARD_SIZE // 8

# Win% curve: ~+300cp ≈ 75%
K_LOG = 0.004

# Style safety caps (Δcp, Δwin%)
ADHERENCE = {
    "Aggressive": (180, 12.0),
    "Normal":     (45,   4.0),
    "Defensive":  (25,   2.5),
}

STYLE_CP_BONUS = {"Aggressive": 60, "Defensive": 40}
DIVERSITY_MARGIN_CP = 60

# ---------- Rating model ----------
ELO_ANCHORS = [
    (800,  (110.0, 110, 0.22, 5)),
    (1000, (120.0, 110, 0.25, 4)),  # warmed vs earlier builds
    (1200, (80.0,   70, 0.14, 4)),
    (1400, (62.0,   55, 0.11, 3)),
    (1600, (48.0,   45, 0.08, 3)),
    (1800, (38.0,   35, 0.06, 3)),
    (2000, (28.0,   28, 0.045,2)),
    (2200, (20.0,   22, 0.03, 2)),
    (2400, (14.0,   16, 0.02, 2)),
    (2600, (10.0,   10, 0.01, 2)),
]

def win_prob(cp_signed: int) -> float:
    return 100.0 / (1.0 + math.exp(-K_LOG * cp_signed))

def fmt_cp(cp: int) -> str:
    return f"{cp/100:.2f}" if cp < 0 else f"+{cp/100:.2f}"

def score_cp_signed(info_score: chess.engine.PovScore, pov_is_white: bool) -> int:
    pov = chess.WHITE if pov_is_white else chess.BLACK
    sc = info_score.pov(pov)
    if sc.is_mate():
        m = sc.mate()
        return 100000 if (m and m > 0) else -100000
    cp = sc.score(mate_score=100000)
    return int(cp if cp is not None else 0)

def mate_in(info_score: chess.engine.PovScore, pov_is_white: bool) -> Optional[int]:
    pov = chess.WHITE if pov_is_white else chess.BLACK
    sc = info_score.pov(pov)
    if sc.is_mate():
        m = sc.mate()
        return int(m) if (m and m > 0) else None
    return None

def find_stockfish_path() -> Optional[str]:
    envp = os.environ.get("STOCKFISH_PATH")
    if envp and os.path.isfile(envp) and os.access(envp, os.X_OK):
        return envp
    which = shutil.which("stockfish")
    if which:
        return which
    cand = "/usr/games/stockfish"
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return None

def lerp(a, b, t): return a + (b - a) * t

def rating_profile(target_elo: int):
    xs = [e for e, _ in ELO_ANCHORS]
    target = max(min(target_elo, xs[-1]), xs[0])
    for i in range(len(xs) - 1):
        e0, p0 = ELO_ANCHORS[i]
        e1, p1 = ELO_ANCHORS[i+1]
        if e0 <= target <= e1:
            t = (target - e0) / (e1 - e0) if e1 > e0 else 0.0
            temp = lerp(p0[0], p1[0], t)
            drop = int(round(lerp(p0[1], p1[1], t)))
            misp = lerp(p0[2], p1[2], t)
            topk = int(round(lerp(p0[3], p1[3], t)))
            return float(temp), int(drop), float(misp), max(2, int(topk))
    return 20.0, 20, 0.03, 2

def depth_profile(target_elo: int) -> Tuple[int, int, int]:
    """(pick_depth, multipv, probe_depth) by target Elo."""
    if target_elo < 1000:  return (9, 6, 5)
    if target_elo < 1200:  return (10, 6, 6)
    if target_elo < 1500:  return (12, 8, 7)
    if target_elo < 1800:  return (14, 8, 7)
    return (16, 8, 7)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish FEN analyser + Rating-Aware Style Coach")
        self.geometry("1100x820")
        self.resizable(True, True)

        # --- Fullscreen bindings ---
        self.is_fullscreen = False
        self.prev_geometry = None
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.exit_fullscreen())

        # ====== EARLY DEFAULTS (prevent race conditions) ======
        self.target_elo = 1000  # safe default; will be updated by on_elo_change()

        # FEN input
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        fen_ent = ttk.Entry(self, textvariable=self.fen_var, width=160)
        fen_ent.pack(padx=10, fill="x")
        fen_ent.bind("<Return>", lambda _ : self.load_fen())

        # Controls row
        ctl = ttk.Frame(self); ctl.pack(anchor="w", padx=10, pady=(6, 0))
        self.coach_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Coach mode", variable=self.coach_on,
                        command=self.redraw).pack(side="left", padx=(0, 10))

        ttk.Label(ctl, text="Style:").pack(side="left")
        self.style_var = tk.StringVar(value="Normal")
        style_cb = ttk.Combobox(
            ctl, textvariable=self.style_var,
            values=["Defensive", "Normal", "Aggressive"],
            state="readonly", width=12
        )
        style_cb.pack(side="left", padx=(4, 12))
        style_cb.bind("<<ComboboxSelected>>", lambda _e: self.update_style_preview())

        # Rating row
        rating_fr = ttk.Frame(self); rating_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(rating_fr, text="My Elo:").pack(side="left")
        self.user_elo_var = tk.StringVar(value="800")
        elo_ent = ttk.Entry(rating_fr, textvariable=self.user_elo_var, width=8)
        elo_ent.pack(side="left", padx=(4, 8))
        elo_ent.bind("<Return>", lambda _e: self.on_elo_change())
        elo_ent.bind("<FocusOut>", lambda _e: self.on_elo_change())
        self.target_elo_lbl = ttk.Label(rating_fr, text="Coach plays at: 1000")
        self.target_elo_lbl.pack(side="left", padx=(8, 12))

        # I'm playing as
        play_fr = ttk.Frame(self); play_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(play_fr, text="I'm playing as:").pack(side="left")
        self.play_color = tk.StringVar(value="white")
        ttk.Radiobutton(play_fr, text="White", value="white",
                        variable=self.play_color, command=self.on_pick_color).pack(side="left")
        ttk.Radiobutton(play_fr, text="Black", value="black",
                        variable=self.play_color, command=self.on_pick_color).pack(side="left")

        # Orientation
        opt_fr = ttk.Frame(self); opt_fr.pack(anchor="w", padx=10, pady=(0, 4))
        self.follow_turn = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_fr, text="Flip with side to move",
                        variable=self.follow_turn, command=self.redraw).pack(side="left", padx=(0, 10))
        ttk.Button(opt_fr, text="Flip now", command=self.flip_once).pack(side="left")

        # Buttons
        btn_fr = ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr, text="Load FEN", command=self.load_fen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_fr, text="Reset",    command=self.reset_pos).grid(row=0, column=1, padx=5)
        ttk.Button(btn_fr, text="Fullscreen", command=self.toggle_fullscreen).grid(row=0, column=2, padx=5)
        self.best_btn = ttk.Button(btn_fr, text="Best move", command=self.preview_best)
        self.best_btn.grid(row=0, column=3, padx=5)

        # Eval label
        self.eval_lbl = ttk.Label(self, text="", font=("Courier New", 12))
        self.eval_lbl.pack(pady=6)

        # Board canvas
        self.canvas = tk.Canvas(self, width=BOARD_SIZE, height=BOARD_SIZE, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self._img = None
        self._sel = None

        # Status bar
        self.status = ttk.Label(self, text="", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # Engine
        path = find_stockfish_path()
        if not path:
            messagebox.showerror("Engine not found",
                                 "Could not find Stockfish.\nTry:  sudo apt install stockfish\n"
                                 "Or set env var STOCKFISH_PATH.")
            self.destroy(); return
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.engine.configure({"Threads": 1, "Hash": 128})
            self.engine_lock = threading.Lock()
        except Exception as e:
            messagebox.showerror("Engine error", f"Failed to start Stockfish at:\n{path}\n\n{e}")
            self.destroy(); return

        # Internal board
        self.board = chess.Board()
        self.base_fen = self.board.fen()
        self.fixed_bottom_is_white = True

        # Analysis caches / guards
        self.last_infos: Optional[List[chess.engine.InfoDict]] = None
        self.permove_cache: Dict[str, chess.engine.PovScore] = {}
        self.style_sugg_move: Optional[chess.Move] = None
        self.style_sugg_reason: str = ""
        self.last_eval_fen: Optional[str] = None

        # Eval throttle
        self.eval_running = False
        self.latest_request = 0

        # Apply initial selections (set Elo BEFORE kicking off eval)
        self.on_pick_color()
        self.on_elo_change()   # sets self.target_elo reliably

        self.redraw()
        self.async_eval()

    # ---------- Safe attr ----------
    def _get_target_elo(self) -> int:
        try:
            return int(getattr(self, "target_elo"))
        except Exception:
            return 1000

    # ---------- Engine wrappers ----------
    def _restart_engine(self):
        try:
            if hasattr(self, "engine") and self.engine:
                self.engine.close()
        except Exception:
            pass
        path = find_stockfish_path()
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": 128})

    def _safe_analyse(self, board: chess.Board, limit: chess.engine.Limit, **kwargs):
        try:
            with self.engine_lock:
                return self.engine.analyse(board, limit, **kwargs)
        except chess.engine.EngineTerminatedError:
            self._restart_engine()
            with self.engine_lock:
                return self.engine.analyse(board, limit, **kwargs)

    # ---------- Fullscreen ----------
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.prev_geometry = self.geometry()
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                self.state("zoomed")
        else:
            self.exit_fullscreen()

    def exit_fullscreen(self):
        try:
            self.attributes("-fullscreen", False)
        except Exception:
            pass
        if self.prev_geometry:
            self.geometry(self.prev_geometry)
        self.is_fullscreen = False

    # ---------- Helpers ----------
    def my_color_is_white(self) -> bool:
        return self.play_color.get() == "white"

    def _invalidate_after_board_change(self):
        self.last_infos = None
        self.permove_cache = {}
        self.last_eval_fen = None
        self.style_sugg_move = None
        self.style_sugg_reason = ""
        self.status.config(text="Thinking…")

    def on_elo_change(self):
        # Parse user Elo and set target (user + 200)
        try:
            user_elo = max(200, min(3000, int(self.user_elo_var.get())))
        except Exception:
            user_elo = 800
            self.user_elo_var.set("800")
        self.target_elo = max(600, min(2600, user_elo + 200))
        self.target_elo_lbl.config(text=f"Coach plays at: {self.target_elo}")
        # Try engine-limited strength (harmless if unsupported)
        try:
            with self.engine_lock:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(self.target_elo)})
        except Exception:
            pass
        if self.last_eval_fen == self.board.fen():
            self.refresh_style_from_cache()
        else:
            self.async_eval()

    # ---------- Orientation ----------
    def bottom_color(self) -> chess.Color:
        return (self.board.turn if self.follow_turn.get()
                else (chess.WHITE if self.fixed_bottom_is_white else chess.BLACK))

    def flip_once(self):
        if self.follow_turn.get():
            self.follow_turn.set(False)
        self.fixed_bottom_is_white = not self.fixed_bottom_is_white
        self.redraw()

    # ---------- Color selection ----------
    def on_pick_color(self):
        self.follow_turn.set(False)
        self.fixed_bottom_is_white = (self.play_color.get() == "white")
        if self.last_eval_fen != self.board.fen():
            self.async_eval()
        else:
            self.refresh_style_from_cache()

    # ---------- Drawing ----------
    def redraw(self, board: Optional[chess.Board] = None, highlight: Optional[chess.Move] = None):
        if board is None:
            board = self.board

        arrows = []
        if self.coach_on.get() and self.style_sugg_move:
            my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
            if my_turn:
                arrows.append(chess.svg.Arrow(
                    self.style_sugg_move.from_square,
                    self.style_sugg_move.to_square,
                    color="#3b82f6"
                ))
        if highlight:
            arrows.append(chess.svg.Arrow(
                highlight.from_square, highlight.to_square, color="#f59e0b"
            ))

        svg = chess.svg.board(board=board, size=BOARD_SIZE,
                              orientation=self.bottom_color(), arrows=arrows)
        png = cairosvg.svg2png(bytestring=svg.encode())
        self._img = ImageTk.PhotoImage(Image.open(io.BytesIO(png)))
        self.canvas.create_image(0, 0, image=self._img, anchor="nw")

    # ---------- Engine eval (MultiPV + per-move probe) ----------
    def async_eval(self):
        self.latest_request += 1
        if self.eval_running:
            return
        self.eval_running = True
        gen = self.latest_request
        brd = self.board.copy()
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_eval, args=(gen, brd), daemon=True).start()

    def _worker_eval(self, gen: int, brd: chess.Board):
        try:
            t_elo = self._get_target_elo()
            pick_depth, mpv, probe_depth = depth_profile(t_elo)
            pick_depth = min(pick_depth, DEPTH)
            mpv = min(mpv, MULTIPV)

            # 1) MultiPV batch
            info_list = self._safe_analyse(
                brd, chess.engine.Limit(depth=pick_depth), multipv=mpv
            )

            # Distinct first moves from MultiPV
            unique_moves = []
            seen = set()
            for inf in info_list:
                pv = inf.get("pv")
                if not pv:
                    continue
                mv = pv[0]
                if mv not in seen:
                    seen.add(mv)
                    unique_moves.append(mv)

            # 2) Per-move probe (if variety low)
            permove_scores: Dict[str, chess.engine.PovScore] = {}
            try_probe = (len(unique_moves) < 2)
            legals = list(brd.legal_moves)

            if try_probe and legals:
                def quick_priority(b: chess.Board, mv: chess.Move) -> float:
                    piece = b.piece_at(mv.from_square)
                    to_piece = b.piece_at(mv.to_square)
                    is_cap = (to_piece is not None) or b.is_en_passant(mv)
                    is_promo = mv.promotion is not None
                    gives_chk = b.gives_check(mv)
                    castle = b.is_castling(mv)
                    def center_w(sq: int) -> float:
                        f = chess.square_file(sq); r = chess.square_rank(sq)
                        return -((f-3.5)**2 + (r-3.5)**2)
                    center_gain = 0.0
                    if piece:
                        center_gain = center_w(mv.to_square) - center_w(mv.from_square)
                    pawn_thrust = 0.0
                    if piece and piece.piece_type == chess.PAWN and not is_cap:
                        delta = abs(chess.square_rank(mv.to_square) - chess.square_rank(mv.from_square))
                        pawn_thrust = float(delta)
                    dev = 0.0
                    if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                        if chess.square_rank(mv.from_square) in (0, 7):
                            dev = 1.0
                    return (4.0 * gives_chk
                            + 3.0 * castle
                            + 3.0 * (is_cap or is_promo)
                            + 1.5 * max(0.0, center_gain)
                            + 1.2 * dev
                            + 1.0 * pawn_thrust)

                legals_sorted = sorted(legals, key=lambda m: quick_priority(brd, m), reverse=True)
                probe_moves: List[chess.Move] = []
                mv_set = set(unique_moves)
                for mv in legals_sorted:
                    if mv not in mv_set:
                        probe_moves.append(mv)
                    if len(probe_moves) >= MAX_PROBE_MOVES:
                        break

                for mv in probe_moves:
                    info = self._safe_analyse(
                        brd, chess.engine.Limit(depth=min(probe_depth, PER_MOVE_DEPTH)),
                        searchmoves=[mv]
                    )
                    if "score" in info:
                        permove_scores[mv.uci()] = info["score"]

            self.after(0, self._finish_eval, gen, brd.fen(), info_list, permove_scores)

        except Exception as e:
            # Surface errors to UI instead of hanging on "Thinking…"
            def _show_err():
                self.status.config(text=f"Engine error: {e}")
                self.eval_running = False
            self.after(0, _show_err)

    def _finish_eval(self, gen: int, fen: str, info_list: list,
                     permove_scores: Dict[str, chess.engine.PovScore]):
        if gen != self.latest_request or fen != self.board.fen():
            self.eval_running = False
            if self.latest_request > gen:
                self.async_eval()
            return

        self.last_infos = info_list
        self.permove_cache = permove_scores
        self.last_eval_fen = fen

        pov_is_white = self.my_color_is_white()
        top = info_list[0] if info_list else None
        cp = score_cp_signed(top["score"], pov_is_white) if top else 0
        self.show_eval(cp)

        self.apply_rating_aware_pick()
        self.redraw()

        self.eval_running = False
        if self.latest_request > gen:
            self.async_eval()

    def show_eval(self, cp_signed: int):
        my_prob = win_prob(cp_signed)
        opp_prob = 100.0 - my_prob
        self.eval_lbl.config(
            text=f"Eval: {fmt_cp(cp_signed)}  |  Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f}"
        )

    # ---------- UI actions ----------
    def load_fen(self):
        fen = self.fen_var.get().strip()
        if not fen:
            messagebox.showinfo("No FEN", "Paste a FEN first."); return
        try:
            self.board = chess.Board(fen)
        except ValueError as e:
            messagebox.showerror("Bad FEN", str(e)); return
        self.base_fen = self.board.fen()
        self._sel = None
        self._invalidate_after_board_change()
        self.redraw()
        self.async_eval()

    def reset_pos(self):
        self.board = chess.Board(self.base_fen)
        self._sel = None
        self._invalidate_after_board_change()
        self.redraw()
        self.async_eval()

    # Click-to-move (with promotion)
    def _make_move_with_optional_promotion(self, from_sq: chess.Square, to_sq: chess.Square) -> chess.Move:
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if to_rank in (0, 7):
                ans = simpledialog.askstring("Promotion", "Promote to (q, r, b, n)?", parent=self)
                promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
                promo = promo_map.get((ans or "q").lower()[0], chess.QUEEN)
                return chess.Move(from_sq, to_sq, promotion=promo)
        return chess.Move(from_sq, to_sq)

    def on_click(self, event):
        bottom = self.bottom_color()
        fx = int(event.x // SQUARE); fy = int(event.y // SQUARE)
        if not (0 <= fx < 8 and 0 <= fy < 8): return
        if bottom == chess.WHITE: file, rank = fx, 7 - fy
        else:                     file, rank = 7 - fx, fy
        sq = chess.square(file, rank)

        if self._sel is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self._sel = sq
        else:
            mv = self._make_move_with_optional_promotion(self._sel, sq)
            self._sel = None
            if self.board.is_legal(mv):
                self.board.push(mv)
                self._invalidate_after_board_change()
                self.redraw()
                self.async_eval()
            else:
                messagebox.showwarning("Illegal", "That move isn't legal.")

    # Best move preview
    def preview_best(self):
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_best, daemon=True).start()

    def _worker_best(self):
        try:
            pick_depth, _, _ = depth_profile(self._get_target_elo())
            info = self._safe_analyse(self.board, chess.engine.Limit(depth=pick_depth))
            best = info["pv"][0]
            brd = self.board.copy(); brd.push(best)
            self.after(0, self._finish_best, brd, best, info)
        except Exception as e:
            self.after(0, lambda: self.status.config(text=f"Engine error: {e}"))

    def _finish_best(self, brd, best, info):
        self.redraw(board=brd, highlight=best)
        pov_is_white = self.my_color_is_white()
        cp = score_cp_signed(info["score"], pov_is_white)
        my_prob = win_prob(cp); opp_prob = 100 - my_prob
        self.eval_lbl.config(
            text=f"Best: {self.board.san(best):6} | {fmt_cp(cp)} | Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f}"
        )
        self.status.config(text="Preview – actual board unchanged (Reset to clear)")

    # ---------- Rating + Style pick ----------
    def update_style_preview(self):
        if self.last_eval_fen != self.board.fen():
            self.async_eval()
        else:
            self.refresh_style_from_cache()

    def refresh_style_from_cache(self):
        self.apply_rating_aware_pick()
        self.redraw()

    def apply_rating_aware_pick(self):
        mv, reason = self.pick_rating_style_move()
        self.style_sugg_move = mv
        self.style_sugg_reason = reason
        my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
        if mv and my_turn:
            try: san = self.board.san(mv)
            except Exception: san = str(mv)
            self.status.config(text=f"{san}  —  {reason}")
        else:
            self.status.config(text="Ready")

    def _style_features(self, board: chess.Board, move: chess.Move) -> dict:
        b = board; mv = move
        moved_piece = b.piece_at(mv.from_square)
        to_piece = b.piece_at(mv.to_square)
        is_capture = to_piece is not None or b.is_en_passant(mv)
        is_promo = mv.promotion is not None
        gives_check = b.gives_check(mv)
        def center_weight(sq: int) -> float:
            f = chess.square_file(sq); r = chess.square_rank(sq)
            return -((f-3.5)**2 + (r-3.5)**2)
        center_gain = 0.0
        if moved_piece:
            center_gain = center_weight(mv.to_square) - center_weight(mv.from_square)
        dev_new_piece = 0
        if moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if chess.square_rank(mv.from_square) in (0, 7):
                dev_new_piece = 1
        pawn_thrust = 0
        if moved_piece and moved_piece.piece_type == chess.PAWN and not is_capture:
            pawn_thrust = abs(chess.square_rank(mv.to_square) - chess.square_rank(mv.from_square))
        castle = b.is_castling(mv)
        quiet_dev = int((not is_capture) and moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP))
        king_shelter_pawn = 0
        if moved_piece and moved_piece.piece_type == chess.PAWN and not is_capture:
            file = chess.square_file(mv.from_square)
            if file in (0, 7): king_shelter_pawn = 1
        simplify_capture = 0
        if is_capture and moved_piece and to_piece:
            simplify_capture = int(moved_piece.piece_type == to_piece.piece_type)
        return {
            "is_check": int(gives_check),
            "is_capture_or_promo": int(is_capture or is_promo),
            "center_gain": center_gain,
            "dev_new_piece": dev_new_piece,
            "pawn_thrust": pawn_thrust,
            "castle": int(castle),
            "quiet_dev": quiet_dev,
            "king_shelter_pawn": king_shelter_pawn,
            "simplify_capture": simplify_capture,
        }

    def _style_metric(self, style: str, f: dict) -> float:
        if style == "Aggressive":
            return (3.0 * f["is_check"]
                    + 2.0 * f["is_capture_or_promo"]
                    + 1.2 * max(0.0, f["center_gain"])
                    + 1.0 * f["dev_new_piece"]
                    + (2.0 if f["pawn_thrust"] >= 2 else (0.8 if f["pawn_thrust"] == 1 else 0.0)))
        if style == "Defensive":
            return (2.5 * f["castle"]
                    + 2.0 * f["quiet_dev"]
                    + 1.5 * f["king_shelter_pawn"]
                    + 1.0 * f["simplify_capture"]
                    + (0.8 if (f["is_check"] == 0 and f["is_capture_or_promo"] == 0) else 0.0))
        return 0.0

    def pick_rating_style_move(self) -> Tuple[Optional[chess.Move], str]:
        style = self.style_var.get()
        pov_is_white = self.my_color_is_white()

        # Candidates from MultiPV (unique first moves)
        cands: List[Tuple[chess.Move, float, dict, List[str], Optional[int], float, str]] = []
        seen = set()
        if self.last_infos:
            for inf in self.last_infos:
                pv = inf.get("pv")
                if not pv: continue
                mv = pv[0]
                if mv in seen: continue
                seen.add(mv)
                cp = score_cp_signed(inf["score"], pov_is_white)
                feats = self._style_features(self.board, mv)
                metric = self._style_metric(style, feats)
                m_in = mate_in(inf["score"], pov_is_white)
                cands.append((mv, float(cp), feats, [], m_in, metric, "pv"))

        # Augment with probes
        for uci, sc in self.permove_cache.items():
            mv = chess.Move.from_uci(uci)
            if mv in seen: continue
            cp = score_cp_signed(sc, pov_is_white)
            feats = self._style_features(self.board, mv)
            metric = self._style_metric(style, feats)
            m_in = mate_in(sc, pov_is_white)
            cands.append((mv, float(cp), feats, [], m_in, metric, "probe"))

        if not cands:
            return None, ""

        # Mate override
        mates = [t for t in cands if t[4] is not None and t[4] > 0]
        if mates:
            best = min(mates, key=lambda t: t[4])
            return best[0], f"mate in {best[4]}"

        # Engine top eval
        top_cp = max(t[1] for t in cands)
        top_mv = max(cands, key=lambda t: t[1])[0]

        # Style "push"
        style_bonus = STYLE_CP_BONUS.get(style, 0)

        # Rating profile
        temp_cp, max_drop_cp, mistake_prob, top_k_floor = rating_profile(self._get_target_elo())

        # State-aware tweak
        if top_cp >= 200:
            max_drop_cp = int(max_drop_cp * 0.6); temp_cp *= 0.85
        elif top_cp <= -80:
            max_drop_cp = int(max_drop_cp * 1.25); temp_cp *= 1.10

        sorted_cands = sorted(cands, key=lambda t: t[1], reverse=True)
        poolA = [t for t in sorted_cands if (top_cp - t[1]) <= max_drop_cp]
        if len(poolA) < top_k_floor:
            poolA = sorted_cands[:max(top_k_floor, len(sorted_cands))]

        def weights(pool):
            w = []
            for mv, cp, feats, _, _, metric, _ in pool:
                eff = cp + style_bonus * metric
                w.append(math.exp((eff - top_cp) / max(1e-9, temp_cp)))
            s = sum(w) or 1.0
            return [x / s for x in w]

        use_mistake = (random.random() < mistake_prob)
        if use_mistake and len(sorted_cands) > len(poolA):
            widen = int(max_drop_cp * 1.8)
            poolB = [t for t in sorted_cands if (top_cp - t[1]) <= widen] or sorted_cands[:max(top_k_floor+1, 3)]
            old_temp = temp_cp
            temp_cp *= 1.4
            probs = weights(poolB)
            pick = random.choices(poolB, probs, k=1)[0]
            temp_cp = old_temp
            reason = f"rating {self._get_target_elo()}, (mistake), Δcap≤{widen}cp, τ≈{int(old_temp)}"
        else:
            probs = weights(poolA)
            pick = random.choices(poolA, probs, k=1)[0]
            reason = f"rating {self._get_target_elo()}, Δcap≤{max_drop_cp}cp, τ≈{int(temp_cp)}"

        pick_mv, pick_cp = pick[0], pick[1]

        # Global safety
        cap_cp, cap_wp = ADHERENCE.get(style, ADHERENCE["Normal"])
        drop_cp = top_cp - pick_cp
        drop_wp = win_prob(top_cp) - win_prob(pick_cp)
        if drop_cp > cap_cp or drop_wp > cap_wp:
            return top_mv, f"engine safety (Δ{int(drop_cp)}cp, Δ{drop_wp:.1f}%)"

        # Diversity nudge
        alts = [t for t in cands if t[0] != pick_mv and (top_cp - t[1]) <= DIVERSITY_MARGIN_CP]
        if alts:
            alt = max(alts, key=lambda t: t[5])
            if alt[5] > pick[5] + 1.0:
                pick_mv = alt[0]
                reason += "; within margin (style)"

        return pick_mv, reason

    # ---------- Cleanup ----------
    def destroy(self):
        try:
            if hasattr(self, "engine"):
                self.engine.close()
        finally:
            super().destroy()

if __name__ == "__main__":
    App().mainloop()
