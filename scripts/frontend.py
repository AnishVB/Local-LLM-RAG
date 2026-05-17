"""
frontend.py — BDL CHATBOT  (Flet 0.85+)
Run:  python scripts/frontend.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-USER LOCAL SERVER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Supports up to 5 concurrent users + 1 admin
  • Per-session auth — each browser tab has its own independent role & state
  • Each session gets its own ChatEngine (own LanceDB connection) so queries
    from different users never block each other
  • Thread-safe file operations throughout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDLE AUTO-LOGOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Users   : kicked after 5 minutes of inactivity  (USER_IDLE_TIMEOUT = 300s)
  • Admins  : kicked after 10 minutes of inactivity (ADMIN_IDLE_TIMEOUT = 600s)
  • "Activity" = any message sent. Typing or scrolling does NOT reset the timer.
  • A background daemon thread (_idle_watchdog) checks every 30 seconds and
    calls _logout() on sessions that have exceeded their idle limit.
  • The watchdog runs per-session (started when _build_chat_ui is called) and
    stops automatically when the session is destroyed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES vs previous version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [FIX-A] FilePicker used for admin document uploads — works from any device
  [FIX-B] "Focus Document" correctly stored in session, reflected in doc bar
  [FIX-C] Upload staging dir cleaned up after indexing (temp files removed)
  [FIX-D] Page vs clause/section citation bug fixed in chat_engine (DOC_SYSTEM)
  [NEW-1] Admin dashboard shows live doc count and chunk count cards
  [NEW-2] FilePicker supports multi-file selection in one pass
  [NEW-3] Upload queue shows individual file status + overall progress
  [NEW-4] Sidebar doc count refreshes after every upload/delete
  [NEW-5] "Focus" button works in the doc list for both admin and user
  [NEW-6] Graceful Ollama-offline banner on login
  [NEW-7] Idle auto-logout: users 5 min, admins 10 min
"""

import os, sys, re, time, threading, base64, socket, json as _json, shutil
from datetime import datetime, date, timedelta
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path   = os.path.normpath(os.path.join(current_dir, "..", "assests", "BDL logo nobg.png"))

# Upload staging directory — Flet web-mode writes uploaded bytes here
UPLOAD_DIR = os.path.normpath(os.path.join(current_dir, "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _get_logo_src(path):
    if not os.path.isfile(path): return None
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except Exception: return None

logo_src = _get_logo_src(logo_path)

ADMIN_PASSWORD = "Admin321!"

# ── Multi-user connection management ──────────────────────────────────────────
MAX_USERS  = 5
MAX_ADMINS = 1
_conn_lock     = threading.Lock()
_active_users  = 0
_active_admins = 0

# ── Live session registry — tracks every connected user for the admin panel ───
# Each entry: { "role", "device", "login_time", "last_active", "kick_fn" }
# kick_fn is a callable set by _build_chat_ui so the admin can force-logout anyone.
_live_sessions: dict[str, dict] = {}   # session_key → info dict
_live_sessions_lock = threading.Lock()

def _register_live_session(session_key: str, role: str, device: str, kick_fn) -> None:
    now = datetime.now()
    with _live_sessions_lock:
        _live_sessions[session_key] = {
            "role":        role,
            "device":      device,
            "login_time":  now,
            "last_active": now,
            "kick_fn":     kick_fn,
        }

def _touch_live_session(session_key: str) -> None:
    """
    Update last_active timestamp whenever the user sends a message.
    Called in _send() so every message resets the idle clock.
    The idle watchdog (_idle_watchdog) reads this timestamp to decide
    whether to kick the session.
    """
    with _live_sessions_lock:
        if session_key in _live_sessions:
            _live_sessions[session_key]["last_active"] = datetime.now()

def _unregister_live_session(session_key: str) -> None:
    with _live_sessions_lock:
        _live_sessions.pop(session_key, None)

def _get_live_sessions_snapshot() -> list[dict]:
    """Return a copy of all live session info for the admin panel."""
    with _live_sessions_lock:
        return [
            {"key": k, **{kk: vv for kk, vv in v.items() if kk != "kick_fn"},
             "kick_fn": v["kick_fn"]}
            for k, v in _live_sessions.items()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Idle auto-logout timeouts
# Users are kicked after 5 minutes idle; admins after 10 minutes.
# "Idle" means no message sent — scrolling/typing does NOT reset the clock.
# The watchdog thread (_idle_watchdog, started per session) checks every 30s.
# ─────────────────────────────────────────────────────────────────────────────
USER_IDLE_TIMEOUT  = 5  * 60   # 5 minutes in seconds
ADMIN_IDLE_TIMEOUT = 10 * 60   # 10 minutes in seconds
IDLE_CHECK_INTERVAL = 30        # watchdog polling interval (seconds)

def _acquire_slot(role):
    global _active_users, _active_admins
    with _conn_lock:
        if role == "admin":
            if _active_admins >= MAX_ADMINS: return False
            _active_admins += 1
        else:
            if _active_users >= MAX_USERS: return False
            _active_users += 1
    return True

def _release_slot(role):
    global _active_users, _active_admins
    with _conn_lock:
        if role == "admin":   _active_admins = max(0, _active_admins - 1)
        elif role == "user":  _active_users  = max(0, _active_users  - 1)

# ── Per-session ChatEngine pool ───────────────────────────────────────────────
# Each browser session gets its own ChatEngine (and therefore its own LanceDB
# connection) so concurrent users never block each other's DB reads.
# The shared singleton pattern caused serialisation: LanceDB's reader lock meant
# that while User A's query was reading, User B's had to wait.
#
# Pool size is capped at MAX_USERS + MAX_ADMINS.  Engines are created on first
# login and destroyed on disconnect.  All engines share the same on-disk LanceDB
# files — reads are concurrent-safe; writes use the per-table _write_lock in
# database.py which serialises only the short write window.
_engine_pool: dict[str, "ChatEngine"] = {}   # session_key → ChatEngine
_engine_pool_lock = threading.Lock()

def _get_engine_for_session(session_key: str) -> "ChatEngine":
    """Return (creating if needed) a dedicated ChatEngine for this session key."""
    with _engine_pool_lock:
        if session_key not in _engine_pool:
            _engine_pool[session_key] = ChatEngine()
        return _engine_pool[session_key]

def _release_engine_for_session(session_key: str) -> None:
    """Remove the engine from the pool when the user disconnects."""
    with _engine_pool_lock:
        _engine_pool.pop(session_key, None)

_SESSION_META_PATH = os.path.normpath(
    os.path.join(current_dir, "..", "chat_history", "session_meta.json"))
_meta_lock = threading.Lock()

def _load_session_meta():
    with _meta_lock:
        if os.path.isfile(_SESSION_META_PATH):
            try:
                with open(_SESSION_META_PATH, "r", encoding="utf-8") as f:
                    return _json.load(f)
            except Exception: pass
        return {}

def _save_session_meta(meta):
    with _meta_lock:
        os.makedirs(os.path.dirname(_SESSION_META_PATH), exist_ok=True)
        with open(_SESSION_META_PATH, "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)

def _get_local_ip():
    # Fully offline LAN-safe IP detection — no external connection needed.
    # Tries each LAN interface in order; falls back to 127.0.0.1.
    try:
        hostname = socket.gethostname()
        candidates = socket.getaddrinfo(hostname, None, socket.AF_INET)
        for item in candidates:
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    # Second fallback: enumerate interfaces via a dummy UDP socket bound to 0.0.0.0
    # This never actually sends any packets — it just asks the OS which source IP
    # it would use to reach a LAN address (10.0.0.1 is just a dummy target).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.0.0.1", 80)); ip = s.getsockname()[0]; s.close()
        if ip and not ip.startswith("127."): return ip
    except Exception:
        pass
    return "127.0.0.1"

def _device_label(client_ip: str) -> str:
    """
    Turn a client IP into a human-readable device label.
    Tries reverse-DNS first to get the machine hostname; falls back to the IP.
    In web/LAN mode each device has its own IP so this uniquely identifies them.
    """
    if not client_ip or client_ip in ("", "unknown"):
        return "Unknown Device"
    try:
        hostname = socket.gethostbyaddr(client_ip)[0]
        # Strip domain suffix — keep only the short machine name
        short = hostname.split(".")[0]
        return short if short else client_ip
    except Exception:
        return client_ip

sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import flet as ft
from chat_engine import (ChatEngine, ChatSession, load_all_sessions, save_all_sessions,
                          delete_session, purge_empty_sessions)

# ── Colours ───────────────────────────────────────────────────────────────────
C = dict(
    bg="#ffffff", sidebar="#E7C39E", card="#ffffff", input="#f0f1f3",
    border="#d0d3de", accent="#3b5bdb", accent2="#4c6ef5",
    user_bg="#3b5bdb", bot_bg="#f4f5f7", fg="#1a1a2e", fg2="#2e3250",
    dim="#9096b0", green="#2f9e44", red="#c92a2a",
    tag_bg="#dbe4ff", tag_fg="#1c4be0", button_accent="#d42727",
)

def _ps(h=0, v=0): return ft.Padding.symmetric(horizontal=h, vertical=v)
def _po(**kw):      return ft.Padding.only(**kw)
def _ms(h=0, v=0): return ft.Margin.symmetric(horizontal=h, vertical=v)
def _trunc(t, n=32): t=(t or "New Chat").strip(); return t if len(t)<=n else t[:n-1]+"…"

def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string like '2h 14m' or '45s'."""
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"

def _date_label(d):
    today = date.today()
    if d == today:                     return "Today"
    if d == today - timedelta(days=1): return "Yesterday"
    if (today - d).days < 7:          return d.strftime("%A")
    if d.year == today.year:          return d.strftime("%b %d")
    return d.strftime("%b %d, %Y")

def _group(sessions):
    groups = {}
    for s in sorted(sessions, key=lambda x: x.updated_at or "", reverse=True):
        try:    d = datetime.fromisoformat(s.updated_at).date()
        except: d = date.today()
        groups.setdefault(d, []).append(s)
    return [(_date_label(d), groups[d]) for d in sorted(groups, reverse=True)]

def _format_suggestions_block(suggestions):
    if not suggestions: return ""
    lines = ["\n\n---\n**Want to explore further?**"]
    for sg in suggestions:
        lines.append(f"- {sg}")
    return "\n".join(lines)


async def main(page: ft.Page):
    page.title      = "BDL CHATBOT"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor    = C["bg"]
    page.window.width, page.window.height     = 1120, 720
    page.window.min_width, page.window.min_height = 800, 480
    page.padding = 0

    page_role = [None]   # "user" or "admin"
    user_ip   = _device_label(getattr(page, 'client_ip', None) or _get_local_ip())

    # ── FilePicker setup (Flet 0.85 Service API) ─────────────────────────────
    # FilePicker is now a Service — do NOT add to page.overlay (causes
    # "Unknown control: FilePicker" error). Just instantiate it.
    # pick_files() is async and returns List[FilePickerFile] directly.
    # with_data=True populates f.bytes with file bytes from the browser.
    file_picker = ft.FilePicker()


    # ── Login ─────────────────────────────────────────────────────────────────
    def _enter_as_user(e):
        if not _acquire_slot("user"):
            _snack_global(f"Server full — max {MAX_USERS} users connected. Try again later.", ok=False)
            return
        page_role[0] = "user"; _launch_chat()

    def _enter_as_admin(e):
        if (admin_pwd_field.value or "") == ADMIN_PASSWORD:
            if not _acquire_slot("admin"):
                admin_error.value = "Another admin is already connected"
                admin_error.visible = True; page.update(); return
            page_role[0] = "admin"; _launch_chat()
        else:
            admin_error.value = "Incorrect password"; admin_error.visible = True; page.update()

    admin_pwd_field = ft.TextField(
        hint_text="Enter admin password", hint_style=ft.TextStyle(color=C["dim"]),
        password=True, can_reveal_password=True, border_color=C["border"],
        focused_border_color=C["accent"], bgcolor=C["input"], color=C["fg"],
        text_style=ft.TextStyle(color=C["fg"], size=13),
        content_padding=_ps(h=14, v=10), width=260,
        on_submit=lambda e: _enter_as_admin(e),
    )
    admin_error   = ft.Text("", color=C["red"], size=12, visible=False)
    admin_section = ft.Column([
        admin_pwd_field, admin_error, ft.Container(height=4),
        ft.FilledButton(
            content=ft.Text("Login as Admin", size=13, color="#ffffff"),
            style=ft.ButtonStyle(bgcolor=C["button_accent"], shape=ft.RoundedRectangleBorder(radius=10)),
            width=260, height=40, on_click=_enter_as_admin,
        ),
    ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER, visible=False)

    login_view = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Image(src=logo_src, width=160, height=55, fit=ft.BoxFit.CONTAIN)
                        if logo_src else ft.Text("BDL", size=28, weight=ft.FontWeight.BOLD, color=C["button_accent"]),
                padding=_po(bottom=6),
            ),
            ft.Text("BDL CHATBOT", size=26, weight=ft.FontWeight.BOLD, color=C["fg"]),
            ft.Text("Select how you'd like to continue", size=13, color=C["dim"]),
            ft.Container(height=16),
            ft.FilledButton(
                content=ft.Row([ft.Icon(ft.Icons.PERSON_ROUNDED, color="#ffffff", size=18),
                                ft.Text("Continue as User", size=14, color="#ffffff", weight=ft.FontWeight.W_500)],
                               spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                style=ft.ButtonStyle(bgcolor=C["accent"], shape=ft.RoundedRectangleBorder(radius=10)),
                width=260, height=44, on_click=_enter_as_user,
            ),
            ft.Container(height=6),
            ft.OutlinedButton(
                content=ft.Row([ft.Icon(ft.Icons.ADMIN_PANEL_SETTINGS_ROUNDED, color=C["button_accent"], size=18),
                                ft.Text("Login as Admin", size=14, color=C["button_accent"], weight=ft.FontWeight.W_500)],
                               spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                style=ft.ButtonStyle(side=ft.BorderSide(1.5, C["button_accent"]), shape=ft.RoundedRectangleBorder(radius=10)),
                width=260, height=44,
                on_click=lambda e: [setattr(admin_section, "visible", not admin_section.visible), page.update()],
            ),
            ft.Container(height=4), admin_section,
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
        alignment=ft.Alignment(0, 0), expand=True, bgcolor=C["bg"],
    )
    page.add(login_view)

    def _on_disconnect(e=None):
        if page_role[0]:
            _release_slot(page_role[0])
            _release_engine_for_session(f"{user_ip}_{page_role[0]}_{id(page)}")
            _unregister_live_session(f"{user_ip}_{page_role[0]}_{id(page)}")
            page_role[0] = None
        # Drop any empty session older than the 5-min TTL
        try:
            all_s   = load_all_sessions()
            cleaned = purge_empty_sessions(all_s)
            if len(cleaned) != len(all_s):
                save_all_sessions(cleaned)
        except Exception:
            pass
    page.on_disconnect = _on_disconnect

    def _logout(e=None):
        if page_role[0]:
            _release_slot(page_role[0])
            _release_engine_for_session(f"{user_ip}_{page_role[0]}_{id(page)}")
            _unregister_live_session(f"{user_ip}_{page_role[0]}_{id(page)}")
            page_role[0] = None
        # Drop any empty session older than the 5-min TTL
        try:
            all_s   = load_all_sessions()
            cleaned = purge_empty_sessions(all_s)
            if len(cleaned) != len(all_s):
                save_all_sessions(cleaned)
        except Exception:
            pass
        admin_pwd_field.value = ""; admin_section.visible = False; admin_error.visible = False
        page.controls.clear(); page.add(login_view); page.update()

    def _launch_chat():
        page.controls.clear(); page.update(); _build_chat_ui()

    def _snack_global(text, ok=True):
        sb = ft.SnackBar(ft.Text(text, color=C["green"] if ok else C["red"]), bgcolor=C["card"], open=True)
        page.overlay.append(sb); page.update()

    # ─────────────────────────────────────────────────────────────────────────
    # _build_chat_ui — called once per login; constructs the entire chat
    # interface (sidebar, chat area, session list, upload dialog, admin
    # dashboard) and wires up all event handlers.
    # Also starts two background daemon threads per session:
    #   1. _periodic_purge  — cleans up empty sessions every 10 minutes
    #   2. _idle_watchdog   — auto-logs out idle sessions
    # ─────────────────────────────────────────────────────────────────────────
    def _build_chat_ui():
        is_admin = page_role[0] == "admin"

        # Unique key for this browser session's engine — based on client IP + role
        # so different tabs from the same device each get their own engine.
        session_key = f"{user_ip}_{page_role[0]}_{id(page)}"
        engine = _get_engine_for_session(session_key)

        sessions_raw = load_all_sessions()
        sessions     = purge_empty_sessions(sessions_raw)
        if len(sessions) != len(sessions_raw):
            save_all_sessions(sessions)

        cur = {
            "session": None, "busy": False, "thread": None,
            "spinner": None, "read_only": False,
            "upload_queue": [], "upload_cancelled": False,
            "stop_event": threading.Event(),
            "current_progress": 0.0, "current_status": "",
        }
        live_md = [None]

        def _save():
            save_all_sessions(sessions)
            if cur["session"]:
                sid  = cur["session"].session_id
                meta = _load_session_meta()
                if sid not in meta:
                    meta[sid] = {"device": user_ip, "role": page_role[0] or "user"}
                    _save_session_meta(meta)

        _lock = threading.Lock()

        # ── Register this session in the live session registry ────────────────
        # kick_fn is defined here so the admin panel can call _logout() on any user.
        def _kick_this_session():
            """Called by the admin to force-logout this specific user."""
            # Stop the idle watchdog for this session immediately
            _idle_stop.set()
            try:
                page.run_thread(_logout)
            except Exception:
                pass

        _register_live_session(session_key, page_role[0], user_ip, _kick_this_session)

        # ── Background periodic purge ─────────────────────────────────────────
        def _periodic_purge():
            while True:
                time.sleep(600)
                try:
                    all_s = load_all_sessions()
                    clean = purge_empty_sessions(all_s)
                    if len(clean) != len(all_s):
                        save_all_sessions(clean)
                        kept_ids  = {s.session_id for s in clean}
                        to_remove = [s for s in sessions if s.session_id not in kept_ids]
                        for s in to_remove:
                            sessions.remove(s)
                        page.run_thread(_refresh_sess_list)
                except Exception:
                    pass

        threading.Thread(target=_periodic_purge, daemon=True).start()

        # ── Idle auto-logout watchdog ─────────────────────────────────────────
        # Runs as a daemon thread for this specific session.
        # Checks every IDLE_CHECK_INTERVAL seconds whether this session has
        # been idle longer than its allowed timeout (role-dependent).
        # If idle limit exceeded → calls _logout() to force this user out.
        # The watchdog automatically dies when the session ends (daemon=True).
        _idle_stop = threading.Event()   # set on logout to stop the watchdog cleanly

        def _idle_watchdog():
            idle_limit = ADMIN_IDLE_TIMEOUT if is_admin else USER_IDLE_TIMEOUT
            while not _idle_stop.is_set():
                _idle_stop.wait(IDLE_CHECK_INTERVAL)   # sleep, but wake early on stop
                if _idle_stop.is_set():
                    break
                # Check how long this session has been idle
                with _live_sessions_lock:
                    info = _live_sessions.get(session_key)
                if info is None:
                    break   # session already unregistered — stop watchdog
                idle_seconds = (datetime.now() - info["last_active"]).total_seconds()
                if idle_seconds >= idle_limit:
                    # Session has been idle too long — force logout
                    role_label = "admin" if is_admin else "user"
                    idle_mins  = int(idle_seconds // 60)
                    print(f"[IDLE] Kicking {role_label} {user_ip} after {idle_mins}m idle")
                    try:
                        page.run_thread(_logout)
                    except Exception:
                        pass   # page may already be disconnected
                    break      # watchdog job done

        threading.Thread(target=_idle_watchdog, daemon=True).start()

        # ── Dialog helpers ────────────────────────────────────────────────────
        # Root cause of remote freeze: show_dialog/pop_dialog call
        # self._dialogs.update() or dialog.update() — both are partial updates
        # that only reach clients who already have the dialog registered.
        # A new dialog created inside a click handler has never been sent to
        # remote clients, so update() on it is a no-op for them.
        # Fix: pre-create ONE persistent AlertDialog, add it to page.overlay
        # on startup, then toggle .open + page.update() to show/hide it.
        # page.update() (no args) does a full-page flush that reaches all clients.
        _dlg = ft.AlertDialog(open=False, title=ft.Text(""))
        page.overlay.append(_dlg)

        def _show_dlg(title_widget, content_widget, actions):
            _dlg.title   = title_widget
            _dlg.content = content_widget
            _dlg.actions = actions
            _dlg.bgcolor  = C["card"]
            _dlg.open    = True
            page.update()

        def _close_dlg():
            _dlg.open = False
            page.update()

        def _snack(text, ok=True):
            sb = ft.SnackBar(ft.Text(text, color=C["green"] if ok else C["red"]), bgcolor=C["card"], open=True)
            page.overlay.append(sb); page.update()

        def _show_sheet(content_widget):
            bs = ft.BottomSheet(content=content_widget, open=True)
            page.overlay.append(bs); page.update(); return bs

        def _close_sheet(bs):
            bs.open = False; page.update()

        # ── Chat widgets ──────────────────────────────────────────────────────
        chat_col   = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO, expand=True, auto_scroll=True)
        doc_bar    = ft.Row(spacing=0, visible=False)
        status_lbl = ft.Text("Ready", size=11, color=C["dim"])
        stop_btn   = ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE_OUTLINED, icon_color=C["red"],
            icon_size=18, tooltip="Stop generation", visible=False,
            on_click=lambda e: _stop_generation(),
        )

        # Upload dialog state — widgets declared inline in _handle_uploaded_files below

        input_box = ft.TextField(
            hint_text="Ask anything about your documents…",
            hint_style=ft.TextStyle(color=C["dim"]),
            border=ft.InputBorder.NONE, bgcolor=C["input"], color=C["fg"],
            text_style=ft.TextStyle(color=C["fg"], size=13),
            multiline=True, min_lines=1, max_lines=5, expand=True,
            content_padding=_ps(h=16, v=12), cursor_color=C["accent"],
            shift_enter=True, on_submit=lambda e: _send(),
        )
        send_btn = ft.IconButton(
            icon=ft.Icons.SEND_ROUNDED, icon_color=C["button_accent"],
            icon_size=22, tooltip="Send",
        )
        sess_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO, expand=True)

        # ── Bubbles ───────────────────────────────────────────────────────────
        def _user_bubble(text):
            return ft.Container(
                content=ft.Row([
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Text(text, color="#ffffff", size=13, selectable=True, no_wrap=False),
                        bgcolor=C["user_bg"], border_radius=ft.BorderRadius(14, 14, 2, 14),
                        padding=_ps(h=14, v=10), width=500,
                    ),
                ]),
                padding=_po(left=16, right=16, top=4, bottom=4),
            )

        def _bot_bubble(thinking=False):
            spinner = ft.ProgressRing(width=16, height=16, visible=thinking)
            md = ft.Markdown(
                value="..." if thinking else "", selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                code_theme=ft.MarkdownCodeTheme.GITHUB, expand=True,
            )
            shell = ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text("● Answer", size=11, weight=ft.FontWeight.W_600, color=C["green"]),
                                spinner,
                            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            md,
                        ], spacing=6, tight=True),
                        bgcolor=C["bot_bg"], border_radius=ft.BorderRadius(2, 14, 14, 14),
                        padding=_ps(h=14, v=10), width=750,
                        border=ft.Border(left=ft.BorderSide(1, C["border"]), right=ft.BorderSide(1, C["border"]), top=ft.BorderSide(1, C["border"]), bottom=ft.BorderSide(1, C["border"])),
                    ),
                    ft.Container(expand=True),
                ]),
                padding=_po(left=16, right=16, top=4, bottom=4),
            )
            return shell, md, spinner

        # ── Render session ────────────────────────────────────────────────────
        def _render():
            chat_col.controls.clear()
            s = cur["session"]
            if not s or not s.messages:
                chat_col.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Text("BDL CHATBOT", size=22, weight=ft.FontWeight.BOLD, color=C["accent"]),
                        ft.Text("Focus on a document using the (+) button, then ask anything about them. \n For more precise answers, try to include specific details in your question.",
                                size=13, color=C["dim"], text_align=ft.TextAlign.CENTER),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                    alignment=ft.Alignment(0, 0), expand=True,
                ))
            else:
                for m in s.messages:
                    if m["role"] == "user":
                        chat_col.controls.append(_user_bubble(m["content"]))
                    else:
                        shell, md, _ = _bot_bubble(thinking=False)
                        md.value = m["content"]
                        chat_col.controls.append(shell)
            page.update()

        def _append_suggestions_to_answer(suggestions, md_widget):
            block = _format_suggestions_block(suggestions)
            if block and md_widget:
                md_widget.value = (md_widget.value or "") + block
                page.update()

        # ── [FIX-B] Doc bar ───────────────────────────────────────────────────
        # Correctly reads session.active_doc and renders it on every device.
        def _refresh_doc_bar():
            doc_bar.controls.clear()
            s = cur["session"]
            if s and s.active_doc:
                doc_bar.visible = True
                doc_bar.controls.append(ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=14, color=C["tag_fg"]),
                        ft.Text(f"  Focused: {s.active_doc}", size=12, color=C["tag_fg"], expand=True),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE, icon_size=14, icon_color=C["dim"],
                            width=26, height=26, on_click=lambda e: _clear_doc(),
                        ),
                    ], spacing=0),
                    bgcolor=C["tag_bg"], border_radius=8,
                    padding=_po(left=10, right=4, top=4, bottom=4), expand=True,
                ))
            else:
                doc_bar.visible = False
            page.update()

        def _clear_doc():
            if cur["busy"]: return
            if cur["session"]:
                cur["session"].active_doc = None; _save()
            _refresh_doc_bar()

        def _stop_generation():
            with _lock:
                if not cur["busy"]: return
                cur["stop_event"].set()
                if cur["spinner"]: cur["spinner"].visible = False
                cur["busy"] = False
            send_btn.disabled = False
            stop_btn.visible  = False
            status_lbl.value  = "Stopped"
            with _lock: md = live_md[0]
            if md: md.value += "\n\n*[Generation stopped]*"
            page.update()

        def _set_busy(busy):
            with _lock: cur["busy"] = busy
            send_btn.disabled = busy
            stop_btn.visible  = busy
            page.update()

        # ── Send ──────────────────────────────────────────────────────────────
        # _send() is the main user-message handler. It:
        #   1. Appends the user bubble to the chat
        #   2. Starts a background thread (_bg) that calls engine.process_message
        #   3. Streams tokens back via _stream callback → Markdown widget
        #   4. On finish: saves the session, refreshes the sidebar, updates status
        #   5. Calls _touch_live_session to reset the idle clock
        def _send(text=None):
            with _lock:
                if cur["busy"] or cur.get("read_only"): return
            raw = (text or input_box.value or "").strip()
            if not raw: return
            input_box.value = ""
            if not cur["session"]: _new_chat()
            cur["session"].add_message("user", raw); _save()
            # Update last-active time for this session in the live registry
            _touch_live_session(session_key)
            chat_col.controls.append(_user_bubble(raw))
            shell, md, spinner = _bot_bubble(thinking=True)
            with _lock:
                cur["spinner"] = spinner
                cur["stop_event"].clear()   # reset any previous stop signal
                live_md[0] = md
            chat_col.controls.append(shell)
            _set_busy(True); status_lbl.value = "Thinking…"; page.update()

            prev_messages = list(cur["session"].messages[:-1])
            last_answer   = ""
            for m in reversed(prev_messages):
                if m["role"] == "assistant":
                    last_answer = m["content"]; break

            history    = prev_messages
            src_filter = cur["session"].active_doc
            target_sid = cur["session"].session_id
            t0, buf    = time.time(), []

            # Capture the stop_event for this specific request so _stop_generation
            # only cancels this user's stream, not any other concurrent request.
            this_stop_event = cur["stop_event"]

            def _stream(chunk):
                """Token callback — called from the background thread for this request."""
                if not chunk: return
                with _lock:
                    stopped = this_stop_event.is_set()
                    md_ref  = live_md[0]
                if stopped: return
                buf.append(chunk)
                # Only update the MD widget if we're still on the same session
                if cur["session"] and cur["session"].session_id == target_sid:
                    if md_ref:
                        md_ref.value = "".join(buf)
                    # page.update() is thread-safe in Flet web mode — each session
                    # has its own WebSocket so this only flushes to this user.
                    page.update()

            def _bg():
                """Background thread: runs process_message, then calls _done on finish."""
                captured_md = None
                with _lock: captured_md = live_md[0]

                def _on_suggestions(suggestions):
                    # suggestions arrive from _BG_POOL after the main answer finishes
                    try:
                        page.run_thread(_append_suggestions_to_answer, suggestions, captured_md)
                    except Exception: pass

                try:
                    answer, ctx_chunks = engine.process_message(
                        raw, history,
                        stream_cb=_stream,
                        followup_cb=_on_suggestions,
                        source_filter=src_filter,
                        last_answer=last_answer,
                        stop_event=this_stop_event,   # per-request cancellation
                    )
                except Exception as ex:
                    answer, ctx_chunks = f"\n\n**Error:** {ex}", []
                    _stream(answer)

                with _lock: stopped = this_stop_event.is_set()
                if not stopped:
                    elapsed  = round(time.time() - t0, 1)
                    full_ans = "".join(buf).strip()
                    tgt = next((s for s in sessions if s.session_id == target_sid), None)
                    if tgt:
                        tgt.messages.append({"role": "assistant", "content": full_ans, "resp_time": elapsed})
                        tgt.updated_at = datetime.now().isoformat(); _save()
                    page.run_thread(_done, elapsed, ctx_chunks)

            def _done(elapsed, ctx_chunks=None):
                with _lock: sp = cur["spinner"]
                if sp: sp.visible = False
                _set_busy(False)
                with _lock: cur["thread"] = None
                status_lbl.value = f"Done in {elapsed}s"
                _refresh_sess_list(); page.update()

            with _lock:
                cur["thread"] = threading.Thread(target=_bg, daemon=True)
                cur["thread"].start()

        # ── Session management ────────────────────────────────────────────────
        def _new_chat():
            if cur["busy"]: return
            s = ChatSession(); sessions.insert(0, s); _save()
            cur["session"] = s; cur["read_only"] = False
            input_area.visible = True
            _render(); _refresh_doc_bar(); _refresh_sess_list()

        def _select(s, read_only=False):
            if cur["busy"]: return
            cur["session"] = s; cur["read_only"] = read_only
            if is_admin and admin_dashboard in page.controls:
                page.controls.clear()
                page.add(ft.Row([sidebar, main_area], spacing=0, expand=True,
                                vertical_alignment=ft.CrossAxisAlignment.STRETCH))
            input_area.visible = not read_only
            _render(); _refresh_doc_bar(); _refresh_sess_list()

        def _delete(sid):
            nonlocal sessions
            sessions = delete_session(sessions, sid)
            if cur["session"] and cur["session"].session_id == sid:
                _new_chat() if not sessions else _select(sessions[0])
            else:
                _refresh_sess_list()

        def _refresh_sess_list():
            sess_col.controls.clear()
            cur_sid   = cur["session"].session_id if cur["session"] else None
            max_items = 999 if is_admin else 5
            shown     = 0
            for label, grp in _group(sessions):
                if shown >= max_items: break
                sess_col.controls.append(ft.Container(
                    content=ft.Text(label, size=10, color=C["dim"], weight=ft.FontWeight.W_600),
                    padding=_po(left=14, top=12, bottom=2),
                ))
                for s in grp:
                    if shown >= max_items: break
                    sel   = s.session_id == cur_sid
                    title = _trunc(s.title or "New Chat")
                    dot   = " ·" if getattr(s, "active_doc", None) else ""
                    sid   = s.session_id
                    row_items = [
                        ft.Text(title + dot, size=12, expand=True,
                                color=C["fg"] if sel else C["fg2"],
                                weight=ft.FontWeight.W_600 if sel else ft.FontWeight.NORMAL,
                                no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                    ]
                    if is_admin:
                        row_items.append(ft.IconButton(
                            icon=ft.Icons.CLOSE, icon_size=12, icon_color=C["dim"],
                            width=26, height=26,
                            on_click=lambda e, s_=sid: _delete(s_),
                        ))
                    sess_col.controls.append(ft.Container(
                        content=ft.Row(row_items, spacing=0),
                        bgcolor=C["accent"] + "22" if sel else "transparent",
                        border_radius=8, padding=_po(left=12, right=2, top=6, bottom=6),
                        margin=_ms(h=6, v=1), on_click=lambda e, s_=s: _select(s_),
                        ink=True, ink_color=C["border"],
                    ))
                    shown += 1
            page.update()

        # ── Upload dialog ─────────────────────────────────────────────────────
        # ── FilePicker upload (Flet 0.85) ────────────────────────────────────
        # pick_files() is async and returns files directly. with_data=True
        # sends file bytes from the browser so this works from any device.
        async def _upload_doc_filepicker():
            if cur.get("busy"): return
            try:
                files = await file_picker.pick_files(
                    dialog_title="Select documents to upload",
                    allow_multiple=True,
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions=["pdf", "txt", "md"],
                    with_data=True,
                )
            except Exception as ex:
                _snack(f"File picker error: {ex}", ok=False)
                return

            if not files:
                return  # User cancelled

            saved_paths = []
            for f_item in files:
                if not f_item.bytes:
                    _snack(f"Could not read '{f_item.name}' — skipping.", ok=False)
                    continue
                dest = os.path.join(UPLOAD_DIR, f_item.name)
                try:
                    with open(dest, "wb") as fh:
                        fh.write(f_item.bytes)
                    saved_paths.append(dest)
                except Exception as ex:
                    _snack(f"Failed to save '{f_item.name}': {ex}", ok=False)

            if saved_paths:
                _handle_uploaded_files(saved_paths)


        # ── Upload dialog widgets (built once, updated in-place) ──────────────
        _prog_bar   = ft.ProgressBar(value=0.0, height=8, color=C["accent"], bgcolor=C["border"])
        _prog_label = ft.Text("Starting…", size=11, color=C["dim"])
        _prog_fname = ft.Text("", size=13, weight=ft.FontWeight.W_600, color=C["fg"],
                              no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS)
        _queue_col  = ft.Column(spacing=6)

        _close_btn  = ft.TextButton("Close",  style=ft.ButtonStyle(color=C["accent"]),
                                    disabled=True, on_click=lambda e: _close_dlg())
        _cancel_btn = ft.TextButton("Cancel", style=ft.ButtonStyle(color=C["red"]),
                                    on_click=lambda e: _cancel_uploads())

        _upload_dlg = ft.AlertDialog(
            title=ft.Text("Uploading Documents", color=C["fg"], weight=ft.FontWeight.BOLD),
            bgcolor=C["card"],
            content=ft.Container(
                content=ft.Column([
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED, size=16, color=C["accent"]),
                                _prog_fname,
                            ], spacing=8),
                            _prog_bar,
                            _prog_label,
                        ], spacing=6, tight=True),
                        bgcolor=C["input"], border_radius=8, padding=ft.Padding.all(12),
                    ),
                    ft.Container(height=4),
                    _queue_col,
                ], spacing=0, tight=True, scroll=ft.ScrollMode.AUTO),
                width=440, height=320,
            ),
            actions=[_cancel_btn, _close_btn],
            open=False,
        )
        page.overlay.append(_upload_dlg)

        def _cancel_uploads(e=None):
            cur["upload_cancelled"] = True
            cur["upload_queue"].clear()
            cur["busy"] = False
            status_lbl.value = "Ready"
            _upload_dlg.open = False
            page.update()

        def _open_upload_dlg():
            _close_btn.disabled  = True
            _cancel_btn.disabled = False
            _upload_dlg.open = True
            page.update()

        def _refresh_upload_dlg():
            _queue_col.controls.clear()
            for path in cur["upload_queue"][1:]:
                _queue_col.controls.append(ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.INSERT_DRIVE_FILE_OUTLINED, size=13, color=C["dim"]),
                        ft.Text(os.path.basename(path), size=12, color=C["fg2"],
                                expand=True, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Text("Queued", size=10, color=C["dim"]),
                    ], spacing=6),
                    bgcolor=C["input"], border_radius=6,
                    padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                ))
            _close_btn.disabled  = cur["busy"]
            _cancel_btn.disabled = not (cur["upload_queue"] or cur["busy"])
            page.update()

        def _tick_upload_progress(done, total, file_name):
            pct = min(1.0, float(done) / float(total)) if total else 0.0
            _prog_bar.value   = pct
            _prog_label.value = f"{done}/{total} chunks indexed ({int(pct*100)}%)"
            page.update()

        def _process_next_upload():
            if not cur["upload_queue"]:
                cur["busy"]      = False
                status_lbl.value = "Ready"
                _close_btn.disabled  = False
                _cancel_btn.disabled = True
                _refresh_sidebar_doc_count()
                _refresh_admin_stats()
                _upload_dlg.open = False
                page.update()
                return

            path      = cur["upload_queue"][0]
            file_name = os.path.basename(path)
            cur["busy"]             = True
            cur["current_status"]   = f"Indexing {file_name}…"
            cur["current_progress"] = 0.0
            _prog_fname.value = file_name
            _prog_bar.value   = 0.0
            _prog_label.value = "Starting…"
            _refresh_upload_dlg()

            def progress_cb(done, total):
                page.run_thread(_tick_upload_progress, done, total, file_name)

            def _bg():
                try:
                    ok, msg = engine.upload_file(path, progress_cb=progress_cb)
                except Exception as e:
                    ok, msg = False, f"Error: {e}"
                try:
                    if path.startswith(UPLOAD_DIR) and os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass

                def on_done():
                    if ok:
                        _prog_bar.value   = 1.0
                        _prog_label.value = f"✓ Done — {msg}"
                        _snack(msg, ok=True)
                    else:
                        _prog_label.value = f"✗ {msg}"
                        _snack(msg, ok=False)
                    page.update()
                    time.sleep(0.6)
                    cur["upload_queue"].pop(0)
                    if cur["upload_queue"] and not cur["upload_cancelled"]:
                        _process_next_upload()
                    else:
                        cur["busy"]             = False
                        cur["upload_cancelled"] = False
                        status_lbl.value        = "Ready"
                        _refresh_sidebar_doc_count()
                        _refresh_admin_stats()
                        _close_btn.disabled  = False
                        _cancel_btn.disabled = True
                        page.update()
                page.run_thread(on_done)

            threading.Thread(target=_bg, daemon=True).start()

        def _handle_uploaded_files(paths):
            new_paths = [p for p in paths if p]
            if not new_paths: return
            cur["upload_cancelled"] = False
            cur["upload_queue"].extend(new_paths)
            if not cur["busy"]:
                _open_upload_dlg()
                _process_next_upload()
            else:
                _refresh_upload_dlg()


        # ── Doc list (View / Focus / Delete) ─────────────────────────────────
        def _show_docs():
            if cur["busy"]: return
            docs = engine.list_documents()
            if not docs: _snack("No documents indexed yet.", ok=False); return

            cur_doc    = cur["session"].active_doc if cur["session"] else None
            doc_items  = sorted(docs.items())
            # Storage per source — proportional estimate from database.py
            storage_by_src = engine.get_storage_by_source() if is_admin else {}
            doc_list_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO)

            def _build_rows(filter_text=""):
                doc_list_col.controls.clear()
                ft_low  = filter_text.strip().lower()
                matched = [(src, cnt) for src, cnt in doc_items if ft_low in src.lower()]
                if not matched:
                    doc_list_col.controls.append(ft.Container(
                        content=ft.Text("No documents match your search.", color=C["dim"],
                                        size=12, text_align=ft.TextAlign.CENTER),
                        padding=_ps(h=12, v=20), alignment=ft.Alignment(0, 0),
                    ))
                else:
                    for src, cnt in matched:
                        act = src == cur_doc; s_ = src
                        mb  = storage_by_src.get(src, 0.0)
                        row_controls = [
                            ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=15,
                                    color=C["tag_fg"] if act else C["dim"]),
                            ft.Text(src, size=13, color=C["fg"], expand=True,
                                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                            # Admin: show MB + chunks; User: show chunks only
                            ft.Text(f"{mb} MB · {cnt} chunks" if is_admin else f"{cnt} chunks",
                                    size=11, color=C["dim"]),
                        ]
                        # Focus button only for regular users — admin doesn't
                        # focus docs since they view all history cross-session
                        if not is_admin:
                            row_controls.append(ft.TextButton(
                                "Unfocus" if act else "Focus",
                                style=ft.ButtonStyle(color=C["dim"] if act else C["accent2"]),
                                on_click=lambda e, s=s_: _focus_doc(s),
                            ))
                        if is_admin:
                            row_controls.append(ft.TextButton(
                                "Delete",
                                style=ft.ButtonStyle(color=C["red"]),
                                on_click=lambda e, s=s_: [
                                    engine.delete_document(s),
                                    _close_dlg(),
                                    _snack(f"Deleted {s}", ok=True),
                                    _refresh_sidebar_doc_count(),
                                ],
                            ))
                        doc_list_col.controls.append(ft.Container(
                            content=ft.Row(row_controls, spacing=6),
                            bgcolor=C["tag_bg"] if act else C["input"],
                            border_radius=8, padding=_ps(h=12, v=8), margin=_po(bottom=4),
                        ))
                page.update()

            def _focus_doc(src):
                if not cur["session"] or cur["busy"]: return
                if cur["session"].active_doc == src:
                    cur["session"].active_doc = None
                else:
                    cur["session"].active_doc = src
                _save()
                _close_dlg()
                _refresh_doc_bar()

            _build_rows()
            search_field = ft.TextField(
                hint_text="Search documents…", hint_style=ft.TextStyle(color=C["dim"]),
                prefix_icon=ft.Icons.SEARCH, border=ft.InputBorder.OUTLINE,
                border_color=C["border"], focused_border_color=C["accent"],
                bgcolor=C["input"], color=C["fg"], text_style=ft.TextStyle(color=C["fg"], size=13),
                content_padding=_ps(h=12, v=8),
                on_change=lambda e: _build_rows(e.control.value),
            )
            subtitle = ("Manage indexed files — storage shown per document"
                        if is_admin else "Choose a file to focus the bot on")
            actions = [
                ft.TextButton("Close", style=ft.ButtonStyle(color=C["accent"]),
                              on_click=lambda e: _close_dlg()),
            ]
            if not is_admin:
                actions.insert(0, ft.TextButton(
                    "Clear Focus", style=ft.ButtonStyle(color=C["dim"]),
                    on_click=lambda e: [_close_dlg(), _clear_doc()],
                ))
            _show_dlg(
                ft.Column([
                    ft.Text("Indexed Documents", color=C["fg"], weight=ft.FontWeight.BOLD),
                    ft.Text(subtitle, color=C["dim"], size=12),
                ], spacing=4),
                ft.Container(
                    content=ft.Column([
                        search_field, ft.Container(height=8),
                        ft.Container(content=doc_list_col,
                                     height=min(56 + len(docs) * 56, 360)),
                    ], spacing=0, tight=True),
                    width=520,
                ),
                actions,
            )

        def _plus_menu():
            container = ft.Container(
                content=ft.Column([
                    ft.Container(
                        content=ft.Text("Options", size=13, weight=ft.FontWeight.W_600, color=C["fg2"]),
                        padding=_po(left=16, top=16, bottom=8),
                    ),
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, color=C["accent"]),
                        title=ft.Text("Focus Document", color=C["fg"]),
                        subtitle=ft.Text("Search only this document", color=C["dim"], size=11),
                        on_click=lambda e: [_close_sheet(bs), _show_docs()],
                    ),
                    ft.Container(height=16),
                ], spacing=0, tight=True),
                bgcolor=C["card"],
            )
            bs = _show_sheet(container)

        # ── Admin: all-user history ───────────────────────────────────────────
        def _show_all_history():
            all_sessions = load_all_sessions(); meta = _load_session_meta()
            history_col  = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO)
            for s in sorted(all_sessions, key=lambda x: x.updated_at or "", reverse=True):
                sid       = s.session_id
                ip_info   = meta.get(sid, {}).get("device") or meta.get(sid, {}).get("ip", "N/A")
                role_info = meta.get(sid, {}).get("role", "user")
                try:    ts = datetime.fromisoformat(s.updated_at).strftime("%Y-%m-%d %H:%M")
                except: ts = s.updated_at or "?"
                history_col.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE_ROUNDED, size=14, color=C["accent"]),
                            ft.Text(_trunc(s.title or "New Chat", 40), size=13, color=C["fg"],
                                    weight=ft.FontWeight.W_600, expand=True, no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                            ft.Container(
                                content=ft.Text(role_info.upper(), size=9, color="#ffffff",
                                                weight=ft.FontWeight.W_600),
                                bgcolor=C["button_accent"] if role_info == "admin" else C["dim"],
                                border_radius=4, padding=_ps(h=6, v=2),
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE_ROUNDED, icon_color=C["red"],
                                icon_size=16, tooltip="Delete Session",
                                on_click=lambda e, sid=sid: [
                                    _delete(sid), _close_dlg(), _show_all_history()
                                ],
                            ),
                        ], spacing=6),
                        ft.Row([
                            ft.Icon(ft.Icons.DEVICES_ROUNDED, size=12, color=C["dim"]),
                            ft.Text(f"Device: {ip_info}", size=11, color=C["dim"]),
                            ft.Container(width=12),
                            ft.Icon(ft.Icons.ACCESS_TIME_ROUNDED, size=12, color=C["dim"]),
                            ft.Text(ts, size=11, color=C["dim"]),
                            ft.Container(width=12),
                            ft.Text(f"{len(s.messages)} msgs", size=11, color=C["dim"]),
                        ], spacing=4),
                    ], spacing=4, tight=True),
                    bgcolor=C["input"], border_radius=8, padding=_ps(h=12, v=8), margin=_po(bottom=4),
                    on_click=lambda e, s_=s: [_close_dlg(), _select(s_, read_only=True)], ink=True,
                ))
            if not history_col.controls:
                history_col.controls.append(
                    ft.Text("No chat history yet.", color=C["dim"], size=13))
            _show_dlg(
                ft.Text("All User History", color=C["fg"], weight=ft.FontWeight.BOLD),
                ft.Container(content=history_col, width=520, height=400),
                [ft.TextButton("Close", style=ft.ButtonStyle(color=C["accent"]),
                               on_click=lambda e: _close_dlg())],
            )

        # ── [NEW-1] Sidebar doc count refresh helper ──────────────────────────
        sidebar_doc_label = [None]   # will be set below

        def _refresh_sidebar_doc_count():
            if sidebar_doc_label[0] is None: return
            try:
                docs    = engine.list_documents()
                n_docs  = len(docs)
                n_chunk = engine.get_chunk_count()
                sidebar_doc_label[0].value = (
                    f"{n_docs} doc{'s' if n_docs != 1 else ''}, {n_chunk} chunks"
                    if n_docs else "No documents indexed"
                )
            except Exception:
                pass
            page.update()
            _refresh_admin_stats()

        # ── Sidebar ───────────────────────────────────────────────────────────
        model_name = engine.get_model_name()
        chunk_n    = engine.get_chunk_count()

        _doc_count_text = ft.Text(
            (f"{len(engine.list_documents())} doc(s), {chunk_n} chunks"
             if chunk_n else "No documents indexed"),
            size=10, color=C["dim"],
        )
        sidebar_doc_label[0] = _doc_count_text

        sidebar_actions = []
        if is_admin:
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([ft.Icon(ft.Icons.DASHBOARD_ROUNDED, size=15, color=C["fg2"]),
                                ft.Text("Admin Dashboard", size=12, color=C["fg2"])], spacing=8),
                on_click=lambda e: [page.controls.clear(), page.add(admin_dashboard), page.update()],
            ))
            # [FIX-A] FilePicker upload button for admin
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED, size=15, color=C["fg2"]),
                                ft.Text("Upload Document", size=12, color=C["fg2"])], spacing=8),
                on_click=lambda e: page.run_task(_upload_doc_filepicker),
            ))
        sidebar_actions.append(ft.TextButton(
            content=ft.Row([ft.Icon(ft.Icons.FOLDER_OPEN_ROUNDED, size=15, color=C["fg2"]),
                            ft.Text("View Documents", size=12, color=C["fg2"])], spacing=8),
            on_click=lambda e: _show_docs(),
        ))
        if is_admin:
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([ft.Icon(ft.Icons.HISTORY_ROUNDED, size=15, color=C["button_accent"]),
                                ft.Text("All User History", size=12, color=C["button_accent"])], spacing=8),
                on_click=lambda e: _show_all_history(),
            ))

        role_badge = ft.Container(
            content=ft.Text("ADMIN" if is_admin else "USER", size=9, color="#ffffff",
                            weight=ft.FontWeight.W_600),
            bgcolor=C["button_accent"] if is_admin else C["accent"],
            border_radius=4, padding=_ps(h=6, v=2),
        )

        # ── Admin Dashboard — live stat refs ──────────────────────────────────
        # Stat Text widgets stored so _refresh_admin_stats() can update in-place
        _stat_docs_val    = ft.Text("0",      size=28, weight=ft.FontWeight.BOLD, color=C["fg"])
        _stat_storage_val = ft.Text("0 MB",   size=28, weight=ft.FontWeight.BOLD, color=C["fg"])
        _stat_sess_val    = ft.Text("0",      size=28, weight=ft.FontWeight.BOLD, color=C["fg"])
        _stat_users_val   = ft.Text("0",      size=28, weight=ft.FontWeight.BOLD, color=C["fg"])

        def _refresh_admin_stats():
            """Update dashboard stat cards in-place — no full rebuild needed."""
            try:
                docs      = engine.list_documents()
                n_docs    = len(docs)
                storage   = engine.get_storage_mb()
                n_sess    = len(load_all_sessions())
                n_users   = len(_get_live_sessions_snapshot())
                _stat_docs_val.value    = str(n_docs)
                _stat_storage_val.value = f"{storage} MB"
                _stat_sess_val.value    = str(n_sess)
                _stat_users_val.value   = str(n_users)
                page.update()
            except Exception:
                pass

        # ── Active Users modal ────────────────────────────────────────────────
        def _show_active_users():
            """
            Show a live list of all connected sessions with:
              - Device name / IP
              - Role badge
              - Time since login (session duration)
              - Time since last message (idle time)
              - Kick button to force-logout that user
            Refreshes every 5 seconds while open via a background ticker.
            """
            users_col   = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)
            ticker_stop = threading.Event()

            def _build_user_rows():
                users_col.controls.clear()
                now      = datetime.now()
                sessions = _get_live_sessions_snapshot()
                if not sessions:
                    users_col.controls.append(
                        ft.Text("No active sessions.", color=C["dim"], size=13))
                for s in sessions:
                    is_self = s["key"] == session_key   # don't show kick for self
                    online_dur  = _fmt_duration((now - s["login_time"]).total_seconds())
                    idle_dur    = _fmt_duration((now - s["last_active"]).total_seconds())
                    role_color  = C["button_accent"] if s["role"] == "admin" else C["accent"]

                    kick_btn = ft.TextButton(
                        "Kick",
                        style=ft.ButtonStyle(color=C["red"]),
                        disabled=is_self,
                        tooltip="Force logout this user" if not is_self else "Cannot kick yourself",
                        on_click=lambda e, fn=s["kick_fn"]: [
                            fn(),
                            _close_dlg(),
                            _snack("User kicked.", ok=True),
                            _refresh_admin_stats(),
                        ],
                    )
                    users_col.controls.append(ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.COMPUTER_ROUNDED, size=16, color=C["dim"]),
                                ft.Text(s["device"], size=13, color=C["fg"],
                                        weight=ft.FontWeight.W_600, expand=True),
                                ft.Container(
                                    content=ft.Text(s["role"].upper(), size=9,
                                                    color="#ffffff", weight=ft.FontWeight.W_600),
                                    bgcolor=role_color, border_radius=4, padding=_ps(h=6, v=2),
                                ),
                                kick_btn,
                            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Row([
                                ft.Icon(ft.Icons.LOGIN_ROUNDED, size=12, color=C["dim"]),
                                ft.Text(f"Online: {online_dur}", size=11, color=C["dim"]),
                                ft.Container(width=16),
                                ft.Icon(ft.Icons.ACCESS_TIME_ROUNDED, size=12, color=C["dim"]),
                                ft.Text(f"Idle: {idle_dur}", size=11,
                                        color=C["red"] if (now - s["last_active"]).total_seconds() > 300
                                        else C["dim"]),
                            ], spacing=4),
                        ], spacing=4, tight=True),
                        bgcolor=C["tag_bg"] if is_self else C["input"],
                        border_radius=8, padding=_ps(h=12, v=8),
                    ))
                try:
                    page.update()
                except Exception:
                    ticker_stop.set()

            _build_user_rows()

            def _ticker():
                """Refresh the user list every 5 s while the dialog is open."""
                while not ticker_stop.is_set():
                    time.sleep(5)
                    if not ticker_stop.is_set():
                        try:
                            page.run_thread(_build_user_rows)
                        except Exception:
                            break

            threading.Thread(target=_ticker, daemon=True).start()

            _show_dlg(
                ft.Column([
                    ft.Text("Active Sessions", color=C["fg"], weight=ft.FontWeight.BOLD),
                    ft.Text("Live view — refreshes every 5 s", color=C["dim"], size=11),
                ], spacing=2),
                ft.Container(content=users_col, width=480, height=360),
                [ft.TextButton("Close", style=ft.ButtonStyle(color=C["accent"]),
                               on_click=lambda e: [ticker_stop.set(), _close_dlg()])],
            )

        def _build_admin_dashboard():
            # Seed initial values
            try:
                docs    = engine.list_documents()
                n_docs  = len(docs)
                storage = engine.get_storage_mb()
                n_sess  = len(load_all_sessions())
                n_users = len(_get_live_sessions_snapshot())
            except Exception:
                n_docs = n_sess = n_users = 0; storage = 0.0
            _stat_docs_val.value    = str(n_docs)
            _stat_storage_val.value = f"{storage} MB"
            _stat_sess_val.value    = str(n_sess)
            _stat_users_val.value   = str(n_users)

            # Auto-refresh Active Users count every 10 s
            def _stat_auto_refresh():
                while True:
                    time.sleep(10)
                    try:
                        page.run_thread(_refresh_admin_stats)
                    except Exception:
                        break
            threading.Thread(target=_stat_auto_refresh, daemon=True).start()

            def _stat_card(icon, color, label, val_widget, on_click=None):
                return ft.Card(content=ft.Container(
                    content=ft.Column([
                        ft.Icon(icon, size=40, color=color),
                        val_widget,
                        ft.Text(label, size=12, color=C["dim"]),
                    ], alignment=ft.MainAxisAlignment.CENTER,
                       horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                    padding=28, width=180, height=160,
                    on_click=on_click, ink=bool(on_click),
                ))

            def _action_card(icon, color, label, sub, on_click):
                return ft.Card(content=ft.Container(
                    content=ft.Column([
                        ft.Icon(icon, size=40, color=color),
                        ft.Text(label, size=15, weight=ft.FontWeight.W_600, color=C["fg"]),
                        ft.Text(sub, size=11, color=C["dim"]),
                    ], alignment=ft.MainAxisAlignment.CENTER,
                       horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6),
                    padding=28, width=200, height=160,
                    on_click=on_click, ink=True,
                ))

            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Container(expand=True),
                        ft.IconButton(icon=ft.Icons.LOGOUT_ROUNDED, icon_color=C["red"],
                                      icon_size=22, tooltip="Logout", on_click=lambda e: _logout()),
                    ]),
                    ft.Image(src=logo_src, width=180, height=60, fit=ft.BoxFit.CONTAIN)
                    if logo_src else ft.Text("BDL CHATBOT", size=28, weight=ft.FontWeight.BOLD, color=C["accent"]),
                    ft.Text("Admin Dashboard", size=22, weight=ft.FontWeight.BOLD, color=C["fg"]),
                    ft.Container(height=20),
                    ft.Row([
                        # Clicking Documents → doc manager; clicking Storage → doc manager (shows sizes)
                        _stat_card(ft.Icons.DESCRIPTION_ROUNDED,  C["accent"],        "Documents",     _stat_docs_val,    on_click=lambda e: _show_docs()),
                        _stat_card(ft.Icons.STORAGE_ROUNDED,       C["green"],         "Storage",       _stat_storage_val, on_click=lambda e: _show_docs()),
                        _stat_card(ft.Icons.CHAT_BUBBLE_ROUNDED,   C["button_accent"], "Sessions",      _stat_sess_val,    on_click=lambda e: _show_all_history()),
                        # Clicking Active Users → live user panel with kick
                        _stat_card(ft.Icons.PEOPLE_ROUNDED,        C["dim"],           "Active Users",  _stat_users_val,   on_click=lambda e: _show_active_users()),
                    ], spacing=16, alignment=ft.MainAxisAlignment.CENTER),
                    ft.Container(height=24),
                    ft.Row([
                        _action_card(ft.Icons.UPLOAD_FILE_ROUNDED,  C["button_accent"], "Upload Document", "Add new files via file picker",    lambda e: page.run_task(_upload_doc_filepicker)),
                        _action_card(ft.Icons.FOLDER_OPEN_ROUNDED,  C["accent"],        "Manage Documents","View, rename, delete indexed docs", lambda e: _show_docs()),
                        _action_card(ft.Icons.HISTORY_ROUNDED,      C["green"],         "User History",    "Browse all chat sessions",          lambda e: _show_all_history()),
                    ], spacing=20, alignment=ft.MainAxisAlignment.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                   alignment=ft.MainAxisAlignment.CENTER),
                alignment=ft.Alignment(0, 0), expand=True, bgcolor=C["bg"],
            )

        admin_dashboard = _build_admin_dashboard()

        sidebar = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Image(src=logo_src, width=100, height=34, fit=ft.BoxFit.CONTAIN)
                            if logo_src else ft.Text("BDL CHATBOT", size=14, weight=ft.FontWeight.BOLD, color=C["accent"]),
                            role_badge,
                            ft.Container(expand=True),
                            ft.IconButton(icon=ft.Icons.CHAT_BUBBLE_ROUNDED, icon_color=C["button_accent"],
                                          icon_size=20, tooltip="New Chat", on_click=lambda e: _new_chat()),
                            ft.IconButton(icon=ft.Icons.LOGOUT_ROUNDED, icon_color=C["red"],
                                          icon_size=20, tooltip="Logout", on_click=lambda e: _logout()),
                        ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        ft.Text(model_name, size=10, color=C["dim"]),
                        _doc_count_text,
                    ], spacing=2),
                    padding=_po(left=16, right=8, top=16, bottom=10),
                ),
                ft.Divider(height=1, color=C["border"]),
                ft.Container(content=sess_col, expand=True),
                ft.Divider(height=1, color=C["border"]),
                ft.Container(content=ft.Column(sidebar_actions, spacing=0), padding=_ps(h=8, v=8)),
            ], spacing=0, expand=True),
            bgcolor=C["sidebar"], width=260,
            border=ft.Border(right=ft.BorderSide(1, C["border"])),
        )

        # ── Main area ─────────────────────────────────────────────────────────
        input_area = ft.Column([
            ft.Container(
                content=ft.Row([
                    ft.IconButton(
                        icon=ft.Icons.ADD_CIRCLE_OUTLINE_ROUNDED, icon_color=C["dim"],
                        icon_size=22, tooltip="Attach / Options",
                        on_click=lambda e: _plus_menu(),
                    ),
                    ft.Container(
                        content=input_box, bgcolor=C["input"],
                        border_radius=14, border=ft.Border(left=ft.BorderSide(1, C["border"]), right=ft.BorderSide(1, C["border"]), top=ft.BorderSide(1, C["border"]), bottom=ft.BorderSide(1, C["border"])), expand=True,
                    ),
                    send_btn,
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.END),
                bgcolor=C["bg"], padding=_po(left=12, right=12, top=8, bottom=8),
            ),
            ft.Container(
                content=ft.Row([status_lbl, ft.Container(width=12), stop_btn], spacing=0),
                bgcolor=C["bg"], padding=_po(left=20, bottom=6),
            ),
        ], spacing=0, visible=True)

        main_area = ft.Column([
            ft.Container(content=chat_col, bgcolor=C["bg"], expand=True),
            ft.Container(content=doc_bar, bgcolor=C["bg"],
                         padding=_po(left=16, right=16, top=4, bottom=0)),
            input_area,
        ], spacing=0, expand=True)

        send_btn.on_click = lambda e: _send()

        if is_admin:
            page.add(admin_dashboard)
        else:
            page.add(ft.Row([sidebar, main_area], spacing=0, expand=True,
                            vertical_alignment=ft.CrossAxisAlignment.STRETCH))
            _new_chat()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BDL CHATBOT — Multi-User Local Server")
    parser.add_argument("--desktop", action="store_true", help="Run as desktop app (single user)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=8550, help="Port number (default: 8550)")
    args = parser.parse_args()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if args.desktop:
            print("\n  BDL CHATBOT — desktop mode (single user)\n")
            ft.app(target=main)
        else:
            import socket as _sock
            print()
            print("  ╔═══════════════════════════════════════════════════════╗")
            print("  ║         BDL CHATBOT — Multi-User Local Server        ║")
            print("  ╠═══════════════════════════════════════════════════════╣")
            print(f"  ║  Max Users: {MAX_USERS}  |  Max Admins: {MAX_ADMINS}                      ║")
            print(f"  ║  Port:      {args.port}                                    ║")
            print("  ╠═══════════════════════════════════════════════════════╣")
            print(f"  ║  Local:   http://localhost:{args.port}                    ║")
            try:
                ip = _sock.gethostbyname(_sock.gethostname())
                print(f"  ║  Network: http://{ip}:{args.port}{' ' * (24 - len(ip))}║")
            except Exception:
                pass
            print("  ╚═══════════════════════════════════════════════════════╝")
            print()
            print("  Share the Network URL with office colleagues to connect.")
            print("  Press Ctrl+C to stop the server.\n")
            ft.app(
                target=main,
                view=ft.AppView.WEB_BROWSER,
                host=args.host,
                port=args.port,
                upload_dir=UPLOAD_DIR,   # Flet writes browser-picked files here
                
            )