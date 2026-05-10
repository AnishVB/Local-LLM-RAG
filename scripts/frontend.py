"""
frontend.py — BDL CHATBOT  (Flet 0.85)
Run:  python scripts/frontend.py

MULTI-USER LOCAL SERVER:
  - Supports up to 5 concurrent users + 1 admin
  - Per-session auth (each browser tab has its own role)
  - Shared ChatEngine singleton for efficiency
  - Thread-safe file operations
  - File upload works in web mode (Flet FilePicker)
"""
import os, sys, re, time, threading, base64, socket, json as _json, shutil
from datetime import datetime, date, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.normpath(os.path.join(current_dir, "..", "assests", "BDL logo nobg.png"))

# Upload staging directory for web-mode file uploads
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
MAX_USERS = 5
MAX_ADMINS = 1
_conn_lock = threading.Lock()
_active_users = 0    # current user sessions
_active_admins = 0   # current admin sessions

def _acquire_slot(role):
    """Try to acquire a connection slot. Returns True if allowed."""
    global _active_users, _active_admins
    with _conn_lock:
        if role == "admin":
            if _active_admins >= MAX_ADMINS:
                return False
            _active_admins += 1
        else:
            if _active_users >= MAX_USERS:
                return False
            _active_users += 1
    return True

def _release_slot(role):
    """Release a connection slot when a user disconnects."""
    global _active_users, _active_admins
    with _conn_lock:
        if role == "admin":
            _active_admins = max(0, _active_admins - 1)
        elif role == "user":
            _active_users = max(0, _active_users - 1)

# ── Shared ChatEngine singleton ───────────────────────────────────────────────
_engine_lock = threading.Lock()
_shared_engine = None

def _get_engine():
    """Get or create the shared ChatEngine singleton (thread-safe)."""
    global _shared_engine
    with _engine_lock:
        if _shared_engine is None:
            _shared_engine = ChatEngine()
        return _shared_engine

_SESSION_META_PATH = os.path.normpath(os.path.join(current_dir, "..", "chat_history", "session_meta.json"))
_meta_lock = threading.Lock()

def _load_session_meta():
    with _meta_lock:
        if os.path.isfile(_SESSION_META_PATH):
            try:
                with open(_SESSION_META_PATH, "r", encoding="utf-8") as f: return _json.load(f)
            except Exception: pass
        return {}

def _save_session_meta(meta):
    with _meta_lock:
        os.makedirs(os.path.dirname(_SESSION_META_PATH), exist_ok=True)
        with open(_SESSION_META_PATH, "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception: return "127.0.0.1"

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
    """
    Issue 3: Format inline suggestion block appended to bot answer markdown.
    Returns markdown string, or empty string if no suggestions.
    """
    if not suggestions:
        return ""
    lines = ["\n\n---\n**Want to explore further?**"]
    for sg in suggestions:
        lines.append(f"- {sg}")
    return "\n".join(lines)


async def main(page: ft.Page):
    page.title      = "BDL CHATBOT"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor    = C["bg"]
    page.window.width, page.window.height = 1120, 720
    page.window.min_width, page.window.min_height = 800, 480
    page.padding    = 0

    # Per-page session state (each browser tab has its own)
    page_role = [None]   # "user" or "admin" — stored in list for closure mutability
    user_ip   = _get_local_ip()

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
                admin_error.visible = True; page.update()
                return
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

    # Release connection slot on page disconnect/close
    def _on_disconnect(e=None):
        if page_role[0]:
            _release_slot(page_role[0])
            page_role[0] = None
    page.on_disconnect = _on_disconnect

    def _logout(e=None):
        if page_role[0]:
            _release_slot(page_role[0])
            page_role[0] = None
        admin_pwd_field.value = ""; admin_section.visible = False; admin_error.visible = False
        page.controls.clear(); page.add(login_view); page.update()

    def _launch_chat():
        page.controls.clear(); page.update(); _build_chat_ui()

    def _snack_global(text, ok=True):
        sb = ft.SnackBar(ft.Text(text, color=C["green"] if ok else C["red"]), bgcolor=C["card"], open=True)
        page.overlay.append(sb); page.update()

    def _build_chat_ui():
        is_admin = page_role[0] == "admin"
        engine   = _get_engine()

        # ── Issue 1: Purge empty sessions older than 1 hour on chat launch ────
        sessions_raw = load_all_sessions()
        sessions     = purge_empty_sessions(sessions_raw)
        if len(sessions) != len(sessions_raw):
            save_all_sessions(sessions)

        cur      = {"session": None, "busy": False, "thread": None, "spinner": None, "read_only": False,
                    "upload_queue": [], "upload_cancelled": False,
                    "stop_event": threading.Event()}
        live_md  = [None]   # active markdown widget for the current streaming response

        def _save():
            save_all_sessions(sessions)
            if cur["session"]:
                sid = cur["session"].session_id
                meta = _load_session_meta()
                if sid not in meta:
                    meta[sid] = {"ip": user_ip, "role": page_role[0] or "user"}
                    _save_session_meta(meta)

        _lock = threading.Lock()

        # ── Issue 1: Background periodic purge (every 10 min) ─────────────────
        def _periodic_purge():
            while True:
                time.sleep(600)   # 10 minutes
                try:
                    all_s  = load_all_sessions()
                    clean  = purge_empty_sessions(all_s)
                    if len(clean) != len(all_s):
                        save_all_sessions(clean)
                        # Refresh local sessions list safely
                        kept_ids = {s.session_id for s in clean}
                        to_remove = [s for s in sessions if s.session_id not in kept_ids]
                        for s in to_remove:
                            sessions.remove(s)
                        page.run_thread(_refresh_sess_list)
                except Exception:
                    pass

        threading.Thread(target=_periodic_purge, daemon=True).start()


        def _show_dlg(dlg): page.show_dialog(dlg)
        def _close_dlg():   page.pop_dialog()

        def _snack(text, ok=True):
            sb = ft.SnackBar(ft.Text(text, color=C["green"] if ok else C["red"]), bgcolor=C["card"], open=True)
            page.overlay.append(sb); page.update()

        def _show_sheet(content_widget):
            bs = ft.BottomSheet(content=content_widget, open=True)
            page.overlay.append(bs); page.update(); return bs

        def _close_sheet(bs): bs.open = False; page.update()

        # ── Chat widgets ──────────────────────────────────────────────────────
        chat_col  = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO, expand=True, auto_scroll=True)
        # Issue 3: followup_row removed — no chip row widget
        doc_bar   = ft.Row(spacing=0, visible=False)
        status_lbl = ft.Text("Ready", size=11, color=C["dim"])
        upload_status_text = ft.Text("", size=11, color=C["dim"], visible=False)
        upload_queue_col = ft.Column(spacing=8, visible=False)
        upload_dialog = [None]
        upload_close_btn = [None]
        upload_cancel_btn = [None]
        stop_btn  = ft.IconButton(icon=ft.Icons.STOP_CIRCLE_OUTLINED, icon_color=C["red"],
                                  icon_size=18, tooltip="Stop generation", visible=False,
                                  on_click=lambda e: _stop_generation())

        input_box = ft.TextField(
            hint_text="Ask anything about your documents…",
            hint_style=ft.TextStyle(color=C["dim"]),
            border=ft.InputBorder.NONE, bgcolor=C["input"], color=C["fg"],
            text_style=ft.TextStyle(color=C["fg"], size=13),
            multiline=True, min_lines=1, max_lines=5, expand=True,
            content_padding=_ps(h=16, v=12), cursor_color=C["accent"],
            shift_enter=True, on_submit=lambda e: _send(),
        )
        send_btn = ft.IconButton(icon=ft.Icons.SEND_ROUNDED, icon_color=C["button_accent"],
                                 icon_size=22, tooltip="Send")
        sess_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO, expand=True)

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
                        border=ft.Border.all(1, C["border"]),
                    ),
                    ft.Container(expand=True),
                ]),
                padding=_po(left=16, right=16, top=4, bottom=4),
            )
            return shell, md, spinner

        # ── Render session ────────────────────────────────────────────────────
        def _render():
            chat_col.controls.clear()
            # Issue 3: no followup_row to clear
            s = cur["session"]
            if not s or not s.messages:
                chat_col.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Text("BDL CHATBOT", size=22, weight=ft.FontWeight.BOLD, color=C["accent"]),
                        ft.Text("Upload documents via the sidebar,\nthen ask anything about them.",
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
            """
            Issue 3: Append inline suggestion block to the bot's markdown answer widget.
            Called from background thread via page.run_thread after streaming completes.
            """
            block = _format_suggestions_block(suggestions)
            if block and md_widget:
                md_widget.value = (md_widget.value or "") + block
                page.update()

        def _refresh_doc_bar():
            doc_bar.controls.clear()
            s = cur["session"]
            if s and s.active_doc:
                doc_bar.visible = True
                doc_bar.controls.append(ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=14, color=C["tag_fg"]),
                        ft.Text(f"  {s.active_doc}", size=12, color=C["tag_fg"], expand=True),
                        ft.IconButton(icon=ft.Icons.CLOSE, icon_size=14, icon_color=C["dim"],
                                      width=26, height=26, on_click=lambda e: _clear_doc()),
                    ], spacing=0),
                    bgcolor=C["tag_bg"], border_radius=8,
                    padding=_po(left=10, right=4, top=4, bottom=4), expand=True,
                ))
            else:
                doc_bar.visible = False
            page.update()

        def _clear_doc():
            if cur["busy"]: return
            if cur["session"]: cur["session"].active_doc = None; _save()
            _refresh_doc_bar()

        def _stop_generation():
            with _lock:
                if not cur["busy"]:
                    return
                cur["stop_event"].set()
                if cur["spinner"]:
                    cur["spinner"].visible = False
                cur["busy"] = False
            send_btn.disabled = False
            stop_btn.visible  = False
            status_lbl.value  = "Stopped"
            with _lock:
                md = live_md[0]
            if md:
                md.value += "\n\n*[Generation stopped]*"
            page.update()

        def _set_busy(busy):
            with _lock:
                cur["busy"] = busy
            send_btn.disabled = busy
            stop_btn.visible  = busy
            page.update()

        # ── Send message ──────────────────────────────────────────────────────
        def _send(text=None):
            with _lock:
                if cur["busy"] or cur.get("read_only"):
                    return
            raw = (text or input_box.value or "").strip()
            if not raw: return
            input_box.value = ""
            if not cur["session"]: _new_chat()
            cur["session"].add_message("user", raw); _save()
            chat_col.controls.append(_user_bubble(raw))
            shell, md, spinner = _bot_bubble(thinking=True)
            with _lock:
                cur["spinner"] = spinner
                cur["stop_event"].clear()
                live_md[0] = md
            chat_col.controls.append(shell)
            _set_busy(True); status_lbl.value = "Thinking…"; page.update()

            prev_messages = list(cur["session"].messages[:-1])
            last_answer = ""
            for m in reversed(prev_messages):
                if m["role"] == "assistant":
                    last_answer = m["content"]
                    break

            history       = prev_messages
            src_filter    = cur["session"].active_doc
            target_sid    = cur["session"].session_id
            t0, buf       = time.time(), []

            def _stream(chunk):
                if not chunk: return
                with _lock:
                    stopped = cur["stop_event"].is_set()
                    md_ref  = live_md[0]
                if stopped:
                    return
                buf.append(chunk)
                if cur["session"] and cur["session"].session_id == target_sid:
                    if md_ref:
                        md_ref.value = "".join(buf)
                    page.update()

            def _bg():
                captured_md = None
                with _lock:
                    captured_md = live_md[0]

                def _on_suggestions(suggestions):
                    """Issue 3: append suggestions inline after streaming is done."""
                    try:
                        if page.session and page.session.connection:
                            page.run_thread(_append_suggestions_to_answer, suggestions, captured_md)
                    except Exception:
                        pass  # Page disconnected, ignore

                try:
                    answer, ctx_chunks = engine.process_message(
                        raw, history, stream_cb=_stream,
                        followup_cb=_on_suggestions,
                        source_filter=src_filter,
                        last_answer=last_answer,
                    )
                except Exception as ex:
                    answer, ctx_chunks = f"\n\n**Error:** {ex}", []
                    _stream(answer)
                with _lock:
                    stopped = cur["stop_event"].is_set()
                if not stopped:
                    elapsed  = round(time.time() - t0, 1)
                    full_ans = "".join(buf).strip()
                    tgt = next((s for s in sessions if s.session_id == target_sid), None)
                    if tgt:
                        # Store the answer WITHOUT the suggestion block in session history
                        # so that follow-up prompts don't see "Want to explore further?" as answer content
                        tgt.messages.append({"role": "assistant", "content": full_ans, "resp_time": elapsed})
                        tgt.updated_at = datetime.now().isoformat(); _save()
                    page.run_thread(_done, elapsed, ctx_chunks)

            def _done(elapsed, ctx_chunks=None):
                with _lock:
                    sp = cur["spinner"]
                if sp: sp.visible = False
                _set_busy(False)
                with _lock:
                    cur["thread"] = None
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
                        row_items.append(ft.IconButton(icon=ft.Icons.CLOSE, icon_size=12,
                            icon_color=C["dim"], width=26, height=26,
                            on_click=lambda e, s_=sid: _delete(s_)))
                    sess_col.controls.append(ft.Container(
                        content=ft.Row(row_items, spacing=0),
                        bgcolor=C["accent"] + "22" if sel else "transparent",
                        border_radius=8, padding=_po(left=12, right=2, top=6, bottom=6),
                        margin=_ms(h=6, v=1), on_click=lambda e, s_=s: _select(s_),
                        ink=True, ink_color=C["border"],
                    ))
                    shown += 1
            page.update()

        # ── Document management ───────────────────────────────────────────────
        def _cancel_uploads(e=None):
            cur["upload_cancelled"] = True
            cur["upload_queue"].clear()
            cur["busy"] = False
            status_lbl.value = "Ready"
            _update_upload_dialog()
            _close_dlg()

        _upload_progress_bar  = [None]
        _upload_status_label  = [None]

        def _show_upload_dialog():
            upload_close_btn[0] = ft.TextButton(
                "Close",
                style=ft.ButtonStyle(color=C["accent"]),
                disabled=True,
                on_click=lambda e: _close_dlg(),
            )
            upload_cancel_btn[0] = ft.TextButton(
                "Cancel",
                style=ft.ButtonStyle(color=C["red"]),
                on_click=_cancel_uploads,
            )
            upload_dialog[0] = ft.AlertDialog(
                title=ft.Text("Upload progress", color=C["fg"], weight=ft.FontWeight.BOLD),
                bgcolor=C["card"],
                content=ft.Container(
                    content=upload_queue_col,
                    height=300,
                    width=400,
                ),
                actions=[upload_cancel_btn[0], upload_close_btn[0]],
            )
            page.show_dialog(upload_dialog[0])
            page.update()

        def _update_upload_dialog():
            upload_queue_col.controls.clear()
            _upload_progress_bar[0] = None
            _upload_status_label[0] = None

            if cur["upload_queue"] or cur["busy"]:
                for i, path in enumerate(cur["upload_queue"]):
                    file_name = os.path.basename(path)
                    if i == 0 and cur["busy"]:
                        bar = ft.ProgressBar(value=cur.get("current_progress", 0.0), height=6)
                        lbl = ft.Text(cur.get("current_status", f"Indexing {file_name}…"), size=10, color=C["dim"])
                        _upload_progress_bar[0] = bar
                        _upload_status_label[0] = lbl
                    else:
                        bar = ft.ProgressBar(value=0.0, height=6)
                        lbl = ft.Text("Queued" if i > 0 else "Starting…", size=10, color=C["dim"])
                    card = ft.Container(
                        content=ft.Column([
                            ft.Text(file_name, size=12, weight=ft.FontWeight.W_600, color=C["fg"]),
                            bar, lbl,
                        ], spacing=4, tight=True),
                        bgcolor=C["input"], border_radius=8, padding=ft.padding.all(8),
                    )
                    upload_queue_col.controls.append(card)
                upload_queue_col.visible = True
            else:
                upload_queue_col.visible = False

            if upload_close_btn[0] is not None:
                upload_close_btn[0].disabled = cur["busy"]
            if upload_cancel_btn[0] is not None:
                upload_cancel_btn[0].disabled = (not cur["upload_queue"] and not cur["busy"])

            if upload_queue_col.visible:
                _show_upload_dialog()
            else:
                _close_dlg()

        def _tick_upload_progress(done, total, file_name):
            pct  = min(1.0, float(done) / float(total)) if total else 0.0
            text = f"{file_name}: {done}/{total} chunks"
            if _upload_progress_bar[0] is not None:
                _upload_progress_bar[0].value = pct
            if _upload_status_label[0] is not None:
                _upload_status_label[0].value = text
            page.update()

        def _process_next_upload():
            if not cur["upload_queue"]:
                cur["busy"] = False
                status_lbl.value = "Ready"
                _close_dlg()
                page.update()
                return

            path = cur["upload_queue"][0]
            file_name = os.path.basename(path)
            cur["busy"] = True
            cur["current_status"] = f"Indexing {file_name}…"
            cur["current_progress"] = 0.0
            _update_upload_dialog()

            def progress_cb(done, total):
                page.run_thread(_tick_upload_progress, done, total, file_name)

            def _bg():
                try:
                    ok, msg = engine.upload_file(path, progress_cb=progress_cb)
                except Exception as e:
                    ok, msg = False, f"Error: {e}"
                def on_done():
                    if ok: _snack(msg, ok=True)
                    else:  _snack(msg, ok=False)
                    cur["upload_queue"].pop(0)
                    if cur["upload_queue"] and not cur["upload_cancelled"]:
                        _process_next_upload()
                    else:
                        cur["busy"] = False
                        cur["upload_cancelled"] = False
                        status_lbl.value = "Ready"
                        _update_upload_dialog()
                        page.update()
                page.run_thread(on_done)
            threading.Thread(target=_bg, daemon=True).start()

        def _handle_uploaded_files(paths):
            """Called after file paths are submitted."""
            new_paths = [p for p in paths if p]
            if not new_paths: return
            cur["upload_cancelled"] = False
            cur["upload_queue"].extend(new_paths)
            _update_upload_dialog()
            if not cur["busy"]:
                _process_next_upload()

        def _upload_doc():
            """Show a dialog for entering file path(s) on the server."""
            path_field = ft.TextField(
                hint_text="e.g. C:\\Documents\\report.pdf",
                hint_style=ft.TextStyle(color=C["dim"]),
                border_color=C["border"], focused_border_color=C["accent"],
                bgcolor=C["input"], color=C["fg"],
                text_style=ft.TextStyle(color=C["fg"], size=13),
                content_padding=_ps(h=14, v=10),
                expand=True, autofocus=True,
            )
            error_txt = ft.Text("", color=C["red"], size=12, visible=False)

            def _submit(e):
                raw = (path_field.value or "").strip()
                if not raw:
                    error_txt.value = "Please enter a file path"
                    error_txt.visible = True; page.update(); return
                # Support multiple paths separated by semicolons
                paths = [p.strip().strip('"') for p in raw.split(";") if p.strip()]
                valid_paths = []
                for p in paths:
                    if not os.path.isfile(p):
                        error_txt.value = f"File not found: {os.path.basename(p)}"
                        error_txt.visible = True; page.update(); return
                    valid_paths.append(p)
                _close_dlg()
                _handle_uploaded_files(valid_paths)

            dlg = ft.AlertDialog(
                title=ft.Text("Upload Document", size=16, weight=ft.FontWeight.BOLD, color=C["fg"]),
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Enter the full file path on the server.", size=13, color=C["dim"]),
                        ft.Text("Separate multiple files with semicolons (;)", size=11, color=C["dim"]),
                        ft.Container(height=8),
                        path_field,
                        error_txt,
                        ft.Text("Supported: .pdf, .txt, .md", size=11, color=C["dim"]),
                    ], spacing=4, tight=True),
                    width=450,
                ),
                actions=[
                    ft.TextButton("Cancel", on_click=lambda e: _close_dlg()),
                    ft.FilledButton(
                        content=ft.Text("Upload", color="#ffffff", size=13),
                        style=ft.ButtonStyle(bgcolor=C["accent"], shape=ft.RoundedRectangleBorder(radius=8)),
                        on_click=_submit,
                    ),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            path_field.on_submit = _submit
            _show_dlg(dlg)

        def _show_docs():
            if cur["busy"]: return
            docs = engine.list_documents()
            if not docs: _snack("No documents indexed yet.", ok=False); return

            cur_doc    = cur["session"].active_doc if cur["session"] else None
            doc_items  = sorted(docs.items())
            doc_list_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO)

            def _build_rows(filter_text=""):
                doc_list_col.controls.clear()
                ft_low   = filter_text.strip().lower()
                matched  = [(src, cnt) for src, cnt in doc_items if ft_low in src.lower()]
                if not matched:
                    doc_list_col.controls.append(ft.Container(
                        content=ft.Text("No documents match your search.", color=C["dim"], size=12,
                                        text_align=ft.TextAlign.CENTER),
                        padding=_ps(h=12, v=20), alignment=ft.Alignment(0, 0),
                    ))
                else:
                    for src, cnt in matched:
                        act = src == cur_doc; s_ = src
                        row_controls = [
                            ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=15, color=C["tag_fg"] if act else C["dim"]),
                            ft.Text(src, size=13, color=C["fg"], expand=True, no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                            ft.Text(f"{cnt} chunks", size=11, color=C["dim"]),
                        ]
                        if not is_admin:
                            row_controls.append(ft.TextButton("Focus",
                                style=ft.ButtonStyle(color=C["accent2"]),
                                on_click=lambda e, s=s_: [
                                    setattr(cur["session"], "active_doc", s) if (cur["session"] and not cur["busy"]) else None,
                                    _save(), _refresh_doc_bar(), _close_dlg(),
                                ] if not cur["busy"] else None))
                        else:
                            row_controls.append(ft.TextButton("Delete",
                                style=ft.ButtonStyle(color=C["red"]),
                                on_click=lambda e, s=s_: [engine.delete_document(s), _close_dlg(), _show_docs()]))
                        doc_list_col.controls.append(ft.Container(
                            content=ft.Row(row_controls, spacing=6),
                            bgcolor=C["tag_bg"] if act else C["input"],
                            border_radius=8, padding=_ps(h=12, v=8), margin=_po(bottom=4),
                        ))
                page.update()

            _build_rows()
            search_field = ft.TextField(
                hint_text="Search documents…", hint_style=ft.TextStyle(color=C["dim"]),
                prefix_icon=ft.Icons.SEARCH, border=ft.InputBorder.OUTLINE,
                border_color=C["border"], focused_border_color=C["accent"],
                bgcolor=C["input"], color=C["fg"], text_style=ft.TextStyle(color=C["fg"], size=13),
                content_padding=_ps(h=12, v=8),
                on_change=lambda e: _build_rows(e.control.value), autofocus=True,
            )
            _show_dlg(ft.AlertDialog(
                title=ft.Text("Indexed Documents", color=C["fg"], weight=ft.FontWeight.BOLD),
                bgcolor=C["card"],
                content=ft.Container(
                    content=ft.Column([search_field, ft.Container(height=8),
                                       ft.Container(content=doc_list_col, height=min(56 + len(docs) * 56, 320))],
                                      spacing=0, tight=True),
                    width=500,
                ),
                actions=[
                    ft.TextButton("Clear Filter", style=ft.ButtonStyle(color=C["dim"]),
                                  on_click=lambda e: [_clear_doc(), _close_dlg()]),
                    ft.TextButton("Close", style=ft.ButtonStyle(color=C["accent"]),
                                  on_click=lambda e: _close_dlg()),
                ],
            ))

        def _plus_menu():
            container = ft.Container(
                content=ft.Column([
                    ft.Container(content=ft.Text("Attach", size=13, weight=ft.FontWeight.W_600,
                                                 color=C["fg2"]), padding=_po(left=16, top=16, bottom=8)),
                    ft.ListTile(leading=ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, color=C["accent"]),
                                title=ft.Text("Focus Document", color=C["fg"]),
                                subtitle=ft.Text("Search only this document", color=C["dim"], size=11),
                                on_click=lambda e: [_close_sheet(bs), _show_docs()]),
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
                sid = s.session_id
                ip_info   = meta.get(sid, {}).get("ip", "N/A")
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
                            ft.Container(content=ft.Text(role_info.upper(), size=9, color="#ffffff",
                                                          weight=ft.FontWeight.W_600),
                                         bgcolor=C["accent"] if role_info == "admin" else C["dim"],
                                         border_radius=4, padding=_ps(h=6, v=2)),
                            ft.IconButton(icon=ft.Icons.DELETE_OUTLINE_ROUNDED, icon_color=C["red"],
                                          icon_size=16, tooltip="Delete Session",
                                          on_click=lambda e, sid=sid: [_delete(sid), _close_dlg(), _show_all_history()]),
                        ], spacing=6),
                        ft.Row([
                            ft.Icon(ft.Icons.COMPUTER_ROUNDED, size=12, color=C["dim"]),
                            ft.Text(f"IP: {ip_info}", size=11, color=C["dim"]),
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
                history_col.controls.append(ft.Text("No chat history yet.", color=C["dim"], size=13))
            _show_dlg(ft.AlertDialog(
                title=ft.Text("All User History", color=C["fg"], weight=ft.FontWeight.BOLD),
                bgcolor=C["card"],
                content=ft.Container(content=history_col, width=520, height=400),
                actions=[ft.TextButton("Close", style=ft.ButtonStyle(color=C["accent"]),
                                       on_click=lambda e: _close_dlg())],
            ))

        # ── Sidebar ───────────────────────────────────────────────────────────
        model_name = engine.get_model_name()
        chunk_n    = engine.get_chunk_count()

        sidebar_actions = []
        if is_admin:
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([ft.Icon(ft.Icons.DASHBOARD_ROUNDED, size=15, color=C["fg2"]),
                                ft.Text("Admin Dashboard", size=12, color=C["fg2"])], spacing=8),
                on_click=lambda e: [page.controls.clear(), page.add(admin_dashboard), page.update()],
            ))
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED, size=15, color=C["fg2"]),
                                ft.Text("Upload Document", size=12, color=C["fg2"])], spacing=8),
                on_click=lambda e: _upload_doc(),
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

        admin_dashboard = ft.Container(
            content=ft.Column([
                ft.Row([ft.Container(expand=True),
                        ft.IconButton(icon=ft.Icons.LOGOUT_ROUNDED, icon_color=C["red"],
                                      icon_size=24, tooltip="Logout", on_click=lambda e: _logout())]),
                ft.Image(src=logo_src, width=200, height=68, fit=ft.BoxFit.CONTAIN)
                    if logo_src else ft.Text("BDL CHATBOT", size=32, weight=ft.FontWeight.BOLD, color=C["accent"]),
                ft.Text("Admin Dashboard", size=24, weight=ft.FontWeight.BOLD, color=C["fg"]),
                ft.Container(height=30),
                ft.Row([
                    ft.Card(content=ft.Container(
                        content=ft.Column([ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED, size=48, color=C["button_accent"]),
                                           ft.Text("Upload Document", size=16, weight=ft.FontWeight.W_600, color=C["fg"]),
                                           ft.Text("Add new files", size=12, color=C["dim"])],
                                          alignment=ft.MainAxisAlignment.CENTER,
                                          horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=30, width=220, height=180, on_click=lambda e: _upload_doc(), ink=True)),
                    ft.Card(content=ft.Container(
                        content=ft.Column([ft.Icon(ft.Icons.FOLDER_OPEN_ROUNDED, size=48, color=C["accent"]),
                                           ft.Text("View Documents", size=16, weight=ft.FontWeight.W_600, color=C["fg"]),
                                           ft.Text("Manage indexing", size=12, color=C["dim"])],
                                          alignment=ft.MainAxisAlignment.CENTER,
                                          horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=30, width=220, height=180, on_click=lambda e: _show_docs(), ink=True)),
                    ft.Card(content=ft.Container(
                        content=ft.Column([ft.Icon(ft.Icons.HISTORY_ROUNDED, size=48, color=C["green"]),
                                           ft.Text("User History", size=16, weight=ft.FontWeight.W_600, color=C["fg"]),
                                           ft.Text("View chat sessions", size=12, color=C["dim"])],
                                          alignment=ft.MainAxisAlignment.CENTER,
                                          horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=30, width=220, height=180, on_click=lambda e: _show_all_history(), ink=True)),
                ], spacing=20, alignment=ft.MainAxisAlignment.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER),
            alignment=ft.Alignment(0, 0), expand=True, bgcolor=C["bg"],
        )

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
                        ft.Text(model_name + (f"  ·  {chunk_n} chunks" if chunk_n else ""),
                                size=10, color=C["dim"]),
                    ], spacing=4),
                    padding=_po(left=16, right=8, top=16, bottom=10),
                ),
                ft.Divider(height=1, color=C["border"]),
                ft.Container(content=sess_col, expand=True),
                ft.Divider(height=1, color=C["border"]),
                ft.Container(content=ft.Column(sidebar_actions, spacing=0), padding=_ps(h=8, v=8)),
            ], spacing=0, expand=True),
            bgcolor=C["sidebar"], width=260, border=ft.Border(right=ft.BorderSide(1, C["border"])),
        )

        # ── Main area ─────────────────────────────────────────────────────────
        input_area = ft.Column([
            ft.Container(
                content=ft.Row([
                    ft.IconButton(icon=ft.Icons.ADD_CIRCLE_OUTLINE_ROUNDED, icon_color=C["dim"],
                                  icon_size=22, tooltip="Attach / Image", on_click=lambda e: _plus_menu()),
                    ft.Container(content=input_box, bgcolor=C["input"],
                                 border_radius=14, border=ft.Border.all(1, C["border"]), expand=True),
                    send_btn,
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.END),
                bgcolor=C["bg"], padding=_po(left=12, right=12, top=8, bottom=8),
            ),
            ft.Container(
                content=ft.Row([status_lbl, ft.Container(width=12), stop_btn], spacing=0),
                bgcolor=C["bg"], padding=_po(left=20, bottom=6),
            ),
        ], spacing=0, visible=True)

        # Issue 3: main_area no longer has a followup_row container
        main_area = ft.Column([
            ft.Container(content=chat_col, bgcolor=C["bg"], expand=True),
            ft.Container(content=doc_bar, bgcolor=C["bg"], padding=_po(left=16, right=16, top=4, bottom=0)),
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
                upload_dir=UPLOAD_DIR,
            )