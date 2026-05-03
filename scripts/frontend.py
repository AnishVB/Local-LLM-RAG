"""
frontend.py — LibrarianBot  (Flet 0.84)
Run:  python scripts/frontend.py

FIXES (v2):
  1. Copy button: uses pyperclip with graceful fallback (avoids flet clipboard API)
  2. ft.alignment.center crash: replaced with ft.Alignment(0, 0) (correct Flet 0.84 API)
  3. Space/time: removed redundant re-builds, reuse source index; lazy matrix build
  4. Speed: classify + rewrite merged into ONE LLM call; follow-ups run async with timeout
  5. Follow-ups: tighter prompt, JSON enforced, no fallback-slop in normal flow
  6. Polish: better hints, status messages, input UX
"""
import os, sys, re, time, threading, base64, mimetypes, socket, json as _json
import tkinter as tk
from tkinter import filedialog as tkfd
from datetime import datetime, date, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.normpath(os.path.join(current_dir, "..", "assests", "BDL logo nobg.png"))


def _get_logo_src(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None

logo_src = _get_logo_src(logo_path)

# ── Role / IP tracking ────────────────────────────────────────────────────────
ADMIN_PASSWORD = "Admin321!"
_user_state = {"role": None}  # "user" or "admin"

_SESSION_META_PATH = os.path.normpath(os.path.join(current_dir, "..", "chat_history", "session_meta.json"))

def _load_session_meta() -> dict:
    if os.path.isfile(_SESSION_META_PATH):
        try:
            with open(_SESSION_META_PATH, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return {}

def _save_session_meta(meta: dict):
    os.makedirs(os.path.dirname(_SESSION_META_PATH), exist_ok=True)
    with open(_SESSION_META_PATH, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import flet as ft
from chat_engine import (ChatEngine, ChatSession,
                         load_all_sessions, save_all_sessions, delete_session)

# ── Clipboard helpers ─────────────────────────────────────────────────────────
def _copy_native(text: str) -> bool:
    """
    Server-side clipboard for desktop mode only.
    Tries pyperclip → xclip → xsel → clip (Win) → pbcopy (mac).
    """
    try:
        import pyperclip; pyperclip.copy(text); return True
    except Exception: pass
    import subprocess
    for cmd, enc in [
        (["xclip", "-selection", "clipboard"], "utf-8"),
        (["xsel", "--clipboard", "--input"],   "utf-8"),
        (["clip"],                              "utf-16"),
        (["pbcopy"],                            "utf-8"),
    ]:
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, close_fds=True)
            p.communicate(input=text.encode(enc))
            if p.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _make_copy_handler(page_ref, text_getter):
    """
    Returns an on_click handler that copies text to clipboard.
    In web mode uses navigator.clipboard (runs in the browser via JS).
    In desktop mode falls back to the server-side native helper.
    """
    def _handler(e):
        text = text_getter()
        if not text:
            return
        if page_ref.web:
            # Escape for JS string literal: backslash, backtick, $
            safe = (text
                    .replace("\\", "\\\\")
                    .replace("`", "\\`")
                    .replace("$", "\\$"))
            page_ref.run_javascript(
                f"navigator.clipboard.writeText(`{safe}`)"
                ".then(()=>{{}})"
                ".catch(err=>console.warn('clipboard:', err));"
            )
            _snack_ref[0]("Copied to clipboard.")
        else:
            ok = _copy_native(text)
            _snack_ref[0](
                "Copied to clipboard." if ok
                else "Clipboard unavailable — select text manually.",
                ok=ok,
            )
    return _handler

# Mutable ref so _make_copy_handler can call _snack before it's defined
_snack_ref = [None]


# ── Colours ───────────────────────────────────────────────────────────────────
C = dict(
    bg      = "#ffffff",
    sidebar = "#E7C39E",
    card    = "#ffffff",
    input   = "#f0f1f3",
    border  = "#d0d3de",
    accent  = "#3b5bdb",
    accent2 = "#4c6ef5",
    user_bg = "#3b5bdb",
    bot_bg  = "#f4f5f7",
    fg      = "#1a1a2e",
    fg2     = "#2e3250",
    dim     = "#9096b0",
    green   = "#2f9e44",
    red     = "#c92a2a",
    tag_bg  = "#dbe4ff",
    tag_fg  = "#1c4be0",
    button_accent = "#d42727",
)

def _ps(h=0, v=0): return ft.Padding.symmetric(horizontal=h, vertical=v)
def _po(**kw):      return ft.Padding.only(**kw)
def _ms(h=0, v=0): return ft.Margin.symmetric(horizontal=h, vertical=v)

def _trunc(t, n=32):
    t = (t or "New Chat").strip()
    return t if len(t) <= n else t[:n-1] + "…"

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


# ── Doc-like keyword check ────────────────────────────────────────────────────
_CASUAL_ONLY_WORDS = {
    "hi","hello","hey","howdy","sup","yo","hiya","heyy","heyyy",
    "thanks","thank you","thx","ty","cheers","appreciated",
    "ok","okay","cool","alright","got it","sure","yep","yup","yeah","nope",
    "lovely","wonderful","awesome","perfect","brilliant","excellent","nice","great",
    "huh","hmm","oh","ah","lol","lmao","haha",
    "bye","goodbye","cya","later","peace",
}


def _looks_like_pure_greeting(text: str) -> bool:
    low = text.strip().lower()
    _MULTI_GREETINGS = {
        "how are you", "how are you doing", "how's it going", "hows it going",
        "what's up", "whats up", "wassup", "wazzup",
        "good morning", "good afternoon", "good evening", "good night",
        "who are you", "what are you", "what can you do",
        "see you", "i see", "right",
    }
    if low in _MULTI_GREETINGS:
        return True
    words = low.split()
    if len(words) == 1 and low in _CASUAL_ONLY_WORDS:
        return True
    if len(words) == 2 and low in _MULTI_GREETINGS:
        return True
    return False


async def main(page: ft.Page):
    page.title       = "BDL CHATBOT"
    page.theme_mode  = ft.ThemeMode.LIGHT
    page.bgcolor     = C["bg"]
    page.window.width  = 1120
    page.window.height = 720
    page.window.min_width  = 800
    page.window.min_height = 480
    page.padding     = 0

    user_ip = _get_local_ip()
    session_meta = _load_session_meta()

    # ── Login / Role Selection Screen ─────────────────────────────────────────
    login_done = threading.Event()

    def _enter_as_user(e):
        _user_state["role"] = "user"
        login_done.set()
        _launch_chat()

    def _enter_as_admin(e):
        pwd = admin_pwd_field.value or ""
        if pwd == ADMIN_PASSWORD:
            _user_state["role"] = "admin"
            login_done.set()
            _launch_chat()
        else:
            admin_error.value = "Incorrect password"
            admin_error.visible = True
            page.update()

    admin_pwd_field = ft.TextField(
        hint_text="Enter admin password",
        hint_style=ft.TextStyle(color=C["dim"]),
        password=True, can_reveal_password=True,
        border_color=C["border"], focused_border_color=C["accent"],
        bgcolor=C["input"], color=C["fg"],
        text_style=ft.TextStyle(color=C["fg"], size=13),
        content_padding=_ps(h=14, v=10),
        width=260,
        on_submit=lambda e: _enter_as_admin(e),
    )
    admin_error = ft.Text("", color=C["red"], size=12, visible=False)

    admin_section = ft.Column([
        admin_pwd_field,
        admin_error,
        ft.Container(height=4),
        ft.FilledButton(
            content=ft.Text("Login as Admin", size=13, color="#ffffff"),
            style=ft.ButtonStyle(bgcolor=C["button_accent"],
                                 shape=ft.RoundedRectangleBorder(radius=10)),
            width=260, height=40,
            on_click=_enter_as_admin,
        ),
    ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER, visible=False)

    def _toggle_admin(e):
        admin_section.visible = not admin_section.visible
        page.update()

    login_view = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Image(src=logo_src, width=160, height=55,
                                 fit=ft.BoxFit.CONTAIN) if logo_src else ft.Text(
                    "BDL", size=28, weight=ft.FontWeight.BOLD, color=C["button_accent"]),
                padding=_po(bottom=6),
            ),
            ft.Text("BDL CHATBOT", size=26, weight=ft.FontWeight.BOLD, color=C["fg"]),
            ft.Text("Select how you'd like to continue", size=13, color=C["dim"]),
            ft.Container(height=16),
            ft.FilledButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.PERSON_ROUNDED, color="#ffffff", size=18),
                    ft.Text("Continue as User", size=14, color="#ffffff",
                            weight=ft.FontWeight.W_500),
                ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                style=ft.ButtonStyle(bgcolor=C["accent"],
                                     shape=ft.RoundedRectangleBorder(radius=10)),
                width=260, height=44,
                on_click=_enter_as_user,
            ),
            ft.Container(height=6),
            ft.OutlinedButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.ADMIN_PANEL_SETTINGS_ROUNDED,
                            color=C["button_accent"], size=18),
                    ft.Text("Login as Admin", size=14, color=C["button_accent"],
                            weight=ft.FontWeight.W_500),
                ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                style=ft.ButtonStyle(
                    side=ft.BorderSide(1.5, C["button_accent"]),
                    shape=ft.RoundedRectangleBorder(radius=10)),
                width=260, height=44,
                on_click=_toggle_admin,
            ),
            ft.Container(height=4),
            admin_section,
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
        alignment=ft.Alignment(0, 0),
        expand=True,
        bgcolor=C["bg"],
    )

    page.add(login_view)

    def _logout(e=None):
        _user_state["role"] = None
        admin_pwd_field.value = ""
        admin_section.visible = False
        admin_error.visible = False
        page.controls.clear()
        page.add(login_view)
        page.update()

    def _launch_chat():
        page.controls.clear()
        page.update()
        _build_chat_ui()


    def _build_chat_ui():
        nonlocal session_meta
        is_admin = _user_state["role"] == "admin"

        engine   = ChatEngine()
        sessions = load_all_sessions()
        cur      = {"session": None, "busy": False, "thread": None, "spinner": None}
        live_md  = [None]

        # Tag new sessions with user IP
        def _save():
            save_all_sessions(sessions)
            if cur["session"]:
                sid = cur["session"].session_id
                if sid not in session_meta:
                    session_meta[sid] = {"ip": user_ip, "role": _user_state["role"]}
                    _save_session_meta(session_meta)

        def _pick_file(title, extensions):
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            filetypes = [(f"{e.upper()} files", f"*.{e}") for e in extensions]
            filetypes.append(("All files", "*.*"))
            path = tkfd.askopenfilename(title=title, filetypes=filetypes, parent=root)
            root.destroy()
            return path or None

        # ── Dialog helpers ────────────────────────────────────────────────────────
        def _show_dlg(dlg):
            page.show_dialog(dlg)

        def _close_dlg():
            page.pop_dialog()

        def _snack(text, ok=True):
            color = C["green"] if ok else C["red"]
            sb = ft.SnackBar(
                ft.Text(text, color=color),
                bgcolor=C["card"],
                open=True,
            )
            page.overlay.append(sb)
            page.update()
        _snack_ref[0] = _snack  # wire clipboard helper

        def _show_sheet(content_widget):
            bs = ft.BottomSheet(content=content_widget, open=True)
            page.overlay.append(bs)
            page.update()
            return bs

        def _close_sheet(bs):
            bs.open = False
            page.update()

        # ── Chat widgets ──────────────────────────────────────────────────────────
        chat_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO,
                             expand=True, auto_scroll=True)
        followup_row = ft.Row(spacing=6, wrap=True, visible=False)
        doc_bar      = ft.Row(spacing=0, visible=False)
        status_lbl   = ft.Text("Ready", size=11, color=C["dim"])
        # FIX #2: small icon button instead of big red text button
        stop_btn     = ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE_OUTLINED,
            icon_color=C["red"],
            icon_size=18,
            tooltip="Stop generation",
            visible=False,
            on_click=lambda e: _stop_generation(),
        )

        input_box = ft.TextField(
            hint_text="Ask anything about your documents…",
            hint_style=ft.TextStyle(color=C["dim"]),
            border=ft.InputBorder.NONE,
            bgcolor=C["input"],
            color=C["fg"],
            text_style=ft.TextStyle(color=C["fg"], size=13),
            multiline=True, min_lines=1, max_lines=5,
            expand=True,
            content_padding=_ps(h=16, v=12),
            cursor_color=C["accent"],
            shift_enter=True,
            on_submit=lambda e: _send(),
        )
        send_btn = ft.IconButton(
            icon=ft.Icons.SEND_ROUNDED,
            icon_color=C["button_accent"],
            icon_size=22,
            tooltip="Send",
        )
        sess_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO, expand=True)

        # ── Bubble builders ───────────────────────────────────────────────────────
        def _user_bubble(text):
            return ft.Container(
                content=ft.Row([
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Text(text, color="#ffffff", size=13,
                                        selectable=True, no_wrap=False),
                        bgcolor=C["user_bg"],
                        border_radius=ft.BorderRadius(14, 14, 2, 14),
                        padding=_ps(h=14, v=10),
                        width=500,
                    ),
                ]),
                padding=_po(left=16, right=16, top=4, bottom=4),
            )

        def _bot_bubble(thinking=False):
            spinner = ft.ProgressRing(width=16, height=16, visible=thinking)
            md = ft.Markdown(
                value="▍" if thinking else "",
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                code_theme=ft.MarkdownCodeTheme.GITHUB,
                expand=True,
            )

            # FIX #1: web-aware clipboard (JS in browser, native on desktop)
            _on_copy = _make_copy_handler(page, lambda md_ref=md: md_ref.value)

            shell = ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text("● Answer", size=11,
                                        weight=ft.FontWeight.W_600,
                                        color=C["green"]),
                                spinner,
                            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            md,
                            ft.Row([
                                ft.Container(expand=True),
                                ft.IconButton(
                                    icon=ft.Icons.CONTENT_COPY,
                                    icon_size=18,
                                    tooltip="Copy response",
                                    on_click=_on_copy,
                                ),
                            ], spacing=0),
                        ], spacing=6, tight=True),
                        bgcolor=C["bot_bg"],
                        border_radius=ft.BorderRadius(2, 14, 14, 14),
                        padding=_ps(h=14, v=10),
                        width=580,
                        border=ft.Border.all(1, C["border"]),
                    ),
                    ft.Container(expand=True),
                ]),
                padding=_po(left=16, right=16, top=4, bottom=4),
            )
            return shell, md, spinner

        # ── Render session ────────────────────────────────────────────────────────
        def _render():
            chat_col.controls.clear()
            followup_row.controls.clear()
            followup_row.visible = False
            s = cur["session"]
            if not s or not s.messages:
                # FIX #2: use ft.Alignment(0, 0) — ft.alignment.center doesn't exist in 0.84
                chat_col.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Text("BDL CHATBOT", size=22,
                                weight=ft.FontWeight.BOLD, color=C["accent"]),
                        ft.Text("Upload documents via the sidebar,\n"
                                "then ask anything about them.",
                                size=13, color=C["dim"],
                                text_align=ft.TextAlign.CENTER),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                       spacing=10),
                    alignment=ft.Alignment(0, 0),   # FIX: was ft.alignment.center
                    expand=True,
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

        def _refresh_followups(qs):
            followup_row.controls.clear()
            if not qs:
                followup_row.visible = False
                page.update()
                return
            followup_row.visible = True
            followup_row.controls.append(
                ft.Text("  Suggested:", size=11, color=C["dim"]))
            for q in qs[:3]:
                short = q[:58] + ("…" if len(q) > 58 else "")
                followup_row.controls.append(
                    ft.OutlinedButton(
                        content=ft.Text(short, size=12, color=C["fg2"]),
                        style=ft.ButtonStyle(
                            side=ft.BorderSide(1, C["border"]),
                            shape=ft.RoundedRectangleBorder(radius=8),
                            padding=_ps(h=10, v=4),
                        ),
                        on_click=lambda e, t=q: _send(t),
                    )
                )
            page.update()

        def _refresh_doc_bar():
            doc_bar.controls.clear()
            s = cur["session"]
            if s and s.active_doc:
                doc_bar.visible = True
                doc_bar.controls.append(ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=14,
                                color=C["tag_fg"]),
                        ft.Text(f"  {s.active_doc}", size=12,
                                color=C["tag_fg"], expand=True),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE, icon_size=14,
                            icon_color=C["dim"], width=26, height=26,
                            on_click=lambda e: _clear_doc(),
                        ),
                    ], spacing=0),
                    bgcolor=C["tag_bg"],
                    border_radius=8,
                    padding=_po(left=10, right=4, top=4, bottom=4),
                    expand=True,
                ))
            else:
                doc_bar.visible = False
            page.update()

        def _clear_doc():
            if cur["busy"]: return
            if cur["session"]:
                cur["session"].active_doc = None
                _save()
            _refresh_doc_bar()

        def _stop_generation():
            if not cur["busy"] or not cur["thread"]:
                return
            cur["thread"] = None
            if cur["spinner"]:
                cur["spinner"].visible = False
            _set_busy(False)
            status_lbl.value = "Stopped"
            if live_md[0]:
                live_md[0].value += "\n\n*[Generation stopped]*"
            page.update()

        def _set_busy(busy: bool):
            cur["busy"] = busy
            send_btn.disabled = busy
            stop_btn.visible = busy
            page.update()

        # ── Send message ──────────────────────────────────────────────────────────
        def _send(text=None):
            if cur["busy"]: return
            raw = (text or input_box.value or "").strip()
            if not raw: return
            input_box.value = ""
            followup_row.controls.clear()
            followup_row.visible = False
            if not cur["session"]: _new_chat()
            cur["session"].add_message("user", raw)
            _save()
            chat_col.controls.append(_user_bubble(raw))
            shell, md, spinner = _bot_bubble(thinking=True)
            cur["spinner"] = spinner
            live_md[0] = md
            chat_col.controls.append(shell)
            _set_busy(True)
            status_lbl.value = "Thinking…"
            page.update()

            history    = list(cur["session"].messages[:-1])
            src_filter = cur["session"].active_doc
            target_sid = cur["session"].session_id
            t0, buf    = time.time(), []

            def _stream(chunk):
                if not chunk: return
                buf.append(chunk)
                if cur["session"] and cur["session"].session_id == target_sid:
                    live_md[0].value = "".join(buf)
                    page.update()

            def _bg():
                try:
                    engine.process_message(
                        raw, history, stream_cb=_stream,
                        followup_cb=lambda qs: page.run_thread(
                            _refresh_followups, qs),
                        source_filter=src_filter,
                    )
                except Exception as ex:
                    _stream(f"\n\n**Error:** {ex}")
                elapsed  = round(time.time() - t0, 1)
                full_ans = "".join(buf).strip()
                tgt = next((s for s in sessions
                            if s.session_id == target_sid), None)
                if tgt:
                    tgt.messages.append({"role": "assistant",
                                         "content": full_ans,
                                         "resp_time": elapsed})
                    tgt.updated_at = datetime.now().isoformat()
                    _save()
                page.run_thread(_done, elapsed)

            def _done(elapsed):
                if cur["spinner"]:
                    cur["spinner"].visible = False
                _set_busy(False)
                cur["thread"] = None
                status_lbl.value = f"Done in {elapsed}s"
                _refresh_sess_list()
                page.update()

            cur["thread"] = threading.Thread(target=_bg, daemon=True)
            cur["thread"].start()

        # ── Image send ────────────────────────────────────────────────────────────
        def _do_image(path, b64, mime, prompt):
            if cur["busy"]: return
            if not cur["session"]: _new_chat()
            label = f"[Image: {os.path.basename(path)}] {prompt}"
            cur["session"].add_message("user", label)
            _save()
            chat_col.controls.append(_user_bubble(label))
            shell, md, spinner = _bot_bubble(thinking=True)
            cur["spinner"] = spinner
            live_md[0] = md
            chat_col.controls.append(shell)
            _set_busy(True)
            status_lbl.value = "Analysing image…"
            page.update()
            buf = []
            def _stream(c):
                buf.append(c); live_md[0].value = "".join(buf); page.update()
            def _bg():
                try:
                    engine.process_image_message(
                        b64, mime, prompt,
                        cur["session"].messages[:-1], stream_cb=_stream)
                except Exception as ex:
                    _stream(f"\n**Error:** {ex}")
                cur["session"].messages.append(
                    {"role": "assistant", "content": "".join(buf).strip()})
                _save()
                page.run_thread(_img_done)
            def _img_done():
                if cur["spinner"]:
                    cur["spinner"].visible = False
                _set_busy(False)
                cur["thread"] = None
                status_lbl.value = "Ready"; page.update()
            cur["thread"] = threading.Thread(target=_bg, daemon=True)
            cur["thread"].start()

        # ── Session management ────────────────────────────────────────────────────
        def _new_chat():
            if cur["busy"]: return
            s = ChatSession(); sessions.insert(0, s); _save()
            cur["session"] = s; _render(); _refresh_doc_bar(); _refresh_sess_list()

        def _select(s):
            if cur["busy"]: return
            cur["session"] = s
            if is_admin and admin_dashboard in page.controls:
                page.controls.clear()
                page.add(ft.Row([sidebar, main_area], spacing=0, expand=True, vertical_alignment=ft.CrossAxisAlignment.STRETCH))
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
            cur_sid = cur["session"].session_id if cur["session"] else None
            max_items = 999 if is_admin else 5   # users see only last 5
            shown = 0
            for label, grp in _group(sessions):
                if shown >= max_items:
                    break
                sess_col.controls.append(ft.Container(
                    content=ft.Text(label, size=10, color=C["dim"],
                                    weight=ft.FontWeight.W_600),
                    padding=_po(left=14, top=12, bottom=2),
                ))
                for s in grp:
                    if shown >= max_items:
                        break
                    sel   = s.session_id == cur_sid
                    title = _trunc(s.title or "New Chat")
                    dot   = " ·" if getattr(s, "active_doc", None) else ""
                    sid   = s.session_id
                    sess_col.controls.append(ft.Container(
                        content=ft.Row([
                            ft.Text(title + dot, size=12, expand=True,
                                    color=C["fg"] if sel else C["fg2"],
                                    weight=(ft.FontWeight.W_600 if sel
                                            else ft.FontWeight.NORMAL),
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE, icon_size=12,
                                icon_color=C["dim"], width=26, height=26,
                                on_click=lambda e, s_=sid: _delete(s_),
                            ),
                        ], spacing=0),
                        bgcolor=C["accent"] + "22" if sel else "transparent",
                        border_radius=8,
                        padding=_po(left=12, right=2, top=6, bottom=6),
                        margin=_ms(h=6, v=1),
                        on_click=lambda e, s_=s: _select(s_),
                        ink=True,
                        ink_color=C["border"],
                    ))
                    shown += 1
            page.update()

        # ── Document management ───────────────────────────────────────────────────
        def _upload_doc():
            if cur["busy"]: return
            path = _pick_file("Select a document", ["pdf", "txt", "md", "docx"])
            if not path:
                return
            status_lbl.value = f"Indexing {os.path.basename(path)}…"
            page.update()
            def _bg():
                ok, msg = engine.upload_file(path)
                def _finish():
                    status_lbl.value = "Ready"
                    _snack(msg, ok=ok)
                page.run_thread(_finish)
            threading.Thread(target=_bg, daemon=True).start()

        # ── FIX #2 applied here too: View docs with search bar ───────────────────
        def _show_docs():
            if cur["busy"]: return
            docs = engine.list_documents()
            if not docs:
                _snack("No documents indexed yet.", ok=False)
                return

            cur_doc = cur["session"].active_doc if cur["session"] else None
            doc_items = sorted(docs.items())

            doc_list_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO)

            def _build_rows(filter_text=""):
                doc_list_col.controls.clear()
                ft_low = filter_text.strip().lower()
                matched = [(src, cnt) for src, cnt in doc_items
                           if ft_low in src.lower()]
                if not matched:
                    # FIX #2: use ft.Alignment(0, 0) instead of ft.alignment.center
                    doc_list_col.controls.append(
                        ft.Container(
                            content=ft.Text("No documents match your search.",
                                            color=C["dim"], size=12,
                                            text_align=ft.TextAlign.CENTER),
                            padding=_ps(h=12, v=20),
                            alignment=ft.Alignment(0, 0),   # FIX: was ft.alignment.center
                        )
                    )
                else:
                    for src, cnt in matched:
                        act = src == cur_doc
                        s_  = src
                        doc_list_col.controls.append(ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, size=15,
                                        color=C["tag_fg"] if act else C["dim"]),
                                ft.Text(src, size=13, color=C["fg"], expand=True,
                                        no_wrap=True,
                                        overflow=ft.TextOverflow.ELLIPSIS),
                                ft.Text(f"{cnt} chunks", size=11, color=C["dim"]),
                                ft.TextButton("Focus",
                                    style=ft.ButtonStyle(color=C["accent2"]),
                                    on_click=lambda e, s=s_: [
                                        setattr(cur["session"], "active_doc", s)
                                        if (cur["session"] and not cur["busy"]) else None,
                                        _save(), _refresh_doc_bar(), _close_dlg(),
                                    ] if not cur["busy"] else None),
                                ft.TextButton("Delete",
                                    style=ft.ButtonStyle(color=C["red"]),
                                    on_click=lambda e, s=s_: [
                                        engine.delete_document(s),
                                        _close_dlg(), _show_docs(),
                                    ]),
                            ], spacing=6),
                            bgcolor=C["tag_bg"] if act else C["input"],
                            border_radius=8,
                            padding=_ps(h=12, v=8),
                            margin=_po(bottom=4),
                        ))
                page.update()

            _build_rows()

            search_field = ft.TextField(
                hint_text="Search documents…",
                hint_style=ft.TextStyle(color=C["dim"]),
                prefix_icon=ft.Icons.SEARCH,
                border=ft.InputBorder.OUTLINE,
                border_color=C["border"],
                focused_border_color=C["accent"],
                bgcolor=C["input"],
                color=C["fg"],
                text_style=ft.TextStyle(color=C["fg"], size=13),
                content_padding=_ps(h=12, v=8),
                on_change=lambda e: _build_rows(e.control.value),
                autofocus=True,
            )

            dlg = ft.AlertDialog(
                title=ft.Text("Indexed Documents", color=C["fg"],
                              weight=ft.FontWeight.BOLD),
                bgcolor=C["card"],
                content=ft.Container(
                    content=ft.Column([
                        search_field,
                        ft.Container(height=8),
                        ft.Container(
                            content=doc_list_col,
                            height=min(56 + len(docs) * 56, 320),
                        ),
                    ], spacing=0, tight=True),
                    width=500,
                ),
                actions=[
                    ft.TextButton("Clear Filter",
                        style=ft.ButtonStyle(color=C["dim"]),
                        on_click=lambda e: [_clear_doc(), _close_dlg()]),
                    ft.TextButton("Close",
                        style=ft.ButtonStyle(color=C["accent"]),
                        on_click=lambda e: _close_dlg()),
                ],
            )
            _show_dlg(dlg)

        def _image_picker():
            path = _pick_file("Select an image",
                              ["png", "jpg", "jpeg", "gif", "webp"])
            if not path:
                return
            mime = mimetypes.guess_type(path)[0] or "image/png"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            prompt_field = ft.TextField(
                value="Describe this image",
                color=C["fg"], bgcolor=C["input"],
                border_color=C["border"], autofocus=True,
            )
            dlg = ft.AlertDialog(
                title=ft.Text(f"Image: {os.path.basename(path)}", color=C["fg"]),
                bgcolor=C["card"],
                content=ft.Container(content=prompt_field, width=360, height=80),
                actions=[
                    ft.TextButton("Cancel",
                        style=ft.ButtonStyle(color=C["dim"]),
                        on_click=lambda e: _close_dlg()),
                    ft.FilledButton(
                        content=ft.Text("Send"),
                        style=ft.ButtonStyle(bgcolor=C["accent"]),
                        on_click=lambda e: [
                            _close_dlg(),
                            _do_image(path, b64, mime,
                                      prompt_field.value or "Describe this image"),
                        ]),
                ],
            )
            _show_dlg(dlg)

        def _plus_menu():
            container = ft.Container(
                content=ft.Column([
                    ft.Container(
                        content=ft.Text("Attach", size=13,
                                        weight=ft.FontWeight.W_600,
                                        color=C["fg2"]),
                        padding=_po(left=16, top=16, bottom=8),
                    ),
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.DESCRIPTION_ROUNDED, color=C["accent"]),
                        title=ft.Text("Focus Document", color=C["fg"]),
                        subtitle=ft.Text("Search only this document",
                                         color=C["dim"], size=11),
                        on_click=lambda e: [_close_sheet(bs), _show_docs()],
                    ),
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.IMAGE_ROUNDED, color=C["accent"]),
                        title=ft.Text("Send an Image", color=C["fg"]),
                        subtitle=ft.Text("Ask the bot about an image",
                                         color=C["dim"], size=11),
                        on_click=lambda e: [_close_sheet(bs), _image_picker()],
                    ),
                    ft.Container(height=16),
                ], spacing=0, tight=True),
                bgcolor=C["card"],
            )
            bs = _show_sheet(container)

        # ── Admin: all-user history dialog ──────────────────────────────────────
        def _show_all_history():
            all_sessions = load_all_sessions()
            meta = _load_session_meta()
            history_col = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO)
            for s in sorted(all_sessions, key=lambda x: x.updated_at or "", reverse=True):
                sid = s.session_id
                ip_info = meta.get(sid, {}).get("ip", "N/A")
                role_info = meta.get(sid, {}).get("role", "user")
                msg_count = len(s.messages)
                title = _trunc(s.title or "New Chat", 40)
                try:
                    ts = datetime.fromisoformat(s.updated_at).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    ts = s.updated_at or "?"
                history_col.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE_ROUNDED,
                                    size=14, color=C["accent"]),
                            ft.Text(title, size=13, color=C["fg"],
                                    weight=ft.FontWeight.W_600, expand=True,
                                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                            ft.Container(
                                content=ft.Text(role_info.upper(), size=9,
                                                color="#ffffff",
                                                weight=ft.FontWeight.W_600),
                                bgcolor=C["accent"] if role_info == "admin" else C["dim"],
                                border_radius=4, padding=_ps(h=6, v=2),
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE_ROUNDED,
                                icon_color=C["red"], icon_size=16,
                                tooltip="Delete Session",
                                on_click=lambda e, sid=sid: [
                                    _delete(sid),
                                    _close_dlg(), _show_all_history()
                                ],
                            ),
                        ], spacing=6),
                        ft.Row([
                            ft.Icon(ft.Icons.COMPUTER_ROUNDED, size=12, color=C["dim"]),
                            ft.Text(f"IP: {ip_info}", size=11, color=C["dim"]),
                            ft.Container(width=12),
                            ft.Icon(ft.Icons.ACCESS_TIME_ROUNDED, size=12, color=C["dim"]),
                            ft.Text(ts, size=11, color=C["dim"]),
                            ft.Container(width=12),
                            ft.Text(f"{msg_count} msgs", size=11, color=C["dim"]),
                        ], spacing=4),
                    ], spacing=4, tight=True),
                    bgcolor=C["input"],
                    border_radius=8,
                    padding=_ps(h=12, v=8),
                    margin=_po(bottom=4),
                    on_click=lambda e, s_=s: [_close_dlg(), _select(s_)],
                    ink=True,
                ))
            if not history_col.controls:
                history_col.controls.append(ft.Text("No chat history yet.",
                                                     color=C["dim"], size=13))
            dlg = ft.AlertDialog(
                title=ft.Text("All User History", color=C["fg"],
                              weight=ft.FontWeight.BOLD),
                bgcolor=C["card"],
                content=ft.Container(
                    content=history_col,
                    width=520, height=400,
                ),
                actions=[
                    ft.TextButton("Close",
                        style=ft.ButtonStyle(color=C["accent"]),
                        on_click=lambda e: _close_dlg()),
                ],
            )
            _show_dlg(dlg)

        # ── Sidebar ───────────────────────────────────────────────────────────────
        model_name = engine.get_model_name()
        chunk_n    = engine.get_chunk_count()

        # Build bottom action buttons based on role
        sidebar_actions = []
        if is_admin:
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED, size=15, color=C["fg2"]),
                    ft.Text("Upload Document", size=12, color=C["fg2"]),
                ], spacing=8),
                on_click=lambda e: _upload_doc(),
            ))
        sidebar_actions.append(ft.TextButton(
            content=ft.Row([
                ft.Icon(ft.Icons.FOLDER_OPEN_ROUNDED, size=15, color=C["fg2"]),
                ft.Text("View Documents", size=12, color=C["fg2"]),
            ], spacing=8),
            on_click=lambda e: _show_docs(),
        ))
        if is_admin:
            sidebar_actions.append(ft.TextButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.HISTORY_ROUNDED, size=15, color=C["button_accent"]),
                    ft.Text("All User History", size=12, color=C["button_accent"]),
                ], spacing=8),
                on_click=lambda e: _show_all_history(),
            ))

        role_badge = ft.Container(
            content=ft.Text(
                "ADMIN" if is_admin else "USER",
                size=9, color="#ffffff", weight=ft.FontWeight.W_600),
            bgcolor=C["button_accent"] if is_admin else C["accent"],
            border_radius=4, padding=_ps(h=6, v=2),
        )

        def _show_admin_dashboard():
            page.controls.clear()
            page.add(admin_dashboard)
            page.update()

        if is_admin:
            sidebar_actions.insert(0, ft.TextButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.DASHBOARD_ROUNDED, size=15, color=C["fg2"]),
                    ft.Text("Admin Dashboard", size=12, color=C["fg2"]),
                ], spacing=8),
                on_click=lambda e: _show_admin_dashboard(),
            ))

        admin_dashboard = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Container(expand=True),
                    ft.IconButton(
                        icon=ft.Icons.LOGOUT_ROUNDED,
                        icon_color=C["red"], icon_size=24,
                        tooltip="Logout",
                        on_click=lambda e: _logout(),
                    )
                ]),
                ft.Image(src=logo_src, width=200, height=68, fit=ft.BoxFit.CONTAIN) if logo_src else ft.Text("BDL CHATBOT", size=32, weight=ft.FontWeight.BOLD, color=C["accent"]),
                ft.Text("Admin Dashboard", size=24, weight=ft.FontWeight.BOLD, color=C["fg"]),
                ft.Container(height=30),
                ft.Row([
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Icon(ft.Icons.UPLOAD_FILE_ROUNDED, size=48, color=C["button_accent"]),
                                    ft.Text("Upload Document", size=16, weight=ft.FontWeight.W_600, color=C["fg"]),
                                    ft.Text("Add new files", size=12, color=C["dim"])
                                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                padding=30, width=220, height=180,
                                on_click=lambda e: _upload_doc(),
                                ink=True
                            )
                        ),
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Icon(ft.Icons.FOLDER_OPEN_ROUNDED, size=48, color=C["accent"]),
                                    ft.Text("View Documents", size=16, weight=ft.FontWeight.W_600, color=C["fg"]),
                                    ft.Text("Manage indexing", size=12, color=C["dim"])
                                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                padding=30, width=220, height=180,
                                on_click=lambda e: _show_docs(),
                                ink=True
                            )
                        ),
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Icon(ft.Icons.HISTORY_ROUNDED, size=48, color=C["green"]),
                                    ft.Text("User History", size=16, weight=ft.FontWeight.W_600, color=C["fg"]),
                                    ft.Text("View chat sessions", size=12, color=C["dim"])
                                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                                padding=30, width=220, height=180,
                                on_click=lambda e: _show_all_history(),
                                ink=True
                            )
                        ),
            ], spacing=20, alignment=ft.MainAxisAlignment.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER),
        alignment=ft.Alignment(0, 0),
        expand=True,
        bgcolor=C["bg"]
    )

        sidebar = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Row([
                                ft.Image(
                                    src=logo_src,
                                    width=100,
                                    height=34,
                                    fit=ft.BoxFit.CONTAIN,
                                ) if logo_src else ft.Text(
                                    "BDL CHATBOT", size=14,
                                    weight=ft.FontWeight.BOLD, color=C["accent"]),
                                role_badge,
                            ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.Text(model_name +
                                    (f"  ·  {chunk_n} chunks" if chunk_n else ""),
                                    size=10, color=C["dim"]),
                        ], spacing=4, expand=True),
                        ft.IconButton(
                            icon=ft.Icons.LOGOUT_ROUNDED,
                            icon_color=C["red"], icon_size=20,
                            tooltip="Logout",
                            on_click=lambda e: _logout(),
                        ),
                        ft.IconButton(
                            icon=ft.Icons.EDIT_NOTE_ROUNDED,
                            icon_color=C["button_accent"], icon_size=20,
                            tooltip="New Chat",
                            on_click=lambda e: _new_chat(),
                        ),
                    ], spacing=0),
                    padding=_po(left=16, right=8, top=16, bottom=10),
                ),
                ft.Divider(height=1, color=C["border"]),
                ft.Container(content=sess_col, expand=True),
                ft.Divider(height=1, color=C["border"]),
                ft.Container(
                    content=ft.Column(sidebar_actions, spacing=0),
                    padding=_ps(h=8, v=8),
                ),
            ], spacing=0, expand=True),
            bgcolor=C["sidebar"],
            width=236,
            border=ft.Border(right=ft.BorderSide(1, C["border"])),
        )

        # ── Main area ─────────────────────────────────────────────────────────────
        main_area = ft.Column([
            ft.Container(content=chat_col, bgcolor=C["bg"], expand=True),
            ft.Container(
                content=followup_row, bgcolor=C["bg"],
                padding=_po(left=16, right=16, top=4, bottom=0),
            ),
            ft.Container(
                content=doc_bar, bgcolor=C["bg"],
                padding=_po(left=16, right=16, top=4, bottom=0),
            ),
            ft.Container(
                content=ft.Row([
                    ft.IconButton(
                        icon=ft.Icons.ADD_CIRCLE_OUTLINE_ROUNDED,
                        icon_color=C["dim"], icon_size=22,
                        tooltip="Attach / Image",
                        on_click=lambda e: _plus_menu(),
                    ),
                    ft.Container(
                        content=input_box,
                        bgcolor=C["input"],
                        border_radius=14,
                        border=ft.Border.all(1, C["border"]),
                        expand=True,
                    ),
                    send_btn,
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.END),
                bgcolor=C["bg"],
                padding=_po(left=12, right=12, top=8, bottom=8),
            ),
            ft.Container(
                content=ft.Row([
                    status_lbl,
                    ft.Container(width=12),
                    stop_btn,
                ], spacing=0),
                bgcolor=C["bg"],
                padding=_po(left=20, bottom=6),
            ),
        ], spacing=0, expand=True)

        send_btn.on_click = lambda e: _send()

        if is_admin:
            page.add(admin_dashboard)
        else:
            page.add(ft.Row(
                [sidebar, main_area],
                spacing=0, expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            ))
            _new_chat()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BDL CHATBOT")
    parser.add_argument("--web",  action="store_true",
                        help="Serve as a web app on the local network")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8550)
    args = parser.parse_args()

    if args.web:
        print(f"\n  BDL CHATBOT  —  web mode")
        print(f"  Open on this machine : http://localhost:{args.port}")
        import socket as _sock
        try:
            local_ip = _sock.gethostbyname(_sock.gethostname())
            print(f"  Open on your network : http://{local_ip}:{args.port}")
        except Exception:
            pass
        print()
        ft.app(target=main, view=ft.AppView.WEB_BROWSER,
               host=args.host, port=args.port)
    else:
        ft.app(target=main)