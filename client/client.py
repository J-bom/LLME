# ----- IMPORTS ZONE -----#
import sys
import socket
from pathlib import Path
from PySide6.QtGui import QAction, QFontMetrics, QIcon
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QStackedWidget, QMainWindow, QLabel, QHBoxLayout, QWidget, QFileDialog, QSizePolicy, QMenu, QPushButton, QFrame, QWidget, QMessageBox
from PySide6.QtCore import QFile, QThread, Signal, Qt, QTimer
import threading
from datetime import datetime
import json
from rsa import RSA_CLASS
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import uuid
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import keyring


# ----- CONSTANTS ZONE -----#
SERVER_PORT = 51225
LOG_SAVE_PATH = './ClientLogs'
DEBUG_MODE = True
LOG_TIMEOUTS = False
FILE_CHUNK_SIZE = 1024
KEYRING_SERVICE = "LLME"
AUTH_FILE = Path.home() / ".llme" / "auth.json"

ERROR_MESSAGES = {
    '1':  ("Authentication failed", "Wrong username or password."),
    '2':  ("Authentication failed", "Could not create the account. The username may already be taken."),
    '04': ("Message failed", "The server couldn't process your message. Please try again."),
    '05': ("Tool call failed", "Something went wrong while running a tool. The conversation has been reset — please try again."),
    '06': ("File too large", "That file exceeds the per-file size limit."),
    '07': ("Storage full", "You don't have enough storage quota left to upload that file."),
    '08': ("Authentication failed", "Account Locked due to too many login attempts."),
    '09': ("Server error", "The server reported an unknown error."),
    'Wrong current password': ("Password change failed", "The current password you entered is incorrect."),
}


# ----- GLOBAL VARS ZONE -----#
message_id = 0
is_chat_open = False
log_file = None
log_lock = threading.Lock()
current_settings = {}
send_lock = threading.Lock()


class MCPManager:
    """Spawns local MCP servers and routes tool calls to them."""

    def __init__(self):
        """
        initiallizer for mcp class
        """
        self.tool_to_session = {}
        self.exit_stack = None
        self.loop = None
        self.thread = None
        self.ready = threading.Event()

    def start(self, config_path='mcp_config.json'):
        """
        Run the MCP event loop on a background thread.
        :param config_path: mcp settings file path
        :return: tool descriptors
        """
        if not Path(config_path).exists():
            log('INFO', 'Client', f'no MCP config at {config_path}, skipping')
            self.ready.set()
            return []

        self.thread = threading.Thread(
            target=self.run, args=(config_path,), daemon=True
        )
        self.thread.start()
        self.ready.wait(timeout=30)
        return self.descriptors

    def run(self, config_path):
        """
        try to run servers
        :param config_path: mcp settings
        :return: tool descriptors
        """
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.descriptors = self.loop.run_until_complete(self.init_servers(config_path))
            self.ready.set()
            self.loop.run_forever()
        except Exception as err:
            log('ERROR', 'Client', f'MCP init failed: {err}')
            self.descriptors = []
            self.ready.set()

    async def init_servers(self, config_path):
        """
        read servers from conf file and start them up
        :param config_path: mcp config file
        """
        with open(config_path) as f:
            config = json.load(f)

        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()

        descriptors = []
        for server_name, params in config.items():
            try:
                server_params = StdioServerParameters(
                    command=params['command'],
                    args=params.get('args', []),
                    env=params.get('env'),
                )
                read, write = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                tools_resp = await session.list_tools()
                for tool in tools_resp.tools:
                    self.tool_to_session[tool.name] = session
                    descriptors.append({
                        'name': tool.name,
                        'description': tool.description or '',
                        'schema': tool.inputSchema or {'type': 'object', 'properties': {}},
                    })
                log('INFO', 'Client',
                    f'MCP server "{server_name}" registered {len(tools_resp.tools)} tools')
            except Exception as err:
                log('WARN', 'Client', f'MCP server "{server_name}" failed: {err}')

        return descriptors

    def call_tool(self, tool_name, args):
        """
        Synchronous wrapper — called from the main thread on MCPC arrival.
        :param tool_name: tool name
        :param args: args for tool
        :return: tool result
        """
        session = self.tool_to_session.get(tool_name)
        if session is None:
            return {'error': f'unknown MCP tool: {tool_name}'}

        future = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool_name, args),
            self.loop,
        )
        try:
            result = future.result(timeout=30)
            text_parts = [
                getattr(c, 'text', str(c))
                for c in (result.content or [])
            ]
            return {'result': '\n'.join(text_parts)}
        except Exception as err:
            return {'error': f'tool call failed: {type(err).__name__}: {err}'}

    def stop(self):
        """Cleanly shut down all MCP server subprocesses."""
        if self.loop is None or not self.loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
        try:
            fut.result(timeout=5)
        except Exception as err:
            log('WARN', 'Client', f'MCP shutdown timeout/error: {err}')
        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)

    async def shutdown(self):
        """Close the exit stack inside the same task that opened it."""
        if self.exit_stack is not None:
            try:
                await self.exit_stack.aclose()
            except Exception as err:
                log('WARN', 'Client', f'exit_stack.aclose error: {err}')
            self.exit_stack = None





def load_auth():
    """
    loads auth file
    :return: a dict with host, username and remember
    """
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return {
        "host": data.get("host", ""),
        "username": data.get("username", ""),
        "remember": bool(data.get("remember", False)),
    }


def save_auth(host, username, password, remember):
    """
    saves auth details
    :param host: server addr
    :param username: username
    :param password: password
    :param remember: is remember
    """
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    if remember:
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump({"host": host, "username": username, "remember": True}, f)
        try:
            keyring.set_password(KEYRING_SERVICE, username, password)
        except Exception as err:
            log(action="WARN", origin="Client", message=f"keyring save failed: {err}")
    else:
        try:
            AUTH_FILE.unlink()
        except FileNotFoundError:
            pass
        prev = load_auth()
        if prev.get("username"):
            try:
                keyring.delete_password(KEYRING_SERVICE, prev["username"])
            except Exception:
                pass


def load_saved_password(username):
    """
    loads saved pass
    :param username: username
    :return: password
    """
    if not username:
        return ""
    try:
        return keyring.get_password(KEYRING_SERVICE, username) or ""
    except Exception:
        return ""


def parse_host(text, default_port):
    """
    parses saved host data to useable host info
    :param text: host text
    :param default_port: default server port
    :return: address + port
    """
    text = (text or "").strip()
    if not text:
        return "127.0.0.1", default_port
    if ":" in text:
        host, _, port = text.rpartition(":")
        try:
            return host.strip(), int(port.strip())
        except ValueError:
            return text, default_port
    return text, default_port



class _ChatWindowProxy:
    """Forwards attribute lookups to whichever child page actually owns the widget."""
    def __init__(self, chat_page, settings_page, main_window):
        """
        initiallizer for proxy
        :param chat_page: chat page
        :param settings_page: settings page
        :param main_window: main window
        """
        self._chat_page = chat_page
        self._settings_page = settings_page
        self._main_window = main_window

    def __getattr__(self, name):
        """
        checks for old widgets
        :param name: widget name
        :return: old widget
        """
        for src in (self._chat_page, self._settings_page, self._main_window):
            w = src.findChild(QWidget, name)
            if w is not None:
                return w
        raise AttributeError(f"No widget named {name!r} in any loaded page")



# ----- GUI MANAGEMENT -----#
class ClientApp(QMainWindow):
    """
    main app
    """
    global current_settings

    new_subtitle_signal = Signal(str)
    authinticated_signal = Signal()
    handshake_done = Signal()
    got_preferences_signal = Signal()

    def __init__(self):
        """
        initiallizer for client app, sets everything up.
        """
        super().__init__()
        create_logfile()
        self.loader = QUiLoader()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.mcp = MCPManager()
        self.mcp_descriptors = []

        self.apply_theme(self.load_theme_pref())

        self.login_window = self.load_ui("ui/login.ui")
        self.main_window = self.load_ui("ui/main_window.ui")

        self.chat_page = self.load_ui("ui/chat_page.ui")
        self.settings_page = self.load_ui("ui/settings_page.ui")
        self.main_window.chatPageHost.layout().addWidget(self.chat_page)
        self.main_window.settingsPageHost.layout().addWidget(self.settings_page)
        self.chat_window = _ChatWindowProxy(self.chat_page, self.settings_page, self.main_window)

        self.setWindowTitle("LLME")

        self.stack.addWidget(self.login_window)
        self.stack.addWidget(self.main_window)
        self.ui = self.login_window
        self.stack.setCurrentWidget(self.ui)
        self.setGeometry(100, 100, 1280, 800)

        self.rsa = RSA_CLASS()
        self.aes_key = None
        self.socket = socket.socket()
        self.listener = Listener(self.socket)
        self.listener.message_received.connect(self.handle_reply)

        self.username = ''
        self.current_session_id = None

        self.new_subtitle_signal.connect(self.update_subtitles)
        self.authinticated_signal.connect(self.get_preferences)
        self.got_preferences_signal.connect(self.get_chat_history)


        self.active_file_path = None

        if self.ui:
            self.login_window.btn_login.clicked.connect(self.handle_login)
            self.login_window.btn_signup.clicked.connect(self.handle_signup)
            self.login_window.in_password.returnPressed.connect(self.handle_login)
            self.login_window.in_username.returnPressed.connect(lambda: self.login_window.in_password.setFocus())

            self.main_window.btn_navchat.clicked.connect(self.show_chat)
            self.main_window.btn_navsettings.clicked.connect(self.show_settings)
            self.main_window.btn_navlogout.clicked.connect(self.handle_quit)

            self.main_window.btn_navchat.clicked.connect(self.show_welcome)
            self.main_window.btn_navsessions.clicked.connect(self.toggle_sessions_panel)
            self.main_window.btn_newchatpanel.clicked.connect(self.new_chat_from_panel)

            self._session_row_btns = {}
            self.main_window.btn_navlogout.clicked.connect(self.handle_quit)

            self.chat_page.btn_attach.clicked.connect(self.add_attachment)
            self.chat_page.obj_promptinputchat.returnPressed.connect(self.send_prompt)
            self.chat_page.btn_send.clicked.connect(self.send_prompt)

            self.chat_page.obj_promptinputwelcome.returnPressed.connect(self.send_prompt)
            self.chat_page.btn_sendwelcome.clicked.connect(self.send_prompt)
            self.chat_page.btn_attachwelcome.clicked.connect(self.add_attachment)

            self.chat_page.btn_suggestion1.clicked.connect(
                lambda: self.use_suggestion(
                    self.chat_page.btn_suggestion1,
                    "Summarize https://en.wikipedia.org/wiki/Cybersecurity"
                )
            )
            self.chat_page.btn_suggestion2.clicked.connect(
                lambda: self.use_suggestion(
                    self.chat_page.btn_suggestion2,
                    "Explain how public-key cryptography works to a 12-year-old"
                )
            )
            self.chat_page.btn_suggestion3.clicked.connect(
                lambda: self.use_suggestion(
                    self.chat_page.btn_suggestion3,
                    "Write me a short poem about late-night coding"
                )
            )

            self.chat_page.obj_chatstack.setCurrentIndex(0)
            self.refresh_welcome_greeting()

            self.settings_page.btn_savesettings.clicked.connect(self.save_settings)
            self.settings_page.btn_logoutsettings.clicked.connect(self.handle_logout)
            self.settings_page.btn_export.clicked.connect(self.export_conversation)
            self.settings_page.btn_changepassword.clicked.connect(self.change_password)
            self.settings_page.btn_deleteaccount.clicked.connect(self.delete_account)

            saved_theme = self.load_theme_pref()
            self.settings_page.obj_dropTheme.blockSignals(True)
            self.settings_page.obj_dropTheme.setCurrentText(saved_theme)
            self.settings_page.obj_dropTheme.blockSignals(False)
            self.settings_page.obj_dropTheme.currentTextChanged.connect(self.apply_theme)
            self.settings_page.obj_dropprovider.currentTextChanged.connect(self.on_provider_changed)

            self.settings_page.obj_ragenabled.toggled.connect(self.on_rag_toggled)
            self.settings_page.btn_raguploaddoc.clicked.connect(self.add_attachment)

            self.main_window.obj_pages.setCurrentIndex(0)

            saved = load_auth()
            if saved["host"]:
                self.login_window.in_serverhost.setText(saved["host"])
            if saved["username"]:
                self.login_window.in_username.setText(saved["username"])
            if saved["remember"]:
                self.login_window.chk_rememberme.setChecked(True)
                pw = load_saved_password(saved["username"])
                if pw:
                    self.login_window.in_password.setText(pw)
                    QTimer.singleShot(50, self.handle_login)
            self.show()

    def load_ui(self, filename):
        """
        loads ui from saved files
        :param filename: file name
        :return: ui object
        """
        ui_file = QFile(filename)
        if not ui_file.open(QFile.ReadOnly):
            print(f"Error: {filename} not found")
            return None
        ui_obj = self.loader.load(ui_file, self)
        ui_file.close()
        return ui_obj

    def add_chat_bubble(self, text, sender="user"):
        """
        adds ui chat bubble
        :param text: message data
        :param sender: user/assistant
        """
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(520)
        bubble.setProperty("bubbleRole", sender)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        wrapper_layout = QHBoxLayout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)
        if sender == "user":
            wrapper_layout.addStretch()
            wrapper_layout.addWidget(bubble)
        else:
            wrapper_layout.addWidget(bubble)
            wrapper_layout.addStretch()

        row_widget = QWidget()
        row_widget.setLayout(wrapper_layout)
        row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.chat_page.obj_chatlayout.addWidget(row_widget)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        """Forces the scroll area to snap to the newest message."""
        scrollbar = self.chat_page.obj_chatscrl.verticalScrollBar()
        QTimer.singleShot(10, lambda: scrollbar.setValue(scrollbar.maximum()))

# ----- ACTIONS (GUI) MANAGEMENT -----#
    def handle_login(self):
        """
        handles authintication of existing user
        """
        host_text = self.login_window.in_serverhost.text()
        self.username = self.login_window.in_username.text().strip()
        password = self.login_window.in_password.text()
        remember = self.login_window.chk_rememberme.isChecked()

        if not self.username or not password:
            self.login_window.lbl_loginstatus.setText("Username and password are required.")
            return

        if not self.ensure_connected(host_text):
            return

        save_auth(self.login_window.in_serverhost.text().strip(),
                   self.username, password, remember)

        send_aes_request(self.socket, 'LGNR', [self.username, password], self.aes_key)

    def handle_signup(self):
        """
        handle authintication and user creation for none existing users
        """
        host_text = self.login_window.in_serverhost.text()
        self.username = self.login_window.in_username.text().strip()
        password = self.login_window.in_password.text()

        if not self.username or not password:
            self.login_window.lbl_loginstatus.setText("Username and password are required.")
            return

        if not self.ensure_connected(host_text):
            return

        send_aes_request(self.socket, 'SGNR', [self.username, password], self.aes_key)

    def ensure_connected(self, host_text):
        """
        Open the socket + wait for the AES handshake to complete.
        :param host_text: host information
        :return: success bool
        """
        if getattr(self, "_socket_connected", False) and self.aes_key is not None:
            return True

        if not getattr(self, "_socket_connected", False):
            host, port = parse_host(host_text, SERVER_PORT)
            self.server_addr = host
            try:
                self.socket.connect((host, port))
            except Exception as err:
                self.login_window.lbl_loginstatus.setText(f"Could not connect to {host}:{port} — {err}")
                log(action='ERROR', origin='Client', message=f'connect failed: {err}')
                self.socket = socket.socket()
                self.listener = Listener(self.socket)
                self.listener.message_received.connect(self.handle_reply)
                return False
            self.listener.start()
            send_requests(self.socket, 'PUBR', [base64.b64encode(self.rsa.public_key).decode()])
            self._socket_connected = True

        self.login_window.lbl_loginstatus.setText("Connecting…")
        QApplication.processEvents()
        deadline = 5.0
        waited = 0.0
        while self.aes_key is None and waited < deadline:
            QThread.msleep(50)
            QApplication.processEvents()
            waited += 0.05

        if self.aes_key is None:
            self.login_window.lbl_loginstatus.setText("Server did not complete handshake. Try again.")
            return False

        self.login_window.lbl_loginstatus.setText("")
        return True

    def change_password(self):
        """
        send password change request
        """
        cur = self.settings_page.in_currentpassword.text()
        new = self.settings_page.in_newpassword.text()
        confirm = self.settings_page.in_newpasswordconfirm.text()
        if not cur or not new or not confirm:
            self.settings_page.lbl_accountstatus.setText("Fill in all three password fields.")
            return
        if new != confirm:
            self.settings_page.lbl_accountstatus.setText("New password fields don't match.")
            return
        self.settings_page.lbl_accountstatus.setText("Changing password…")
        send_aes_request(self.socket, 'CHPW', [cur, new], self.aes_key)

    def delete_account(self):
        """
        send account deletion request
        """
        confirm = QMessageBox.question(
            self, "Delete account",
            f"Permanently delete account '{self.username}'?\n\n"
            "All your chats, settings, and uploaded documents will be erased.\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        send_aes_request(self.socket, 'DELU', [], self.aes_key)

    def add_attachment(self):
        """
        send attachment to server for RAG
        """
        file_path, file_types_filter = QFileDialog.getOpenFileName(None, "Select a file to attach", '', '')
        file_path = Path(file_path)
        if file_path.exists():
            self.active_file_path = file_path
            file_size = file_path.stat().st_size
            chunk_amount = file_size // FILE_CHUNK_SIZE
            if chunk_amount == 0: chunk_amount = 1
            send_aes_request(self.socket,'RADD',[file_path.name, str(file_size), str(chunk_amount)], self.aes_key)

    def send_prompt(self):
        """
        send prompt to server for model processing
        """

        if (self.chat_window.obj_dropprovider.currentText() == 'Gemini'
                and not self.chat_window.in_geminiapi.text().strip()):
            QMessageBox.warning(self, "Gemini API key required","Add a Gemini API key in Settings → Model → Gemini API.")
            return

        message = ''
        message = (
                self.chat_page.obj_promptinputchat.text().strip()
                or self.chat_page.obj_promptinputwelcome.text().strip()
        )
        if not message:
            return
        if self.current_session_id is None:
            self.start_new_session()
            self._record_new_local_session(self.current_session_id, message)

        self.show_messages()

        if self.current_session_id is None:
            self.current_session_id = str(uuid.uuid4())
            self._record_new_local_session(self.current_session_id, preview=message)

        self.add_chat_bubble(message, sender="user")
        send_aes_request(self.socket, 'SMSG', [self.current_session_id,message], self.aes_key)
        self.chat_page.obj_promptinputchat.clear()
        self.chat_page.obj_promptinputwelcome.clear()

    def handle_quit(self):
        """Quit the app gracefully — keeps remember-me intact."""
        try:
            self.mcp.stop()
        except Exception as e:
            log('WARN', 'Client', f'MCP stop error: {e}')
        try:
            if getattr(self, 'aes_key', None):
                send_aes_request(self.socket, 'EXIT', [self.username], aes_key=self.aes_key)
        except Exception:
            pass
        try:
            self.socket.close()
        except Exception:
            pass
        sys.exit()

    def closeEvent(self, event):
        """X button — same effect as quit."""
        self.handle_quit()
        event.accept()

    def handle_logout(self):
        """True logout — clear remembered creds, return to login screen."""
        save_auth("", "", "", remember=False)
        self.handle_quit()

    def apply_theme(self, theme_name):
        """
        Load the QSS file for the given theme and apply it to the whole app.
        :param theme_name: name of theme to apply
        """
        path = "ui/style.qss" if theme_name == "Dark" else "ui/style_light.qss"
        try:
            with open(path, "r", encoding="utf-8") as f:
                QApplication.instance().setStyleSheet(f.read())
        except FileNotFoundError:
            log('WARN', 'Client', f'theme {path} not found')
        self._current_theme = theme_name
        self.persist_theme(theme_name)

    def persist_theme(self, theme_name):
        """
        Write theme to ~/.llme/auth.json without touching the rest of the file.
        :param theme_name: name of theme to apply
        """
        try:
            if AUTH_FILE.exists():
                with open(AUTH_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data["theme"] = theme_name
        AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_theme_pref(self):
        """Read saved theme. Defaults to Dark."""
        try:
            with open(AUTH_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("theme", "Dark")
        except (FileNotFoundError, json.JSONDecodeError):
            return "Dark"

    def on_provider_changed(self, provider):
        """
        Gray out the active-model dropdown for engines that have only one model.
        :param provider: model provider
        """
        cb = self.settings_page.obj_dropmodelllamacpp
        if provider in ("AlterEgo", "Gemini"):
            cb.setEnabled(False)
            if provider == "AlterEgo" and cb.findText("AlterEgo") >= 0:
                cb.setCurrentText("AlterEgo")
            elif provider == "Gemini" and cb.findText("gemini-2.5-flash") >= 0:
                cb.setCurrentText("gemini-2.5-flash")
        else:
            cb.setEnabled(True)

    def refresh_rag_panel(self):
        """Ask the server for the document list + quota."""
        if getattr(self, "aes_key", None):
            send_aes_request(self.socket, 'RLST', [], self.aes_key)

    def on_rag_toggled(self, checked):
        """
        User toggled RAG enabled — push to server preferences.
        :param checked: toggle status
        """
        if not getattr(self, "aes_key", None):
            return
        payload = json.dumps({"rag_enabled": 1 if checked else 0})
        send_aes_request(self.socket, 'SETP', [payload], self.aes_key)
        self.settings_page.ragQuotaCard.setVisible(checked)
        self.settings_page.ragDocsCard.setVisible(checked)

    def delete_rag_doc(self, doc_name):
        """
        send request to delete doc from rag
        :param doc_name: document name
        """
        confirm = QMessageBox.question(
            self,
            "Delete document",
            f"Delete '{doc_name}' from RAG?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        send_aes_request(self.socket, 'RDEL', [doc_name], self.aes_key)

    def populate_rag_panel(self, docs, used_bytes, quota_bytes):
        """
        show rag progress bar quota
        :param docs: user documents
        :param used_bytes: user used bytes
        :param quota_bytes: max quota bytes
        """
        pct = int(100 * used_bytes / quota_bytes) if quota_bytes else 0
        pct = max(0, min(100, pct))
        self.settings_page.obj_ragquota.setValue(pct)

        def fmt(b):
            if b < 1024:        return f"{b} B"
            if b < 1024 ** 2:     return f"{b / 1024:.1f} KB"
            if b < 1024 ** 3:     return f"{b / (1024 ** 2):.1f} MB"
            return f"{b / (1024 ** 3):.2f} GB"

        self.settings_page.lbl_ragquotatext.setText(f"{fmt(used_bytes)} / {fmt(quota_bytes)} used")

        layout = self.settings_page.obj_ragdocslayout
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not docs:
            empty = QLabel("(no documents uploaded yet)")
            empty.setProperty("helpText", True)
            empty.setAlignment(Qt.AlignCenter)
            layout.insertWidget(0, empty)
            return

        for doc in docs:
            row = QFrame()
            row.setProperty("ragDocRow", True)
            h = QHBoxLayout(row)
            h.setContentsMargins(10, 8, 10, 8)
            name_lbl = QLabel(doc["name"])
            chunks_lbl = QLabel(f"{doc['chunks']} chunks")
            chunks_lbl.setProperty("helpText", True)
            del_btn = QPushButton("Delete")
            del_btn.setProperty("danger", True)
            del_btn.setCursor(Qt.PointingHandCursor)
            del_btn.setMinimumHeight(28)
            del_btn.clicked.connect(lambda _checked=False, n=doc["name"]: self.delete_rag_doc(n))

            h.addWidget(name_lbl, stretch=1)
            h.addWidget(chunks_lbl)
            h.addWidget(del_btn)

            layout.insertWidget(layout.count() - 1, row)


    def show_chat(self):
        """
        show chat page
        """
        self.main_window.obj_pages.setCurrentIndex(0)
        self.main_window.btn_navchat.setChecked(True)
        self.main_window.btn_navsettings.setChecked(False)

    def show_settings(self):
        """
        show settings page
        """
        self.close_sessions_panel()
        self.main_window.obj_pages.setCurrentIndex(1)
        self.main_window.btn_navchat.setChecked(False)
        self.main_window.btn_navsettings.setChecked(True)
        self.refresh_rag_panel()

    def hide_settings(self):
        """
        hide settings page
        """
        self.show_welcome()

    def toggle_sessions_panel(self):
        """Show the sessions slide-out panel."""
        panel = self.main_window.sessionsPanel
        panel.setVisible(not panel.isVisible())
        self.main_window.btn_navsessions.setChecked(panel.isVisible())

    def close_sessions_panel(self):
        """hide the sessions slide-out panel."""
        self.main_window.sessionsPanel.setVisible(False)
        self.main_window.btn_navsessions.setChecked(False)

    def new_chat_from_panel(self):
        """+ New chat button at the top of the sessions panel."""
        self.close_sessions_panel()
        self.show_welcome()

    def render_sessions_panel(self):
        """Re-render the session row buttons from self._known_sessions."""
        layout = self.main_window.obj_sessionslayout
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._session_row_btns = {}

        if not self._known_sessions:
            empty = QLabel("(no saved sessions yet)")
            empty.setProperty("helpText", True)
            empty.setAlignment(Qt.AlignCenter)
            layout.insertWidget(0, empty)
            return

        for sess in self._known_sessions:
            sid = sess.get("session_id") or sess.get("id")
            if not sid:
                continue
            preview = sess.get("preview") or sess.get("title") or sess.get("name") or ""
            label = self.format_session_label(preview, sid)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            btn = QPushButton(label)
            btn.setProperty("sessionRow", True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(preview or sid)
            btn.setMinimumHeight(40)
            btn.setProperty("active", sid == self.current_session_id)
            btn.clicked.connect(lambda _checked=False, _sid=sid: self.switch_to_session(_sid))

            del_btn = QPushButton("×")
            del_btn.setProperty("sessionDelete", True)
            del_btn.setCursor(Qt.PointingHandCursor)
            del_btn.setToolTip("Delete session")
            del_btn.setFixedSize(28, 28)
            del_btn.clicked.connect(
                lambda _checked=False, _sid=sid, _preview=preview: self.delete_session(_sid, _preview))

            row_layout.addWidget(btn, stretch=1)
            row_layout.addWidget(del_btn)

            layout.insertWidget(layout.count() - 1, row)
            self._session_row_btns[sid] = btn

    def delete_session(self, session_id, preview):
        """
        delete session
        :param session_id: session id
        :param preview: preiew of session
        """
        label = preview if preview else f"chat {str(session_id)[:8]}"
        confirm = QMessageBox.question(
            self, "Delete session",
            f"Delete this session?\n\n\"{label[:80]}\"\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        send_aes_request(self.socket, 'DELS', [session_id], self.aes_key)

    def format_session_label(self, preview, sid):
        """
        Truncate the preview so it fits one row.
        Falls back to a short id snippet if the preview is empty.
        :param preview: current preview
        :param sid: session id
        :return: session label
        """
        text = (preview or "").strip().replace("\n", " ")
        if not text:
            return f"chat {sid[:8]}…" if isinstance(sid, str) else f"chat {sid}"
        if len(text) > 40:
            text = text[:38].rstrip() + "…"
        return text

    def switch_to_session(self, session_id):
        """User picked an existing session — load it and show the messages view."""
        self.current_session_id = session_id

        while self.chat_page.obj_chatlayout.count():
            item = self.chat_page.obj_chatlayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sess = next(
            (s for s in (self._known_sessions or [])
             if (s.get("session_id") or s.get("id")) == session_id),
            None
        )
        title = self.format_session_label(
            (sess or {}).get("preview", ""),
            session_id
        ) if sess else f"chat {str(session_id)[:8]}…"
        self.chat_page.lbl_messagessessiontitle.setText(title)

        for sid, btn in self._session_row_btns.items():
            btn.setProperty("active", sid == session_id)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        send_aes_request(self.socket, 'GETH', [session_id], self.aes_key)

        self.close_sessions_panel()
        self.show_messages()

    def refresh_welcome_greeting(self):
        """Set the time-aware greeting line."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            part = "Good morning"
        elif 12 <= hour < 18:
            part = "Good afternoon"
        elif 18 <= hour < 22:
            part = "Good evening"
        else:
            part = "Working late"
        name = self.username or "there"
        self.chat_page.lbl_welcomegreeting.setText(f"{part}, {name}")

    def use_suggestion(self, card_btn, text):
        """
        Flash the card briefly and populate the welcome input.
        :param card_btn: card button obj
        :param text: prompt
        """
        self.chat_page.obj_promptinputwelcome.setText(text)
        self.chat_page.obj_promptinputwelcome.setFocus()
        self.chat_page.obj_promptinputwelcome.selectAll()

        card_btn.setProperty("suggestionFlash", True)
        card_btn.style().unpolish(card_btn)
        card_btn.style().polish(card_btn)
        QTimer.singleShot(700, lambda: self.end_suggestion_flash(card_btn))

    def end_suggestion_flash(self, card_btn):
        """
        end flash on card button suggestion
        :param card_btn: card object
        """
        card_btn.setProperty("suggestionFlash", False)
        card_btn.style().unpolish(card_btn)
        card_btn.style().polish(card_btn)

    def show_welcome(self):
        """Show the welcome screen. New chat starts here."""
        self.close_sessions_panel()
        self.current_session_id = None
        self.chat_page.obj_chatstack.setCurrentIndex(0)
        self.refresh_welcome_greeting()
        self.main_window.btn_navchat.setChecked(True)
        self.main_window.btn_navsettings.setChecked(False)
        self.main_window.obj_pages.setCurrentIndex(0)

    def show_messages(self):
        """Show the message scroll view (used after first send or session switch)."""
        self.chat_page.obj_chatstack.setCurrentIndex(1)
        self.main_window.btn_navchat.setChecked(True)
        self.main_window.btn_navsettings.setChecked(False)
        self.main_window.obj_pages.setCurrentIndex(0)

    def export_conversation(self):
        """Save the currently-loaded chat history to a markdown file."""
        if self.chat_page.obj_chatlayout.count() == 0:
            QMessageBox.information(self, "Export", "No messages to export.")
            return

        default_name = "conversation.md"
        if self.current_session_id:
            default_name = f"conversation_{str(self.current_session_id)[:8]}.md"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export conversation", default_name, "Markdown (*.md);;Text (*.txt)"
        )
        if not path:
            return

        lines = [f"# Conversation export\n"]
        if self.current_session_id:
            lines.append(f"Session: `{self.current_session_id}`\n")
        lines.append("")

        layout = self.chat_page.obj_chatlayout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget() if item else None
            if w is None:
                continue
            for child in w.findChildren(QLabel):
                role = child.property("bubbleRole") or "assistant"
                text = child.text()
                if not text:
                    continue
                if role == "user":
                    lines.append(f"**You:** {text}\n")
                else:
                    lines.append(f"**Assistant:** {text}\n")
                lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            QMessageBox.information(self, "Export", f"Saved to {path}")
        except Exception as err:
            QMessageBox.warning(self, "Export failed", str(err))

    def load_models(self, models):
        """
        populates active model dropdown
        :param models: model list
        """
        self.chat_window.obj_dropmodelllamacpp.addItems(models['Llama'])

    def apply_settings(self, preferences):
        """
        apply settings received from server
        :param preferences: new settings
        """
        global current_settings
        self.chat_window.obj_dropprovider.setCurrentText(preferences['model_provider'])
        self.chat_window.obj_dropmodelllamacpp.setCurrentText(preferences['model_active'])
        self.chat_window.obj_tempselect.setValue(float(preferences['model_temperature']))
        self.chat_window.obj_pselect.setValue(float(preferences['model_p']))
        self.chat_window.obj_kselect.setValue(float(preferences['model_k']))
        self.chat_window.obj_maxtselect.setValue(preferences['model_maxoutputtokens'])
        self.chat_window.in_sysprompt.setPlainText(preferences['model_systemprompt'])
        self.chat_window.in_geminiapi.setText(preferences['gemini_apikey'])

        cb = self.settings_page.obj_ragenabled
        cb.blockSignals(True)
        cb.setChecked(bool(preferences.get('rag_enabled', 1)))
        cb.blockSignals(False)

        self.settings_page.ragQuotaCard.setVisible(cb.isChecked())
        self.settings_page.ragDocsCard.setVisible(cb.isChecked())
        self.dump_settings()


    def dump_settings(self):
        """
        dump current settings
        """
        current_settings['model_provider'] = self.chat_window.obj_dropprovider.currentText()
        current_settings['model_active'] = self.chat_window.obj_dropmodelllamacpp.currentText()
        current_settings['model_temperature'] = self.chat_window.obj_tempselect.value()
        current_settings['model_p'] = self.chat_window.obj_pselect.value()
        current_settings['model_k'] = self.chat_window.obj_kselect.value()
        current_settings['model_maxoutputtokens'] = self.chat_window.obj_maxtselect.value()
        current_settings['model_systemprompt'] = self.chat_window.in_sysprompt.toPlainText()
        current_settings['gemini_apikey'] = self.chat_window.in_geminiapi.text()


    def check_updated_settings(self):
        """
        check if updated settings
        :return: list of updated settings
        """
        global current_settings
        new_settings = {}
        if current_settings['model_provider'] != self.chat_window.obj_dropprovider.currentText():
            new_settings['model_provider'] = self.chat_window.obj_dropprovider.currentText()
            current_settings['model_provider'] = self.chat_window.obj_dropprovider.currentText()
        if current_settings['model_active'] != self.chat_window.obj_dropmodelllamacpp.currentText():
            new_settings['model_active'] = self.chat_window.obj_dropmodelllamacpp.currentText()
            current_settings['model_active'] = self.chat_window.obj_dropmodelllamacpp.currentText()
        if current_settings['model_temperature'] != self.chat_window.obj_tempselect.value():
            new_settings['model_temperature'] = self.chat_window.obj_tempselect.value()
            current_settings['model_temperature'] = self.chat_window.obj_tempselect.value()
        if current_settings['model_p'] != self.chat_window.obj_pselect.value():
            new_settings['model_p'] = self.chat_window.obj_pselect.value()
            current_settings['model_p'] = self.chat_window.obj_pselect.value()
        if current_settings['model_k'] != self.chat_window.obj_kselect.value():
            new_settings['model_k'] = self.chat_window.obj_kselect.value()
            current_settings['model_k'] = self.chat_window.obj_kselect.value()
        if current_settings['model_maxoutputtokens'] != self.chat_window.obj_maxtselect.value():
            new_settings['model_maxoutputtokens'] = self.chat_window.obj_maxtselect.value()
            current_settings['model_maxoutputtokens'] = self.chat_window.obj_maxtselect.value()
        if current_settings['model_systemprompt'] != self.chat_window.in_sysprompt.toPlainText():
            new_settings['model_systemprompt'] = self.chat_window.in_sysprompt.toPlainText()
            current_settings['model_systemprompt'] = self.chat_window.in_sysprompt.toPlainText()
        if current_settings['gemini_apikey'] != self.chat_window.in_geminiapi.text():
            new_settings['gemini_apikey'] = self.chat_window.in_geminiapi.text()
            current_settings['gemini_apikey'] = self.chat_window.in_geminiapi.text()
        new_settings["rag_enabled"]: 1 if self.settings_page.obj_ragenabled.isChecked() else 0
        current_settings['rag_enabled']: 1 if self.settings_page.obj_ragenabled.isChecked() else 0

        return new_settings

    def save_settings(self):
        """
        save updated settings
        """
        settings = self.check_updated_settings()
        encoded_settings = json.dumps(settings)
        send_aes_request(self.socket, 'SETP', [encoded_settings], self.aes_key)

    def update_subtitles(self, text):
        """
        show reply from Ai on screen
        :param text: reply from Ai
        """
        self.add_chat_bubble(text, sender="assistant")

    def get_preferences(self):
        """
        ask for user settings from the server
        """
        send_aes_request(self.socket, 'GETP', [], self.aes_key)

    def get_chat_history(self):
        """
        ask for chat history from the server
        :return:
        """
        send_aes_request(self.socket, 'RSES', [], self.aes_key)

    def handle_mcp_call(self, call_id, tool_name, args):
        """
        handles mcp calls from the server
        :param call_id: tool call id
        :param tool_name: tool name
        :param args: tool args
        """
        result = self.mcp.call_tool(tool_name, args)
        result64 = base64.b64encode(json.dumps(result).encode()).decode()
        send_aes_request(
            self.socket,
            'MCPR',
            [call_id, result64],
            self.aes_key,
        )

    def register_mcp_tools(self):
        """
        register all avilable mcp tools.
        """
        if getattr(self, "_mcp_registered", False):
            return
        self.mcp_descriptors = self.mcp.start('mcp_config.json')
        if self.mcp_descriptors:
            send_aes_request(
                self.socket,
                'MCPL',
                [json.dumps(self.mcp_descriptors)],
                self.aes_key,
            )
        self._mcp_registered = True

    def update_history(self, messages):
        """
        load session history
        :param messages: message history
        """
        self.clear_chat_bubbles()
        dict_messages = json.loads(messages)
        for message in dict_messages:
            self.add_chat_bubble(message['content'], message['role'])
        if self.chat_page.obj_chatlayout.count() == 0:
            self.show_welcome()
        else:
            self.show_messages()

    def handle_session_change(self, index):
        """
        Triggered when the user picks a different session from the dropdown.
        :param index: session index
        """
        if index < 0:
            return
        new_session_id = self.chat_window.obj_activesession.itemData(index)
        if new_session_id is None or new_session_id == self.current_session_id:
            return

        self.current_session_id = new_session_id
        self.clear_chat_bubbles()
        send_aes_request(self.socket, 'GETH', [self.current_session_id], self.aes_key)

    def start_new_session(self):
        """Create a fresh session id locally. The server-side row gets created on first SMSG."""
        self.current_session_id = str(uuid.uuid4())
        while self.chat_page.obj_chatlayout.count():
            item = self.chat_page.obj_chatlayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


    def clear_chat_bubbles(self):
        """Remove all chat bubbles from the layout."""
        layout = self.chat_page.obj_chatlayout
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _record_new_local_session(self, session_id, preview):
        """
        Insert a brand-new session at the top of our local list and refresh the panel.
        :param session_id: session id
        :param preview: session preview
        :return:
        """
        if not hasattr(self, "_known_sessions") or self._known_sessions is None:
            self._known_sessions = []
        if not any((s.get("session_id") or s.get("id")) == session_id for s in self._known_sessions):
            self._known_sessions.insert(0, {
                "session_id": session_id,
                "preview": preview,
                "last_updated": "",
            })
        self.render_sessions_panel()

    def populate_sessions(self, sessions):
        """
        populate sessions screen
        :param sessions: list of sessions
        """
        self.populate_sessions_data(sessions)
        self.render_sessions_panel()


    def populate_sessions_data(self, sessions_list):
        """
        populates sessions data for sessions screen
        :param sessions_list: list of sessions
        :return:
        """
        self._known_sessions = sessions_list or []


# ----- NETWORKING (GUI) MANAGEMENT -----#
    def handle_reply(self, message):
        """
        handle reply from the server
        :param message: message
        """
        global message_id, send_lock
        split_message = (message.decode()).split('~')
        current_message_id = int(split_message[0])
        with send_lock:
            if current_message_id != message_id:
                log(action='ERROR', origin='Server', recipient='Client', message='message id missmatch error!')
            message_id += 1
            code = split_message[1]
            data_fields = split_message[2:]

            if code == 'RSAM':
                code, data_fields = decrypt_rsa(data_fields[0], self.rsa)

            if code == 'AESM':
                code, data_fields = decrypt_aes(data_fields[0], self.aes_key, data_fields[2], data_fields[1])
                print(data_fields)


        match code:
            case 'LGNS':
                self.authinticated_signal.emit()
                self.stack.setCurrentWidget(self.main_window)
                self.show_welcome()
            case 'SGNS':
                self.authinticated_signal.emit()
                self.stack.setCurrentWidget(self.main_window)

            case 'CHPS':
                self.settings_page.lbl_accountstatus.setText("Password changed.")
                self.settings_page.in_currentpassword.clear()
                self.settings_page.in_newpassword.clear()
                self.settings_page.in_newpasswordconfirm.clear()
                saved = load_auth()
                if saved.get("remember"):
                    new_pw = self.settings_page.in_newpassword.text()
                    save_auth("", "", "", remember=False)

            case 'CHPF':
                self.settings_page.lbl_accountstatus.setText(f"Password change failed: {data_fields[0]}")

            case 'DELA':
                save_auth("", "", "", remember=False)
                QMessageBox.information(self, "Account deleted", "Your account was removed. The app will now close.")
                self.handle_quit()

            case 'SSES':
                sessions = json.loads(data_fields[0])
                self.populate_sessions(sessions)
                self.register_mcp_tools()

            case 'DELP':
                ok, deleted_sid = data_fields[0], data_fields[1]
                if ok == '1':
                    self._known_sessions = [
                        s for s in (self._known_sessions or [])
                        if (s.get("session_id") or s.get("id")) != deleted_sid
                    ]
                    if self.current_session_id == deleted_sid:
                        self.current_session_id = None
                        while self.chat_page.obj_chatlayout.count():
                            item = self.chat_page.obj_chatlayout.takeAt(0)
                            if item.widget():
                                item.widget().deleteLater()
                        self.show_welcome()
                    self.render_sessions_panel()

            case 'CHAH':
                self.register_mcp_tools()
                self.update_history(data_fields[0])
            case 'SREP':
                model_response = data_fields[1]
                self.new_subtitle_signal.emit(model_response)
            case 'USRP':
                preferences = json.loads(data_fields[0])
                available_models = json.loads(data_fields[1])
                self.apply_settings(preferences)
                self.load_models(available_models)
                self.got_preferences_signal.emit()
            case 'PUBK':
                server_public = base64.b64decode(data_fields[0])
                self.rsa.set_other_public(server_public)
                send_rsa_request(self.socket,'AESR',[],self.rsa)


            case 'AESK':
                encoded_key = data_fields[0]
                key = base64.b64decode(encoded_key)
                self.aes_key = key
                self.handshake_done.emit()

            case 'MCPC':
                call_id = data_fields[0]
                tool_name = data_fields[1]
                args = json.loads(base64.b64decode(data_fields[2]).decode())

                threading.Thread(
                    target=self.handle_mcp_call,
                    args=(call_id, tool_name, args),
                    daemon=True,
                ).start()

            case 'RLSP':
                docs = json.loads(data_fields[0])
                used = int(data_fields[1])
                quota = int(data_fields[2])
                self.populate_rag_panel(docs, used, quota)

            case 'RDLP':
                if data_fields[0] == '1':
                    log(action='INFO', origin='Client', message='document deleted')
                self.refresh_rag_panel()

            case 'RACK':
                port = int(data_fields[0])
                t = threading.Thread(target=send_file_attachment, args=(self.active_file_path, self.server_addr, port, self.aes_key))
                t.start()
                self.refresh_rag_panel()
            case 'SUCC':
                log('INFO', 'Server', 'Command success!')
            case 'EROR':
                code = data_fields[0] if data_fields else '06'
                title, body = ERROR_MESSAGES.get(
                    code,
                    ("Error", f"The server reported an error (code: {code}).")
                )
                log('ERROR', 'Server', f'received EROR code={code}')
                QMessageBox.warning(self, title, body)
            case _:
                print('unknown message.')



# ----- NETWORKING MANAGEMENT -----#

def send_file_attachment(file_path, server_address, port, aes_key):
    """
    send file attachment to server in chunks
    :param file_path: file path
    :param server_address: server address
    :param port: server port
    :param aes_key: aes key
    """
    file_send_socket = socket.socket()
    try:
        file_send_socket.connect((server_address, port))
        file_size = file_path.stat().st_size
        data_read = 0

        with open(file_path, 'rb') as f:
            while data_read < file_size:
                chunk = f.read(FILE_CHUNK_SIZE)
                data_read += len(chunk)
                encoded_chunk = base64.b64encode(chunk).decode()

                message = build_requests('RCHK', [encoded_chunk]).encode()
                cipher = AES.new(aes_key, AES.MODE_GCM)
                encrypted_message, tag = cipher.encrypt_and_digest(message)
                encoded_msg = base64.b64encode(encrypted_message).decode()
                encoded_tag = base64.b64encode(tag).decode()
                encoded_nonce = base64.b64encode(cipher.nonce).decode()
                final_payload = f"0~AESM~{encoded_msg}~{encoded_tag}~{encoded_nonce}\n".encode()

                try:
                    file_send_socket.sendall(final_payload)
                except (ConnectionResetError, BrokenPipeError, OSError) as err:
                    log('WARN', 'Client',
                        f'Upload aborted — server closed the connection: {err}')
                    break

    finally:
        file_send_socket.close()




class Listener(QThread):
    """
    listener class for messages from the server
    """
    message_received = Signal(bytes)

    def __init__(self, socket):
        """
        initiallizer for listener
        :param socket: socket to listen on
        """
        super().__init__()
        self.socket = socket
        self.running = True
        self.socket.settimeout(0.2)
        self.buffer = b''

    def run(self):
        """
        listener to messages, signal when got one
        """
        while self.running:
            try:
                data = self.socket.recv(65536)
                if not data:
                    self.running = False
                    break
                self.buffer += data
                while b'\n' in self.buffer:
                    message, self.buffer = self.buffer.split(b'\n', 1)
                    log(action='MESSAGE_RECV', origin='Server',
                        recipient='Client', message=message)
                    self.message_received.emit(message)
            except socket.timeout:
                if LOG_TIMEOUTS:
                    log(action='WARN', origin='Client',
                        message='timeout from listener thread')
            except Exception as err:
                log(action='ERROR', origin='Server', recipient='Client',
                    message=err)
                self.running = False

def build_requests(message_code, fields, final=False):
    """
    build requests to the server
    :param message_code: message code
    :param fields: daa fields
    :param final: is it the final build?
    :return: formatted message
    """
    global message_id
    message = str(message_id) + '~' + message_code
    if fields or fields != None or fields != []:
        message += '~' + '~'.join(field for field in fields)
    if final:
        message_id += 1
    return message

def send_requests(server_socket, code, fields):
    """
    send requests to the server
    :param server_socket: server socket
    :param code: message code
    :param fields: data fields
    """
    global send_lock
    with send_lock:
        message = build_requests(code, fields, True)
        encoded_message = message.encode() + b'\n'
        server_socket.send(encoded_message)
        log(action='MESSAGE_SEND', origin='Client', recipient='Server', message=message)

def send_aes_request(server_socket, code, fields, aes_key):
    """
    wrapper for sending AES encrypted messages to the server
    :param server_socket: server socket
    :param code: message code
    :param fields: data fields
    :param aes_key: aes key
    """
    global send_lock
    with send_lock:
        message = build_requests(code, fields).encode()
        cipher = AES.new(aes_key, AES.MODE_GCM)
        encrypted_message, tag = cipher.encrypt_and_digest(message)
        nonce = cipher.nonce
        encoded_message = base64.b64encode(encrypted_message).decode()
        encoded_tag = base64.b64encode(tag).decode()
        encoded_nonce = base64.b64encode(nonce).decode()
        outer_message = build_requests('AESM', [encoded_message, encoded_tag, encoded_nonce], True)
        encoded_outer = outer_message.encode() + b'\n'
        server_socket.send(encoded_outer)
        log(action='MESSAGE_SEND', origin='Client', recipient='Server', message=outer_message)


def send_rsa_request(server_socket, code, fields, rsa_obj):
    """
    wrapper for sending RSA encrypted messages to the server
    :param server_socket: server socket
    :param code: message code
    :param fields: data fields
    :param rsa_obj: RSA managment object
    """
    global send_lock
    with send_lock:
        message = build_requests(code, fields)
        encrypted_message = rsa_obj.encrypt_RSA(message)
        encoded_message = base64.b64encode(encrypted_message).decode()
        outer_message = build_requests('RSAM', [encoded_message], True)
        encoded_outer = outer_message.encode() + b'\n'
        server_socket.send(encoded_outer)
        log(action='MESSAGE_SEND', origin='Client', recipient='Server', message=outer_message)

def decrypt_rsa(message, rsa_obj):
    """
    decrypt RSA messages
    :param message: encrypted message
    :param rsa_obj: rsa object
    :return: message code + fields decrypted
    """
    decoded_message = base64.b64decode(message)
    decrypted_message = rsa_obj.decrypt_RSA(decoded_message).decode()

    splitted_message = decrypted_message.split('~')
    message_code = splitted_message[1]
    data_fields = splitted_message[2:]
    return message_code, data_fields


def decrypt_aes(message, aes_key, nonce, tag):
    """
    decrypt AES messages
    :param message: encrypted message
    :param aes_key: aes key
    :param nonce: aes nonce
    :param tag: aes verification tag
    :return: message code + fields decrypted
    """
    decoded_message = base64.b64decode(message)
    decoded_tag = base64.b64decode(tag)
    decoded_nonce = base64.b64decode(nonce)
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=decoded_nonce)

    decrypted_message = cipher.decrypt_and_verify(decoded_message, decoded_tag).decode()
    splitted_message = decrypted_message.split('~')

    message_code = splitted_message[1]
    data_fields = splitted_message[2:]

    return message_code, data_fields

    message_code = splitted_message[1]
    data_fields = splitted_message[2:]

    for i in range(len(data_fields)):
        data_fields[i] = base64.b64decode(data_fields[i])

    return message_code, data_fields



# ----- PROGRAM MANAGEMENT -----#

def create_logfile():
    """
    creates the log file
    """
    global log_file
    log_dir = Path(LOG_SAVE_PATH)

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file_name = f'SERVER_LOG_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    log_file = log_dir / log_file_name

    try:
        with open(log_file, 'a') as f:
            pass

    except IOError as err:
        raise Exception(f'[ERROR] LOG CREATION {log_file} FAILED')


def log(action, origin, message, recipient=None):
    """
    logs actions done on the client
    :param action: action
    :param origin: origin
    :param message: message to log
    :param recipient: if there is a network action with a recipient mention them
    """
    global log_file
    global log_lock

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    message_data = ''
    if recipient:
        log_message_data = f'Origin: {origin} | Recipient: {recipient} | {message}'
    else:
        log_message_data = message
    log_message = f'[{action} | {current_time} | {log_message_data}]\n'

    if DEBUG_MODE:
        print(log_message, end='')

    log_lock.acquire()

    try:
        with open(log_file, 'a') as f:
            f.write(log_message)

    except IOError as err:
        print(f'[ERROR] UNABLE TO WRITE LOG FILE TO {log_file}')

    log_lock.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("assets/yblogo.ico"))
    main_app = ClientApp()
    sys.exit(app.exec())