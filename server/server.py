# ----- IMPORTS ZONE -----#
import errno
import json
import random
import socket
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import base64
import torch.cuda
from userdata import database
import os
import hashlib
from RAG import RAG
from Ai import AI_Manager as AI
from rsa import RSA_CLASS
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from llama_cpp import Llama
from config import RAG_USER_QUOTA_BYTES, RAG_MAX_FILE_SIZE_BYTES, system_prompt_for
from tools import registry, executor, tools as builtin_tools
# ----- CONSTANTS ZONE -----#
DEBUG_MODE = True
LOG_TIMEOUTS = False
SERVER_PORT = 51225
LOG_SAVE_PATH = './ServerLogs'
USERDATA_PATH = './users'
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
LLAMA_MODEL = "./llamacpp_models/llama-pro-8b-instruct.Q5_K_M.gguf"
BRUTE_USER_SOFT_TRY = 5
BRUTE_USER_HARD_TRY = 10
BRUTE_LOGIN_IP_SOFT_TRY = 10
BRUTE_LOGIN_IP_HARD_TRY = 50
BRUTE_SIGNUP_IP_SOFT_TRY = 5
BRUTE_MIN_SOFTDELAY = 120
BRUTE_MAX_SOFTDELAY = 1800
BRUTE_TIMEOUT = timedelta(seconds=3600)
MAX_CONNECTIONS_PER_IP = 10
PRE_AUTH_TIMEOUT_SEC = 20

# ----- GLOBAL VARS ZONE -----#
kill_all = False
log_file = None
log_lock = threading.Lock()
client_id_by_address = {}
db = None
active_users = []
user_lock = threading.Lock()
RAG_Engine = None
Ai = None
active_models = {}
pending_uploads = {}
pending_uploads_lock = threading.Lock()
tool_executor = None
recv_buffers = {}
recv_buffers_lock = threading.Lock()
failure_by_user = {}
failure_by_login_ip = {}
failure_by_signup_ip = {}
brute_force_lock = threading.Lock()
connection_count_by_ip = {}
connection_count_lock = threading.Lock()

class User:
    """
    class for managing user info
    """
    def __init__(self):
        """
        initiallizer for the user class
        """
        self.client_sock = None
        self.pending_prompt = None
        self.authinticated = False
        self.user_id = -1
        self.username = ''
        self.preferences = None
        self.user_data_path = None
        self.vector_db = None
        self.aes_key = None
        self.rsa_obj = RSA_CLASS()

    def update_preferneces(self, encoded_preferences):
        """
        updates user preferences
        :param encoded_preferences: new perferences
        :return none
        """
        global db
        new_preferences = json.loads(encoded_preferences)
        for preference in new_preferences.keys():
            db.update_user_preference(self.user_id, preference, new_preferences[preference])
        self.preferences = self.get_preferences()

    def get_preferences(self):
        """
        gets user preferences
        :return: user preferences
        """
        global db
        return db.get_user_preferences(self.user_id)

    def get_preference(self, column):
        """
        get a single user setting
        :param column: setting name
        :return: setting value
        """
        global db
        return db.get_user_preference(self.user_id, column)

    def get_ai_stats(self):
        """
        get all relevant ai settings for user
        :return: dict with ai settings for user
        """
        stats = {
            'provider': self.get_preference('model_provider'),
            'active_model': self.get_preference('model_active'),
            'temperature': self.get_preference('model_temperature'),
            'p': self.get_preference('model_p'),
            'k': self.get_preference('model_k'),
            'max_output_tokens': self.get_preference('model_maxoutputtokens'),
            'system_prompt': self.get_preference('model_systemprompt')
        }

        return stats

    def get_gemini_api(self):
        """
        returns the gemini api key of user
        :return: api key
        """
        return self.get_preference('gemini_apikey')

    def create_user_data(self):
        """
        creates relevant user data paths
        """
        docspath = Path(str(self.user_data_path) + '/docs')
        self.user_data_path.parent.mkdir(parents=True, exist_ok=True)
        docspath.parent.mkdir(parents=True, exist_ok=True)


    def authinticate(self, id, name):
        """
        changes state to authinticated user from unauthinticated
        :param id: user id
        :param name: username
        """
        global RAG_Engine
        self.user_id = id
        self.username = name
        self.preferences = self.get_preferences()
        self.user_data_path = Path(USERDATA_PATH + f'/{self.user_id}')
        self.vector_db = RAG_Engine.initallize_user_vdb(self.user_data_path)
        self.authinticated = True

    def create_user(self, id, name):
        """
        creates a new user
        :param id: user id
        :param name: user name
        """
        self.user_id = id
        self.username = name
        self.user_data_path = Path(USERDATA_PATH + f'/{self.user_id}')
        self.create_user_data()
        self.preferences = self.get_preferences()
        self.vector_db = RAG_Engine.initallize_user_vdb(self.user_data_path)
        self.authinticated = True


# ----- CLIENTS MANAGEMENT -----#
def handle_client(client_sock, client_address, client_id):
    """
    handles all interaction with a certain client
    :param client_sock: socket for client
    :param client_address: ip address of client
    :param client_id: id of client
    """
    global kill_all
    global active_users
    global user_lock
    use_aes = False
    kill = False
    user_obj = User()
    user_obj.client_sock = client_sock
    expected_message_id = 0
    log(action='INFO', origin='server', message=f'client {client_id} joined from {client_address}')
    client_sock.settimeout(0.2)
    rsa = RSA_CLASS()

    connect_time = datetime.now()
    try:
        while not kill_all and not kill:
                try:
                    if not user_obj.authinticated and (datetime.now() - connect_time) > timedelta(PRE_AUTH_TIMEOUT_SEC):
                        log(action='WARN', origin='Server', message=f'client {client_id} from {client_address} dropped: pre-auth timeout')
                        break
                    message = recieve_message(client_sock, client_id, client_address)
                    message_code, data_fields = parse_request(message, expected_message_id)
                    expected_message_id += 1

                    response_code, response_fields = handle_request(message_code, data_fields, user_obj, expected_message_id, client_address)
                    if response_code == 'EXIT':
                        tool_executor.registry.unregister_user_tools(user_obj.user_id)
                        active_users.remove(response_fields[0])
                        log(action='INFO', origin='client_address', message=f'client {client_id} from {client_address} left.')
                        break

                    if response_code:
                        if use_aes:
                            response_fields = build_aes_reply(response_code, response_fields, expected_message_id, user_obj.aes_key)
                            response_code = 'AESM'
                        if user_obj.authinticated:
                            use_aes = True
                        reply = build_reply(response_code, response_fields, expected_message_id)
                        send_message(client_sock, reply, client_id, client_address)
                        expected_message_id += 1

                except socket.timeout as err:
                    if LOG_TIMEOUTS:
                        log(action='WARN', origin='server', message=f'timeout from client {client_id}')
                    continue
    finally:
        with connection_count_lock:
            if client_address[0] in connection_count_by_ip:
                connection_count_by_ip[client_address[0]] -=1
                if connection_count_by_ip[client_address[0]] <= 0:
                    del connection_count_by_ip[client_address[0]]
        try:
            client_sock.close()
        except OSError:
            pass

def parse_request(message, expected_message_id):
    """
    parses client request
    :param message: message from client
    :param expected_message_id: message id from client
    :return: parsed message
    """
    decoded_message = message.decode()
    split_message = decoded_message.split('~')
    message_id = int(split_message[0])
    message_code = split_message[1]
    message_fields = split_message[2:]
    if message_id != expected_message_id:
        #raise ValueError('bad message id!!')
        expected_message_id +=1

    return message_code, message_fields


def handle_request(message_code, data_fields, user_obj, message_id, client_address):
    """
    if message is encrypted sends to decryption if not sends to relevant handler
    :param message_code: message code
    :param data_fields: data fields for message
    :param user_obj: object with user info
    :param message_id: id of message
    :return: response code and fields
    """

    if message_code == 'RSAM':
        message_code, data_fields = decrypt_rsa(data_fields[0], user_obj.rsa_obj)
    elif message_code == 'AESM':
        message_code, data_fields = decrypt_aes(data_fields[0], user_obj.aes_key, data_fields[2], data_fields[1])


    if message_code == 'EXIT':
        return 'EXIT', data_fields

    if user_obj.authinticated:
        return handle_authinticated_request(message_code, data_fields, user_obj)
    else:
        return handle_unauthinticated_request(message_code, data_fields, user_obj, message_id, client_address)

def handle_unauthinticated_request(message_code, data_fields, user_obj, message_id, client_address):
    """
    handle requests from unauthinticated users
    :param message_code: message code
    :param data_fields: data fields for message
    :param user_obj: object with user info
    :param message_id: id of message
    :return: response code and fields
    """
    response_code = None
    response_fields = None
    match message_code:
        case 'PUBR':
            encoded_key = data_fields[0]
            key = base64.b64decode(encoded_key)
            user_obj.rsa_obj.set_other_public(key)
            response_code = 'PUBK'
            response_fields = [(base64.b64encode(user_obj.rsa_obj.public_key)).decode()]

        case 'AESR':
            key = get_random_bytes(16)
            user_obj.aes_key = key
            encoded_key = base64.b64encode(key).decode()
            reply = build_rsa_reply('AESK', [encoded_key], message_id, user_obj.rsa_obj)
            response_code = 'RSAM'
            response_fields = [reply]

        case 'AESN':
            encoded_nonce = data_fields[0]
            client_nonce = base64.b64decode(encoded_nonce)
            user_obj.client_nonce = client_nonce


        case 'LGNR':
            username = data_fields[0]
            password = data_fields[1]
            client_ip = client_address[0] if client_address else 'unknown'

            allowed, delay, reason = check_login_brute_force(username, client_ip)
            if not allowed:
                log(action='WARN', origin='Server', message=f'Blocked login attempt for "{username}" from {client_ip}: {reason}')
                response_code = 'EROR'
                response_fields = ['08']
            else:
                if delay > 0:
                    log(action='INFO', origin='Server', message=f'throttling login for {username} from {client_address} by {delay}s')
                    time.sleep(delay)

            if login(username, password, user_obj):
                save_auth_success(username,client_ip)
                response_code = 'AESM'
                response_fields = build_aes_reply('LGNS', [], message_id, user_obj.aes_key)
            else:
                save_auth_fail(username, client_ip)
                response_code = 'EROR'
                response_fields = ['01']

        case 'SGNR':
            username = data_fields[0]
            password = data_fields[1]
            client_ip = client_address[0] if client_address else 'unknown'

            allowed, delay, reason = check_signup_brute_force(client_ip)
            if not allowed:
                log(action='WARN', origin='Server', message=f'Blocked signup from {client_ip}: {reason}')
                response_code = 'EROR'
                response_fields = ['08']
            else:
                if delay > 0:
                    time.sleep(delay)

                if sign_up(username, password, user_obj):
                    save_auth_success(username, client_ip)
                    response_code = 'AESM'
                    response_fields = build_aes_reply('SGNS', [], message_id, user_obj.aes_key)
                else:
                    save_signup_fail(username, client_ip)
                    response_code = 'EROR'
                    response_fields = ['02']

        case _:
            log(action='ERROR', origin='Server', message=f'Unknown message code {message_code}')
            response_code = 'EROR'
            response_fields = ['09']

    return response_code, response_fields


def handle_authinticated_request(message_code, data_fields, user_obj):
    """
    handles requests from authinticated users
    :param message_code: message code
    :param data_fields: data fields for message
    :param user_obj: object with user info
    :param message_id: id of message
    :return: response code and fields
    """
    global Ai
    response_code = None
    response_fields = None
    match message_code:
        case 'MCPL':
            descriptors = json.loads(data_fields[0])
            for d in descriptors:
                tool_executor.registry.register(registry.Tool(
                    name=d['name'],
                    description=d.get('description', ''),
                    schema=d.get('schema') or {'type': 'object', 'properties': {}},
                    implementation=None,
                    remote=True,
                    owner_user_id=user_obj.user_id,
                ))
            log('INFO', 'Server',
                f'user {user_obj.user_id} registered {len(descriptors)} MCP tools')
            response_code = 'SUCC'
            response_fields = []


        case 'GETP':
            response_code = 'USRP'
            response_fields = [json.dumps(user_obj.get_preferences()), json.dumps(Ai.Models)]

        case 'RSES':
            sessions = db.get_user_sessions(user_obj.user_id)
            response_code = 'SSES'
            response_fields = [json.dumps(sessions)]

        case 'GETH':
            if data_fields and data_fields[0]:
                session_id = data_fields[0]
                history = get_message_history(user_obj.user_id, 50, session_id=session_id)
            else:
                history = get_recent_history(user_obj.user_id, 20)
            response_code = 'CHAH'
            response_fields = [history]

        case 'CHPW':
            current_pw = data_fields[0]
            new_pw = data_fields[1]
            user_row = db.get_user(user_obj.username)
            saved_hash = user_row['password']
            salt = user_row['salt']
            if hash_password(current_pw, salt) != saved_hash:
                response_code = 'EROR'
                response_fields = ['Wrong current password']
            else:
                new_salt = generate_salt()
                new_hash = hash_password(new_pw, new_salt)
                db.update_user_password(user_obj.user_id, new_hash, new_salt)
                response_code = 'CHPS'
                response_fields = ['1']

        case 'DELU':
            user_id = user_obj.user_id
            try:
                db.delete_user(user_id)
                response_code = 'DELA'
                response_fields = ['1']
            except Exception as err:
                response_code = "EROR"
                response_fields = ["03"]
                log('ERROR', 'Server', f'delete_user db error: {err}')

            try:
                shutil.rmtree(user_obj.user_data_path, ignore_errors=True)
            except Exception as err:
                log('ERROR', 'Server', f'delete_user dir error: {err}')


            user_obj.authinticated = False

        case 'DELS':
            session_id = data_fields[0]
            try:
                db.delete_session(user_obj.user_id, session_id)
                response_code = 'DELP'
                response_fields = ['1', session_id]
            except Exception as err:
                log('ERROR', 'Server', f'delete_session error: {err}')
                response_code = 'DELP'
                response_fields = ['0', session_id]

        case 'SMSG':
            session_id = data_fields[0]
            prompt = data_fields[1]

            save_message(user_obj.user_id, session_id, 'user', prompt)

            history_rows = get_session_history_rows(user_obj.user_id, session_id, 20)
            history = [(row['role'], row['content']) for row in history_rows]
            if history and history[-1] == ('user', prompt):
                history = history[:-1]

            ai_preferences = user_obj.get_ai_stats()
            if not ai_preferences.get('system_prompt'):
                ai_preferences['system_prompt'] = system_prompt_for(ai_preferences['provider'])
            if ai_preferences['provider'] == 'Gemini':
                ai_preferences['api'] = user_obj.get_gemini_api()

            if user_obj.preferences.get('rag_enabled', 1):
                context_items = RAG_Engine.search(prompt, user_obj.vector_db)
            else:
                context_items = []
            formatted_prompt = AI.build_model_request(
                ai_preferences, history, context_items, prompt
            )

            try:
                state = Ai.start_prompt(formatted_prompt,
                                        executor=tool_executor,
                                        user_obj=user_obj)

                if state['type'] == 'tool_call':
                    state['_session_id'] = session_id
                    user_obj.pending_prompt = state
                    call = state['calls'][0]
                    response_code = 'MCPC'
                    tool_name = call['name']
                    prefix = f'u{user_obj.user_id}_'
                    if tool_name.startswith(prefix):
                        tool_name = tool_name[len(prefix):]
                    response_fields = [call['id'], tool_name, base64.b64encode(json.dumps(call['args']).encode()).decode()]
                else:
                    reply = state['text']
                    save_message(user_obj.user_id, session_id, 'assistant', reply)
                    response_code = 'SREP'
                    response_fields = ['0', reply]
            except Exception as err:
                import traceback
                log('ERROR', 'Server', f'SMSG failed: {err}\n{traceback.format_exc()}')
                response_code = 'EROR'
                response_fields = ['04']
        case 'MCPR':
            call_id = data_fields[0]
            result = base64.b64decode(data_fields[1]).decode()

            state = user_obj.pending_prompt
            if state is None:
                log('WARN', 'Server',
                    f'user {user_obj.user_id} sent MCPR but no prompt is paused')
                response_code = 'EROR'
                response_fields = ['05']
            else:
                session_id = state['_session_id']
                try:
                    state = Ai.continue_prompt(state, call_id, result,
                                               executor=tool_executor,
                                               user_obj=user_obj)
                except Exception as err:
                    log('ERROR', 'Server', f'continue_prompt failed: {err}')
                    user_obj.pending_prompt = None
                    response_code = 'EROR'
                    response_fields = ['05']
                else:
                    if state['type'] == 'tool_call':
                        state['_session_id'] = session_id
                        user_obj.pending_prompt = state
                        call = state['calls'][0]
                        response_code = 'MCPC'
                        response_fields = [call['id'], call['name'], base64.b64encode(json.dumps(call['args']).encode()).decode()]
                    else:
                        user_obj.pending_prompt = None
                        reply = state['text']
                        save_message(user_obj.user_id, session_id, 'assistant', reply)
                        response_code = 'SREP'
                        response_fields = ['0', reply]


        case 'SETP':
            user_obj.update_preferneces(data_fields[0])
            response_code = 'SUCC'
            response_fields = []

        case 'RADD':
            file_name = data_fields[0]
            file_size = int(data_fields[1])
            expected_chunk_amount = int(data_fields[2])

            if file_size > RAG_MAX_FILE_SIZE_BYTES:
                log('WARN',origin='Server',message=f'User {user_obj.user_id} tried to upload {file_size} bytes (limit {RAG_MAX_FILE_SIZE_BYTES})')
                response_code = 'EROR'
                response_fields = ['06']
            elif get_user_storage_used(user_obj.user_data_path) + get_pending(user_obj.user_id) + file_size > RAG_USER_QUOTA_BYTES:
                log(action='WARN', origin='Server',
                    message=f'User {user_obj.user_id} quota exceeded: {file_size} > {RAG_USER_QUOTA_BYTES}')
                response_code = 'EROR'
                response_fields = ['07']
            else:
                reserve_upload(user_obj.user_id, file_size)
                new_socket = socket.socket()
                port = random.randint(1024, 65535)
                port_chosen = False
                while not port_chosen:
                    try:
                        new_socket.bind(('0.0.0.0', port))
                        port_chosen = True
                    except OSError as err:
                        if err.errno == errno.EADDRINUSE or err.errno == errno.EADDRNOTAVAIL:
                            port = random.randint(1024, 65535)

                file_download = threading.Thread(
                    target=file_downloader,
                    args=(new_socket, user_obj.user_data_path, user_obj.vector_db,
                          file_name, expected_chunk_amount, user_obj.aes_key, user_obj.user_id, file_size)
                )
                response_code = 'RACK'
                response_fields = [str(port)]
                file_download.start()

        case 'RLST':
            docs = user_obj.vector_db.list_documents()
            used = get_user_storage_used(user_obj.user_data_path)
            response_code = 'RLSP'
            response_fields = [json.dumps(docs), str(used), str(RAG_USER_QUOTA_BYTES)]

        case 'RDEL':
            doc_name = data_fields[0]
            doc_name = os.path.basename(doc_name).replace('..', '').replace('\x00', '')
            ok = user_obj.vector_db.delete_document(doc_name)
            file_name = os.path.basename(file_name)
            file_name = file_name.replace('..', '').replace('\x00', '')
            file_path = Path(user_obj.user_data_path) / doc_name
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
            response_code = 'RDLP'
            response_fields = ['1' if ok else '0']


        case 'GUSE':
            used = get_user_storage_used(user_obj.user_data_path)
            response_code = 'USAG'
            response_fields = [str(used), str(RAG_USER_QUOTA_BYTES)]

        case _:
            log(action='ERROR', origin='Server', message=f'Unknown message code {message_code}')
            response_code = 'EROR'
            response_fields = ['09']

    return response_code, response_fields


def save_message(user_id, session_id, role, message):
    """
    saves message in session history
    :param user_id: user id
    :param session_id: session id
    :param role: role of the message origin (user/assistant)
    :param message: message
    """
    global db
    db.save_message(user_id, session_id, role, message)


def build_reply(response_code, response_fields, message_id):
    """
    build reply for client
    :param response_code: reply code
    :param response_fields: reply data fields
    :param message_id: reply message id
    :return: prepared reply to send to client
    """
    message = str(message_id) + '~' + response_code
    if response_fields != None and response_fields != []:
        message += '~' + '~'.join(field for field in response_fields)
    return message

def build_rsa_reply(response_code, response_fields, message_id, rsa_obj):
    """
    wrapper for build reply to add rsa encryption
    :param response_code: reply code
    :param response_fields: reply data fields
    :param message_id: reply message id
    :param rsa_obj: rsa managment object
    :return: prepared rsa reply to send to the client
    """
    message = build_reply(response_code, response_fields, message_id)
    encrypted_message = rsa_obj.encrypt_RSA(message)
    encoded_message = base64.b64encode(encrypted_message).decode()
    return encoded_message

def build_aes_reply(response_code, response_fields, message_id, aes_key):
    """
    wrapper for build reply to add aes encryption
    :param response_code: reply code
    :param response_fields: reply data fields
    :param message_id: reply message id
    :param aes_key: aes GCM key
    :return: prepared aes reply to send to the client
    """
    message = build_reply(response_code, response_fields, message_id).encode()
    cipher = AES.new(aes_key, AES.MODE_GCM)
    encrypted_message, tag = cipher.encrypt_and_digest(message)
    nonce = cipher.nonce
    encoded_message = base64.b64encode(encrypted_message).decode()
    encoded_tag = base64.b64encode(tag).decode()
    encoded_nonce = base64.b64encode(nonce).decode()
    return [encoded_message, encoded_tag, encoded_nonce]

def sign_up(username, password, user_obj):
    """
    handles registration
    :param username: username
    :param password: password
    :param user_obj: object with user info
    :return: bool succss
    """
    global db
    global active_users
    salt = generate_salt()
    hashed_password = hash_password(password, salt)
    userid = db.add_user(username, hashed_password, salt)
    if userid:
        user_obj.create_user(userid, username)
        user_lock.acquire()
        active_users.append(username)
        user_lock.release()
        return True

    return False


def login(username, password, user_obj):
    """
    handles login
    :param username: username
    :param password: password
    :param user_obj: object with user info
    :return: bool succss
    """
    global active_users
    global db
    if not username in active_users:
        user = db.get_user(username)
        if user:
            userid = user[0]
            saved_hash = user['password']
            salt = user['salt']
            hashed_user_password = hash_password(password, salt)
            if saved_hash == hashed_user_password:
                user_obj.authinticate(userid, username)
                user_lock.acquire()
                active_users.append(username)
                user_lock.release()
                return True

    return False

def purge_old_fails(timestamps, now):
    """
    resets bruteforce countdowns after enough time have passed
    :param timestamps: attempt time
    :param now: current time
    """
    cutoff = now - BRUTE_TIMEOUT
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)

def delay_for(count, soft):
    """
    calculates how much time to delay action based on offense
    :param count: offense count
    :param soft: soft treshold
    :return: amount of time. if none, 0
    """
    over = count - soft
    if over < 0:
        return 0
    return min(BRUTE_MIN_SOFTDELAY * (2 ** over), BRUTE_MAX_SOFTDELAY)

def check_login_brute_force(username, ip):
    """
    check if auth attempt is allowed. if not, report so
    :param username: username ascociated with auth attempt
    :param ip: ip address of the client ascociated with auth attempt
    :return: is allowed to login, if needs to delay before auth, fail reason
    """
    global failure_by_user
    global failure_by_login_ip
    global brute_force_lock
    now = datetime.now()
    with brute_force_lock:
        user_fails = failure_by_user.setdefault(username, [])
        ip_fails = failure_by_login_ip.setdefault(ip, [])
        purge_old_fails(user_fails, now)
        purge_old_fails(ip_fails, now)

        if len(user_fails) >= BRUTE_USER_HARD_TRY:
            return False, 0, f"Account locked due to too many invalid login attempts"
        if len(ip_fails) >= BRUTE_LOGIN_IP_HARD_TRY:
            return False, 0, f"Ip blocked due to too many invalid auth attempts"

        delay = max(delay_for(len(user_fails), BRUTE_USER_SOFT_TRY), delay_for(len(ip_fails), BRUTE_LOGIN_IP_SOFT_TRY))
        if delay > 0:
            return True, delay, f"throttled login (user:{len(user_fails)}, ip:{len(ip_fails)})"
        return True, 0, "ok"

def check_signup_brute_force(ip):
    """
    check if sign up attempt is allowed. if not, report so
    :param ip: ip address of the client ascociated with sign up attempt
    :return: is allowed to sign up, if needs to delay before proceed, fail reason
    """
    global failure_by_signup_ip
    global brute_force_lock
    now = datetime.now()
    with brute_force_lock:
        ip_fails = failure_by_signup_ip.setdefault(ip, [])
        purge_old_fails(ip_fails, now)
        delay = delay_for(len(ip_fails), BRUTE_SIGNUP_IP_SOFT_TRY)
        if delay > 0:
            return True, delay, f"throttled signup (ip:{len(ip_fails)})"
        return True, 0, "ok"



def save_login_fail(username, ip):
    """
    record a fail auth attempt
    :param username: username ascociated with auth attempt
    :param ip: ip address of the client ascociated with auth attempt
    """
    global failure_by_user
    global failure_by_login_ip
    global brute_force_lock
    now = datetime.now()
    with brute_force_lock:
        failure_by_user.setdefault(username, []).append(now)
        failure_by_login_ip.setdefault(ip, []).append(now)
        u = len(failure_by_user[username])
        i = len(failure_by_login_ip[ip])
    log(action='WARN', origin='Server', message=f'failed login for {username} from {ip}. current fails: [user:{u}, ip:{i}]')

def save_signup_fail(username, ip):
    """
    record a fail auth attempt
    :param username: username ascociated with auth attempt
    :param ip: ip address of the client ascociated with auth attempt
    """
    global failure_by_user
    global failure_by_signup_ip
    global brute_force_lock
    now = datetime.now()
    with brute_force_lock:
        failure_by_signup_ip.setdefault(ip, []).append(now)
        i = len(failure_by_signup_ip[ip])
    log(action='WARN', origin='Server',
        message=f'failed sign up from {ip}. current fails: {i}')


def save_auth_success(username, ip):
    """
    clear the username's fail record on login
    :param username: username ascociated with auth attempt
    :param ip: ip address of the client ascociated with auth attempt
    """
    global brute_force_lock
    brute_force_lock.acquire()
    failure_by_user.pop(username, [])
    brute_force_lock.release()
    log(action='INFO', origin='Server', message=f'successful login for {username} from {ip}')


def decrypt_rsa(message, rsa_obj):
    """
    decrypts rsa messages
    :param message: encrypted message
    :param rsa_obj: RSA manegment obj
    :return: decrypted message
    """
    decoded_message = base64.b64decode(message)
    decrypted_message = rsa_obj.decrypt_RSA(decoded_message).decode()

    splitted_message = decrypted_message.split('~')
    message_code = splitted_message[1]
    data_fields = splitted_message[2:]
    return message_code, data_fields

def decrypt_aes(message, aes_key, nonce, tag):
    """
    decrypts AES GCM messaegs
    :param message: encrypted message
    :param aes_key: aes key
    :param nonce: nonce
    :param tag: verification tag
    :return: decrypted message
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


# ----- AI MANAGEMENT -----#
def send_mcp_call(user_obj, call_id, tool_name, args):
    """
    Sends MCPC to the user's client over their existing socket.
    :param user_obj: object with username info
    :param call_id: tool call id
    :param tool_name: tool name
    :param args: args for tool
    """
    payload = [call_id, tool_name, ase64.b64encode(args)]
    send_aes_request(user_obj.client_sock, 'MCPC', payload, user_obj.aes_key)


def get_session_history_rows(user_id, session_id, limit):
    """
    Return raw history rows (list of dicts) — used internally by the prompt path.
    :param user_id: user id
    :param session_id: session id
    :param limit: message limit
    :return: chat history
    """
    global db
    return db.get_chat_history(user_id, session_id, limit)

def get_recent_history(user_id, message_amount):
    """
    retrives recent history
    :param user_id: user id
    :param message_amount: amount of messages
    :return: recent history
    """
    return get_message_history(user_id, message_amount)

def get_message_history(user_id, message_amount, session_id = None):
    """
    retrives message history from database
    :param user_id: user id
    :param message_amount: amount of messages to return
    :param session_id: session id
    :return: message history
    """
    global db
    messages = None
    if not session_id:
        messages = db.get_recent_history(user_id, message_amount)
    else:
        messages = db.get_chat_history(user_id, session_id,message_amount)

    return json.dumps(messages)



# ----- PROGRAM MANAGEMENT -----#


def get_user_storage_used(user_data_path):
    """
    Sum the size of every uploaded document, ignoring rag internals.
    :param user_data_path:  path to user data
    :return: amount of storage used
    """
    user_path = Path(user_data_path)
    if not user_path.exists():
        return 0
    total = 0
    for entry in user_path.iterdir():
        if not entry.is_file():
            continue
        if entry.name in ('rag.db', 'embeddings.pt'):
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            pass
    return total

def file_downloader(download_socket, user_path, user_vdb, file_name, chunk_amount, aes_key, user_id, file_size):
    """
    file downloader thread
    :param download_socket: socket for download
    :param user_path: user data path
    :param user_vdb: user vector db
    :param file_name: file name to download
    :param chunk_amount: amount of chunks
    :param aes_key: aes key
    :param user_id: id of user
    :param file_size: file size
    """
    log('INFO', 'Server', f'file_downloader started for {file_name}, expecting {chunk_amount} chunks')
    global RAG_Engine
    file_name = os.path.basename(file_name)
    file_name = file_name.replace('..', '').replace('\x00', '')
    file_path = user_path / file_name
    try:
        download_socket.listen()
        user_socket, addr = download_socket.accept()
        user_socket.settimeout(15.0)

        buffer = ""
        received_chunks = 0

        with open(file_path, 'wb') as f:
            while received_chunks < chunk_amount:
                while '\n' not in buffer:
                    try:
                        data = user_socket.recv(16384)
                        if not data:
                            break
                        buffer += data.decode()
                    except socket.timeout:
                        break

                if '\n' not in buffer:
                    print("[ERROR] Stream broke before receiving full chunk.")
                    break

                message_str, buffer = buffer.split('\n', 1)

                parts = message_str.split('~')
                if len(parts) < 5:
                    continue

                enc_msg = parts[2]
                enc_tag = parts[3]
                enc_nonce = parts[4]

                decrypted_code, data_fields = decrypt_aes(enc_msg, aes_key, enc_nonce, enc_tag)
                if decrypted_code == 'RCHK':
                    chunk_data = base64.b64decode(data_fields[0])

                    f.write(chunk_data)
                    received_chunks += 1

        user_socket.close()
        log('INFO', 'Server', f'file_downloader finished {file_name}, received {received_chunks}/{chunk_amount}')
        download_socket.close()

        RAG_Engine.add_document(file_path, user_vdb)

    except Exception as err:
        log('ERROR', 'Server', f'file_downloader crashed: {err}')

    finally:
        release_upload(user_id, file_size)

def reserve_upload(user_id, size):
    """
    reserve pending upload for user
    :param user_id: user id
    :param size: upload size
    """
    with pending_uploads_lock:
        pending_uploads[user_id] = pending_uploads.get(user_id, 0) + size

def release_upload(user_id, size):
    """
    release done upload for user
    :param user_id: user id
    :param size: upload size
    """
    with pending_uploads_lock:
        pending_uploads[user_id] = max(0, pending_uploads.get(user_id, 0) - size)

def get_pending(user_id):
    """
    get all pending uploads for user
    :param user_id: user id
    :return: list of all pending uploads
    """
    with pending_uploads_lock:
        return pending_uploads.get(user_id, 0)

def recieve_message(client_sock, client_id, client_address):
    """
    recv wrapper to get user messages
    :param client_sock: client sock
    :param client_id: client id
    :param client_address: client address
    :return: message
    """
    sock_key = id(client_sock)

    with recv_buffers_lock:
        buf = recv_buffers.get(sock_key, b'')

    while b'\n' not in buf:
        data = client_sock.recv(65536)
        if not data:
            with recv_buffers_lock:
                recv_buffers.pop(sock_key, None)
            raise ConnectionError('client disconnected')
        buf += data

    message, remainder = buf.split(b'\n', 1)
    with recv_buffers_lock:
        recv_buffers[sock_key] = remainder

    log(action='MESSAGE_RECV', origin=f'{client_id}, {client_address}',
        recipient='Server', message=message)
    return message


def send_message(client_sock, message, client_id, client_address):
    """
    send wrapper to send messages
    :param client_sock: client socke
    :param message: message
    :param client_id: client id
    :param client_address: client address
    """
    if isinstance(message, str):
        encoded_message = message.encode()
    else:
        encoded_message = message
    encoded_message += b'\n'
    client_sock.send(encoded_message)
    log(action='MESSAGE_SEND', origin='Server',
        recipient=f'{client_id}, {client_address}', message=message)


# ----- MISC -----#

def generate_salt():
    """
    generates random salt
    :return: salt
    """
    return os.urandom(16)


def hash_password(password, salt):
    """
    simple hash func for passwords
    :param password: password
    :param salt: salt
    :return: hashed password
    """
    encoded_password = password.encode()
    hashed_password = hashlib.sha256(encoded_password + salt).hexdigest()
    return hashed_password


def dict_update(source, overrides):
    """
    simple dict update util
    :param source: original dict
    :param overrides: overrides to do in the dict
    :return: new dict
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = dict_update(source[key], value)
        else:
            source[key] = value
    return source


# ----- PROGRAM MANAGEMENT -----#

def main():
    """
    main function
    """
    global tool_executor
    global active_models
    global db
    global RAG_Engine
    global Ai
    global kill_all

    threads = []
    client_cnt = 0

    create_logfile()
    db = database.Database()
    RAG_Engine = RAG.RAG_Engine(EMBEDDING_MODEL)
    Ai = AI.Model_Manager()

    tool_executor = executor.ToolExecutor(
        registry.ToolRegistry(),
        log_fn=log
    )

    tool_executor.registry.register(registry.Tool(
        name='get_current_time',
        description='Returns the current server time in ISO 8601 format.',
        schema={'type': 'object', 'properties': {}, 'required': []},
        implementation=builtin_tools.get_current_time,
    ))

    tool_executor.registry.register(registry.Tool(
        name='calculate',
        description='Evaluates a basic arithmetic expression. Supports + - * / ** % //.',
        schema={
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': 'A simple arithmetic expression, e.g. (2 + 3) * 4',
                },
            },
            'required': ['expression'],
        },
        implementation=builtin_tools.calculate,
    ))

    tool_executor.registry.register(registry.Tool(
        name='search_user_documents',
        description="Search the user's uploaded documents for relevant information. "
                    'Use this when the user asks about content from files they uploaded.',
        schema={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Natural language search query',
                },
            },
            'required': ['query'],
        },
        implementation=builtin_tools.make_search_user_documents(RAG_Engine),
    ))


    server_sock = socket.socket()
    server_sock.bind(('0.0.0.0', SERVER_PORT))
    server_sock.listen()
    log(action='INFO', origin='Server', message='Server started Successfully!')
    try:
        while True:
            connected_client_sock, client_address = server_sock.accept()
            client_ip = client_address[0]
            with connection_count_lock:
                current = connection_count_by_ip.get(client_ip, 0)
                if current > MAX_CONNECTIONS_PER_IP:
                    log(action='WARN', origin='Server', message=f'Rejected connection from {client_ip}')
                    connected_client_sock.close()
                    continue
                connection_count_by_ip[client_ip] = current + 1
            thread = threading.Thread(target=handle_client, args=(connected_client_sock, client_address, client_cnt))
            threads.append(thread)
            thread.start()
            client_cnt += 1

    except OSError:
        pass

    except KeyboardInterrupt:
        log(action='INFO', origin='Server', message='Shutdown requested (KeyboardInterrupt). Shutting down gracefully...')
    finally:
        kill_all = True
        for t in threads:
            t.join()
        server_sock.close()
        log(action='INFO', origin='Server', message='Server shut down.')


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
    logs everything that happens
    :param action: action to log
    :param origin: origin of action
    :param message: message/explanation
    :param recipient: if the aciton was regarding a networked message, who was the recipient
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


if __name__ == '__main__':
    main()
