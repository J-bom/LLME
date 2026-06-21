import sqlite3
import threading

#Default model params for user
DEFAULT_PARAMS = {
    'model_provider' : "AlterEgo",
    'model_active' : '',
    'model_temperature' : 0.7,
    'model_p' : 0.95,
    'model_k' : 40,
    'model_maxoutputtokens' : 1024,
    'model_systemprompt' : 'you are a cool ai assistant',
    'gemini_apikey' : ''
}

class Database:
    """
    database management class
    """
    def __init__(self, db_name="LLME.db"):
        """
        initiallizer for the database class
        :param db_name: name for database file
        """
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self.lock = threading.Lock()

        self.setup()

    def setup(self):
        """
        sets up the database if not exists.
        """
        self.lock.acquire()
        self.conn.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password TEXT NOT NULL,salt TEXT NOT NULL)                """)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS user_settings (id INTEGER PRIMARY KEY,model_provider TEXT NOT NULL,model_active TEXT,model_temperature REAL,model_p REAL,model_k INTEGER,model_maxoutputtokens INTEGER,model_systemprompt TEXT,gemini_apikey TEXT,rag_enabled INTEGER DEFAULT 1,FOREIGN KEY (id) REFERENCES users (id))""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS user_usage (user_id INTEGER PRIMARY KEY,tokens_sent INTEGER DEFAULT 0,tokens_received INTEGER DEFAULT 0,total_requests INTEGER DEFAULT 0,FOREIGN KEY (user_id) REFERENCES users (id))""")

        self.conn.execute("""CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,session_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY (user_id) REFERENCES users (id));""")
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_user_session ON chat_messages(user_id, session_id);')
        self.conn.commit()
        self.lock.release()

    def add_user(self, username, password, salt):
        """
        writes user login data to database in a thread safe way
        :param username: username
        :param password: hashed pass
        :param salt: salt
        :return user id
        """
        toret = None
        self.lock.acquire()
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password, salt) VALUES (?, ?, ?)",(username, password, salt))
            user_id = cursor.lastrowid
            cursor.execute("INSERT INTO user_settings (id, model_provider, model_active, model_temperature, model_p, model_k, model_maxoutputtokens, model_systemprompt, gemini_apikey) values(?,?,?,?,?,?,?,?,?)",(int(user_id),DEFAULT_PARAMS['model_provider'], DEFAULT_PARAMS['model_active'], DEFAULT_PARAMS['model_temperature'], DEFAULT_PARAMS['model_p'], DEFAULT_PARAMS['model_k'], DEFAULT_PARAMS['model_maxoutputtokens'], DEFAULT_PARAMS['model_systemprompt'], DEFAULT_PARAMS['gemini_apikey']))
            cursor.execute("INSERT INTO user_usage (user_id) VALUES (?)",(user_id,))
            self.conn.commit()
            toret = user_id
        except sqlite3.IntegrityError:
            print(f"Error: Username '{username}' already exists.")
        finally:
            self.lock.release()
            return toret

    def get_user(self, username):
        """
        gets a user by username
        :param username: user
        :return: user info
        """
        cursor = self.conn.execute("SELECT * FROM users WHERE username = ?",(username,))
        return cursor.fetchone()

    def get_user_preferences(self, userid):
        """
        gets user preferences
        :param userid: user id
        :return: dict with user preferences
        """
        cursor = self.conn.execute("SELECT * FROM user_settings WHERE id = ?", (int(userid),))
        return self.settings_to_dict(cursor.fetchone())

    def get_user_preference(self, userid, column):
        """
        get a single user preference
        :param userid: user id
        :param column: preference
        :return: user preference
        """
        cursor = self.conn.execute(f"SELECT {column} FROM user_settings WHERE id = ?", (userid,))
        return cursor.fetchone()[0]

    def update_user_preference(self, userid, column, value):
        """
        update a user's preference
        :param userid: user id
        :param column: preference
        :param value: new value
        """
        ALLOWED = {
            'model_provider', 'model_active', 'model_temperature',
            'model_p', 'model_k', 'model_maxoutputtokens',
            'model_systemprompt', 'gemini_apikey', 'rag_enabled'
        }
        if column not in ALLOWED:
            raise ValueError(f"Unknown preference column: {column}")
        self.lock.acquire()
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE user_settings SET {column} = ? WHERE id = ?", (value, userid))
        self.conn.commit()
        self.lock.release()

    def close(self):
        """
        close connection to db
        """
        self.conn.close()

    def settings_to_dict(self, settings):
        """
        convert user perferences to dict form
        :param settings: user preferences
        :return: user preferences in dict form
        """
        dict = {
            'id': settings[0],
            'model_provider': settings[1],
            'model_active': settings[2],
            'model_temperature': settings[3],
            'model_p': settings[4],
            'model_k': settings[5],
            'model_maxoutputtokens': settings[6],
            'model_systemprompt': settings[7],
            'gemini_apikey': settings[8],
            'rag_enabled': settings[9]
        }
        return dict

    def update_user_password(self, user_id, new_hashed_password, new_salt):
        """
        updates a user's password and salt
        :param user_id: user id
        :param new_hashed_password: new password
        :param new_salt: new salt
        """
        self.lock.acquire()
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET password = ?, salt = ? WHERE id = ?",
            (new_hashed_password, new_salt, int(user_id))
        )
        self.conn.commit()
        self.lock.release()

    def delete_user(self, user_id):
        """
        deletes user's account
        :param user_id: user id
        """
        self.lock.acquire()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_settings WHERE id = ?", (user_id,))
        cursor.execute("DELETE FROM user_usage WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        self.lock.release()

    def delete_session(self, user_id, session_id):
        """
        delete a chat
        :param user_id: user id
        :param session_id: session id
        """
        self.lock.acquire()
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?",
            (user_id, session_id)
        )
        self.conn.commit()
        self.lock.release()

    def save_message(self, user_id, session_id, role, content):
        """
        save message into message history
        :param user_id: user id
        :param session_id: session id
        :param role: chat role (user/assistant)
        :param content: text
        """
        self.lock.acquire()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
            (user_id, session_id, role, content)
        )
        self.conn.commit()
        self.lock.release()

    def get_chat_history(self, user_id, session_id, limit=50):
        """
        retrives chat history from a session
        :param user_id: user id
        :param session_id: session id
        :param limit: message limit
        :return: dict of messages
        """
        cursor = self.conn.execute("SELECT role, content FROM chat_messages WHERE user_id = ? AND session_id = ? ORDER BY id DESC LIMIT ?",(user_id, session_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows][::-1]





    def get_usage_stats(self, user_id):
        """
        retrives usage stats per user
        :param user_id: user id
        :return: dict with info
        """
        cursor = self.conn.execute("SELECT tokens_sent, tokens_received, total_requests FROM user_usage WHERE user_id = ?",(user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


    def get_recent_history(self, user_id, limit=50):
        """
        retrives the last limit amount of messages no matter the sesison
        :param user_id: user id
        :param limit: amount of messages to retrive
        :return: dict of messages
        """
        cursor = self.conn.execute("SELECT role, content FROM chat_messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",(user_id, limit))
        rows = cursor.fetchall()
        return [dict(row) for row in rows][::-1]

    def get_user_sessions(self, user_id):
        """
        gets metadata and info about all sessions of user
        :param user_id: user id
        :return: dict of sessions
        """
        cursor = self.conn.execute("""SELECT session_id,MIN(content) FILTER (WHERE role = 'user') AS preview,MAX(created_at) AS last_updated FROM chat_messages WHERE user_id = ? GROUP BY session_id ORDER BY last_updated DESC""", (user_id,))
        return [dict(row) for row in cursor.fetchall()]

    def delete_session(self, user_id, session_id):
        """
        delete a session from history
        :param user_id: user id
        :param session_id: session id
        """
        self.lock.acquire()
        self.conn.execute("DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?",(user_id, session_id))
        self.conn.commit()
        self.lock.release()