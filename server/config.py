import os
from dotenv import load_dotenv
from pathlib import Path

#env file path
ENV_PATH = Path('.env')

#default settings
DEFAULT_ENV = """\
# Storage limits (in MB)
RAG_USER_QUOTA_MB=100
RAG_MAX_FILE_SIZE_MB=20

# Per-provider system prompts
SYSTEM_PROMPT_ALTEREGO=You are Alter Ego, a small AI built from scratch. You're casual and direct. You're not great with facts, math, or current events — when you don't know something, just say so. You're better at chatting than at answering questions.
SYSTEM_PROMPT_LLAMA=You are a helpful assistant. Answer clearly and concisely.
SYSTEM_PROMPT_GEMINI=You are a helpful assistant. Answer clearly and concisely.
"""

if not ENV_PATH.exists():
    with open(ENV_PATH, 'w') as f:
        f.write(DEFAULT_ENV)
load_dotenv()


def mb_to_bytes(env_var, default_mb):
    """
    simple convert mb to bytes
    :param env_var: variable
    :param default_mb: defualt size
    :return: mb in bytes
    """
    return int(os.getenv(env_var, default_mb)) * 1024 * 1024

def system_prompt_for(provider):
    """
    Return the right system prompt for a provider, with a generic fallback.
    :param provider: Ai provider
    :return: system prompt
    """
    return DEFAULT_SYSTEM_PROMPTS.get(provider, 'You are a helpful assistant.')


RAG_USER_QUOTA_BYTES   = mb_to_bytes('RAG_USER_QUOTA_MB', 100)
RAG_MAX_FILE_SIZE_BYTES = mb_to_bytes('RAG_MAX_FILE_SIZE_MB', 20)
DEFAULT_SYSTEM_PROMPTS = {
    'AlterEgo': os.getenv('SYSTEM_PROMPT_ALTEREGO', "You are Alter Ego, a small AI built from scratch. You're casual and direct. You're not great with facts, math, or current events — when you don't know something, just say so. You're better at chatting than at answering questions."),
    'Llama':    os.getenv('SYSTEM_PROMPT_LLAMA',    'You are a helpful assistant.'),
    'Gemini':   os.getenv('SYSTEM_PROMPT_GEMINI',   'You are a helpful assistant.'),
}


