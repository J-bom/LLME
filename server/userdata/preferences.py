import json
from pathlib import Path

USERDATA_PATH = './USERINFO'
DEFAULT_STARCT = {'settings': {'general': {}, 'model' : {'global' : {'provider' : 'llme', 'current_model' : '', 'temp' : 0.7, 'p' : 0.95, 'k' : 40, 'maxtoken' : 1024, 'system_prompt' : ''}, 'gemini' : {'api' : ''}, 'llama' : {}, 'llme': {}}}}
def initiallize_data():
    data_dir = Path(USERDATA_PATH)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

def create_user_data(username):
    user_folder = Path(str(USERDATA_PATH) + f'/{username}')
    if not user_folder.exists():
        user_folder.mkdir(parents=True, exist_ok=True)

    doc_folder = Path(str(user_folder) + '/docs')
    if not doc_folder.exists():
        doc_folder.mkdir(parents=True, exist_ok=True)

    file = Path(str(user_folder) + '/preferences.json')

    with open(file, 'w') as f:
        json.dump(DEFAULT_STARCT,f,indent=4)

def get_user_preferences(username):
    user_file = Path(str(USERDATA_PATH) + f'/{username}/preferences.json')
    data = None
    with open(user_file, 'r') as f:
        data = json.load(f)
    return data

def save_settings(settings, username):
    user_file = Path(str(USERDATA_PATH) + f'/{username}/preferences.json')
    with open(user_file, 'w') as f:
        json.dump(settings,f,indent=4)
    print('done')



