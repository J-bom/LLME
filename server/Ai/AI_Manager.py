from google import genai
from google.genai import types
from threading import Lock
from datetime import datetime, timedelta
import torch
from collections import OrderedDict
import os
import json
import uuid
import gc
import threading
import time

torch_bin_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_bin_path):
    os.add_dll_directory(torch_bin_path)
from llama_cpp import Llama
from Ai.Alter_Ego import AlterEgo

import re
import ast



#models path
MODEL_PATH = './Ai/Models'

#when to check for background actions
EVICT_INTERVAL = 60

#how long til kick model out
EVICT_TIMEOUT = 20

#Llama.cpp models context length
LLAMA_CTX_SIZE = 16384

#AlterEgo context length
ALTEREGO_CTX_SIZE = 2048

#Gemini context length
GEMINI_CTX_SIZE = 1_000_000

#tool_call scheme
_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

class Model:
    """
    hold information about the model including model
    """
    def __init__(self, id, model, engine):
        """
        initiallizer
        :param id: model name
        :param model: model
        :param engine: model engine
        """
        self.model_id = id
        self.engine = engine
        self.model = model
        self.model_lock = Lock()
        self.last_used = datetime.now()



class Model_Loader:
    """
    the model loader class, handles the loading of models to memory.
    """
    def __init__(self, device):
        """
        Initiallizer for the model lodaer, sets up the engines and device
        :param device: device to use
        """
        self.device = device
        self.engines = {
            'Llama': self.load_llama,
            'AlterEgo': self.load_alterego,
        }

    def load_model(self, model_id, path, engine):
        """
        loads model
        :param model_id: model id
        :param path: model path
        :param engine: model engine
        :return: model object
        """
        return Model(model_id, self.engines[engine](path), engine)

    def load_llama(self, path):
        """
        loads llama based models
        :param path: model path
        :return: model object
        """
        gpu_layers = -1 if self.device == 'cuda' else 0
        return Llama(model_path=path, n_ctx=LLAMA_CTX_SIZE,
                     n_gpu_layers=gpu_layers, verbose=False)

    def load_alterego(self, path):
        """
        loads AlterEgo
        :param path: AlterEgo checkpoint path
        :return: AlterEgo object
        """
        return AlterEgo(checkpoint_path=path, device=self.device)


class Model_Manager:
    """
    model manager. handles all Ai related stuff
    """
    def __init__(self, safety_buffer=1.5):
        """
        initiallizer for the model manager.
        :param safety_buffer: safety VRAM buffer to ensure stability
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loader = Model_Loader(self.device)
        self.Models = scan_models()
        self.buffer = safety_buffer * 1024 ** 3
        self.loaded_models = OrderedDict()
        self.stop_eviction = False
        self.eviction_thread = threading.Thread(
            target=self.eviction_loop,
            args=(EVICT_INTERVAL, timedelta(seconds=EVICT_TIMEOUT)),
            daemon=True,
        )
        self.eviction_thread.start()


    def load(self, model_id, path, engine):
        """
        loads model if there is avilable memory
        :param model_id: model id
        :param path: path to model
        :param engine: model engine
        :return: model object
        """
        if model_id in self.loaded_models:
            self.loaded_models.move_to_end(model_id)
            return self.loaded_models[model_id]

        model_size = os.path.getsize(path)

        if self.device == 'cuda':
            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            if total_vram < model_size + self.buffer:
                raise MemoryError('Model too large for GPU')
            while self.get_free_vram()[0] < model_size + self.buffer:
                if not self.loaded_models:
                    raise MemoryError('No available memory for model')
                # Evict the LRU model (the first item in the OrderedDict)
                lru_id = next(iter(self.loaded_models))
                lru_model = self.loaded_models[lru_id]
                if not lru_model.model_lock.acquire(blocking=False):
                    # If the LRU is busy, can't reclaim memory right now
                    raise MemoryError('No available memory for model — all models busy')
                try:
                    print(f'[load] evicting LRU model to make room: {lru_id}')
                    self.eject(lru_id)
                finally:
                    try:
                        lru_model.model_lock.release()
                    except RuntimeError:
                        pass

        model = self.loader.load_model(model_id, path, engine)
        self.loaded_models[model_id] = model
        return model

    def get_free_vram(self):
        """
        gets amount of free VRAM
        :return: amount of free VRAM
        """
        if self.device == 'cuda':
            return torch.cuda.mem_get_info()
        return (0, 0)

    def eject(self, model_id):
        """
        ejects a model by removing it from the list and cleaning the garbage
        :param model_id: model id
        """
        del self.loaded_models[model_id]
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def eviction_loop(self, interval_seconds, idle_timeout):
        """
        eviction loop for auto ejecting unused models
        :param interval_seconds: interval for when to work
        :param idle_timeout: how much time until kicking inactive models
        """
        while not self.stop_eviction:
            time.sleep(interval_seconds)
            self.Models = scan_models()
            self.sweep_idle_models(idle_timeout)

    def sweep_idle_models(self, idle_timeout):
        """
        check for inactive models and kick them
        :param idle_timeout: how much time until kicking inactive models
        """
        now = datetime.now()
        for model_id, model in list(self.loaded_models.items()):
            if now - model.last_used >= idle_timeout:
                if model.model_lock.acquire(blocking=False):
                    try:
                        if model_id in self.loaded_models:
                            print(f'[evictor] ejecting idle model: {model_id}')
                            self.eject(model_id)
                    finally:
                        try:
                            model.model_lock.release()
                        except RuntimeError:
                            pass


    def start_prompt(self, request, executor=None, user_obj=None):
        """
        Begin a prompt. Returns one of:
            {'type': 'text', 'text': str}              — final answer
            {'type': 'tool_call', ...state...}         — needs a remote tool

        For 'tool_call', the server should send MCPC to the client and then
        call continue_prompt() with the result.
        :param request: prompt in the request format
        :param executor: executor object for local tools
        :param user_obj: object with user info
        :returns another tool call or reply
        """
        provider = request['provider']

        if provider == 'AlterEgo':
            return {'type': 'text', 'text': self.prompt_alterego(request)}

        if provider not in ('Llama', 'Gemini'):
            raise ValueError(f"Unknown provider: {provider}")

        ctx_window = LLAMA_CTX_SIZE if provider == 'Llama' else GEMINI_CTX_SIZE
        messages = build_messages(
            request, ctx_window,
            reserve_for_output=request['sampling']['max_tokens'],
        )
        msgs = [{'role': r, 'content': c} for r, c in messages]

        return self.step_or_finish(request, msgs, executor, user_obj,
                                    iterations_used=0)

    def continue_prompt(self, state, call_id, result, executor=None, user_obj=None):
        """
        Resume a paused prompt with a tool result. Returns next step.
        :param state: state of prompt
        :param call_id: tool call id
        :param result: tool result
        :param executor: local tool executor
        :param user_obj: object with user info
        :returns another tool call or reply
        """
        msgs = state['msgs']
        request = state['request']

        msgs.append({
            'role': 'tool',
            'tool_call_id': call_id,
            'name': state['_pending_call_name'],
            'content': json.dumps(result),
        })

        return self.step_or_finish(request, msgs, executor, user_obj,
                                    iterations_used=state['iterations_used'])


    def prompt(self, request, executor=None, user_obj=None):
        """
        main prompting fnction
        :param request: prompt in the request format
        :param executor: local tool executor
        :param user_obj: object with user info
        :return: prompt response
        """
        state = self.start_prompt(request, executor, user_obj)
        if state['type'] == 'tool_call':
            return ("[error] this code path cannot handle remote tools — "
                    "use start_prompt/continue_prompt for MCP")
        return state['text']


    def prompt_alterego(self, request):
        """
        prompts AlterEgo
        :param request: prompt in the request format
        :return: reply
        """
        model_id = 'AlterEgo'
        messages = build_messages(request,context_window=ALTEREGO_CTX_SIZE,reserve_for_output=request['sampling']['max_tokens'],)
        system = next(c for r, c in messages if r == 'system')
        history = [(r, c) for r, c in messages[1:-1]]
        user = messages[-1][1]

        model = self.load(model_id, MODEL_PATH + f'/{model_id}.pt',engine='AlterEgo')
        model.model_lock.acquire()
        output = model.model.prompt(
            user,
            history=history,
            system_prompt=system,
            max_new_tokens=request['sampling']['max_tokens'],
            temperature=request['sampling']['temperature'],
            top_k=request['sampling']['top_k'],
            top_p=request['sampling']['top_p'],
        )
        model.model_lock.release()
        model.last_used = datetime.now()
        return output

    def step_or_finish(self, request, msgs, executor, user_obj, iterations_used):
        """
        Run one LLM step. If text is produced, return it.
        If a local tool is requested, execute it inline and recurse.
        If a remote tool is requested, return a tool_call state for the server.
        :param request: prompt in request format
        :param msgs: messages so far
        :param executor: local tool executor
        :param user_obj: object with user info
        :param iterations_used: iterations so far
        """
        max_iterations = (executor.MAX_TOOL_CALLS_PER_TURN if executor is not None else 1)
        if iterations_used >= max_iterations:
            return {'type': 'text', 'text': "Tool iteration limit reached."}

        if executor is not None:
            descriptors = executor.registry.descriptors_for_user(user_obj)
        else:
            descriptors = []

        provider = request['provider']

        if provider == 'Llama':
            model_id = request['model_id']
            model = self.load(model_id, MODEL_PATH + f'/{model_id}.gguf',engine='Llama')
            model.model_lock.acquire()
            step = self.llama_step(request, msgs, descriptors, model)
            model.last_used = datetime.now()
            model.model_lock.release()
        elif provider == 'Gemini':
            step = self.gemini_step(request, msgs, descriptors)
        else:
            raise ValueError(f"unsupported provider in _step_or_finish: {provider}")

        if step['type'] == 'text':
            return {'type': 'text', 'text': step['text']}

        msgs.append(step['assistant_msg'])
        first_call = step['calls'][0]

        tool = executor.registry.get(first_call['name']) if executor else None
        if tool is not None and not tool.remote:
            result = executor.execute(first_call['name'], first_call['args'], user_obj)
            msgs.append({'role': 'tool', 'tool_call_id': first_call['id'], 'name': first_call['name'], 'content': json.dumps(result),})
            return self.step_or_finish(request, msgs, executor, user_obj, iterations_used=iterations_used + 1)

        return {
            'type': 'tool_call',
            'calls': step['calls'],
            'msgs': msgs,
            'request': request,
            'iterations_used': iterations_used + 1,
            '_pending_call_name': first_call['name'],
        }

    def llama_step(self, request, msgs, tool_descriptors, model):
        """
        One round-trip with a llama-cpp model. Caller holds model.model_lock.
        :param request: prompt in request format
        :param msgs: messages so far
        :param tool_descriptors: tool info
        :param model: model obj
        :return: response
        """
        model_id = request['model_id'].lower()
        needs_hermes = 'qwen' in model_id or 'hermes' in model_id
        if needs_hermes:
            rendered_msgs = to_hermes_format(msgs)
        else:
            rendered_msgs = msgs
        kwargs = dict(messages=rendered_msgs, max_tokens=request['sampling']['max_tokens'], temperature=request['sampling']['temperature'], top_k=request['sampling']['top_k'], top_p=request['sampling']['top_p'],)
        if tool_descriptors:
            kwargs['tools'] = tool_descriptors
            kwargs['tool_choice'] = 'auto'

        output = model.model.create_chat_completion(**kwargs)
        msg = output['choices'][0]['message']

        raw_content = msg.get('content') or ''
        if isinstance(raw_content, list):
            content = ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in raw_content)
        else:
            content = raw_content

        if msg.get('tool_calls'):
            tool_calls = []
            for tc in msg['tool_calls']:
                try:
                    args = json.loads(tc['function']['arguments'])
                except (json.JSONDecodeError, TypeError):
                    args = {'_parse_error': 'model produced malformed JSON'}
                tool_calls.append({'id': tc.get('id') or str(uuid.uuid4()), 'name': tc['function']['name'], 'args': args,})
            return {
                'type': 'tool_call',
                'calls': tool_calls,
                'assistant_msg': {
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': msg.get('tool_calls'),
                },
            }

        parsed_calls = parse_hermes_tool_calls(content)
        if parsed_calls:
            return {
                'type': 'tool_call',
                'calls': parsed_calls,
                'assistant_msg': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [
                        {
                            'id': c['id'],
                            'type': 'function',
                            'function': {'name': c['name'], 'arguments': json.dumps(c['args'])},
                        }
                        for c in parsed_calls
                    ],
                },
            }

        return {'type': 'text', 'text': content, 'assistant_msg': {'role': 'assistant', 'content': content},}

    def sanitize_gemini_schema(self, schema):
        """
        strip JSON schema for gemini supported schema
        :param schema: old schema to refactor
        :return: gemini safe schema
        """
        if not isinstance(schema, dict):
            return schema
        DROP = {
            'exclusiveMinimum', 'exclusiveMaximum',
            'multipleOf', 'patternProperties', 'additionalProperties',
            'const', 'examples', '$schema', '$id', '$ref', 'definitions',
            'allOf', 'anyOf', 'oneOf', 'not',
            'contentEncoding', 'contentMediaType',
            'minLength', 'maxLength', 'pattern',
        }
        cleaned = {}
        for k, v in schema.items():
            if k in DROP:
                continue
            if isinstance(v, dict):
                cleaned[k] = self.sanitize_gemini_schema(v)
            elif isinstance(v, list):
                cleaned[k] = [
                    self.sanitize_gemini_schema(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                cleaned[k] = v
        return cleaned

    def gemini_step(self, request, msgs, tool_descriptors):
        """
        One round-trip with Gemini.
        :param request: prompt in request format
        :param msgs: messages so far
        :param tool_descriptors: tool info
        :return: response
        """
        if 'api' in request.keys():
            api_key = request['api']
            if not api_key:
                return {
                    'finish_reason': 'error',
                    'message': "No Gemini API key configured. Add one in Settings → Model → Gemini API.",
                    'tool_calls': [],
                }


        system = next((m['content'] for m in msgs if m['role'] == 'system'), None)

        contents = []
        for m in msgs:
            role = m['role']
            if role == 'system':
                continue

            if role == 'user':
                contents.append(types.Content(role='user',parts=[types.Part.from_text(text=m['content'])],))

            elif role == 'assistant':
                parts = []
                if m.get('content'):
                    parts.append(types.Part.from_text(text=m['content']))
                for tc in m.get('tool_calls') or []:
                    if 'function' in tc:
                        name = tc['function']['name']
                        args_raw = tc['function'].get('arguments')
                        if isinstance(args_raw, str):
                            try:
                                args = json.loads(args_raw)
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = args_raw or {}
                    else:
                        name = tc['name']
                        args = tc.get('args') or {}
                    parts.append(types.Part.from_function_call(name=name, args=args))
                contents.append(types.Content(role='model', parts=parts))

            elif role == 'tool':
                resp = m['content']
                if isinstance(resp, str):
                    try:
                        resp = json.loads(resp)
                    except json.JSONDecodeError:
                        resp = {'result': resp}
                contents.append(types.Content(role='user',parts=[types.Part.from_function_response(name=m.get('name', 'unknown'),response=resp,)],))

        gemini_tools = None
        if tool_descriptors:
            decls = []
            for d in tool_descriptors:
                f = d['function']
                sanitized = self.sanitize_gemini_schema(f['parameters'])
                decls.append(types.FunctionDeclaration(
                    name=f['name'],
                    description=f['description'],
                    parameters=sanitized,))
                gemini_tools = [types.Tool(function_declarations=decls)]

        client = genai.Client(api_key=request['api'])
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=gemini_tools,
                max_output_tokens=request['sampling']['max_tokens'],
                temperature=request['sampling']['temperature'],
                top_k=request['sampling']['top_k'],
                top_p=request['sampling']['top_p'],
                seed=67,
            ),
        )

        candidate = response.candidates[0]
        tool_calls = []
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                tool_calls.append({
                    'id': str(uuid.uuid4()),
                    'name': fc.name,
                    'args': dict(fc.args) if fc.args else {},
                })
            elif hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

        if tool_calls:
            return {
                'type': 'tool_call',
                'calls': tool_calls,
                'assistant_msg': {
                    'role': 'assistant',
                    'content': ''.join(text_parts),
                    'tool_calls': tool_calls,
                },
            }

        final_text = ''.join(text_parts)
        return {'type': 'text','text': final_text,'assistant_msg': {'role': 'assistant', 'content': final_text},}


def scan_models():
    """
    scans for models in model path
    :return: model list
    """
    result = {"Llama": [], "AlterEgo": []}
    for filename in os.listdir(MODEL_PATH):
        name, ext = os.path.splitext(filename)
        if ext == ".gguf":
            result["Llama"].append(name)
        elif ext == ".pt":
            result["AlterEgo"].append(name)
    return result


def build_model_request(user_preferences, history, rag_context, prompt):
    """
    converts prompt and settings to request format
    :param user_preferences: user settings
    :param history: convo history
    :param rag_context: rag context
    :param prompt: prompt
    :return: prompt in request format
    """
    request = {
        'provider': user_preferences['provider'],
        'model_id': user_preferences['active_model'],
        'system_prompt': user_preferences['system_prompt'],
        'history': history,
        'context_items': rag_context,
        'sampling': {
            'temperature': user_preferences['temperature'],
            'top_k': user_preferences['k'],
            'top_p': user_preferences['p'],
            'max_tokens': user_preferences['max_output_tokens'],
        },
        'prompt': prompt,
    }
    if request['provider'] == 'Gemini':
        request['api'] = user_preferences['api']
    return request


def build_messages(request, context_window, reserve_for_output):
    """
    build a proper message that ai can understand + cut if too big for context
    :param request: prompt in request format
    :param context_window: ai context window
    :param reserve_for_output: how much to keep for output out of context
    :return: built message
    """
    system = request['system_prompt']
    if request.get('context_items'):
        ctx = "\n".join(f"- {x}" for x in request['context_items'])
        system = f"{system}\n\nUse this context to answer:\n{ctx}"

    messages = [('system', system)]
    messages.extend(request.get('history') or [])
    messages.append(('user', request['prompt']))

    budget = context_window - reserve_for_output
    while estimate_tokens(messages) > budget and len(messages) > 2:
        del messages[1]

    return messages


def estimate_tokens(messages):
    """
    estimate how many tokens convo takes
    :param messages: messages so far
    :return: estimated amount of tokens
    """
    return sum(len(c) for _, c in messages) // 4


def parse_hermes_tool_calls(text):
    """
    Extract Hermes-style <tool_call>{...}</tool_call> blocks from text.
    :param text: request for tool
    :returns:a list of {'id', 'name', 'args'} dicts, or [] if none found.
    """
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1)
        obj = lenient_json_parse(raw)
        if obj is None:
            continue
        calls.append({
            'id': str(uuid.uuid4()),
            'name': obj.get('name', ''),
            'args': obj.get('arguments') or obj.get('args') or {},
        })
    return calls


def lenient_json_parse(s):
    """
    Try strict JSON first, then loosen for common LLM mistakes.
    :param s: serialized string
    :return: object
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r"(?<![\\\w])'([^']*?)'(?![\\\w])", r'"\1"', s)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    return None

def to_hermes_format(msgs):
    """
    Convert OpenAI-style msgs to Hermes/Qwen text-tag format.
    :param msgs: messages so far
    :return: messages in correct format
    """
    out = []
    for m in msgs:
        if m['role'] == 'tool':
            tool_name = m.get('name', 'unknown')
            content = m.get('content', '')
            out.append({
                'role': 'user',
                'content': f'<tool_response>\n{{"name": "{tool_name}", "content": {content}}}\n</tool_response>',
            })
        elif m['role'] == 'assistant' and m.get('tool_calls'):
            text_parts = []
            if m.get('content'):
                text_parts.append(m['content'])
            for tc in m['tool_calls']:
                fn = tc.get('function', {})
                name = fn.get('name', '')
                args = fn.get('arguments', '{}')
                if not isinstance(args, str):
                    args = json.dumps(args)
                text_parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>')
            out.append({'role': 'assistant', 'content': '\n'.join(text_parts)})
        else:
            out.append(m)
    return out