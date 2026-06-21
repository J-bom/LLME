import json
import jsonschema
from threading import Lock


class ToolExecutor:
    """
    Excecutes server owned and provided tools.
    remote tools are handled by the server's back and forth tool call logic
    """

    MAX_TOOL_CALLS_PER_TURN = 5
    MAX_OUTPUT_LENGTH = 4000

    def __init__(self, registry, log_fn=None):
        """
        Tool executor initallizer
        :param registry: tool registry
        :param log_fn: logging function
        """
        self.registry = registry
        self.audit_lock = Lock()
        self.log_fn = log_fn or (lambda *a, **kw: None)

    def execute(self, tool_name, args, user_obj):
        """
        executes local tools on the server
        :param tool_name: name of the tool
        :param args: args for the tool
        :param user_obj: object containing user information
        :return: result of the operation
        """
        tool = self.registry.get(tool_name)
        if tool is None:
            return {'error': f'unknown tool: {tool_name}'}

        #Safety net
        if tool.remote:
            self.log_fn('WARN', 'Server',
                        f'remote tool {tool_name} reached executor.execute() — '
                        f'Security Alert!!!')
            return {'error': 'remote tool not handled by executor'}

        try:
            jsonschema.validate(args, tool.schema)
        except jsonschema.ValidationError as err:
            return {'error': f'invalid arguments: {err.message}'}

        self.audit(user_obj, tool_name, args)

        try:
            result = tool.impl(args, user_obj)
            return self.sanitize_output(result)
        except Exception as err:
            return {'error': f'tool execution failed: {type(err).__name__}'}

    def sanitize_output(self, result):
        """
        sanitizes result output.
        :param result: tool call result
        :return: sanitized results in json format
        """
        try:
            s = json.dumps(result)
        except TypeError:
            return {'error': 'tool returned non-serializable data'}
        if len(s) > self.MAX_OUTPUT_LENGTH:
            return {'truncated_result': s[:self.MAX_OUTPUT_LENGTH] + '...[truncated]'}
        return result

    def audit(self, user_obj, tool_name, args):
        """
        logs executions.
        :param user_obj: an object containing a user's information
        :param tool_name: name of the tool
        :param args: args for the tool
        """
        user_id = getattr(user_obj, 'user_id', '?')
        self.audit_lock.acquire()
        self.log_fn('TOOL_CALL', 'Server', f'user={user_id} tool={tool_name} args={args}')
        self.audit_lock.release()
