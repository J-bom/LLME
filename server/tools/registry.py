class Tool:
    """
    Contains informaiton about the tool
    """
    def __init__(self, name, description, schema, implementation=None, requires_user=True, remote=False, owner_user_id=None):
        """
        initiallizer of the tool class
        :param name: tool name
        :param description: tool description
        :param schema: tool usage schema
        :param implementation: tool implementation
        :param requires_user: user requirement
        :param remote: is remote?
        :param owner_user_id: owner of tool (if remote)
        """
        self.name = name
        self.description = description
        self.schema = schema
        self.impl = implementation
        self.requires_user = requires_user
        self.remote = remote
        self.owner_user_id = owner_user_id

    def to_descriptor(self):
        """
        packs tool object into a dict object
        :return: dict object with info about the tool
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            }
        }


class ToolRegistry:
    """
    registry of tools, holds informaiton about all registerd tools.
    """
    def __init__(self):
        """
        initiallizer of the registry class
        """
        self._tools = {}
        self._user_tools = {}

    def register(self, tool):
        """
        registers a new tool
        :param tool: tool to be registerd
        """
        if tool.remote and tool.owner_user_id is not None:
            qualified_name = f"u{tool.owner_user_id}_{tool.name}"
            tool.name = qualified_name
            self._user_tools.setdefault(tool.owner_user_id, []).append(qualified_name)
        self._tools[tool.name] = tool

    def get(self, name):
        """
        returns tool
        :param name: tool name
        :return: tool
        """
        return self._tools.get(name)

    def descriptors_for_user(self, user_obj):
        """
        return tools visible to this user: globals + their own remote tools.
        :param user_obj: object containing user information
        :return: a list filled with dicts describing all the tools avilable to the user.
        """
        user_id = getattr(user_obj, 'user_id', None) if user_obj else None
        result = []
        for tool in self._tools.values():
            if not tool.remote:
                result.append(tool.to_descriptor())
            elif tool.owner_user_id == user_id:
                result.append(tool.to_descriptor())
        return result

    def descriptors(self):
        """Legacy — returns globals only. Used for fallback."""
        return [t.to_descriptor() for t in self._tools.values() if not t.remote]

    def unregister_user_tools(self, user_id):
        """Called on user disconnect — remove their MCP tools from registry."""
        for name in self._user_tools.pop(user_id, []):
            self._tools.pop(name, None)