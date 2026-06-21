import ast
import operator
from datetime import datetime


"""
some quick standalone server-run tools the user can access to see the wonderful agentic capabilites of LLME
"""

#supported opperations for math tool
OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def get_current_time(args, user_obj):
    """
    return current server time
    :param args: tool args
    :param user_obj: info of calling user
    :return: server time
    """
    return {'time': datetime.now().isoformat()}


def calculate(args, user_obj):
    """
    calculates a math expression
    :param args: tool args
    :param user_obj: info of calling user
    :return: results
    """
    expr = args.get('expression', '')
    if not isinstance(expr, str) or len(expr) > 200:
        return {'error': 'invalid expression'}
    try:
        tree = ast.parse(expr, mode='eval')
        result = safe_eval(tree.body)
        return {'result': result}
    except Exception as e:
        return {'error': f'could not evaluate: {e}'}

def safe_eval(node):
    """
    must make sure that the expression is supported otherwise some models go crazy
    :param node: current node
    :return:
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError(f"forbidden expression element: {ast.dump(node)}")


def make_search_user_documents(rag_engine):
    """
    factory for a method that lets the ai search through user documents
    this is a factory because the RAG_ENGINE is created in runtime
    :param rag_engine: RAG_ENGINE object
    :return: tool function
    """

    def search_user_documents(args, user_obj):
        query = args.get('query', '')
        if not isinstance(query, str) or len(query) > 500:
            return {'error': 'invalid query'}
        if user_obj is None or user_obj.vector_db is None:
            return {'error': 'no user document store available'}
        results = rag_engine.search(query, user_obj.vector_db)
        return {'results': results[:5]}
    return search_user_documents

