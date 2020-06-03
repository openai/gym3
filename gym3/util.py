import importlib


def call_func(fn_path, **kwargs):
    """
    Look up a function for a function path, e.g. fn_path='gym3:call_func' will
    return this function, then call it with the specified kwargs.
    """
    assert isinstance(fn_path, str)
    module_path, fn_name = fn_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)(**kwargs)
