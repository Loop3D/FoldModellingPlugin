import functools
import inspect

def type_check(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.parameters
        for arg, param in zip(args, params.values()):
            if param.annotation is not param.empty and not isinstance(arg, param.annotation):
                raise TypeError(f"Argument {param.name} must be {param.annotation}")
        result = func(*args, **kwargs)
        if sig.return_annotation is not sig.empty and not isinstance(result, sig.return_annotation):
            raise TypeError(f"Return value must be {sig.return_annotation}")
        return result
    return wrapper