from collections.abc import Callable
import importlib.util
from functools import wraps
from typing import overload


# Overload 1: Decorator with single module string
@overload
def needs[**P, R](
    module_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


# Overload 2: Decorator with list of modules
@overload
def needs[**P, R](
    module_name: list[str],
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


# Overload 3: Decorator without arguments (bare decorator)
@overload
def needs[**P, R](module_name: Callable[P, R]) -> Callable[P, R]: ...


def needs[**P, R](
    module_name: str | list[str] | Callable[P, R] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """
    Check if a module is installed. Can be used as a decorator.

    Usage as decorator with single module:
        @needs("numpy")
        def my_function(x: int) -> str:
            import numpy as np
            ...

    Usage as decorator with multiple modules:
        @needs(["numpy", "pandas"])
        def my_function(x: int) -> str:
            ...

    Usage as decorator without parentheses:
        @needs
        def my_function():
            # Will check for "my_function" module - probably not useful
            ...
    """

    def _check_single_module(module: str) -> None:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                error_msg = (
                    f"\n{'=' * 60}\n"
                    f"❌ Module Not Found: '{module}'\n"
                    f"{'-' * 60}\n"
                    f"Please install it using one of the following commands:\n\n"
                    f"  📦 pip install {module}\n"
                    f"  🚀 uv add {module}\n"
                    f"{'=' * 60}"
                )
                raise ImportError(error_msg)
        except (AttributeError, ImportError) as e:
            if "No module named" in str(e):
                error_msg = (
                    f"\n{'=' * 60}\n"
                    f"❌ Module Not Found: '{module}'\n"
                    f"{'-' * 60}\n"
                    f"Please install it using one of the following commands:\n\n"
                    f"  📦 pip install {module}\n"
                    f"  🚀 uv add {module}\n"
                    f"{'=' * 60}"
                )
                raise ImportError(error_msg)
            raise

    # Used as decorator with string or list of modules
    if isinstance(module_name, (str, list)):
        modules: list[str] = (
            [module_name] if isinstance(module_name, str) else module_name
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                for module in modules:
                    _check_single_module(module)
                return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    # Used as decorator without arguments: @needs
    if callable(module_name):
        func = module_name

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _check_single_module(func.__name__)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    # Fallback - shouldn't happen
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# Standalone function for checking modules without decoration
def check_modules(*module_names: str) -> None:
    """
    Check if modules are installed without using as a decorator.

    Usage:
        check_modules("numpy", "pandas", "matplotlib")
    """

    def _check_single_module(module: str) -> None:
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                error_msg = (
                    f"\n{'=' * 60}\n"
                    f"❌ Module Not Found: '{module}'\n"
                    f"{'-' * 60}\n"
                    f"Please install it using one of the following commands:\n\n"
                    f"  📦 pip install {module}\n"
                    f"  🚀 uv add {module}\n"
                    f"{'=' * 60}"
                )
                raise ImportError(error_msg)
        except (AttributeError, ImportError) as e:
            if "No module named" in str(e):
                error_msg = (
                    f"\n{'=' * 60}\n"
                    f"❌ Module Not Found: '{module}'\n"
                    f"{'-' * 60}\n"
                    f"Please install it using one of the following commands:\n\n"
                    f"  📦 pip install {module}\n"
                    f"  🚀 uv add {module}\n"
                    f"{'=' * 60}"
                )
                raise ImportError(error_msg)
            raise

    for module_name in module_names:
        _check_single_module(module_name)
