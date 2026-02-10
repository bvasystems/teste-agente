from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable)


def terminal(message_param: str | None = None) -> Callable[[T], T]:
    """
    Decorator to mark a tool as terminal.
    
    When a tool marked with this decorator is executed by the agent, the agent's
    execution loop will terminate immediately after the tool completes.
    
    Args:
        message_param: Optional name of the parameter in the decorated function
                      that contains the message to be treated as the assistant's
                      response. If provided, the value of this argument will be
                      used as the final response from the agent.
    """
    def decorator(func: T) -> T:
        setattr(func, "_is_terminal", True)
        setattr(func, "_terminal_message_param", message_param)
        return func

    return decorator
