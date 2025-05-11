from typing import Callable, TypeVar, Any, Optional
from functools import wraps

T = TypeVar('T', bound=Callable[..., Any])


class NamedMethod:
    """
    A descriptor that wraps a method and provides multiple accessible properties
    """

    def __init__(self, func: Callable, name: str, rule_type: Optional[str] = None,
                 reversible: bool = False, allows_weak_ranking: bool = False):
        self.func = func
        self.name = name
        self.rule_type = rule_type
        self.reversible = reversible
        self.allows_weak_ranking = allows_weak_ranking
        wraps(func)(self)

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        # Create a bound method that maintains access to all properties
        bound_method = self.func.__get__(obj, cls)
        bound_method.name = self.name
        bound_method.rule_type = self.rule_type
        bound_method.reversible = self.reversible
        bound_method.allows_weak_ranking = self.allows_weak_ranking
        return bound_method

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the descriptor directly callable"""
        return self.func(*args, **kwargs)


def method_name(name: str, *, rule_type: Optional[str] = None,
                reversible: bool = False, allows_weak_ranking = False) -> Callable[[T], T]:
    """
    A decorator that assigns multiple accessible properties to a method.

    Args:
        name: The string name to assign to the method
        rule_type: Optional type classification for the method
        reversible: Whether the method's operation can be reversed
        allows_weak_ranking: Whether the method incorporates some ability to handle ties in rankings.
    """

    def decorator(func: T) -> T:
        return NamedMethod(func, name, rule_type, reversible)

    return decorator
