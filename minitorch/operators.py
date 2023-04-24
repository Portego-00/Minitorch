"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    """
    Multiplication operator

    Args:
      x: Input float
      y: Input float

    Returns:
       Float resulting of computing x * y
    """
    return x * y
    raise NotImplementedError("Need to implement for Task 0.1")


def id(x: float) -> float:
    "$f(x) = x$"
    """
    Id operator

    Args:
      x: Input float

    Returns:
       Input float x
    """
    return x
    # TODO: Implement for Task 0.1.
    raise NotImplementedError("Need to implement for Task 0.1")


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    """
    Addition operator

    Args:
      x: Input float
      y: Input float

    Returns:
       Float resulting of computing x + y
    """
    return x + y
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    """
    Negative operator

    Args:
      x: Input float

    Returns:
       Float resulting of inverting the sign of the input float
    """
    return -x
    raise NotImplementedError("Need to implement for Task 0.1")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    """
    Lower than operator

    Args:
      x: Input float
      y: Input float

    Returns:
       0: x is greater or equal than y
       1: x is lower than y
    """
    return x < y
    # raise NotImplementedError("Need to implement for Task 0.1")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    """
    Equal operator

    Args:
      x: Input float
      y: Input float

    Returns:
       0: if x equals y
       1: if x is different than y
    """
    return x == y
    # raise NotImplementedError("Need to implement for Task 0.1")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    """
    Max operator

    Args:
      x: Input float
      y: Input float

    Returns:
       x: if x is greater than y
       y: if x is lower than y
    """
    if x >= y:
        return x
    else:
        return y
    # raise NotImplementedError("Need to implement for Task 0.1")


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    """
    Is close operator

    Args:
      x: Input float
      y: Input float

    Returns:
       0: x and y are different values and not close
       1: x and y are close or equal values
    """
    return abs(x - y) < 1e-2
    # raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    """
    Sigmoid operator

    Args:
      x: Input float

    Returns:
       \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}
    """
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))
    # raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    """
    Relu operator

    Args:
      x: Input float

    Returns:
       0: x is lower than 0
       x: x is greater than 0
    """
    return (x > 0) * x
    # raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    """
    Log Back operator

    Args:
      x: Input float
      d: Input float

    Returns:
       d times the derivative of the log(x)
    """
    return d / (x + EPS)
    # raise NotImplementedError("Need to implement for Task 0.1")


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    """
    Inv operator

    Args:
      x: Input float

    Returns:
       The inverse of x, which is 1/x
    """
    return 1 / x
    # raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    """
    Inv Back operator

    Args:
      x: Input float
      d: Input float

    Returns:
       d times the derivative of the inverse function
    """
    return -(d / x**2)
    # raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    """
    Relu Back operator

    Args:
      x: Input float
      d: Input float

    Returns:
       d times the derivative of the relu function
    """
    return (x > 0) * d
    # raise NotImplementedError("Need to implement for Task 0.1")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # TODO: Implement for Task 0.3.
    def func(list: Iterable[float]) -> Iterable[float]:
        newList = []
        for val in list:
            newList.append(fn(val))
        return newList

    return func
    # raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    return map(neg)(ls)
    # raise NotImplementedError("Need to implement for Task 0.3")


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    def func(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
        newlist = []
        for cont1, val1 in enumerate(list1):
            for cont2, val2 in enumerate(list2):
                if cont1 == cont2:
                    newlist.append(fn(val1, val2))
                    break
        return newlist

    return func


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    def func(list1: Iterable[float]) -> float:
        newlist = list(list1).copy()
        if len(newlist) == 0:
            return start
        val = newlist.pop()
        return fn(val, func(newlist))

    return func
    # raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    func = reduce(add, 0)
    return func(ls)
    # raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    func = reduce(mul, 1)
    return func(ls)
    # raise NotImplementedError("Need to implement for Task 0.3")
