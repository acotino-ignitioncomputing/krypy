"""
Compatibility layer for different numpy and scipy versions.

This module provides a consistent API across different versions of numpy and scipy,
handling deprecated functions and API changes.
"""

import numpy


def find_common_dtype(*args):
    """Find common dtype from numpy and scipy objects.

    This function provides compatibility between numpy < 2.0 (which has
    find_common_type) and numpy >= 2.0 (which deprecated it in favor of
    result_type).

    Args:
        *args: Objects that may have dtype attributes

    Returns:
        numpy.dtype: Common dtype for the objects
    """
    dtypes = []
    for arg in args:
        if type(arg) is list:
            # In case entry in args is a list of objects, add list to dtypes
            dtypes = dtypes + arg
        else:
            dtypes.append(arg)

    # Try numpy 2.0+ approach first
    try:
        return numpy.result_type(*dtypes)
    except (AttributeError, TypeError):
        # Fall back to numpy < 2.0 approach
        return numpy.find_common_type(dtypes, [])


def isintlike(x):
    """Check if x is integer-like.

    This function provides compatibility for scipy.sparse.sputils.isintlike
    which was removed in scipy 1.14+.

    Args:
        x: Value to check

    Returns:
        bool: True if x is integer-like
    """
    try:
        # Try scipy < 1.14 approach
        from scipy.sparse.sputils import isintlike as scipy_isintlike

        return scipy_isintlike(x)
    except ImportError:
        """
        # Fall back to numpy approach for scipy >= 1.14
        try:
            return numpy.issubdtype(type(x), numpy.integer)
        except (TypeError, AttributeError):
        """
        # Ultimate fallback
        try:
            return int(x) == x
        except (TypeError, ValueError):
            return False
