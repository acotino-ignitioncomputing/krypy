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
        if arg is not None and hasattr(arg, "dtype"):
            dtypes.append(arg.dtype)

    if not dtypes:
        return numpy.dtype(numpy.float64)

    # Try numpy 2.0+ approach first
    try:
        return numpy.result_type(*dtypes)
    except (AttributeError, TypeError):
        # Fall back to numpy < 2.0 approach
        try:
            return numpy.find_common_type(dtypes, [])
        except (AttributeError, TypeError):
            # Ultimate fallback - just use the first dtype
            return dtypes[0] if dtypes else numpy.dtype(numpy.float64)


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


def get_blas_functions():
    """Get BLAS functions with compatibility handling.

    Returns:
        tuple: (drotg, zrotg) functions or None if not available
    """
    try:
        import scipy.linalg.blas as blas

        # Test if the functions we need exist
        drotg = getattr(blas, "drotg", None)
        zrotg = getattr(blas, "zrotg", None)
        if drotg is not None and zrotg is not None:
            return drotg, zrotg
    except ImportError:
        pass

    # Return None if BLAS functions are not available
    return None, None


def safe_asanyarray(x):
    """Safely convert to array with compatibility handling.

    Args:
        x: Input to convert to array

    Returns:
        numpy.ndarray: Array representation
    """
    try:
        return numpy.asanyarray(x)
    except (AttributeError, TypeError):
        # Fallback for very old numpy versions
        return numpy.asarray(x)


def safe_atleast_2d(x):
    """Safely ensure at least 2D with compatibility handling.

    Args:
        x: Input array

    Returns:
        numpy.ndarray: At least 2D array
    """
    try:
        return numpy.atleast_2d(x)
    except (AttributeError, TypeError):
        # Fallback for very old numpy versions
        x = numpy.asarray(x)
        if x.ndim == 0:
            return x.reshape(1, 1)
        elif x.ndim == 1:
            return x.reshape(1, -1)
        else:
            return x
