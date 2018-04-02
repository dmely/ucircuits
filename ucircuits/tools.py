"""
API-related tools.
"""
import types

def use_base_docstrings(cls):
    """
    Override the documentation of all methods overriding (implementing)
    the base class' abstract methods and properties with documentation from
    those.

    Parameters
    ----------
    cls : types.ClassType
        Derived class implementing a base class whose docstring methods we
        would like to inherit.

    Returns
    -------
    cls : types.ClassType
        Derived class with filled-in docstrings.
    """
    for name, function in vars(cls).items():
        if isinstance(function, types.FunctionType):
            try:
                parent_function = getattr(cls.__bases__[-1], name)
                parent_docs = parent_function.__doc__
            except AttributeError:
                continue
            else:
                function.__doc__ = parent_docs
                break

        elif isinstance(function, property):
            try:
                parent_property = getattr(cls.__bases__[-1], name)
                parent_docs = getattr(parent_property.fget, '__doc__')
            except AttributeError:
                continue
            else:
                new_property = property(fget=function.fget,
                                        fset=function.fset,
                                        fdel=function.fdel,
                                        doc=parent_property.fget.__doc__)
                setattr(cls, name, new_property)
                break
    return cls
