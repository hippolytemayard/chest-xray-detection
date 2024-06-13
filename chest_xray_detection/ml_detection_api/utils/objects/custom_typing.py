"""Module related to any customiztion of typing."""

from typing import Any, Literal, Optional

from pydantic.fields import ModelField


def get_json_schema_compatible_custom_typing(class_: type) -> type:
    """Transform any class into json schema compatible pydantic type. This function
    hacks pydantic validation mechanism to allow arbitrary type not to raise an error.

    Context: type `numpy.ndarray` and other types from module `shapely`, even if excluded=True,
    will trigger an error when schema is called to create openapi model.


    The example below would not work with numpy.ndarray inside an instance of pydantic.BaseModel.

    Example:
    --------
    ```
    import numpy
    from pydantic import BaseModel, Field

    Ndarray = get_json_schema_compatible_custom_typing(numpy.ndarray)

    class StuffWithArray(BaseModel):
        name: str
        array: Ndarray = Field(None, excluded=True)

    stuff = StuffWithArray(name="oscar", array=numpy.ones((2, 2)))
    stuff.schema()
    ```
    """

    class class_type(class_):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v):
            if not isinstance(v, class_):
                raise ValueError(f"Not a valid class {class_}")
            return v

        @classmethod
        def __modify_schema__(cls, field_schema: dict[str, Any], field: Optional[ModelField]):
            # this hack enables json schema to work in pydantic"""
            field.type_ = Any

    return class_type
