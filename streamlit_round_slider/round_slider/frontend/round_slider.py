import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "round_slider",
    path=os.path.join(os.path.dirname(__file__), "frontend/build"),
)

def round_slider(key=None) -> int:
    return _component_func(key=key, default=0)

