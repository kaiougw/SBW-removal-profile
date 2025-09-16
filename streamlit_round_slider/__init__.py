import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "round_slider",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)

def round_slider(
    value: int = 0,
    min: int = 0,
    max: int = 360,
    step: int = 1,
    radius: int = 110,
    width: int = 12,
    start_angle: int = 0,
    circle_shape: str = "full",
    key: str | None = None,
) -> int:
    """Return the current angle in degrees."""
    return _component_func(
        value=value,
        min=min,
        max=max,
        step=step,
        radius=radius,
        width=width,
        startAngle=start_angle,
        circleShape=circle_shape,
        key=key,
        default=value,
    )

