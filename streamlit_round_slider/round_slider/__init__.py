import os
import streamlit.components.v1 as components

# Point Streamlit to the static frontend (no build step needed)
_component_func = components.declare_component(
    "round_slider",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)

def round_slider(
    value: int = 0,
    min: int = 0,
    max: int = 360,
    step: int = 1,
    radius: int = 90,        # slider radius in px
    width: int = 10,         # track width
    start_angle: int = 0,    # where 0° sits; 0 is to the right (east)
    circle_shape: str = "full",  # "full" | "pie"
    key: str | None = None,
) -> int:
    """
    Returns the current angle in degrees [min..max].
    """
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
