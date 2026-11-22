"""Static Streamlit component: a single rectangle over an EXIF-corrected preview."""
import base64
from io import BytesIO
from pathlib import Path

import streamlit.components.v1 as components

_selector = components.declare_component("repair_region_selector", path=str(Path(__file__).parent / "components/region_selector"))


def select_region(image, image_id, region=None, disabled=False):
    preview = image.copy()
    preview.thumbnail((1024, 1024))
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return _selector(image="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                     image_id=image_id, region=region, disabled=disabled,
                     default=None, key="repair_region_" + str(image_id))
