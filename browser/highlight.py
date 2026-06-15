"""Set-of-marks — draw numbered boxes over clickable elements on a screenshot."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont


BOX_COLORS = {
    'a': (70, 130, 230),      # blue for links
    'button': (50, 180, 80),  # green for buttons
    'input': (230, 150, 50),  # orange for inputs
    'textarea': (230, 150, 50),
    'select': (160, 80, 200), # purple for selects
    'label': (70, 130, 230),
    'default': (220, 50, 50), # red for everything else
}
TEXT_COLOR = (255, 255, 255)
BOX_OUTLINE_WIDTH = 3
LABEL_PADDING = 3

FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def _load_font(size: int):
    """Load the best available font, scaled for readability on high-DPR screenshots."""
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def draw_set_of_marks(
    screenshot_bytes: bytes,
    elements: list[dict],
    dpr: float = 2.0,
) -> tuple[bytes, dict[int, dict]]:
    """Draw numbered boxes on screenshot for VLM consumption.

    Args:
        screenshot_bytes: Raw PNG bytes from Playwright screenshot.
        elements: List of {index, tag, text, role, bbox: {x, y, width, height}} in CSS coords.
        dpr: Device pixel ratio (CSS coords * dpr = image pixel coords).

    Returns:
        (annotated_png_bytes, index_to_element_map)
    """
    img = Image.open(io.BytesIO(screenshot_bytes))
    draw = ImageDraw.Draw(img)

    font_size = max(16, int(img.width / 80))
    font = _load_font(font_size)

    index_map: dict[int, dict] = {}

    for el in elements:
        idx = el["index"]
        bbox = el["bbox"]

        x = bbox["x"] * dpr
        y = bbox["y"] * dpr
        w = bbox["width"] * dpr
        h = bbox["height"] * dpr

        if x < 0 or y < 0 or w < 5 or h < 5:
            continue
        if x + w > img.width or y + h > img.height:
            continue

        tag = el.get('tag', 'default')
        color = BOX_COLORS.get(tag, BOX_COLORS['default'])
        # Draw dashed box (4 sides with gaps)
        dash_len = 8
        gap_len = 5
        for side in [(x, y, x+w, y), (x+w, y, x+w, y+h), (x, y+h, x+w, y+h), (x, y, x, y+h)]:
            sx, sy, ex, ey = side
            length = max(abs(ex-sx), abs(ey-sy))
            if length == 0: continue
            dx = (ex-sx)/length if length > 0 else 0
            dy = (ey-sy)/length if length > 0 else 0
            pos = 0
            while pos < length:
                end = min(pos + dash_len, length)
                draw.line(
                    [(sx + dx*pos, sy + dy*pos), (sx + dx*end, sy + dy*end)],
                    fill=color, width=BOX_OUTLINE_WIDTH
                )
                pos += dash_len + gap_len

        label = f"#{idx}"
        label_bbox = font.getbbox(label)
        lw = label_bbox[2] - label_bbox[0] + LABEL_PADDING * 2
        lh = label_bbox[3] - label_bbox[1] + LABEL_PADDING * 2

        label_x = x
        label_y = max(0, y - lh - 1)

        draw.rectangle(
            [label_x, label_y, label_x + lw, label_y + lh],
            fill=color,
        )
        draw.text(
            (label_x + LABEL_PADDING, label_y + LABEL_PADDING),
            label,
            fill=TEXT_COLOR,
            font=font,
        )

        index_map[idx] = el

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), index_map
