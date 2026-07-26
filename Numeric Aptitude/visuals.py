"""
visuals.py 

Renders triangle and grid puzzles as actual SVG graphics (matching the
classic "number in triangle" exam format) instead of showing raw arrays.
Returns HTML/SVG strings meant to be passed to st.markdown(..., unsafe_allow_html=True).
"""

def draw_triangle_svg(left, right, bottom, center, highlight_center=False):
    """
    left, right, bottom: the three outer numbers
    center: the center value, or "?" for the unknown
    """
    center_color = "#e63946" if highlight_center else "#111111"
    center_display = "?" if center is None else str(center)

    svg = f"""
    <svg width="320" height="260" xmlns="http://www.w3.org/2000/svg">
        <polygon points="160,30 60,200 260,200"
                 fill="white" stroke="black" stroke-width="4" />
        <text x="40" y="120" font-size="26" font-weight="bold" fill="white">{left}</text>
        <text x="250" y="120" font-size="26" font-weight="bold" fill="white">{right}</text>
        <text x="145" y="240" font-size="26" font-weight="bold" fill="white">{bottom}</text>
        <text x="145" y="150" font-size="28" font-weight="bold" fill="{center_color}">{center_display}</text>
    </svg>
    """
    return svg


def draw_grid_svg(a, b, c, d, highlight_d=False):
    """
    2x2 grid:  a  b
               c  d
    d can be None to show "?"
    """
    d_color = "#171010" if highlight_d else "#111111"
    d_display = "?" if d is None else str(d)

    svg = f"""
    <svg width="180" height="180" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="80" height="80" fill="white" stroke="black" stroke-width="3"/>
        <rect x="90" y="10" width="80" height="80" fill="white" stroke="black" stroke-width="3"/>
        <rect x="10" y="90" width="80" height="80" fill="white" stroke="black" stroke-width="3"/>
        <rect x="90" y="90" width="80" height="80" fill="white" stroke="red" stroke-width="3"/>
        <text x="45" y="60" font-size="24" font-weight="bold" fill="black" text-anchor="middle">{a}</text>
        <text x="125" y="60" font-size="24" font-weight="bold" fill="black" text-anchor="middle">{b}</text>
        <text x="45" y="140" font-size="24" font-weight="bold" fill="black" text-anchor="middle">{c}</text>
        <text x="125" y="140" font-size="24" font-weight="bold" fill="{d_color}" text-anchor="middle">{d_display}</text>
    </svg>
    """
    return svg


def draw_series_svg(numbers, next_val=None, highlight_next=False):
    """
    Renders a horizontal row of boxes for a number series, with the
    final box showing '?' if next_val is None.
    """
    boxes = list(numbers) + [next_val if next_val is not None else "?"]
    box_w, gap, start_x = 60, 15, 10
    total_w = len(boxes) * (box_w + gap) + gap
    next_color = "#e63946" if highlight_next else "#111111"

    rects = ""
    for i, val in enumerate(boxes):
        x = start_x + i * (box_w + gap)
        is_last = (i == len(boxes) - 1)
        fill = "#fff7e6" if is_last else "white"
        text_color = next_color if is_last else "black"
        rects += f'<rect x="{x}" y="10" width="{box_w}" height="60" fill="{fill}" stroke="black" stroke-width="2"/>'
        rects += f'<text x="{x + box_w/2}" y="48" font-size="20" font-weight="bold" fill="{text_color}" text-anchor="middle">{val}</text>'

    svg = f"""
    <svg width="{total_w}" height="80" xmlns="http://www.w3.org/2000/svg">
        {rects}
    </svg>
    """
    return svg
