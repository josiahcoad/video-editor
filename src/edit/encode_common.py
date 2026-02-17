"""
Shared encoding options for edit-pipeline outputs.

Use these so exports look correct on Twitter, Instagram, and other platforms
that re-encode or rely on color metadata. Without correct tags, video can look
washed out or have a green/red cast.
"""

# BT.709 primaries + transfer + matrix, TV (limited) range. Most platforms
# expect this for HD SDR; missing or wrong metadata causes color/gamma issues.
H264_SOCIAL_COLOR_ARGS: list[str] = [
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-colorspace",
    "bt709",
    "-color_range",
    "tv",
]
