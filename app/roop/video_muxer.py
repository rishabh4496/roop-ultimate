"""Color metadata shared by Pre-Scan and the existing render/mux writers.

ffprobe calls the transfer tag color_transfer; FFmpeg's encoder option is
color_trc. Keep both names explicit on the API, without assigning HDR tags to
tonemapped SDR pixels. HDR decode/tonemapping remains in hdr_pipeline.
"""
from roop.source_color import ColorPlan, plan_for, read_tags


def extract_color_tags(stream):
    tags = {key: str(stream.get(key) or '') for key in
            ('color_space', 'color_transfer', 'color_primaries', 'color_range')}
    tags['color_transfer'] = tags['color_transfer'] or str(stream.get('color_trc') or '')
    return tags


def probe_color(path):
    from roop.capturer import _probe_video
    info = _probe_video(str(path))
    if info is None:
        raise ValueError('Cannot probe source color metadata')
    tags = read_tags(info)
    transfer = tags['color_transfer']
    return {**tags, 'color_trc': transfer,
            'hdr': transfer in ('smpte2084', 'arib-std-b67'),
            'bit_depth': info.get('bit_depth', 8),
            'color_badge': info.get('color_badge', ''),
            'width': info.get('w'), 'height': info.get('h')}


def output_plan(info, height=0, hdr_to_sdr=None):
    if hdr_to_sdr is None:
        from roop.hdr_pipeline import output_to_sdr
        hdr_to_sdr = output_to_sdr()
    if hdr_to_sdr:
        return ColorPlan('bt709', 'bt709', 'tv', 'bt709', 'bt709')
    return plan_for(info, height)
