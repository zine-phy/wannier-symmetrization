"""Portable plotting defaults; no private fonts or machine paths are required."""
import os

def configure_plot_style():
    from matplotlib import font_manager, rcParams
    requested = os.environ.get('WANNSYM_FONT')
    if requested:
        path = font_manager.findfont(requested, fallback_to_default=False)
    else:
        try:
            path = font_manager.findfont('Times New Roman', fallback_to_default=False)
        except ValueError:
            path = font_manager.findfont('DejaVu Serif', fallback_to_default=False)
    family = font_manager.FontProperties(fname=path).get_name()
    rcParams.update({'font.family': family, 'font.size': 30,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    return path
