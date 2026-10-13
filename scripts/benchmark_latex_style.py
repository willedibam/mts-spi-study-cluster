"""Consistent Computer Modern text and math for the benchmark's LaTeX figures."""
import re
import shutil
import matplotlib as mpl
from matplotlib.text import Text


def latex_style():
    if not (shutil.which('latex') and shutil.which('dvipng')):
        raise RuntimeError('Benchmark LaTeX figures require the existing latex and dvipng tools.')
    mpl.rcParams.update({'text.usetex':True,'font.family':'serif',
                         # TeX recognises the first name; native Matplotlib font lookup
                         # needs an installed fallback even when usetex is enabled.
                         'font.serif':['Computer Modern Roman','DejaVu Serif'],'mathtext.fontset':'cm'})


def latex_text(text):
    """Escape prose only; preserve existing math and escaped specials. Idempotent."""
    chunks=re.split(r'(?<!\\)\$',text)
    symbols={'→':r'$\rightarrow$','×':r'$\times$','±':r'$\pm$',
             'ρ':r'$\rho$','η':r'$\eta$','ε':r'$\varepsilon$','−':r'$-$'}
    for j in range(0,len(chunks),2):
        chunks[j]=re.sub(r'(?<!\\)([_%&#])',r'\\\1',chunks[j])
        for old,new in symbols.items():chunks[j]=chunks[j].replace(old,new)
    return '$'.join(chunks)


def prepare_latex_figure(fig):
    for text in fig.findobj(match=Text):
        text.set_text(latex_text(text.get_text()))
        text.set_usetex(True)
        text.set_fontfamily('serif')
