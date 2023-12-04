from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from holoviews import opts


### CONSTANTS

n_per_ftr_typical = 5

### SOURCE DATA

def save_source_data(di, subfolder, base_folder='fig/source_data/'):

    folder = Path(base_folder) / subfolder
    folder.mkdir(exist_ok=True)

    for label, df in di.items():
        # print(label)
        df.to_csv(folder / (label + '.csv'))

### PLOTTING

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']

cmap_n_per_ftr = 'cool'
cmap_r = 'winter_r'

clr_cca = 'peru'
clr_pls = 'lightseagreen'

clr_inSample = 'steelblue'
clr_5cv = 'tomato'
clr_20x5cv = 'gold'
clr_5cvMean = 'maroon'
clr_20x5cvMean = 'darkgoldenrod'

default_fontsizes = dict(title=8, labels=8, ticks=7, minor_ticks=7, legend=7)


fig_opts = [
    opts.Layout(aspect_weight=1, fig_inches=(3.42, None), sublabel_format="{alpha}", sublabel_size=10, fontsize=8), 
    opts.Overlay(fontsize=default_fontsizes,),
    opts.Area(fontsize=default_fontsizes),
    opts.Arrow(textsize=default_fontsizes),
    opts.Curve(fontsize=default_fontsizes),
    opts.HexTiles(fontsize=default_fontsizes),
    opts.Histogram(fontsize=default_fontsizes),
    opts.Raster(fontsize=default_fontsizes),
    opts.Scatter(fontsize=default_fontsizes),
    opts.Text(fontsize=default_fontsizes),
    opts.QuadMesh(fontsize=default_fontsizes),
    opts.Violin(fontsize=default_fontsizes),
    opts.VLine(fontsize=default_fontsizes),
]

# hv hooks

class Suptitle:
    def __init__(self, suptitle, color, y=1.175, fontsize=10, fontweight='bold'):
        self.suptitle = suptitle
        self.color = color
        self.y = y
        self.fontsize = fontsize
        self.fontweight = fontweight
    def __call__(self, plot, element):
        ax = plot.handles['axis']
        ax.text(.5, self.y, self.suptitle, ha='center', va='bottom', color=self.color,
                fontdict=dict(size=self.fontsize, weight=self.fontweight), transform=ax.transAxes)
    
def suptitle_cca(plot, element):
    Suptitle('CCA', clr_cca)(plot, element)
    
    
def suptitle_pls(plot, element):
    Suptitle('PLS', clr_pls)(plot, element)
    
    
def legend_frame_off(plot, element):
    try:
        legend = plot.handles['legend']
    except KeyError:
        pass
    else:
        legend.set_frame_on(False)
    

class Format_log_axis:
    def __init__(self, dim, major_subs=None, label_minor=False,
                 major_numticks=None, minor_numticks=None):
        if dim not in ['x', 'y']:
            raise ValueError('Invalid dim')
        self.dim = dim
        
        if major_subs is None:
            self.major_subs = (1,)
        else:
            self.major_subs = major_subs

        self.label_minor = label_minor
        self.major_numticks = major_numticks
        self.minor_numticks = minor_numticks
        
    def __call__(self, plot, element):
        
        if self.dim == 'x':
            ax = plot.handles['axis'].xaxis
        elif self.dim == 'y':
            ax = plot.handles['axis'].yaxis

        if not self.label_minor:
            minor_formatter = matplotlib.ticker.NullFormatter()
        else:
            minor_formatter = matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(1.3, .01))
            plot.handles['axis'].tick_params(axis=self.dim, which='minor', labelsize=7)
        ax.set_minor_formatter(minor_formatter)
        ax.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.setdiff1d(np.arange(1, 10), self.major_subs), numticks=self.minor_numticks))
        ax.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.set_major_locator(matplotlib.ticker.LogLocator(subs=self.major_subs, numticks=self.major_numticks))


class Ax_ticks:
    def __init__(self, dim, locs, labels, minor=False):
        if dim not in ['x', 'y']:
            raise ValueError('Invalid dim')
        self.dim = dim
        self.locs = locs
        self.labels = labels
        self.minor = minor

    def __call__(self, plot, element):

        if self.dim == 'x':
            ax = plot.handles['axis'].xaxis
        elif self.dim == 'y':
            ax = plot.handles['axis'].yaxis

        ax.set_ticks(self.locs, minor=self.minor)
        ax.set_ticklabels(self.labels, minor=self.minor)
        ax.set_tick_params(which='minor' if self.minor else 'major',
                           labelsize=7)
