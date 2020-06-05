import sys
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

def set_matplotlib_numpy_prefs(linewidth=2,xticksize=16,yticksize=18,fontsize=18,titlesize=16,labelsize=20):
    # use pdf backend for command line plotting
    if sys.stdout.isatty():
        mpl.use('pdf')
        print('setting pdf renderer for command line usage')
    # set preferences before importing matplotlib
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams.update({'font.size': fontsize, 'font.weight':'normal'}) 
    mpl.rcParams.update({'axes.titlesize':titlesize, 'font.weight':'normal'})
    mpl.rcParams.update({'axes.labelsize':labelsize, 'font.weight':'normal'})
    mpl.rcParams['lines.linewidth'] = linewidth
    mpl.rcParams['xtick.labelsize'] = xticksize
    mpl.rcParams['ytick.labelsize'] = yticksize
    mpl.rc('image', cmap='jet')

    np.set_printoptions(linewidth=120, precision=4)
    return 

"""
removeaxes: 
This function is used to add lables and titles to a figure
while also removing the x and y tickmarks
To make tight subplots use : plt.tight_layout()
"""
def removeaxes(xlabel=None, ylabel=None, title=None):
    import matplotlib.pylab as plt
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)

    plt.xticks([])
    plt.yticks([])
    plt.box(on=None)
    return


def add_colorbar(location="bottom", orientation="horizontal", labelsize=6):
    import matplotlib.pylab as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes(location, size="5%", pad=0.08)
    h = plt.colorbar(cax=cax, orientation=orientation)
    h.ax.tick_params(labelsize=labelsize)
    return

