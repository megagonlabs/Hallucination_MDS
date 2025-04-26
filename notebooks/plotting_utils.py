import os


FULL_WIDTH = 6.75133
COL_WIDTH  = 3.25063


def adjust(fig, left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0):
    fig.subplots_adjust(
        left   = left,  # the left side of the subplots of the figure
        right  = right,  # the right side of the subplots of the figure
        bottom = bottom,  # the bottom of the subplots of the figure
        top    = top,  # the top of the subplots of the figure
        wspace = wspace,  # the amount of width reserved for blank space between subplots
        hspace = hspace,  # the amount of height reserved for white space between subplots
    )

def remove_axes(ax):
    ax.spines[['right', 'top']].set_visible(False)

def save_fig(fig, name, dpi=200, base_dir="../paper-oct/images", **kwargs):
    import os
    path = f"{base_dir}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Image saved in {path}")
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)