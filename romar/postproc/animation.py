import os
import matplotlib
import matplotlib.pyplot as plt

from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from .. import const

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


# Animation
# =====================================
# Initialize lines for levels distribution
def _init_lines(
  labels,
  ax,
  markersize
):
  # Set up axes
  ax.set_xlabel(r"$\epsilon_i$ [eV]")
  ax.set_ylabel(r"$n_i\//\/g_i$ [m$^{-3}$]")
  # Initialize lines
  style = dict(
    linestyle="",
    marker="o",
    fillstyle="full"
  )
  # Colors
  i = 0
  colors = []
  for l in labels:
    if (("FOM" in l.upper()) or ("STS" in l.upper())):
      colors.append("k")
    else:
      colors.append(COLORS[i])
      i += 1
  # Lines
  lines = []
  for c in colors:
    lines.append(ax.semilogy([], [], c=c, markersize=markersize, **style)[0])
  # Legend
  ax.legend(
    [ax.semilogy([], [], c=c, markersize=6, **style)[0] for c in colors],
    labels=labels,
    fancybox=True,
    framealpha=0.9
  )
  return ax, lines

# Create animation
def _create_animation(
  t,
  x,
  y,
  frames,
  markersize
):
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Initialize levels distribution lines objects
  ax, lines = _init_lines(list(y.keys()), ax, markersize)
  # Initialize text in ax
  txt = ax.text(0.05, 0.05, "", transform=ax.transAxes, fontsize=25)
  # Tight layout
  plt.tight_layout()

  def _animate(frame):
    i = int(frame/frames*len(t))
    # Write time instant
    txt.set_text(r"$t$ = %.1e s" % t[i])
    # Loop over models
    for (j, yj) in enumerate(y.values()):
      lines[j].set_data(x, yj[i])
    # Rescale axis limits
    ax.relim()
    ax.autoscale_view(tight=True)
    return lines

  # Get animation
  return FuncAnimation(
    fig,
    _animate,
    frames=frames,
    blit=True
  )

def animate(
  t,
  x,
  y,
  markersize=6,
  frames=10,
  fps=10,
  filename="./lev_dist.gif",
  dpi=600,
  save=True,
  show=False
):
  # Create animation
  anim = _create_animation(t, x, y, frames, markersize)
  # Save animation
  if save:
    anim.save(filename, fps=fps, dpi=dpi)
  # Display animation
  if show:
    HTML(anim.to_jshtml())

def animate_dist(
  path,
  t,
  y,
  species,
  markersize=6
):
  for s in species.keys():
    if (species[s].nb_comp > 1):
      # Path to saving
      spath = path + f"/dist/{s}/"
      os.makedirs(spath, exist_ok=True)
      # Number densities
      n = {k: yk["dist"][s] for (k, yk) in y.items()}
      for (k, nk) in n.items():
        if (nk.shape[0] != len(t)):
          if (s == "Ar"):
            nk = nk[1:]
          n[k] = nk.T
      x = species[s].lev["E"]/const.eV_to_J
      if (s == "Ar"):
        x = x[1:]
      animate(
        t=t,
        x=x,
        y=n,
        markersize=markersize,
        frames=100,
        fps=10,
        filename=spath + "/video.mp4",
        dpi=600,
        save=True,
        show=False
      )
