import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from .. import const

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
LINESTYLES = ('-', '--', '-.', '-', ':')


# Plotting
# =====================================
# Cumulative energy
def plot_cum_energy(
  y,
  figname=None,
  save=False,
  show=True
):
  # Initialize figure
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  x = np.arange(1,len(y)+1)
  ax.set_xlabel("$i$-th basis")
  # y axis
  ax.set_ylabel("Cumulative energy")
  # Plotting
  ax.plot(x, y, marker="o")
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

# Time evolution
def plot_evolution(
  x,
  y,
  ls=None,
  xlim=None,
  ylim=None,
  hline=None,
  labels=[r"$t$ [s]", r"$n$ [m$^{-3}$]"],
  scales=["log", "linear"],
  legend_loc="best",
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  if (xlim is None):
    xlim = (np.amin(x), np.amax(x))
  ax.set_xlim(xlim)
  xmin, xmax = xlim
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  if (ylim is not None):
    ax.set_ylim(ylim)
  # Plotting
  if isinstance(y, dict):
    i = 0
    for (k, yk) in y.items():
      if (k.upper() == "FOM"):
        _c = "k"
        _ls = "-" if (ls is None) else ls
      else:
        _c = COLORS[i]
        _ls = "--" if (ls is None) else ls
        i += 1
      if isinstance(yk, dict):
        ax.plot([], [], c=_c, label=k)
        for (j, yl) in enumerate(yk.values()):
          ax.plot(x, yl, ls=LINESTYLES[j], c=_c)
      else:
        ax.plot(x, yk, ls=_ls, c=_c, label=k)
    ax.legend(loc=legend_loc, fancybox=True, framealpha=0.9)
  else:
    ax.plot(x, y, c="k")
  if (hline is not None):
    ax.text(
      0.99, hline,
      r"{:0.1f} \%".format(hline),
      va="bottom", ha="right",
      transform=ax.get_yaxis_transform(),
      fontsize=20
    )
    ax.hlines(hline, xmin, xmax, colors="grey", lw=1.0)
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

def plot_temp_evolution(
  path,
  t,
  y,
  err,
  tlim=None,
  ylim_err=None,
  err_scale="linear",
  hline=None
):
  path = path + "/temp/"
  os.makedirs(path, exist_ok=True)
  # Temperatures
  plot_evolution(
    x=t,
    y={k: yk["temp"] for (k, yk) in y.items()},
    xlim=tlim,
    labels=[r"$t$ [s]", "$T$ [K]"],
    legend_loc="center left",
    scales=["log", "linear"],
    figname=path + "/sol",
    save=True,
    show=False
  )
  # Temperatures error
  plot_evolution(
    x=t,
    y={k: ek["temp"] for (k, ek) in err.items()},
    xlim=tlim,
    ylim=ylim_err,
    hline=hline["temp"],
    labels=[r"$t$ [s]", "$T$ error [\%]"],
    legend_loc="best",
    scales=["log", err_scale],
    figname=path + "/err",
    save=True,
    show=False
  )

# Moments
def plot_mom_evolution(
  path,
  t,
  y,
  err,
  species,
  labels,
  tlim=None,
  ylim_err=None,
  err_scale="linear",
  hline=None,
  max_mom=2
):
  path = path + "/moments/"
  os.makedirs(path, exist_ok=True)
  # Plot moments
  for m in range(max_mom):
    for s in species.keys():
      if (m == 0):
        yscale = "log"
        label_sol = f"$n_{labels[s]}$ [m$^{{-3}}$]"
        label_err = f"$n_{labels[s]}$ error [\%]"
      else:
        yscale = "linear"
        if (m == 1):
          label_sol = f"$e_{labels[s]}$ [eV]"
          label_err = f"$e_{labels[s]}$ error [\%]"
        else:
          label_sol = fr"$\gamma_{m}$ [eV$^{m}$]"
          label_err = fr"$\gamma_{m}$ error [\%]"
      # > Moment
      plot_evolution(
        x=t,
        y={k: yk["mom"][s][f"m{m}"] for (k, yk) in y.items()},
        xlim=tlim[f"m{m}"] if isinstance(tlim, dict) else tlim,
        labels=[r"$t$ [s]", label_sol],
        legend_loc="best",
        scales=["log", yscale],
        figname=path + f"/m{m}_{s}",
        save=True,
        show=False
      )
      # > Moment error
      plot_evolution(
        x=t,
        y={k: ek["mom"][s][f"m{m}"] for (k, ek) in err.items()},
        xlim=tlim[f"m{m}"] if isinstance(tlim, dict) else tlim,
        ylim=ylim_err,
        hline=hline["mom"],
        labels=[r"$t$ [s]", label_err],
        legend_loc="best",
        scales=["log", err_scale],
        figname=path + f"/m{m}_{s}_err",
        save=True,
        show=False
      )

def plot_err_ci_evolution(
  x,
  mean,
  sem,
  size,
  alpha=0.95,
  xlim=None,
  ylim=None,
  hline=None,
  labels=[r"$t$ [s]", r"$n$ [m$^{-3}$]"],
  scales=["log", "linear"],
  legend_loc="best",
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  if (xlim is None):
    xlim = (np.amin(x), np.amax(x))
  ax.set_xlim(xlim)
  xmin, xmax = xlim
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  if (ylim is not None):
    ax.set_ylim(ylim)
  # Plotting
  y1, y2 = sp.stats.t.interval(
    alpha=alpha,
    df=size-1,
    loc=mean,
    scale=sem
  )
  # y1, y2 = [np.clip(z, 0, None) for z in (y1, y2)]
  ci_lbl = "${}\\%$ CI".format(int(100*alpha))
  ax.fill_between(x=x, y1=y1, y2=y2, alpha=0.2, label=ci_lbl)
  ax.plot(x, mean)
  if (hline is not None):
    ax.text(
      0.99, hline,
      r"{:0.1f} \%".format(hline),
      va="bottom", ha="right",
      transform=ax.get_yaxis_transform(),
      fontsize=20
    )
    ax.hlines(hline, xmin, xmax, colors="grey", lw=1.0)
  ax.legend(loc=legend_loc)
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

def plot_err_evolution(
  path,
  t,
  error,
  species,
  labels,
  tlim=None,
  ylim_err=None,
  err_scale="linear",
  hline=None,
  max_mom=2
):
  os.makedirs(path, exist_ok=True)
  rlist = sorted(list(error.keys()), key=int)
  # Temperatures
  for k in ("Th", "Te"):
    plot_evolution(
      x=t,
      y={f"$r={r}$": error[r]["temp"][k]["mean"] for r in rlist},
      xlim=tlim,
      # ylim=ylim_err["temp"] if (ylim_err is not None) else None,
      ls="-",
      hline=hline["temp"],
      # legend_loc="lower left",
      legend_loc="best",
      labels=[r"$t$ [s]", fr"$T_{k[1]}$ error [\%]"],
      scales=["log", err_scale],
      figname=path + f"/err_temp_{k}",
      save=True,
      show=False
    )
  # Moments
  for s in species.keys():
    for m in range(max_mom):
      if (m == 0):
        label = fr"$n_{labels[s]}$ error [\%]"
      else:
        if (m == 1):
          label = fr"$e_{labels[s]}$ error [\%]"
        else:
          label = fr"$\gamma_{m}$ error [\%]"
      plot_evolution(
        x=t,
        y={f"$r={r}$": error[r]["mom"][s][f"m{m}"]["mean"] for r in rlist},
        xlim=tlim,
        # ylim=ylim_err["temp"] if (ylim_err is not None) else None,
        ls="-",
        hline=hline["mom"],
        # legend_loc="lower left",
        legend_loc="best",
        labels=[r"$t$ [s]", label],
        scales=["log", err_scale],
        figname=path + f"/err_mom_{s}_m{m}",
        save=True,
        show=False
      )
  # Distribution
  plot_evolution(
    x=t,
    y={f"$r={r}$": error[r]["dist"]["mean"] for r in rlist},
    xlim=tlim,
    # ylim=ylim_err,
    ls="-",
    hline=hline["dist"],
    # legend_loc="lower left",
    legend_loc="best",
    labels=[r"$t$ [s]", fr"$w_i$ error [\%]"],
    scales=["log", err_scale],
    figname=path + "/err_dist",
    save=True,
    show=False
  )

# 2D distribution
def as_si(x, ndp=0):
  s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
  m, e = s.split("e")
  return r"{m:s}\times 10^{{{e:d}}}".format(m=m, e=int(e))

def plot_dist_2d(
  x,
  y,
  t=None,
  scales=["linear", "log"],
  labels=None,
  markersize=6,
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  if (labels is None):
    labels = [
      "$\epsilon_i$ [eV]",
      "$n_i/g_i$ [m$^{-3}$]"
    ]
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  style = dict(
    linestyle="",
    marker="o",
    rasterized=True
  )
  if isinstance(y, dict):
    i = 0
    lines = []
    for (k, yk) in y.items():
      if (k.upper() == "FOM"):
        c = "k"
        ymin = yk.min()
      else:
        c = COLORS[i]
        i += 1
      yk[yk<ymin*1e-1] = 0.0
      ax.plot(x, yk, c=c, markersize=markersize, **style)
      lines.append(plt.plot([], [], c=c, **style)[0])
    ax.legend(lines, list(y.keys()), fancybox=True, framealpha=0.9)
  else:
    ax.plot(x, y, c="k", markersize=markersize, **style)
  if (t is not None):
    ax.text(
      0.05, 0.05,
      r"$t = {0:s}$ s".format(as_si(t)),
      transform=ax.transAxes,
      fontsize=25
    )
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname, dpi=300)
  if show:
    plt.show()
  plt.close()

def plot_multi_dist_2d(
  path,
  t,
  y,
  teval,
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
      # Interpolate at "teval" points
      n_eval = {}
      for (k, nk) in n.items():
        if (nk.shape[0] != len(t)):
          if (s == "Ar"):
            nk = nk[1:]
          nk = nk.T
        n_eval[k] = sp.interpolate.interp1d(t, nk, kind="cubic", axis=0)(teval)
      x = species[s].lev["E"]/const.eV_to_J
      if (s == "Ar"):
        x = x[1:]
      for i in range(len(teval)):
        plot_dist_2d(
          x=x,
          y={k: nk[i] for (k, nk) in n_eval.items()},
          t=float(teval[i]),
          scales=["linear", "log"],
          markersize=markersize,
          figname=spath + f"/t{i+1}",
          save=True,
          show=False
        )
