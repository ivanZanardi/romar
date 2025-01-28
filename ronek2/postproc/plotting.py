import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from .. import const
from .. import utils

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


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
      ax.plot(x, yk, ls=_ls, c=_c, label=k)
    ax.legend(loc=legend_loc, fancybox=True, framealpha=0.9)
  else:
    ax.plot(x, y, "-", c="k")
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

# Moments
def plot_mom_evolution(
  path,
  t,
  n_m,
  molecule,
  molecule_label,
  tlim=None,
  ylim_err=None,
  err_scale="linear",
  hline=None,
  max_mom=2
):
  path = path + "/moments/"
  os.makedirs(path, exist_ok=True)
  # Compute moments
  # > Check if MT model is present
  moms_mt = {}
  keys = list(n_m.keys())
  for k in keys:
    if ("MT" in k):
      moms_mt[k] = n_m.pop(k)
  # > Order 0
  moms = [{k: molecule.compute_mom(nk, m=0) for (k, nk) in n_m.items()}]
  # > Order 1-max_mom
  for m in range(1,max_mom):
    moms.append(
      {k: molecule.compute_mom(nk, m=1)/moms[0][k] for (k, nk) in n_m.items()}
    )
  # Include MT model
  if (len(moms_mt) > 0):
    for k in moms_mt.keys():
      for m in range(2):
        moms[m][k] = moms_mt[k][m]
  # Plot moments
  for m in range(max_mom):
    if (m == 0):
      yscale = "log"
      label_sol = fr"$n_{molecule_label}$ [m$^{{-3}}$]"
      label_err = fr"$n_{molecule_label}$ error [\%]"
    else:
      yscale = "linear"
      if (m == 1):
        label_sol = fr"$e_{molecule_label}$ [eV]"
        label_err = fr"$e_{molecule_label}$ error [\%]"
      else:
        label_sol = fr"$\gamma_{m}$ [eV$^{m}$]"
        label_err = fr"$\gamma_{m}$ error [\%]"
    # > Moment
    plot_evolution(
      x=t,
      y=moms[m],
      xlim=tlim[f"m{m}"] if isinstance(tlim, dict) else tlim,
      labels=[r"$t$ [s]", label_sol],
      # legend_loc="lower left" if (m == 0) else "lower right",
      legend_loc="best",
      scales=["log", yscale],
      figname=path + f"/m{m}",
      save=True,
      show=False
    )
    # > Moment error
    for (k, momk) in moms[m].items():
      if (k.upper() == "FOM"):
        mom_fom = momk
        break
    moms_err = {}
    for (k, momk) in moms[m].items():
      if (k.upper() != "FOM"):
        moms_err[k] = utils.absolute_percentage_error(mom_fom, momk)
    plot_evolution(
      x=t,
      y=moms_err,
      xlim=tlim[f"m{m}"] if isinstance(tlim, dict) else tlim,
      ylim=ylim_err,
      hline=hline,
      labels=[r"$t$ [s]", label_err],
      # legend_loc="lower center",
      legend_loc="best",
      scales=["log", err_scale],
      figname=path + f"/m{m}_err",
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
  err,
  eval_err,
  molecule_label,
  tlim=None,
  ylim_err=None,
  subscript="i",
  err_scale="linear",
  hline=None,
  max_mom=2
):
  os.makedirs(path, exist_ok=True)
  rlist = sorted(list(err.keys()), key=int)
  for r in err.keys():
    t = err[r]["t"]
    break
  if (eval_err == "mom"):
    for m in range(max_mom):
      if (m == 0):
        label = fr"$n_{molecule_label}$ error [\%]"
      else:
        if (m == 1):
          label = fr"$e_{molecule_label}$ error [\%]"
        else:
          label = fr"$\gamma_{m}$ error [\%]"
      plot_evolution(
        x=t,
        y={f"$r={r}$": err[r]["mean"][m] for r in rlist},
        xlim=tlim,
        ylim=ylim_err,
        ls="-",
        hline=hline,
        # legend_loc="lower left",
        legend_loc="best",
        labels=[r"$t$ [s]", label],
        scales=["log", err_scale],
        figname=path + f"/mean_m{m}_err",
        save=True,
        show=False
      )
  else:
    plot_evolution(
      x=t,
      y={f"$r={r}$": err[r]["mean"] for r in rlist},
      xlim=tlim,
      ylim=ylim_err,
      ls="-",
      hline=hline,
      # legend_loc="lower left",
      legend_loc="best",
      labels=[r"$t$ [s]", fr"$w_{subscript}$ error [\%]"],
      scales=["log", err_scale],
      figname=path + "/mean_dist_err",
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
  subscript="i",
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
      fr"$\epsilon_{subscript}$ [eV]",
      fr"$n_{subscript}/g_{subscript}$ [m$^{{-3}}$]"
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
      elif ("MT" in k.upper()):
        continue
      else:
        c = COLORS[i]
        i += 1
      yk[yk<ymin*0.1] = 0.0
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
  teval,
  t,
  n_m,
  molecule,
  subscript="i",
  markersize=6,
  shift_diss_en=False
):
  path = path + "/dist/"
  os.makedirs(path, exist_ok=True)
  # Interpolate at "teval" points
  n_m_eval = {}
  for (k, nk) in n_m.items():
    if (nk.shape[-1] != molecule.nb_comp):
      nk = nk.T
    nk = sp.interpolate.interp1d(t, nk, kind="cubic", axis=0)(teval)
    n_m_eval[k] = nk / molecule.lev["g"]
  if shift_diss_en:
    x = (molecule.lev["e"] - molecule.e_d) / const.eV_to_J
  else:
    x = molecule.lev["e"] / const.eV_to_J
  for i in range(len(teval)):
    plot_dist_2d(
      x=x,
      y={k: nk[i] for (k, nk) in n_m_eval.items()},
      t=float(teval[i]),
      subscript=subscript,
      scales=["linear", "log"],
      markersize=markersize,
      figname=path + f"/t{i+1}",
      save=True,
      show=False
    )
