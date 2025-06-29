import numpy as np


# == margin ==
def calculate_margin_rect(s, x_y_w_h, negativeInside=True):
  """Calculates the margin to a rectangular box in the x-y state space."""
  x, y, w, h = x_y_w_h
  delta_x = np.abs(s[0] - x)
  delta_y = np.abs(s[1] - y)
  margin = max(delta_y - h/2, delta_x - w/2)

  if negativeInside:
    return margin
  else:
    return -margin


def calculate_margin_circle(s, c_r, negativeInside=True):
  """Calculates the margin to a circle in the x-y state space."""
  center, radius = c_r
  dist_to_center = np.linalg.norm(s[:2] - center)
  margin = dist_to_center - radius

  if negativeInside:
    return margin
  else:
    return -margin


# == Plotting ==
def plot_arc(
    center, r, thetaParam, ax, c='b', lw=1.5, orientation=0., zorder=0
):
  """Plots an arc given a center, a radius and the (theta_init, theta_final)."""
  x, y = center
  thetaInit, thetaFinal = thetaParam

  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(thetaInit + orientation, thetaFinal + orientation, 100)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)

  ax.plot(xs, ys, c=c, lw=lw, zorder=zorder)


def plot_circle(
    center, r, ax, c='b', lw=1.5, ls='-', orientation=0, scatter=False,
    zorder=0
):
  """Plots a circle given a center and a radius."""
  x, y = center
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)

  theta = np.linspace(0, 2 * np.pi, 200)
  xs = xtilde + r * np.cos(theta)
  ys = ytilde + r * np.sin(theta)
  ax.plot(xs, ys, c=c, lw=lw, linestyle=ls, zorder=zorder)
  if scatter:
    ax.scatter(xtilde + r, ytilde, c=c, s=80)
    ax.scatter(xtilde - r, ytilde, c=c, s=80)
    print(xtilde + r, ytilde, xtilde - r, ytilde)


def rotatePoint(state, orientation):
  """Rotates the point counter-clockwise by a given angle."""
  x, y, theta = state
  xtilde = x * np.cos(orientation) - y * np.sin(orientation)
  ytilde = y * np.cos(orientation) + x * np.sin(orientation)
  thetatilde = theta + orientation

  return np.array([xtilde, ytilde, thetatilde])
