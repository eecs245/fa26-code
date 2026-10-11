"""Visualization helpers for EECS 245 Homework 5, Fall 2026.

Adapted from the matrix visualizations in Math 124 Lab 6.
"""
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("retina")
__all__ = ["show_projection", "projection_explorer", "rotation_explorer"]
_vectors = np.array([[2., 1.], [-1., 2.]])
_colors = ['#e69f00', '#3d81f6']


def _matrix_explorer(kind):
    """Display a native Jupyter widget for projections or rotations."""
    rotation = kind == 'rotation'
    symbol = 'R' if rotation else 'P'
    slider = widgets.IntSlider(
        value=60 if rotation else 45, min=-180 if rotation else 0,
        max=180, step=1, description='Angle θ (°):',
        continuous_update=False, style={'description_width': 'initial'},
        layout=widgets.Layout(width='90%'))
    formula = widgets.HTML()
    output = widgets.Output()

    def draw(change=None):
        angle = slider.value
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        M = np.array([[c, -s], [s, c]]) if rotation else np.array([[c*c, c*s], [c*s, s*s]])
        d = str(angle) + '°'
        co, si = 'cos(' + d + ')', 'sin(' + d + ')'
        entries = [co, '−' + si, si, co] if rotation else ['cos²(' + d + ')', co + si, co + si, 'sin²(' + d + ')']
        formula.value = '<div style="font:18px Georgia,serif;text-align:center;padding:10px">' + symbol + ' = <span style="display:inline-grid;grid-template-columns:auto auto;gap:8px 22px;padding:5px 12px;border-left:2px solid;border-right:2px solid;vertical-align:middle">' + ''.join('<span>' + e + '</span>' for e in entries) + '</span></div>'
        transformed = np.array([M @ vector for vector in _vectors])
        with output:
            output.clear_output(wait=True)
            with plt.rc_context({'font.family': 'serif', 'font.size': 13, 'mathtext.fontset': 'cm'}):
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=140)
                for panel, (ax, vectors) in enumerate(zip(axes, [_vectors, transformed])):
                    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), xlabel='First coordinate', ylabel='Second coordinate', title=['Before', 'After'][panel])
                    ax.set_aspect('equal')
                    ax.grid(color='#e7e7e7', linewidth=.7)
                    ax.axhline(0, color='#999', linewidth=1)
                    ax.axvline(0, color='#999', linewidth=1)
                    for spine in ax.spines.values(): spine.set_visible(False)
                    if not rotation:
                        direction = np.array([c, s])
                        extent = 3.1 / np.max(np.abs(direction))
                        ends = np.outer([-extent, extent], direction)
                        ax.plot(ends[:, 0], ends[:, 1], color='#888', lw=1.2)
                        ax.annotate(r'$\ell$', xy=.85*extent*direction, xytext=(8, 8), textcoords='offset points', color='#666')
                    for i, vector in enumerate(vectors):
                        name = [r'\vec x', r'\vec y'][i]
                        if panel: name = symbol + name
                        ax.annotate('', xy=vector, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=_colors[i], lw=2.4))
                        ax.annotate('$' + name + '$', xy=vector, xytext=(7, 8) if i == 0 else (-24, -22), textcoords='offset points', color=_colors[i], fontsize=15)
                fig.tight_layout(pad=1.5)
                display(fig)
                plt.close(fig)
    slider.observe(draw, names='value')
    view = widgets.VBox([slider, formula, output])
    display(view)
    draw()
    return view

def rotation_explorer():
    """Rotate the fixed vectors with a slider for theta."""
    return _matrix_explorer('rotation')

def projection_explorer():
    """Project the fixed vectors onto a line controlled by theta."""
    return _matrix_explorer('projection')

def show_projection(P, u, v):
    """Draw the input, its image, and the proposed projection line."""
    if P is Ellipsis:
        print("Replace ... with your matrix from Problem 5(a), then run this cell again.")
        return
    P, u, v = np.asarray(P, dtype=float), np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if P.shape != (2, 2):
        raise ValueError("P must have shape (2, 2).")
    p = P @ u
    limit = max(3., 1.25 * np.max(np.abs(np.r_[u, p])))
    direction = v / np.linalg.norm(v)
    extent = limit / np.max(np.abs(direction))
    with plt.rc_context({'font.family': 'serif', 'font.size': 13, 'mathtext.fontset': 'cm'}):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=160)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel='First coordinate', ylabel='Second coordinate')
        ax.set_aspect('equal')
        ax.grid(color='#e7e7e7')
        ax.axhline(0, color='#aaa', lw=1)
        ax.axvline(0, color='#aaa', lw=1)
        ends = np.outer([-extent, extent], direction)
        ax.plot(ends[:, 0], ends[:, 1], color='#777', label=r'$\ell=\mathrm{span}(\vec v)$')
        for vector, color, name in [(u, _colors[0], r'$\vec u$'), (p, _colors[1], r'$P\vec u$')]:
            ax.annotate('', xy=vector, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.annotate(name, xy=vector, xytext=(8, 8), textcoords='offset points', color=color, fontsize=16)
        ax.plot([u[0], p[0]], [u[1], p[1]], '--', color='#888')
        ax.legend(loc='upper left')
        fig.tight_layout()
        plt.show()
