"""Publication plots for both articles, using existing exported results only.
Run from any directory: python make_article_figures.py
Dependencies: pandas, matplotlib, numpy. No model files required.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'results' / 'tables'
FIGURES = ROOT / 'results' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
PERIODS = [f'{year}-е' for year in range(1800, 1920, 10)]
X = np.arange(len(PERIODS))
LABELS = {'liberty': 'Деспотизм — свобода', 'civilisation': 'Варварство — цивилизация'}
# Colors from the project's colorblind-friendly palette (lwm/config.py).
BLUE, RED, GREEN, ORANGE, PURPLE = '#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.labelsize': 8.5, 'axes.titlesize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .6, 'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'svg.fonttype': 'none',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
WIDTH = 16 / 2.54

def decimal(y, _):
    if abs(y) < 1e-10:
        return '0'
    return f'{y:.2f}'.replace('-', '−').replace('.', ',')

def format_axis(ax, ylim, ticks):
    ax.set_xlim(-.48, 11.48)
    ax.set_ylim(*ylim)
    ax.set_xticks(X, PERIODS)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(decimal))
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#dedede', lw=.55)
    ax.tick_params(length=2.5, width=.6, pad=3)
    ax.spines[['left', 'bottom']].set_color('#777777')

def save(fig, stem):
    for suffix in ('png', 'svg'):
        fig.savefig(FIGURES / f'{stem}.{suffix}', dpi=600)
    plt.close(fig)

POLITICAL = ['imperialism', 'imperialist', 'imperialists', 'jingoism']
GENERAL = ['empire', 'imperial', 'emperor', 'empress']
STYLES = [(BLUE, 'o', '-'), (RED, 's', '--'), (GREEN, '^', '-.'), (ORANGE, 'D', ':')]


def make_frequency(freq, counts):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.55))
    fig.subplots_adjust(left=.115, right=.985, bottom=.15, top=.77)
    format_axis(ax, (0, 4.5), np.arange(0, 4.5, 1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    for w, (color, marker, style) in zip(POLITICAL, STYLES):
        y = freq[w].to_numpy(dtype=float)
        n = counts[w].to_numpy(dtype=float)
        # NaN breaks lines; never connect across absent vocabulary entries.
        ax.plot(X, y, color=color, ls=style, lw=1.4, label=w, zorder=2)
        for mask, face in [(n >= 50, color), ((n < 50) & np.isfinite(n), 'white')]:
            ax.plot(X[mask], y[mask], ls='none', marker=marker, markersize=4.4,
                    markerfacecolor=face, markeredgecolor=color, markeredgewidth=.9, zorder=4)
    handles = [Line2D([], [], color=c, marker=m, ls=s, lw=1.4, ms=4.4, label=w)
               for w, (c,m,s) in zip(POLITICAL, STYLES)]
    handles.append(Line2D([], [], color='#555555', marker='o', ls='none', ms=4.4,
                          markerfacecolor='white', label='Менее 50 употреблений'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55,.98),
               ncol=3, frameon=False, handlelength=2.3, columnspacing=1.2)
    ax.set_ylabel('На миллион токенов')
    ax.set_xlabel('Десятилетие')
    save(fig, 'fig1_imperial_frequencies_article')


def make_balance(bal):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.985, bottom=.095, top=.86, hspace=.55)
    for ax, words, title in zip(axes, (POLITICAL, GENERAL),
            ('а) Политическая лексика империализма', 'б) Более ранняя имперская лексика')):
        format_axis(ax, (-.34, .10), np.arange(-.3, .101, .1))
        ax.axhline(0, color='#888888', lw=.75, zorder=1)
        ax.plot(X, bal['базовая_линия'], color='#444444', ls=(0, (1, 2)),
                lw=1.2, label='Базовая линия', zorder=2)
        for w, (c,m,s) in zip(words, STYLES):
            ax.plot(X, bal[w], color=c, marker=m, ls=s, lw=1.35, ms=4.1,
                    label=w, zorder=3)
        # The panel title and legend occupy separate bands above the data.
        ax.set_title(title, loc='left', pad=38, fontweight='medium')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
                  loc='lower left', bbox_to_anchor=(0, 1.015), ncol=3,
                  frameon=False, borderaxespad=0, handlelength=2.3, columnspacing=1.2)
        ax.set_ylabel('Баланс')
    axes[-1].set_xlabel('Десятилетие')
    save(fig, 'fig2_imperial_balance_article')


def make_imperial():
    data = pd.read_csv(TABLES / '01_word_frequencies.csv')
    data = data[data['слово'].isin(POLITICAL+GENERAL)]
    freq = data.pivot(index='десятилетие',columns='слово',values='per_million_corpus').loc[PERIODS]
    counts = data.pivot(index='десятилетие',columns='слово',values='count').loc[PERIODS]
    bal = pd.read_csv(TABLES / '04_empire_balance.csv',index_col=0).loc[PERIODS]
    assert freq.shape == (12,8) and counts.shape == (12,8)
    assert bal[POLITICAL+GENERAL].notna().sum().sum() == 75
    assert (bal[POLITICAL+GENERAL].notna() == counts[POLITICAL+GENERAL].ge(50)).all().all()
    make_frequency(freq, counts)
    make_balance(bal)


def make_russia():
    # Figure 1: the exported relative differences; lines guide the eye only.
    diff = pd.read_csv(TABLES / '05_russia_minus_others.csv').pivot(index='десятилетие', columns='ось', values='разность').loc[PERIODS]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.25))
    fig.subplots_adjust(left=.135, right=.985, bottom=.16, top=.84)
    format_axis(ax, (-.13,.09), np.arange(-.12,.081,.04))
    ax.axhline(0, color='#555555', lw=.9, zorder=2)
    for axis, color, marker, style in [('liberty',BLUE,'o','-'),('civilisation',RED,'s','--')]:
        ax.plot(X, diff[axis], label=LABELS[axis], color=color, marker=marker,
                ls=style, lw=1.5, ms=4, zorder=4, clip_on=False)
    ax.set_ylabel('Разность проекций')
    ax.set_xlabel('Десятилетие')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.55,.98),
               ncol=2, frameon=False, handlelength=2.5, columnspacing=1.4)
    save(fig, 'fig1_russia_relative_article')

    # Figure 2: raw projection values from the exported tables. No trajectories
    # between independent decade models; tiny fixed horizontal offsets distinguish
    # names within each decade, without altering any vertical coordinate.
    series = {'russia': ('Россия', RED, 'o'), 'britain': ('Британия', BLUE, 's'),
              'germany': ('Германия', GREEN, '^'), 'prussia': ('Пруссия', PURPLE, 'D'),
              'turkey': ('Турция', ORANGE, 'v')}
    others = ['france','austria','spain','italy','england','america']
    offsets = dict(zip(['france','austria','spain','italy','england','america','turkey','prussia','germany','britain','russia'], np.linspace(-.30,.30,11)))
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.985, bottom=.095, top=.855, hspace=.26)
    for ax, axis, panel in zip(axes, ('liberty','civilisation'), ('а','б')):
        data = pd.read_csv(TABLES / f'05_projections_{axis}.csv', index_col=0).loc[PERIODS]
        format_axis(ax, (-.30,.18), np.arange(-.3,.151,.1))
        ax.axhline(0, color='#999999', lw=.65, zorder=1)
        for boundary in np.arange(.5,11,1):
            ax.axvline(boundary, color='#eeeeee', lw=.55, zorder=0)
        for w in others:
            ax.scatter(X+offsets[w], data[w], s=11, marker='o', facecolors='none',
                       edgecolors='#777777', linewidths=.6, zorder=2)
        for w, (label, color, marker) in reversed(list(series.items())):
            ax.scatter(X+offsets[w], data[w], s=27 if w=='russia' else 20,
                       c=color, marker=marker, edgecolors='white', linewidths=.35, zorder=4 if w=='russia' else 3)
        ax.set_title(f'{panel}) {LABELS[axis]}', loc='left', pad=7, fontweight='medium')
        ax.set_ylabel('Проекция')
    axes[-1].set_xlabel('Десятилетие')
    handles = [Line2D([],[], marker=m, color='none', markerfacecolor=c, markeredgecolor=c,
                      markersize=5, label=l) for l,c,m in series.values()]
    handles.append(Line2D([],[],marker='o',color='none',markerfacecolor='none',markeredgecolor='#777777',markersize=4.5,label='Остальные названия'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55,.985), ncol=3, frameon=False,
               handletextpad=.4,columnspacing=1.4, labelspacing=.5)
    save(fig, 'fig2_countries_two_axes_article')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--article', choices=('all', 'imperial', 'russia'), default='all')
    args = parser.parse_args()
    if args.article in ('all', 'imperial'):
        make_imperial()
    if args.article in ('all', 'russia'):
        make_russia()

if __name__ == '__main__':
    main()
