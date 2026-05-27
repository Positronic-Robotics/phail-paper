"""Figure for Appendix I (Sup-Sign Intransitivity).

Plots three step-function CDFs and marks the pairwise supremum points,
showing the Condorcet cycle A > B > C > A under sup-sign ordering.

Output: out/paper/fig_supsign.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'out' / 'paper'


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))

    # Three step CDFs (Appendix I construction).
    # F_A: jumps to 0.4 at t=1.
    # F_B: jumps to 0.3 at t=5, then 0.7 at t=20.
    # F_C: jumps to 0.2 at t=10, then 0.9 at t=50.
    t_max = 70
    color_a = '#1f77b4'
    color_b = '#ff7f0e'
    color_c = '#2ca02c'

    ax.plot([0, 1, 1, t_max], [0, 0, 0.4, 0.4],
            color=color_a, linewidth=2.2, label=r'$F_A$')
    ax.plot([0, 5, 5, 20, 20, t_max], [0, 0, 0.3, 0.3, 0.7, 0.7],
            color=color_b, linewidth=2.2, label=r'$F_B$')
    ax.plot([0, 10, 10, 50, 50, t_max], [0, 0, 0.2, 0.2, 0.9, 0.9],
            color=color_c, linewidth=2.2, label=r'$F_C$')

    # Pairwise sup-points: vertical arrows between the two CDFs at the sup,
    # annotated with the winner at the sup.
    arrow_color = '#444'
    arrow_kwargs = dict(arrowstyle='<->', color=arrow_color, lw=1.5)
    text_kwargs = dict(fontsize=10, va='center', color='#222')

    # A vs B: sup on t in [1, 5), value 0.4, F_A > F_B → A ≻ B
    ax.annotate('', xy=(3, 0.4), xytext=(3, 0.0),
                arrowprops=arrow_kwargs)
    ax.text(3.5, 0.20,
            r'$\sup|F_A{-}F_B|=0.4$' + '\n' + r'$A \succ B$',
            **text_kwargs)

    # B vs C: sup on t in [20, 50), value 0.5, F_B > F_C → B ≻ C
    ax.annotate('', xy=(35, 0.7), xytext=(35, 0.2),
                arrowprops=arrow_kwargs)
    ax.text(36, 0.45,
            r'$\sup|F_B{-}F_C|=0.5$' + '\n' + r'$B \succ C$',
            **text_kwargs)

    # A vs C: sup on t >= 50, value 0.5, F_C > F_A → C ≻ A
    ax.annotate('', xy=(60, 0.9), xytext=(60, 0.4),
                arrowprops=arrow_kwargs)
    ax.text(60.5, 0.65,
            r'$\sup|F_A{-}F_C|=0.5$' + '\n' + r'$C \succ A$',
            **text_kwargs)

    ax.set_xlabel('Time $t$', fontsize=11)
    ax.set_ylabel('CDF $F(t)$', fontsize=11)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        r'Sup-sign on three CDFs yields a Condorcet cycle: '
        r'$A \succ B \succ C \succ A$',
        fontsize=11,
    )

    plt.tight_layout()
    pdf_path = OUT / 'fig_supsign.pdf'
    png_path = OUT / 'fig_supsign.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Wrote {pdf_path}')
    print(f'Wrote {png_path}')


if __name__ == '__main__':
    main()
