"""
GTO Blackjack: Optimal Stopping / Cashout Strategy

Key questions:
1. EV is always negative - can you still "profitably" cash out?
2. How does setting a stop-win target affect outcomes?
3. What's the optimal cashout target given an x/y ratio?

The core insight: the game is a biased random walk (drift = -0.005 * bet/hand).
Variance dominates early, so most trajectories peak above start before
the drift takes over. If you commit to cashing out at target T, you
lock in that early variance spike — this is pure optimal stopping.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from blackjack_sim import build_shoe, play_hand, simulate, analyze


# ---------------------------------------------------------------------------
# Simulate with a stop-win target
# ---------------------------------------------------------------------------

def simulate_with_target(bankroll, bet, target_mult, stop_loss_mult=0.0,
                          num_hands=2000, num_sims=500):
    """
    Simulate with:
      - Cash out if bankroll >= bankroll * target_mult
      - Bust/stop if bankroll <= bankroll * stop_loss_mult (0 = play to bust)

    Returns: (outcomes, hands_played)
      outcomes: array of final bankroll values (target = cashed out at target)
      hands_played: how many hands until stopping
    """
    target = bankroll * target_mult
    floor = bankroll * stop_loss_mult

    outcomes = np.zeros(num_sims)
    hands_played = np.zeros(num_sims)

    for sim in range(num_sims):
        shoe = build_shoe()
        balance = bankroll
        stopped = False
        for h in range(num_hands):
            if balance >= target:
                outcomes[sim] = balance
                hands_played[sim] = h
                stopped = True
                break
            if balance <= floor or balance <= 0:
                outcomes[sim] = balance
                hands_played[sim] = h
                stopped = True
                break
            actual_bet = min(bet, balance)
            profit = play_hand(shoe, actual_bet)
            balance += profit

        if not stopped:
            outcomes[sim] = balance
            hands_played[sim] = num_hands

    return outcomes, hands_played


def cashout_stats(outcomes, bankroll, target_mult):
    target = bankroll * target_mult
    hit_target = outcomes >= target * 0.99  # small tolerance
    bust = outcomes <= 1

    return {
        'hit_target_pct': hit_target.mean() * 100,
        'bust_pct': bust.mean() * 100,
        'mean_outcome': outcomes.mean(),
        'median_outcome': np.median(outcomes),
        'ev_vs_start': (outcomes.mean() - bankroll) / bankroll * 100,
    }


# ---------------------------------------------------------------------------
# Theory: Gambler's Ruin probability
# ---------------------------------------------------------------------------

def gambler_ruin_prob(start, target, floor, p_win, p_lose):
    """
    Classic gambler's ruin: probability of reaching `target` before `floor`
    in a +1/-1 random walk, starting at `start`.
    p_win: probability of winning each unit
    p_lose: probability of losing each unit
    Assumes integer units.
    """
    if abs(p_win - p_lose) < 1e-10:  # fair game
        return (start - floor) / (target - floor)
    r = p_lose / p_win
    rn = r ** (start - floor)
    rd = r ** (target - floor)
    return (1 - rn) / (1 - rd)


def main():
    bankroll = 1000
    num_sims = 600
    num_hands = 3000

    # Approximate blackjack win/loss probabilities with basic strategy
    # ~43% win, ~49% loss, ~8% push (ignoring pushes for ruin calc)
    p_win = 0.4648   # approx after adjusting for BJ bonus, doubles, splits
    p_lose = 0.4752  # house edge ~0.5%

    print("=" * 65)
    print("BLACKJACK OPTIMAL STOPPING ANALYSIS")
    print("=" * 65)
    print(f"\nHouse edge per hand ≈ {(p_lose - p_win) * 100:.2f}% of bet")
    print(f"EV is ALWAYS negative — basic strategy minimises loss, doesn't flip it.\n")

    # -----------------------------------------------------------------------
    # Part 1: Gambler's ruin theory vs simulation for a few targets
    # -----------------------------------------------------------------------
    bet = 20  # x/y = 0.02
    print(f"Bet=${bet}, Bankroll=${bankroll} (x/y={bet/bankroll:.3f})\n")

    targets = [1.05, 1.10, 1.20, 1.30, 1.50, 2.0]
    print(f"{'Target':>8} {'Theory%':>10} {'Sim%':>10} {'AvgHands':>10} {'EV%':>8}")
    print("-" * 50)

    sim_results = []
    for t in targets:
        outcomes, hands = simulate_with_target(bankroll, bet, t,
                                                num_hands=num_hands, num_sims=num_sims)
        stats = cashout_stats(outcomes, bankroll, t)

        # Theory: treat as units of bet size, walk from y/x=50, target=t*y/x, floor=0
        units_start = bankroll // bet
        units_target = int(t * bankroll // bet)
        theory_pct = gambler_ruin_prob(units_start, units_target, 0, p_win, p_lose) * 100

        print(f"{t:>8.2f}x {theory_pct:>10.1f} {stats['hit_target_pct']:>10.1f} "
              f"{hands[outcomes >= t*bankroll*0.99].mean() if (outcomes >= t*bankroll*0.99).any() else 0:>10.0f} "
              f"{stats['ev_vs_start']:>8.2f}%")
        sim_results.append((t, stats, outcomes, hands))

    print("\nNote: EV% = (mean final bankroll - start) / start")
    print("Even though EV per hand is negative, expected cashout can be positive")
    print("because you STOP when ahead — asymmetric stopping rule.\n")

    # -----------------------------------------------------------------------
    # Part 2: Sweep targets for different x/y ratios
    # -----------------------------------------------------------------------
    target_range = np.linspace(1.01, 2.0, 30)
    configs = [
        (5,   '#4fc3f7', 'x/y=0.005'),
        (20,  '#81c784', 'x/y=0.02'),
        (100, '#ffb74d', 'x/y=0.10'),
        (250, '#e57373', 'x/y=0.25'),
    ]

    print("Sweeping cashout targets for different x/y ratios...")
    sweep_data = {}
    for bet_size, color, label in configs:
        hit_probs = []
        ev_pcts = []
        for t in target_range:
            outcomes, _ = simulate_with_target(bankroll, bet_size, t,
                                               num_hands=num_hands, num_sims=300)
            stats = cashout_stats(outcomes, bankroll, t)
            hit_probs.append(stats['hit_target_pct'])
            ev_pcts.append(stats['ev_vs_start'])
        sweep_data[label] = (hit_probs, ev_pcts, color)
        best_idx = np.argmax(ev_pcts)
        print(f"  {label}: best EV {ev_pcts[best_idx]:.1f}% at target {target_range[best_idx]:.2f}x")

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 14), facecolor='#1a1a2e')
    fig.suptitle('GTO Blackjack: Optimal Stopping — When Should You Cash Out?',
                 color='white', fontsize=14, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_style = dict(facecolor='#0d1117')
    tick_style = dict(colors='#aaaaaa', labelsize=8)
    spine_color = '#333355'

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor('#0d1117')
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xlabel(xlabel, color='#aaaaaa', fontsize=8)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(spine_color)

    # --- Plot 1: Sample trajectories with stop-win marked ---
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Sample Trajectories (bet=$20, stop at +20%)', 'Hand #', 'Bankroll ($)')
    bet_eg = 20
    target_eg = 1.20
    traj_raw = simulate(bankroll, bet_eg, num_hands=500, num_sims=80)
    hands = np.arange(traj_raw.shape[1])
    for i in range(80):
        traj = traj_raw[i]
        # Find first time it hits target
        hit = np.where(traj >= bankroll * target_eg)[0]
        bust_pt = np.where(traj <= 0)[0]
        if len(hit):
            ax1.plot(hands[:hit[0]+1], traj[:hit[0]+1], alpha=0.3, color='#81c784', lw=0.8)
            ax1.plot(hit[0], traj[hit[0]], 'o', color='#81c784', markersize=3, alpha=0.6)
        elif len(bust_pt):
            ax1.plot(hands[:bust_pt[0]+1], traj[:bust_pt[0]+1], alpha=0.2, color='#e57373', lw=0.8)
        else:
            ax1.plot(hands, traj, alpha=0.15, color='#aaaaaa', lw=0.7)
    ax1.axhline(bankroll, color='white', lw=1, ls='--', alpha=0.5, label='Start')
    ax1.axhline(bankroll * target_eg, color='#81c784', lw=1.5, ls='-', alpha=0.8, label=f'Target (+20%)')
    ax1.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')

    # --- Plot 2: Hit target % vs target level for different x/y ---
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'P(Cash Out at Target) vs Target Level', 'Target (× start)', 'Hit target %')
    for label, (hit_probs, ev_pcts, color) in sweep_data.items():
        ax2.plot(target_range, hit_probs, color=color, lw=2, label=label)
    ax2.axvline(1.0, color='white', lw=0.5, ls='--', alpha=0.4)
    ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # --- Plot 3: EV% vs target level (the key insight) ---
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Expected Return vs Cashout Target\n(asymmetric stopping boosts EV)',
             'Target (× start)', 'Expected return vs start (%)')
    for label, (hit_probs, ev_pcts, color) in sweep_data.items():
        ax3.plot(target_range, ev_pcts, color=color, lw=2, label=label)
        best_i = np.argmax(ev_pcts)
        ax3.plot(target_range[best_i], ev_pcts[best_i], '*', color=color, markersize=12)
    ax3.axhline(0, color='white', lw=1, ls='--', alpha=0.5)
    ax3.fill_between(target_range, 0, 0, alpha=0)  # placeholder
    ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax3.set_ylim(bottom=None)

    # --- Plot 4: Theory vs sim (gambler's ruin) ---
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Gambler's Ruin Theory vs Simulation (x/y=0.02)",
             'Target (× start)', 'P(reach target before bust) %')
    theory_probs = []
    sim_probs_line = []
    targets_line = np.linspace(1.01, 2.5, 25)
    units_start = bankroll // bet
    for t in targets_line:
        units_target = int(t * bankroll // bet)
        tp = gambler_ruin_prob(units_start, units_target, 0, p_win, p_lose) * 100
        theory_probs.append(tp)
    # sim points
    for t in [1.05, 1.10, 1.20, 1.40, 1.60, 2.0, 2.5]:
        outcomes, _ = simulate_with_target(bankroll, 20, t, num_hands=num_hands, num_sims=400)
        stats = cashout_stats(outcomes, bankroll, t)
        sim_probs_line.append((t, stats['hit_target_pct']))
    ax4.plot(targets_line, theory_probs, color='#ffb74d', lw=2, label="Gambler's Ruin theory")
    ax4.scatter([s[0] for s in sim_probs_line], [s[1] for s in sim_probs_line],
                color='#4fc3f7', s=50, zorder=5, label='Simulation')
    ax4.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # --- Plot 5: The bust/ever-up paradox explained ---
    ax5 = fig.add_subplot(gs[2, :])
    style_ax(ax5, 'The Bust/Ever-Up Paradox: Bankroll Distribution Over Time (x/y=0.02, 500 sims)',
             'Hand #', 'Bankroll / Start')
    traj_big = simulate(bankroll, 20, num_hands=2000, num_sims=500)
    hands_full = np.arange(traj_big.shape[1])
    p10 = np.percentile(traj_big, 10, axis=0) / bankroll
    p25 = np.percentile(traj_big, 25, axis=0) / bankroll
    p50 = np.median(traj_big, axis=0) / bankroll
    p75 = np.percentile(traj_big, 75, axis=0) / bankroll
    p90 = np.percentile(traj_big, 90, axis=0) / bankroll
    pct_above_start = (traj_big > bankroll).mean(axis=0) * 100  # right axis

    ax5.fill_between(hands_full, p10, p90, alpha=0.15, color='#81c784', label='10th–90th pct')
    ax5.fill_between(hands_full, p25, p75, alpha=0.3, color='#81c784', label='25th–75th pct')
    ax5.plot(hands_full, p50, color='#81c784', lw=2.5, label='Median')
    ax5.axhline(1.0, color='white', lw=1, ls='--', alpha=0.6)

    ax5b = ax5.twinx()
    ax5b.plot(hands_full, pct_above_start, color='#ffb74d', lw=1.5, ls=':', alpha=0.9,
              label='% sims currently above start')
    ax5b.set_ylabel('% currently above start', color='#ffb74d', fontsize=8)
    ax5b.tick_params(colors='#ffb74d', labelsize=7)
    ax5b.set_ylim(0, 100)

    # Annotate the paradox
    early_peak_hand = np.argmax(pct_above_start)
    early_peak_val = pct_above_start[early_peak_hand]
    ax5b.annotate(f'Peak: {early_peak_val:.0f}% above start\n(hand ~{early_peak_hand})',
                  xy=(early_peak_hand, early_peak_val),
                  xytext=(early_peak_hand + 150, early_peak_val - 15),
                  color='#ffb74d', fontsize=8,
                  arrowprops=dict(arrowstyle='->', color='#ffb74d', lw=1.2))

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7.5, facecolor='#1a1a2e', labelcolor='white', loc='upper right')

    plt.savefig('/home/user/Misc/blackjack_cashout.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: blackjack_cashout.png")

    # -----------------------------------------------------------------------
    # Summary intuition
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("INTUITION SUMMARY")
    print("=" * 65)
    print("""
EV per hand:     always negative (~-0.5% × bet)
Kelly criterion: says bet ZERO in a -EV game

But: if you commit to a stop-win target T, you're playing a
different game — you're asking "will the random walk touch T
before it touches 0?" This is Gambler's Ruin, and:

  P(reach T before bust) ≈ (1 - (q/p)^start) / (1 - (q/p)^target)

where p=P(win hand), q=P(lose hand), expressed in bet-size units.

Key insight: with a modest target (e.g. +10-20%), this probability
is HIGH (70-85%), even though EV is negative. The strategy works
because you're selectively stopping at the UP tail of the distribution
and playing through the DOWN tail. That asymmetry dominates for
small targets.

BUT: the mean outcome is still negative — you can't manufacture
a positive EV. What you can do is increase P(positive session)
at the cost of average magnitude. The optimal target maximises
expected outcome, NOT EV per hand.

Bust=35% but ever>start=97%: most trajectories spike above start
in the first ~50-200 hands (variance dominates drift early).
Then negative drift takes over. If you had cashed out at +5%
when you first crossed it, you'd capture that early spike.
""")


if __name__ == '__main__':
    main()
