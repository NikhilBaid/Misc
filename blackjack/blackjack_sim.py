"""
GTO Blackjack Bankroll Simulator

Models bankroll trajectories using basic strategy (optimal GTO play).
Explores how bet/bankroll ratio (x/y) affects the probability of
cashing out ahead at some point before going bust.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

def build_shoe(num_decks=6):
    ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # 11 = Ace
    shoe = ranks * 4 * num_decks
    random.shuffle(shoe)
    return shoe


def hand_value(hand):
    total = sum(hand)
    aces = hand.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def is_soft(hand):
    total = sum(hand)
    return 11 in hand and total <= 21


# ---------------------------------------------------------------------------
# Basic Strategy (GTO)
# ---------------------------------------------------------------------------
# Returns: 'H'=hit, 'S'=stand, 'D'=double(else hit), 'P'=split, 'R'=surrender(else hit)

PAIR_STRATEGY = {
    # (pair_card, dealer_upcard) -> action
    # Aces and 8s: always split
    11: 'P',   # Aces - always split (handled specially)
    8:  None,  # handled below per dealer card
}

def basic_strategy(hand, dealer_up, can_double=True, can_split=True, can_surrender=True):
    val = hand_value(hand)

    # Pair splitting
    if can_split and len(hand) == 2 and hand[0] == hand[1]:
        pair = hand[0] if hand[0] != 11 else 11
        # Normalize face cards to 10
        pair = min(pair, 10)
        d = min(dealer_up, 10)

        if pair == 11:  return 'P'  # Always split aces
        if pair == 8:   return 'P'  # Always split 8s
        if pair == 10:  return 'S'  # Never split 10s
        if pair == 5:
            # Treat as hard 10
            if d in range(2, 10): return 'D' if can_double else 'H'
            return 'H'
        if pair == 4:
            if d in (5, 6): return 'P'
            return 'H'
        if pair == 9:
            if d in (7, 10, 11): return 'S'
            return 'P'
        if pair == 7:
            if d in range(2, 8): return 'P'
            return 'H'
        if pair == 6:
            if d in range(2, 7): return 'P'
            return 'H'
        if pair == 3 or pair == 2:
            if d in range(2, 8): return 'P'
            return 'H'

    # Soft hands
    if is_soft(hand):
        non_ace = [c for c in hand if c != 11]
        soft_val = sum(non_ace)  # e.g. soft 18 -> soft_val=7
        d = min(dealer_up, 10)

        if val >= 19: return 'S'
        if val == 18:
            if d in (2, 7, 8): return 'S'
            if d in (3, 4, 5, 6): return 'D' if can_double else 'S'
            return 'H'
        if val == 17:
            if d in (3, 4, 5, 6): return 'D' if can_double else 'H'
            return 'H'
        if val in (15, 16):
            if d in (4, 5, 6): return 'D' if can_double else 'H'
            return 'H'
        if val in (13, 14):
            if d in (5, 6): return 'D' if can_double else 'H'
            return 'H'
        return 'H'

    # Hard hands
    d = min(dealer_up, 10)

    if val >= 17: return 'S'
    if val == 16:
        if can_surrender and d in (9, 10, 11): return 'R'
        if d in range(2, 7): return 'S'
        return 'H'
    if val == 15:
        if can_surrender and d == 10: return 'R'
        if d in range(2, 7): return 'S'
        return 'H'
    if val in (13, 14):
        if d in range(2, 7): return 'S'
        return 'H'
    if val == 12:
        if d in (4, 5, 6): return 'S'
        return 'H'
    if val == 11:
        return 'D' if can_double else 'H'
    if val == 10:
        if d in range(2, 10): return 'D' if can_double else 'H'
        return 'H'
    if val == 9:
        if d in range(3, 7): return 'D' if can_double else 'H'
        return 'H'
    return 'H'


# ---------------------------------------------------------------------------
# Single hand simulation
# ---------------------------------------------------------------------------

def play_hand(shoe, bet):
    """Play one hand of blackjack. Returns net profit (can be negative)."""

    def draw():
        if len(shoe) < 15:
            shoe.extend(build_shoe())
        return shoe.pop()

    dealer_hand = [draw(), draw()]
    player_hand = [draw(), draw()]
    dealer_up = dealer_hand[0]

    # Player blackjack
    if hand_value(player_hand) == 21:
        if hand_value(dealer_hand) == 21:
            return 0  # push
        return bet * 1.5

    # Dealer blackjack
    if hand_value(dealer_hand) == 21:
        return -bet

    # Play out player hands (supports splits)
    def play_player_hand(hand, current_bet, can_split=True, can_double=True, can_surrender=True):
        """Returns list of (bet, final_hand) tuples."""
        results = []

        while True:
            action = basic_strategy(hand, dealer_up, can_double=can_double,
                                    can_split=can_split, can_surrender=can_surrender)

            if action == 'S':
                results.append((current_bet, hand))
                break
            elif action == 'H':
                hand = hand + [draw()]
                can_double = False
                can_surrender = False
                if hand_value(hand) >= 21:
                    results.append((current_bet, hand))
                    break
            elif action == 'D':
                hand = hand + [draw()]
                results.append((current_bet * 2, hand))
                break
            elif action == 'P':
                card = hand[0]
                hand1 = [card, draw()]
                hand2 = [card, draw()]
                # Aces get one card only after split
                if card == 11:
                    results.append((current_bet, hand1))
                    results.append((current_bet, hand2))
                else:
                    results.extend(play_player_hand(hand1, current_bet, can_split=False))
                    results.extend(play_player_hand(hand2, current_bet, can_split=False))
                break
            elif action == 'R':
                results.append((-current_bet / 2, None))  # surrender
                break

        return results

    player_results = play_player_hand(player_hand, bet)

    # Dealer plays
    while hand_value(dealer_hand) < 17:
        dealer_hand.append(draw())
    dealer_val = hand_value(dealer_hand)
    dealer_bust = dealer_val > 21

    total_profit = 0
    for hand_bet, hand in player_results:
        if hand is None:  # surrender
            total_profit += hand_bet  # already negative
            continue
        if hand_bet < 0:  # already surrendered
            total_profit += hand_bet
            continue

        pval = hand_value(hand)
        if pval > 21:
            total_profit -= hand_bet
        elif dealer_bust or pval > dealer_val:
            total_profit += hand_bet
        elif pval == dealer_val:
            pass  # push
        else:
            total_profit -= hand_bet

    return total_profit


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(bankroll, bet, num_hands=2000, num_sims=200):
    """
    Run multiple simulations tracking bankroll trajectory.
    Returns array of shape (num_sims, num_hands+1).
    """
    trajectories = np.zeros((num_sims, num_hands + 1))
    trajectories[:, 0] = bankroll

    for sim in range(num_sims):
        shoe = build_shoe()
        balance = bankroll
        for h in range(num_hands):
            if balance <= 0:
                trajectories[sim, h+1:] = 0
                break
            actual_bet = min(bet, balance)
            profit = play_hand(shoe, actual_bet)
            balance += profit
            trajectories[sim, h + 1] = balance

    return trajectories


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(trajectories, bankroll):
    """Compute stats across trajectories."""
    ever_up = np.any(trajectories > bankroll, axis=1).mean()
    bust = (trajectories[:, -1] == 0).mean()
    final = trajectories[:, -1]
    peak = trajectories.max(axis=1)

    return {
        'ever_up_pct': ever_up * 100,
        'bust_pct': bust * 100,
        'median_final': np.median(final),
        'mean_final': np.mean(final),
        'median_peak': np.median(peak),
        'p10_final': np.percentile(final, 10),
        'p90_final': np.percentile(final, 90),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sim(bankroll, bet, trajectories, ax_traj, ax_dist, color, label):
    ratio = bet / bankroll
    stats = analyze(trajectories, bankroll)

    n_show = min(50, len(trajectories))
    hands = np.arange(trajectories.shape[1])

    # Trajectories
    for i in range(n_show):
        ax_traj.plot(hands, trajectories[i] / bankroll, alpha=0.12, color=color, lw=0.7)

    median_traj = np.median(trajectories, axis=0)
    ax_traj.plot(hands, median_traj / bankroll, color=color, lw=2,
                 label=f'x/y={ratio:.2f} | ever>start: {stats["ever_up_pct"]:.0f}%')
    ax_traj.axhline(1.0, color='white', lw=0.5, ls='--', alpha=0.4)

    # Final distribution
    finals = trajectories[:, -1] / bankroll
    ax_dist.hist(finals, bins=40, alpha=0.5, color=color,
                 label=f'x/y={ratio:.2f}', orientation='horizontal', density=True)

    return stats


def main():
    bankroll = 1000
    num_hands = 1500
    num_sims = 300

    # Different bet sizes (x/y ratios)
    configs = [
        (bankroll, 5,    '#4fc3f7', 'x/y=0.005 (min bet)'),
        (bankroll, 25,   '#81c784', 'x/y=0.025'),
        (bankroll, 100,  '#ffb74d', 'x/y=0.10'),
        (bankroll, 250,  '#e57373', 'x/y=0.25 (aggressive)'),
    ]

    print(f"Running blackjack simulations: {num_sims} sims × {num_hands} hands each")
    print(f"Starting bankroll: ${bankroll}\n")

    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    fig.suptitle('GTO Blackjack: Bankroll Trajectory vs Bet/Bankroll Ratio (x/y)',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    all_stats = []
    all_trajectories = []

    for i, (y, x, color, label) in enumerate(configs):
        print(f"  Simulating x={x}, y={y} (x/y={x/y:.3f})...", end=' ', flush=True)
        traj = simulate(y, x, num_hands=num_hands, num_sims=num_sims)
        all_trajectories.append(traj)
        stats = analyze(traj, y)
        all_stats.append((x/y, stats))
        print(f"ever_up={stats['ever_up_pct']:.1f}%  bust={stats['bust_pct']:.1f}%")

    # --- Top row: individual trajectory plots ---
    for i, (traj, (y, x, color, label)) in enumerate(zip(all_trajectories, configs)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor('#0d1117')

        ratio = x / y
        stats = analyze(traj, y)
        hands = np.arange(traj.shape[1])
        n_show = min(80, len(traj))

        for j in range(n_show):
            ax.plot(hands, traj[j] / y, alpha=0.1, color=color, lw=0.6)

        median_traj = np.median(traj, axis=0)
        p25 = np.percentile(traj, 25, axis=0) / y
        p75 = np.percentile(traj, 75, axis=0) / y

        ax.fill_between(hands, p25, p75, alpha=0.25, color=color)
        ax.plot(hands, median_traj / y, color=color, lw=2, label='Median')
        ax.axhline(1.0, color='white', lw=1, ls='--', alpha=0.6, label='Start')
        ax.axhline(0, color='#e57373', lw=0.8, ls='-', alpha=0.5)

        ax.set_title(f'Bet ${x} | Start ${y} | x/y = {ratio:.3f}',
                     color='white', fontsize=10)
        ax.set_xlabel('Hand #', color='#aaaaaa', fontsize=8)
        ax.set_ylabel('Bankroll / Start', color='#aaaaaa', fontsize=8)
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

        textstr = (f'Ever > start: {stats["ever_up_pct"]:.1f}%\n'
                   f'Bust rate:    {stats["bust_pct"]:.1f}%\n'
                   f'Median peak: {np.median(traj.max(axis=1))/y:.2f}x')
        props = dict(boxstyle='round', facecolor='#0d1117', alpha=0.8, edgecolor=color)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, color='white', family='monospace')
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

    plt.savefig('/home/user/Misc/blackjack_trajectories.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    print("\nSaved: blackjack_trajectories.png")

    # --- Summary plot: x/y ratio vs "ever up" probability ---
    fig2, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#1a1a2e')
    fig2.suptitle('GTO Blackjack: Effect of Bet/Bankroll Ratio (x/y)',
                  color='white', fontsize=13, fontweight='bold')

    # Fine-grained sweep of ratios
    print("\nSweeping x/y ratios for summary stats...")
    sweep_ratios = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]
    sweep_stats = []
    for r in sweep_ratios:
        bet_size = int(bankroll * r)
        if bet_size < 1: bet_size = 1
        traj = simulate(bankroll, bet_size, num_hands=num_hands, num_sims=num_sims)
        s = analyze(traj, bankroll)
        sweep_stats.append(s)
        print(f"  x/y={r:.3f}  ever_up={s['ever_up_pct']:.1f}%  bust={s['bust_pct']:.1f}%")

    ever_up = [s['ever_up_pct'] for s in sweep_stats]
    bust = [s['bust_pct'] for s in sweep_stats]
    median_peak = [s['median_peak'] / bankroll for s in sweep_stats]

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    ax1.plot(sweep_ratios, ever_up, 'o-', color='#81c784', lw=2, markersize=6, label='Ever above start')
    ax1.plot(sweep_ratios, [100 - b for b in bust], 's--', color='#4fc3f7', lw=1.5, markersize=5, label='Survival rate')
    ax1.plot(sweep_ratios, bust, '^--', color='#e57373', lw=1.5, markersize=5, label='Bust rate')
    ax1.set_xlabel('Bet / Bankroll ratio (x/y)', color='#aaaaaa')
    ax1.set_ylabel('Probability (%)', color='#aaaaaa')
    ax1.set_title('Cashout-Up Probability vs x/y Ratio', color='white')
    ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax1.xaxis.label.set_color('#aaaaaa')

    ax2.plot(sweep_ratios, median_peak, 'D-', color='#ffb74d', lw=2, markersize=6)
    ax2.axhline(1.0, color='white', ls='--', lw=0.8, alpha=0.5)
    ax2.set_xlabel('Bet / Bankroll ratio (x/y)', color='#aaaaaa')
    ax2.set_ylabel('Median peak bankroll / start', color='#aaaaaa')
    ax2.set_title('Median Peak Bankroll vs x/y Ratio', color='white')
    ax2.fill_between(sweep_ratios, 1.0, median_peak,
                     where=[p > 1 for p in median_peak],
                     alpha=0.2, color='#81c784', label='Profitable peak zone')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/user/Misc/blackjack_summary.png', dpi=150,
                bbox_inches='tight', facecolor='#1a1a2e')
    print("Saved: blackjack_summary.png")

    # Print final table
    print("\n" + "="*70)
    print(f"{'x/y':>8} {'Ever>start%':>12} {'Bust%':>8} {'MedianFinal':>13} {'MedPeak':>10}")
    print("="*70)
    for r, s in zip(sweep_ratios, sweep_stats):
        print(f"{r:8.3f} {s['ever_up_pct']:12.1f} {s['bust_pct']:8.1f} "
              f"{s['median_final']:13.0f} {s['median_peak']:10.0f}")
    print("="*70)
    print(f"\nNote: House edge ~0.5% with basic strategy. {num_hands} hands per sim.")


if __name__ == '__main__':
    main()
