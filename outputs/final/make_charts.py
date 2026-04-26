import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG = "/Users/I764361/openenv_hack/astrof/outputs/final/training_log.json"
OUT = "/Users/I764361/openenv_hack/astrof/outputs/final/"

with open(LOG) as f:
    data = json.load(f)

# ── palette ──────────────────────────────────────────────────────────────────
BLUE   = "#4C9BE8"
AMBER  = "#F5A623"
TEAL   = "#2EC4B6"
CORAL  = "#E84855"
PURPLE = "#9B5DE5"
GREY   = "#A0AEC0"
BG     = "#0F1117"
PANEL  = "#1A1D27"
TEXT   = "#E2E8F0"
GRID   = "#2D3148"

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 1 — Grouped bar chart: Random / Greedy / Zero-shot / Trained
# ─────────────────────────────────────────────────────────────────────────────
tasks   = ["easy", "medium", "hard", "expert"]
random_  = [0.4170, 0.6802, 0.7447, 0.3000]
greedy   = [0.7068, 0.5069, 0.6540, 0.2609]
zeroshot = [0.6206, 0.5185, 0.6598, 0.3790]
trained  = [0.956,  0.791,  0.821,  0.731 ]

x = np.arange(len(tasks))
w = 0.19

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(PANEL)

bars = [
    ax.bar(x - 1.5*w, random_,  w, label="Random",       color=GREY,   zorder=3, alpha=0.85),
    ax.bar(x - 0.5*w, greedy,   w, label="Greedy",        color=AMBER,  zorder=3, alpha=0.85),
    ax.bar(x + 0.5*w, zeroshot, w, label="Zero-shot LLM", color=TEAL,   zorder=3, alpha=0.85),
    ax.bar(x + 1.5*w, trained,  w, label="Trained (GRPO)",color=BLUE,   zorder=3),
]

# value labels on trained bars only
for bar in bars[3]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=BLUE)

ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in tasks], color=TEXT, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", color=TEXT, fontsize=12)
ax.set_title("ASTROF — Scores by Task & Method", color=TEXT, fontsize=14, fontweight="bold", pad=14)
ax.tick_params(colors=TEXT)
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)

legend = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10,
                   loc="upper right", framealpha=0.9)

fig.tight_layout(pad=1.6)
fig.savefig(OUT + "results_chart.png", dpi=160, facecolor=BG)
plt.close()
print("results_chart.png done")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 2 — Training curves: reward per phase + SFT loss inset
# ─────────────────────────────────────────────────────────────────────────────
phases = [
    ("easy",   data["grpo_easy"]["log"],   BLUE,   "Easy"),
    ("medium", data["grpo_medium"]["log"], TEAL,   "Medium"),
    ("hard",   data["grpo_hard"]["log"],   AMBER,  "Hard"),
    ("expert", data["grpo_expert"]["log"], CORAL,  "Expert"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG,
                         gridspec_kw={"width_ratios": [2.4, 1]})

# ── left: reward curves ───────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(PANEL)

offset = 0
for name, log, color, label in phases:
    steps   = [e["step"] + offset for e in log]
    rewards = [e["reward"] for e in log]
    stds    = [e["reward_std"] for e in log]
    sa = np.array(steps); ra = np.array(rewards); std = np.array(stds)
    ax.plot(sa, ra, color=color, linewidth=2.2, label=label, zorder=3)
    ax.fill_between(sa, ra - std, ra + std, color=color, alpha=0.12, zorder=2)
    # phase label at end
    ax.annotate(f"  {ra[-1]:.3f}", xy=(sa[-1], ra[-1]),
                fontsize=8.5, color=color, va="center")
    offset += 100

# phase dividers
for i, (_, _, color, label) in enumerate(phases[:-1]):
    xd = (i + 1) * 100
    ax.axvline(xd, color=GRID, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(xd + 2, 0.07, phases[i+1][3], fontsize=8, color=GREY, alpha=0.7)

ax.set_xlabel("Global Training Step", color=TEXT, fontsize=11)
ax.set_ylabel("Reward", color=TEXT, fontsize=11)
ax.set_title("GRPO Curriculum Reward Curves", color=TEXT, fontsize=13, fontweight="bold")
ax.set_xlim(0, 400)
ax.set_ylim(0, 1.05)
ax.tick_params(colors=TEXT)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10,
          loc="lower right", framealpha=0.9)

# phase labels at top
for i, (_, _, color, label) in enumerate(phases):
    mid = i * 100 + 50
    ax.text(mid, 1.015, label, ha="center", fontsize=9, color=color, fontweight="bold", clip_on=False)

# ── right: SFT loss ───────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(PANEL)
sft_log = data["sft_warmstart"]["log"]
sft_steps  = [e["step"] for e in sft_log]
sft_losses = [e["loss"] for e in sft_log]
ax2.plot(sft_steps, sft_losses, color=PURPLE, linewidth=2.2, marker="o",
         markersize=4, zorder=3)
ax2.fill_between(sft_steps, sft_losses, min(sft_losses),
                 color=PURPLE, alpha=0.12, zorder=2)
ax2.set_xlabel("SFT Step", color=TEXT, fontsize=11)
ax2.set_ylabel("Loss", color=TEXT, fontsize=11)
ax2.set_title("SFT Warm-start Loss", color=TEXT, fontsize=13, fontweight="bold")
ax2.tick_params(colors=TEXT)
ax2.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID)

fig.tight_layout(pad=1.8)
fig.savefig(OUT + "training_curves.png", dpi=160, facecolor=BG)
plt.close()
print("training_curves.png done")
