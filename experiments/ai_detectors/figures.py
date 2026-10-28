"""Figures for chapter 3.7: (a) AUC per detector per arm, (b) TPR@5%FPR vs bits per word, (c) key-selection transfer."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
R = json.load(open(HERE / "results.json"))
rows = R["rows"]; bpw = R["bpw"]; M = R["matrix"]
dets = sorted({r["detector"] for r in rows})
LAB = {"D1_roberta_openai": "D1 RoBERTa-GPT-2", "D2_chatgpt_roberta": "D2 RoBERTa-ChatGPT", "D3_desklib": "D3 DeBERTa-v3 (RAID)",
       "D4_binoculars": "D4 Binoculars", "D5_fast_detectgpt": "D5 Fast-DetectGPT"}
ARMS = ["cover", "stego_b1", "stego_b2", "stego_b3", "stego_b4", "stego_b5"]
ARM_LAB = {"cover": "контейнер", **{f"stego_b{b}": f"стего, b={b}" for b in range(1, 6)}}
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})

# (a) grouped bars: AUC
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.13; x = np.arange(len(dets))
for k, arm in enumerate(ARMS):
    vals = [next(r["auc"] for r in rows if r["detector"] == d and r["arm"] == arm) for d in dets]
    ax.bar(x + (k - 2.5) * w, vals, w, label=ARM_LAB[arm], color=plt.cm.viridis(k / 6))
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([LAB[d] for d in dets], fontsize=9)
ax.set_ylabel("AUC (людський проти оцінюваного)"); ax.set_ylim(0, 1.05); ax.legend(ncol=3, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(HERE / "fig_auc.png", dpi=200)

# (b) TPR@5%FPR vs actual bpw
fig, ax = plt.subplots(figsize=(7, 4))
for d in dets:
    xs = [0] + [bpw[a]["bpw"] for a in ARMS[1:]]
    ys = [next(r["tpr5"] for r in rows if r["detector"] == d and r["arm"] == a) for a in ARMS]
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=LAB[d])
ax.set_xlabel("фактична ємність, біт/слово контейнера"); ax.set_ylabel("частка виявлень при FPR = 5 %")
ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "fig_tpr_bpw.png", dpi=200)

# (c) transfer matrix heatmap: AUC(sel) - AUC(rand)
guides = list(M); evals = list(M[guides[0]])
mat = np.array([[M[g][e]["auc_sel"] - M[g][e]["auc_rand"] for e in evals] for g in guides])
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(mat, cmap="RdBu", vmin=-0.25, vmax=0.25)
ax.set_xticks(range(len(evals))); ax.set_xticklabels([LAB[e] for e in evals], rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(guides))); ax.set_yticklabels([LAB.get(g, "ансамбль") for g in guides], fontsize=9)
for i in range(len(guides)):
    for j in range(len(evals)):
        ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=8)
ax.set_xlabel("детектор, яким оцінюється стеготекст"); ax.set_ylabel("детектор-провідник вибору ключа")
fig.colorbar(im, ax=ax, label="ΔAUC (вибір ключа − випадковий ключ)")
fig.tight_layout(); fig.savefig(HERE / "fig_transfer.png", dpi=200)
print("saved")
