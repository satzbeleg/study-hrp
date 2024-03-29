import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker

MODELS = [
    'paraphrase-multilingual-mpnet-base-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'distiluse-base-multilingual-cased-v2',
    'sentence-transformers_LaBSE',
    'm-use',
    'laser-de',
]
MODELNAME = [
    "SBert MPNet v2",
    "SBert MiniLM v2",
    "SBert DistilUSE v2",
    "LaBSE",
    "m-USE",
    "LASER de"
]
MODELDIM = [
    768, 
    384, 
    512, 
    768, 
    512, 
    1024
]


NFEATS = [256, 384, 512, 768, 1024, 1536, 2048]

SEEDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def compression_rate(num_bit, num_flt):
    return num_bit / (num_flt * 32)


TESTS = [
    'VMWE', 
    'ABSD-2', 
    'MIO-P', 
    'ARCHI', 
]

metric = "acc"
# metric = "f1-balanced"


# lookup tables: tab[model,size,test]
res_mu = np.empty((len(MODELS), len(NFEATS), len(TESTS)))
res_sd = np.empty((len(MODELS), len(NFEATS), len(TESTS)))
res_min = np.empty((len(MODELS), len(NFEATS), len(TESTS))) 
res_max =np.empty((len(MODELS), len(NFEATS), len(TESTS)))

for i, model in enumerate(MODELS):
    for j, numbool in enumerate(NFEATS):
        # lookup: matrix[seed, test-results]
        res_seeds = []
        for seed in SEEDS:
            FILE = f"numbool={numbool}-randomstate={seed}-outputtype=hrp-seeg.json"
            dat = json.load(open(f"{model}/{FILE}", "r"))
            tmp = [
                [d for d in dat if d.get("task") == test][0]["test"][metric]
                for test in TESTS]
            res_seeds.append(tmp)
        # mean, std, min, max of seeds
        res_seeds = np.array(res_seeds)
        res_mu[i, j, :] = res_seeds.mean(axis=0)
        res_sd[i, j, :] = res_seeds.std(axis=0)
        res_min[i, j, :] = res_seeds.min(axis=0)
        res_max[i, j, :] = res_seeds.max(axis=0)


# read baselines 1 (original)
baselines1 = []
FILE = "numbool=None-randomstate=None-outputtype=float-seeg.json"
for i, model in enumerate(MODELS):
    dat = json.load(open(f"{model}/{FILE}", "r"))
    tmp = [
        [d for d in dat if d.get("task") == test][0]["test"][metric]
        for test in TESTS]
    baselines1.append(tmp)

baselines1 = np.array(baselines1)
avg_bases1 = baselines1.mean(axis=1)


# read baselines 2 (sigmoid)
baselines2 = []
FILE = "numbool=None-randomstate=None-outputtype=sigmoid-seeg.json"
for i, model in enumerate(MODELS):
    dat = json.load(open(f"{model}/{FILE}", "r"))
    tmp = [
        [d for d in dat if d.get("task") == test][0]["test"][metric]
        for test in TESTS]
    baselines2.append(tmp)

baselines2 = np.array(baselines2)
avg_bases2 = baselines2.mean(axis=1)


# save tables
with open("table-seeg.tex", "w") as fp:
    fp.write("Model & " + " & ".join(TESTS) + " & Avg \\\\\n")
    for i in range(len(MODELS)):
        fp.write(f"% {MODELS[i]} \n")
        # original baseline
        line = np.array2string(
            np.array(baselines1[i].tolist() + [avg_bases1[i]]) * 100., 
            separator=" & ", formatter={'float_kind':lambda x: "%.2f" % x})
        fp.write(f"{MODELS[i]} & {line[1:-1]} \\\\\n")
        # sigmoid baseline
        line = np.array2string(
            np.array(baselines2[i].tolist() + [avg_bases2[i]]) * 100., 
            separator=" & ", formatter={'float_kind':lambda x: "%.2f" % x})
        fp.write(f"sigmoid & {line[1:-1]} \\\\\n")
        # HRP results
        for j in range(len(NFEATS)):
            line = np.array2string(
                np.array(res_mu[i, j, :].tolist() + [res_mu[i, j, :].mean()]) * 100., 
                separator=" & ", formatter={'float_kind':lambda x: "%.2f" % x})
            fp.write(f"{NFEATS[j]} & {line[1:-1]} \\\\\n")
        # end of table
        fp.write(f"\\midrule \n")


# line styles
styles = [
    ("dotted", "v"),
    ("dashed", "s"),
    ("dashdot", "."),
    ((0, (2, 1)), "^"),
    ((0, (5, 1)), "x"),
    ((0, (3, 1, 1, 1)), "d"),
]

colors = list(matplotlib.colors.TABLEAU_COLORS.keys())




# average accurancy for all tests; by compression rate
avg_mu = res_mu.mean(axis=2)

fig = plt.figure(figsize=(5, 4), dpi=144)
ax = fig.add_subplot(111)
for i, modelname in enumerate(MODELNAME):
    # modelname = f"{modelname} ({avg_bases[i]:.1f}%)"
    x = [100 * compression_rate(numbool, MODELDIM[i]) for numbool in NFEATS]
    ax.plot(x, avg_mu[i, :], label=modelname, marker=styles[i][1], linestyle=styles[i][0], color=colors[i])

for i, modelname in enumerate(MODELNAME):
    ax.plot([100.], [avg_bases1[i]], marker=styles[i][1], color="darkgrey", alpha=.9)

for i, modelname in enumerate(MODELNAME):
    ax.plot([100 * 0.03125], [avg_bases2[i]], marker=styles[i][1], color="black", alpha=.9)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("memory consumption rate in %")
ax.set_ylabel("accuracy")  # avg. accuracy for all tests
ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, 1.2),
    ncol=2, fancybox=True, shadow=True)
fig.tight_layout()
fig.savefig("res-seeg-memorate-averaged.png")




# error bar diagrams; by compression rate
for k, testname in enumerate(TESTS):
    # new figure
    fig = plt.figure(figsize=(5, 4), dpi=144)
    ax = fig.add_subplot(111)
    # loop over each model
    for i, modelname in enumerate(MODELNAME):
        modelname = f"{modelname} ({100 * baselines1[i, k]:.1f}%)"
        x = [100 * compression_rate(numbool, MODELDIM[i]) for numbool in NFEATS]
        ax.fill_between(
            x, res_min[i, :, k], res_max[i, :, k], 
            alpha=0.1)
        ax.errorbar(
            x, res_mu[i, :, k], np.stack((
                res_mu[i, :, k] - res_min[i, :, k], 
                res_max[i, :, k] - res_mu[i, :, k])), 
            label=modelname, capsize=5, elinewidth=.5, capthick=.5,
            linestyle=styles[i][0], marker=styles[i][1])
    ax.set_xlabel("memory consumption rate in %")
    ax.set_ylabel("accuracy")
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.35),
        ncol=2, fancybox=True, shadow=True)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(f"res-seeg-memorate-error-{testname}.png")

