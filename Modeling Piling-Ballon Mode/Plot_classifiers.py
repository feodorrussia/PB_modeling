import os

import numpy as np
from joblib import load
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Functions import get_data_fromNPZ, get_data_fromDAT, get_isoline, pack_df_fromArrays

models_dir = "models/"
list_df_files = list(filter(lambda x: ".joblib" in x, os.listdir(models_dir)))
#  ["MLP_classifier-2L_s-wo_aug-all.joblib",
#                  "MLP_classifier-2L_s-aug-h_l.joblib"]

names = []
classifiers = []

for i in range(len(list_df_files)):
    names.append(list_df_files[i][:-7])
    classifiers.append(load(models_dir + list_df_files[i]))

datasets = [
    get_data_fromNPZ("output-data_triangularity=1.8e-1.npz",
                     meta={"triangularity": 0.18,
                           "elongation": 1.95,
                           "I": 300.,
                           "B": 0.7}, verbose=False),
    get_data_fromNPZ("output-data_triangularity=2.1e-1.npz",
                     meta={"triangularity": 0.21,
                           "elongation": 1.9,
                           "I": 300.,
                           "B": 0.7}, verbose=False),
    get_data_fromDAT(meta={"triangularity": 0.35,
                           "elongation": 1.83,
                           "I": 400.,
                           "B": 0.8}, verbose=False)
]

n_new = 20j
x_min, x_max = 0.01, 0.15
y_min, y_max = 0.025, 6.

xgrid = np.mgrid[x_min:x_max:n_new, y_min:y_max:n_new]
xflat = xgrid.reshape(2, -1).T
x, y = xflat[:, 0], xflat[:, 1]

growth = np.zeros((x.shape[0], y.shape[0]))
new_df = pack_df_fromArrays({"A": x,
                             "P": y,
                             "growth": growth})

# datasets.append((0.35, 1.83, 250, 0.7, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 350, 0.7, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 400, 0.7, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 450, 0.7, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 500, 0.6, x, y, growth, new_df))
#
# datasets.append((0.35, 1.83, 350, 0.6, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 350, 0.65, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 350, 0.7, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 350, 0.75, x, y, growth, new_df))
# datasets.append((0.35, 1.83, 350, 0.8, x, y, growth, new_df))

coeff_x, coeff_y = 3.5, 3
width, height = int((len(classifiers) + 1) * coeff_x), int((len(datasets) + 0) * coeff_x)
figure = plt.figure(figsize=(width, height))
i = 1

colorbars = []

for ds_cnt, ds in enumerate(datasets):
    sigma, kappa, I, B, x, y, growth, df = ds

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")

    vmin, vmax = growth.min(), growth.max()
    pcm = ax.pcolormesh(x, y, growth.T, vmin=vmin, vmax=vmax, cmap='YlGnBu')

    ax.set_ylabel(f"$\\sigma$={sigma}, $\\kappa$={kappa}, $I$={I} $kA$, $B$={B} $T$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    i += 1

    x_, y_ = np.meshgrid(x, y)
    xobs = np.stack([x_.flat, y_.flat], axis=1)

    growth_edge = 0.1
    if np.count_nonzero(df.growth.to_numpy()) > 0:
        plot_x, plot_y, plot_z = get_isoline(df.growth.to_numpy(), df[["A", "P", "growth"]].to_numpy(), growth_edge)

    n_new = 100j
    xgrid = np.mgrid[x_.min():x_.max():n_new, y_.min():y_.max():n_new]
    xflat = xgrid.reshape(2, -1).T

    X = np.stack([xflat[:, 0], xflat[:, 1],
                  np.array([sigma] * xflat.shape[0]),
                  np.array([kappa] * xflat.shape[0]),
                  np.array([I] * xflat.shape[0]),
                  np.array([B] * xflat.shape[0])], axis=1)

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        probs = clf.predict(X)
        yflat = probs  # > probs[:, 0]).astype(int)
        ygrid = yflat.reshape(int(n_new.imag), int(n_new.imag))

        ax.pcolormesh(*xgrid, ygrid, shading='gouraud', vmin=vmin, vmax=vmax, cmap='YlGnBu')
        if np.count_nonzero(df.growth.to_numpy()) > 0:
            ax.plot(plot_x, plot_y, marker=".", c="orange", label=f"Isoline from DS: $\\gamma={growth_edge}$")
            ax.legend(loc="lower right")

        if ds_cnt == 0:
            ax.set_title(f"{name}")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        i += 1

    colorbars.append((pcm, ax))

for p_i, ax_i in colorbars:
    figure.colorbar(p_i, ax=ax_i)
plt.tight_layout()
plt.show()
