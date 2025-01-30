import os

import numpy as np
from joblib import load
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Functions import get_data_fromNPZ, get_data_fromDAT, get_isoline

models_dir = "models/"
list_df_files = list(filter(lambda x: ".joblib" in x, os.listdir(models_dir)))

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

figure = plt.figure(figsize=(27, 9))
i = 1

for ds_cnt, ds in enumerate(datasets):
    sigma, kappa, I, B, x, y, growth, df = ds

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    pcm = ax.pcolormesh(x, y, growth.T, cmap='YlGnBu')
    ax.set_ylabel(f"$\\sigma$={sigma}, $\\kappa$={kappa}, $I$={I} $kA$, $B$={B} $T$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    i += 1

    x_, y_ = np.meshgrid(x, y)
    xobs = np.stack([x_.flat, y_.flat], axis=1)

    growth_edge = 0.1
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

        probs = clf.predict_proba(X)
        yflat = probs[:, 1]  # > probs[:, 0]).astype(int)
        ygrid = yflat.reshape(int(n_new.imag), int(n_new.imag))

        p = ax.pcolormesh(*xgrid, ygrid, shading='gouraud', vmin=0., vmax=1., cmap='gist_yarg')
        ax.plot(plot_x, plot_y, marker=".", c="orange")

        ax.set_title(f"{name}\n" + f"$\\sigma$={sigma}, $\\kappa$={kappa}, $I$={I} $kA$, $B$={B} $T$\n" +
                     f"Isoline by growth: {growth_edge}\n")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        i += 1

# figure.colorbar(p)
plt.tight_layout()
plt.show()
