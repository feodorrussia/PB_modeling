import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data_fromString(filename, regular_mask, data_name=None):
    match = re.search(regular_mask, filename)

    if match:
        data = float(match.group(1))
        print(f"Current data{('(' + data_name + ')') if data_name is not None else ''}:", data)
    else:
        print(f"Data{('(' + data_name + ')') if data_name is not None else ''} is not found")
        try:
            data = float(input(f"Input data{('(' + data_name + ')') if data_name is not None else ''}: "))
            print(f"Current data{('(' + data_name + ')') if data_name is not None else ''}:", data)
        except Exception as e:
            print(e)
            data = 0.1
            print(f"Data set as {data}")
    print()
    return data


def pack_df_fromArrays(data):
    N, M = data["growth"].shape

    data_arr = []
    for i in range(N):
        for j in range(M):
            data_arr.append([data["A"][i], data["P"][j], data["growth"][i, j]])
    return pd.DataFrame(data_arr, columns=['A', 'P', 'growth'])


def get_data_fromNPZ(filename, meta=None, verbose=True):
    if meta is None:
        triangularity = get_data_fromString(filename, r"triangularity=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]",
                                            "triangularity")
        elongation = get_data_fromString(filename, r"elongation=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]",
                                         "elongation")
        I = get_data_fromString(filename, r"I=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]", "I")
        B = get_data_fromString(filename, r"B=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]", "B")
    else:
        triangularity = meta["triangularity"]
        elongation = meta["elongation"]
        I = meta["I"]
        B = meta["B"]

    data = np.load(f'data/{filename}')

    df_ = pack_df_fromArrays({"A": data["delta_arr"],
                              "P": data["p_multy_arr"],
                              "growth": data["growth"]})

    if verbose:
        plot_diagram(data["delta_arr"], data["p_multy_arr"], data["growth"], data["unstable_mode"],
                     meta={'sigma': triangularity,
                           'kappa': elongation,
                           'I': I,
                           'B': B,
                           'growth_edge': 0.1})
    # , data["unstable_mode"]

    return triangularity, elongation, I, B, data["delta_arr"], data["p_multy_arr"], data["growth"], df_


def get_data_fromDAT(index="", meta=None, verbose=True):
    if meta is None:
        triangularity = get_data_fromString(index, r"triangularity=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]",
                                            "triangularity")
        elongation = get_data_fromString(index, r"elongation=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]",
                                         "elongation")
        I = get_data_fromString(index, r"I=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]", "I")
        B = get_data_fromString(index, r"B=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)[^0-9]", "B")
    else:
        triangularity = meta["triangularity"]
        elongation = meta["elongation"]
        I = meta["I"]
        B = meta["B"]

    index_A = pd.read_table(f"data/A{index}.dat", sep=' ', header=None).to_numpy()[:, 0]
    header_P = pd.read_table(f"data/P{index}.dat", sep=' ', header=None).to_numpy()[0]

    growth = pd.read_table(f"data/growth{index}.dat", sep=' ', header=None)
    mode = pd.read_table(f"data/unstable_mode{index}.dat", sep=' ', header=None)

    df_ = pack_df_fromArrays({"A": index_A,
                              "P": header_P,
                              "growth": growth.to_numpy()})

    if verbose:
        plot_diagram(index_A, header_P, growth.to_numpy(), mode.to_numpy(), meta={'sigma': triangularity,
                                                                                  'kappa': elongation,
                                                                                  'I': I,
                                                                                  'B': B,
                                                                                  'growth_edge': 0.1})
    # , index_A, header_P, growth.to_numpy(), mode.to_numpy()

    return triangularity, elongation, I, B, index_A, header_P, growth.to_numpy(), df_


def get_isoline(np_growth, df_data, growth_edge):
    zero_mask = abs(np_growth - growth_edge) < np_growth.std() / 4
    plot_data = df_data[zero_mask, :]

    plot_x = [plot_data[0, 0]]
    plot_y = [plot_data[0, 1]]
    plot_z = [plot_data[0, 2]]
    n_of_same_x = 1
    for x, y, z in plot_data[1:, :3]:
        if x == plot_x[-1]:
            plot_y[-1] += y
            plot_z[-1] += z
            n_of_same_x += 1
        else:
            plot_y[-1] /= n_of_same_x
            plot_z[-1] /= n_of_same_x

            plot_x.append(x)
            plot_y.append(y)
            plot_z.append(z)
            n_of_same_x = 1

    plot_y[-1] /= n_of_same_x
    plot_z[-1] /= n_of_same_x

    return plot_x, plot_y, plot_z


def plot_diagram(A_data, P_data, growth_data, mode_data, meta):
    df = pack_df_fromArrays({"A": A_data,
                             "P": P_data,
                             "growth": growth_data})

    plot_x, plot_y, plot_z = get_isoline(df.growth.to_numpy(), df.to_numpy(), meta["growth_edge"])

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(7)

    pcm = ax.pcolormesh(A_data, P_data, growth_data.T, cmap='winter')
    fig.colorbar(pcm, ax=ax)

    X, Y = np.meshgrid(A_data, P_data)
    # print(X)
    # print(Y)
    for i in range(1, P_data.shape[0]):
        for j in range(1, P_data.shape[0]):
            ax.annotate(f"{mode_data[i, j].astype(int)}",
                        (X.flat[(i - 1) * P_data.shape[0] + j - 1], Y.flat[(i - 1) * P_data.shape[0] + j]),
                        textcoords="offset points",
                        fontsize=10, xytext=(5, 7), alpha=0.6)

    # ax.plot(plot_data[:, 0], plot_data[:, 1], marker="+", label='Direct', alpha=0.8)
    # for i, (xi, yi) in enumerate(zip(plot_data[:, 0], plot_data[:, 1])):
    #     ax.annotate(f"{plot_data[i, 2]:.2f}",
    #                  (xi, yi), textcoords="offset points",
    #                  fontsize=8, xytext=(-8, 5))

    ax.plot(plot_x, plot_y, marker=".", c="orange")  # , label='Average', alpha=0.8, linewidth=.5
    for i, (xi, yi) in enumerate(zip(plot_x, plot_y)):
        ax.annotate(f"{plot_z[i]:.2f}",
                    (xi, yi), textcoords="offset points",
                    fontsize=8, xytext=(-8, (5 if i % 2 else -11)), c="k")

    # ax2.legend()
    # ax.set_xlim([df.A.min(), df.A.max()])
    # ax.set_ylim([df.P.min(), df.P.max()])

    # ax.grid(which='major', color='#DDDDDD', linewidth=0.9)
    # ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
    # ax.minorticks_on()
    # ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.set_title(
        f"$\\sigma$={meta['sigma']}, $\\kappa$={meta['kappa']}, $I$={meta['I']} $kA$, $B$={meta['B']} $T$\nIsoline by growth: {meta['growth_edge']}")

    plt.show()


def variate_points(points, n, var_x, var_y):
    rng = np.random.default_rng(seed=42)
    new_points = []
    for i in range(points.shape[0]):
        variances = 2 * rng.random((n, 2)) - 1
        variate_from_point = np.stack([points[i, 0] + variances[:, 0] * var_x, points[i, 1] + variances[:, 1] * var_y],
                                      axis=1)
        new_points.append(variate_from_point)
    return np.concatenate(new_points)
