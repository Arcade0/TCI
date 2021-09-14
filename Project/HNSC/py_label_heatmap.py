import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def label_heatmap(dfi,
                  color_pal,
                  dfi_labels,
                  rc_labels,
                  dfi_pal,
                  rc_pal,
                  df_legend_position=(0, 0.4),
                  rc_legend_position=(0, 0.57),
                  label_column="subtype",
                  continous=False,
                  cbar_location=(0.15, 0.4, 0.02, 0.1),
                  r_c=False,
                  c_c=False,
                  y_t=False,
                  x_t=False,
                  show_tick=False,
                  tick_l=None,
                  col_name="Protein",
                  row_name="Sample",
                  dfi_legend_title="Protein State",
                  rc_legend_title="Subtype",
                  figure_name="Test.png",
                  dp=600):
    """Notes for label_heatmap.

    A modification which is based on sns.clustermap.

    Args:
        dfi: 2-D dfi, index are sample IDs, columns are SGAs or gene names.
        dfi's value is 0 or 1.It needs contain a columns with label information
        in the last column.
        color_pal:A list of color. Used in sns.clustermap, value of "cmap".
        dfi_labels, rc_labels: List of elements in heatmap and labeled rows legend.
        dfi_pal, rc_pal: List of elements' colors in heatmap and labeled rows legend.
        df_legend_position:
        rc_legend_position:
        label_column:
        continous: Bool value, data type.
        cbar_location: (x, y, width, height) of color bar.
        r_c, c_c, y_t, x_t: Bool value, True or False, parameters which are used in
        sns.clustermap. values of "row_cluster", "col_cluster", "yticklabels", "xticklabels".
        Default is False.
        show_tick: Bool value, if show some special x ticks. Default is False.
        tick_l: A list of x tick labels. Default is None.
        col_name, row_name: String.
        dfi_legend_title, rc_legend_title: String.
        figure_name: String. File names of the figure used to save figure.
        dp: int, dpi used in save figure.

    Returns:
        Save a figure heatmap with rows are labeled to the path.
    """

    # set overall font style
    plt.rc('font', family='Times New Roman')

    # set heatmap color panel
    dfi_lut = dict(zip(dfi_labels, dfi_pal))  # one by one

    # set row_color panel, this two line is for tcga paper
    #     labs = ['Atypical','Basal','Classical','Mesenchymal']
    #     rc_lut = dict(zip(labs, rc_pal))  # one by one
    rc_lut = dict(zip(rc_labels.unique(), rc_pal))  # one by one
    rc_colors = rc_labels.map(rc_lut)  # lut to all labels

    # plot step
    g = sns.clustermap(
        dfi.drop(label_column, axis=1),
        figsize=(1.8, 1.8),
        row_cluster=r_c,
        col_cluster=c_c,
        yticklabels=y_t,
        xticklabels=x_t,
        row_colors=[rc_colors],
        # Add colored class labels using data frame created from node and network colors
        cmap=color_pal)  # Make the plot look better when many rows/cols

    ax0 = g.ax_heatmap
    ax0.set_xlabel(col_name, fontsize=10)
    ax0.set_ylabel(row_name, fontsize=10)

    # show some special gene
    if show_tick is True:
        if c_c is False:
            b = list(dfi.columns)
        else:
            b = list(dfi.iloc[:, g.dendrogram_col.reordered_ind].columns)
        print(b)

        c = set(b) & set(tick_l)
        d = [b.index(ele) for ele in c]
        ax0.set_xticks(d)
        ax0.set_xticklabels(c, rotation=90, fontsize=6)

    ax1 = g.cax

    # set legend of heatmap if the data is discrete data, continous data legend in ax4
    if continous is False:
        for label in dfi_labels:
            ax0.bar(0, 0, color=dfi_lut[label], label=label, linewidth=0)
        ax0_legend = ax0.legend(loc="center",
                                ncol=1,
                                bbox_transform=plt.gcf().transFigure,
                                bbox_to_anchor=df_legend_position,
                                prop={'size': 6})
        ax0_legend.set_title(dfi_legend_title, prop={'size': 6})
        ax1.set_visible(False)
    else:
        ax1.set_visible(True)
        ax1.set_title("Expression", fontsize=6)
        min_v = np.min(np.min(dfi.drop(label_column, axis=1)))
        max_v = np.max(np.max(dfi.drop(label_column, axis=1)))
        ax1.yaxis.set_ticks([min_v, (min_v + max_v) / 2, max_v])
        ax1.yaxis.set_ticklabels(["Low", "Normal", "High"], fontsize=6)
        ax1.set_position(cbar_location)

    # set legend of row color bars
    ax2 = g.ax_row_colors
    for label in rc_labels.unique():
        #     for label in labs:
        ax2.bar(0, 0, color=rc_lut[label], label=label, linewidth=0)
    ax2_legend = ax2.legend(loc="center",
                            ncol=1,
                            bbox_transform=plt.gcf().transFigure,
                            bbox_to_anchor=rc_legend_position,
                            prop={'size': 6})
    ax2_legend.set_title(rc_legend_title, prop={'size': 6})

    ax3 = g.ax_row_dendrogram
    ax3.set_visible(False)

    ax4 = g.ax_col_dendrogram
    ax4.set_visible(False)

    g.savefig(figure_name, dpi=dp)


# dfi_labels_2 = dfi["path:" + sga]
# dfi_pal_2 = sns.cubehelix_palette(dfi_labels_2.unique().size, light=.9, dark=.1, reverse=True, start=1, rot=-2)
# dfi_lut_2 = dict(zip(dfi_labels_2.unique(), dfi_pal_2))
# dfi_colors_2 = dfi_labels_2.map(dfi_lut_2)

# dfi_labels_3 = dfi["deg:" + sga]
# dfi_colors_3 = dfi_labels_3.map(dfi_lut_2)

# ax1.legend(
#     title="path: %s States" % sga, loc="lower left", ncol=1,
#     bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1.0, 0.4))

# ax2 = g.ax_row_colors
# for label in dfi_labels_2.unique():
#     ax2.bar(0, 0, color=dfi_lut_2[label], label=label, linewidth=0)
# legend2 = ax2.legend(
#     title="%s state" % sga, loc="lower left", ncol=1,
#     bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1.0, 0.2))
# ax2.set_xticklabels(["TCGA","DEG"])
# ax2.set_xticks([0.5,1.5])
