import matplotlib
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
warnings.filterwarnings("ignore")


def hist_plot(df, params=None):
    if not params:
        params = dict(x='Amount', data=df, color='red')

    sns.histplot(**params)
    plt.show()


def scatter_plot(df, params=None):
    sns.set_style('dark')

    if not params:
        params = dict(x='Time', y='Amount', data=df,  hue='Class')

    sns.scatterplot(**params)
    plt.show()


def line_plot(df, params=None):
    if not params:
        params = dict(x='Time', y='Amount', data=df, ci=None, lw=2, color='#aa00aa', alpha=0.6)

    sns.lineplot(**params)
    plt.show()


def dist_plot(column, title):
    plt.figure(figsize=(6, 6))
    sns.distplot(column)
    plt.title(title)
    plt.show()


def countplot(df, params=None):
    if not params:
        params = dict(x='sex', data=df)

    sns.countplot(**params)
    plt.show()





