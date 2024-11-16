import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_bar_charts(data, x, y, hue, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=hue)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def plot_line_charts(data, x, y, hue, title, xlabel, ylabel, save_path=None):

    sns.lineplot(data=data, x=x, y=y, hue=hue, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=hue)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def plot_heatmap(data, x, y, z, title, xlabel, ylabel, save_path=None):
    pivot_table = data.pivot(index=y, columns=x, values=z)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def plot_sample_ratio_line_chart(data, x, y1, y2, title, xlabel, ylabel1, ylabel2, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x], data[y1], marker='o', label=ylabel1)
    plt.plot(data[x], data[y2], marker='s', label=ylabel2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('MAF 分数')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
