import pandas as pd
import os
import json
from scipy.stats import spearmanr
from collections import Counter
import seaborn as sns
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.rcParams['font.size'] = 16

data_git = os.path.dirname(__file__)+'/../'
csv_path = data_git+'/Annotations/table.csv'
TT_names = json.load(open(data_git+'TT_labels.json'))

path_fig = 'T:/Studies/Uveitis/data/Statistics'
path_agreement = 'T:/Studies/Uveitis/data/Agreement_plots'
main_items = ['optic disk hyperfluorescence', 'macular edema', 'vascular leakage', 'capillary leakage']
main_periphery = main_items + ['vascular leakage (periphery 2)', 'capillary leakage (score 4)']
annotators = ['Teodora', 'Muriel', 'Shalini']

def distribution(item, split_filter='Shalini'):
    df = df_used(pd.read_csv(csv_path,index_col=0)[['key', item, split_filter]])
    df = df[df[split_filter]!='excluded']
    data_counts = Counter(df[item])
    outcomes = TT_names['short_labels'][item] if item in TT_names['short_labels'] else TT_names[item]
    heights = [data_counts[label] for label in TT_names[item]]
    plt.bar(outcomes, heights, edgecolor='black')
    x_pos = np.arange(len(outcomes))
    for i in range(len(heights)):
        plt.text(x_pos[i], heights[i]+10, str(heights[i]), ha='center')
    plt.gca().axes.get_yaxis().set_visible(False)
    for loc in ['top', 'right', 'left']:
        plt.gca().spines[loc].set_visible(False)
    plt.xlabel(item)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path_fig+'/distributions/'+item+'.png')
    plt.show()
    
def show_distributions():
    for item in main_items:
        distribution(item)
        
def plot_items(item1, item2, style='heat'):
    if item1 != item2:
        df = df_used(pd.read_csv(csv_path, index_col=0)[['key', item1, item2]])
        plt.xlabel(item1), plt.ylabel(item2)
        plot(df, item1, item2, style, item1+' -- '+item2)
    
def plot(df, item1, item2, style, name):
    if style=='heat':
        heatmap(df, item1, item2)
    else:
        scatter(df, item1, item2, style)
    plt.xticks(rotation=20)
    plt.savefig(path_fig+'/correlations/'+name, bbox_inches='tight')
    plt.show()
    
def heatmap(df, item1, item2):
    list1, list2 = list(df[item1]), list(df[item2])
    uniques1 = sorted(np.unique(list1), key=points) 
    uniques2 = sorted(np.unique(list2), key=points)
    matrix = confusion_matrix(list1, list2, len(uniques1), len(uniques2))
    plot_matrix(matrix, uniques1, uniques2)

def confusion_matrix(list1, list2, n1, n2):
    matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            matrix[i, j] = sum(1 for x, y in zip(list1, list2) if (x, y)==(i, j))
    return matrix

def matrix_from_predictions(labels, preds, item, save_folder):
    outcomes = TT_names['short_labels'][item] if item in TT_names['short_labels'] else TT_names[item]
    matrix = confusion_matrix(labels, preds, len(outcomes), len(outcomes))
    vmax = [83, 72, 84, 84][(annotators+['model']).index(save_folder)]
    plot_matrix(matrix, outcomes, outcomes, vmax=vmax)
    plt.gca().yaxis.set_label_position("right")
    plt.ylabel('senior uveitis expert', fontweight='bold')
    plt.gca().xaxis.set_label_position("top")
    xlabel = f'Grader {annotators.index(save_folder)+1}' if save_folder in annotators else 'model predictions'
    plt.xlabel(xlabel, fontweight='bold')
    plt.title(item, pad=35, fontsize=20)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    add_border(len(outcomes))
    np.save(save_folder+f'/{item}.npy', matrix)
    plt.savefig(save_folder+f'/{item}.png', bbox_inches='tight')
    plt.close()

def plot_matrix(matrix, uniques1, uniques2, vmax=''):
    n1, n2 = len(uniques1), len(uniques2)
    norm = mpl.colors.PowerNorm(gamma=0.4, vmin=0, vmax=matrix.max() if not vmax else vmax)
    plt.imshow(matrix, cmap=plt.get_cmap('OrRd'), interpolation='nearest', norm=norm)
    cbar = plt.colorbar(pad=0.15)
    normalized_ticks = np.linspace(0, 1, 5)
    cbar_ticks = np.unique([int(round(tick)) for tick in norm.inverse(normalized_ticks)])
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([str(tick) for tick in cbar_ticks])
    plt.xticks(np.arange(n2), uniques2)
    plt.yticks(np.arange(n1), uniques1)
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=20, ha='right')
    for i in range(n1):
        for j in range(n2):
            color = 'black' if i+j else 'white'
            plt.text(j, i, str(int(matrix[i, j])), ha='center', va='center', color=color, fontsize=12)

def add_border(n, a_0=0.02, b=0.5, c=1.33):
    ax = plt.gca()
    f = lambda z: a_0 if z==0 else (n-c*a_0 if z==n else z)
    def line(x, y, dir):
        ax.add_patch(Rectangle((f(x)-b, f(y)-b), dir=='h', dir=='v', fill=False, edgecolor='black', lw=3))
    line(0, 0, 'h'), line(0, 0, 'v')
    line(n-1, n, 'h'), line(n, n-1, 'v')
    for i in range(n):
        line(i+1, i, 'h'), line(i+2, i, 'v')
        line(i-1, i+1, 'h'), line(i-1, i, 'v')

def scatter(df, item1, item2, style):
    df_score = score(df)
    n = len(df_score[item1])
    (coef, p_value) = spearmanr(df_score[item1], df_score[item2])
    plt.scatter(df_score[item1], df_score[item2], s=0)
    if type(df[item1][0])==type(''):
        plt.xticks(np.unique(df_score[item1]), sorted(np.unique(df[item1]), key=points))
    if type(df[item2][0])==type(''):
        plt.yticks(np.unique(df_score[item2]), sorted(np.unique(df[item2]), key=points))
    if style == 'dots':
        df_score[item1] += np.random.normal(scale=0.05, size=n)
        df_score[item2] += np.random.normal(scale=0.05, size=n)
        plt.scatter(df_score[item1], df_score[item2], c='black', s=10)
    else:
        dots = [(df_score[item1][i], df_score[item2][i]) for i in range(n)]
        uniques = list(set(dots))
        sizes = np.array([dots.count(dot) for dot in uniques])
        plt.scatter([dot[0] for dot in uniques], [dot[1] for dot in uniques], 
                    s=5*sizes, cmap=plt.get_cmap('Purples'), c=np.log(sizes+10))
    exp = int(np.log(p_value)/np.log(10))
    p_value /= 10**exp
    plt.title(f"correlation: r={coef:.2f}, p-value: {p_value:.2f} e{exp}")
    
def boxplot(item1, item2):
    df = df_used(pd.read_csv(csv_path, index_col=0)[['key', item1, item2]])
    M_dist = []
    df_score = score(df)
    for answer in np.unique(df_score[item1]):
        M_dist.append(list(df_score[item2][df_score[item1]==answer]))
    sns.boxplot(data=M_dist)
    for (column, label) in [(df_score[item1], df[item1]), (df_score[item2], df[item2])]:
        if type(label[0])==type(''):
            plt.xticks(np.unique(column), sorted(np.unique(label), key=points))
    plt.xlabel(item1), plt.ylabel(item2)
    plt.xticks(rotation=20)
    plt.savefig(path_fig+'/correlations/'+item1+'   '+item2+' (box plot)')
    plt.show()

def points(answer):
    for k, answers in enumerate(TT_names['T-T score']):
        if answer in answers:
            return k
    return int(answer)

def df_used(df):
    df = df.iloc[[i for i in range(len(df)) if not df['key'][i].endswith('Ex')],:]
    for na in ['na', 'not assessable']:
        df = df.loc[~(df==na).any(axis=1)]
    return df.dropna()

def score(df):
    for k, answers in enumerate(TT_names['T-T score']):
        for answer in answers:
            df = df.replace(to_replace=answer, value=k)
    return df
    
def correlation(split_filter='Shalini', periphery=False):
    mpl.rcParams['font.size'] = 12
    df = df_used(pd.read_csv(csv_path, index_col=0))
    items = main_periphery if periphery else main_items
    df = df[df[split_filter]!='excluded'][items]
    df.rename(columns={'vascular leakage (periphery 2)': 'periph. vas. leakage',
                       'capillary leakage (score 4)': 'periph. cap. leakage'}, inplace=True)
    df = df.replace('none', 0).replace('0', 0)
    df = df.applymap(lambda x: 1 if x != 0 else 0)
    map_plot = sns.clustermap if periphery else sns.heatmap
    map_plot(df.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(path_fig+'/correlation_matrix'+'_periphery'*periphery+'.png')

def all_TT_corr():
    for item in main_items:
        plot_items(item,'T-T score', style='dots')
        boxplot(item,'T-T score')

def plot_result_table(df, name):
    df = df.applymap(lambda x: f"{x:.3f}")
    _, ax = plt.subplots(figsize=(10, 5)) 
    ax.xaxis.set_visible(False), ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    cell_name = f'Grader {annotators.index(name)+1}' if name in annotators else name
    table.add_cell(0, -1, width=0.2, height=0.045, text=cell_name, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.tight_layout()
    image_path = f'{data_git}/Metrics/table/{name}.png'
    plt.savefig(image_path, bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.close()
    img = Image.open(image_path)
    img_array = np.array(img)
    non_white_pixels = np.where(np.any(img_array[:, :, :3]<255, axis=-1))
    top_left = np.min(non_white_pixels, axis=1)
    bottom_right = np.max(non_white_pixels, axis=1)
    cropped_img = img.crop((top_left[1], top_left[0], bottom_right[1]+1, bottom_right[0]+1))
    cropped_img.save(image_path)