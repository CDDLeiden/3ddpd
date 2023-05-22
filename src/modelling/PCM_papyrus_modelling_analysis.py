import os
import glob
import argparse
from itertools import chain
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import LinearLocator, AutoMinorLocator

SMALL_SIZE = 14
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def tabulate_results(pcm_qsar_folder: str, prot_desc_list: list, evaluate: str = 'Test set') -> pd.DataFrame:
    """Gather performance metrics of QSAR and PCM models.
    :param pcm_qsar_folder: folder containing the results of the QSAR and PCM modelling
    :param evaluate: part of the results to be summarized:
                       - 'Mean' for average cross-validation performance
                       - 'Test set' for performance on held-out test set
    :return: the tabulated results per descriptor and model type (QSAR results averaged per target family)
    """
    if not os.path.isdir(pcm_qsar_folder):
        raise ValueError(f'folder does not exist: {pcm_qsar_folder}')
    if not evaluate in ['Test set', 'Mean']:
        raise ValueError('evaluate value must be one of: [\'Test set\', \'Mean\']')
    results = pd.DataFrame(None,
                           columns=pd.MultiIndex.from_product([['random', 'year'],
                                                               prot_desc_list,
                                                               ['MCC', 'r', 'RMSE']]
                                                              ),
                           index=pd.MultiIndex.from_product([['QSAR', 'PCM'],
                                                            ['0','1','2','3','4','5','6','7','8','9']]
                                                            )
                           )
    for file_ in chain(glob.glob(os.path.join(pcm_qsar_folder, '*/*_results_*.tsv'))):
        # Identify the type of results
        if os.path.basename(file_).startswith('QSAR'):
            model = 'QSAR'
        elif os.path.basename(file_).startswith('PCM'):
            model = 'PCM'
        else:
            raise ValueError('model type (QSAR/PCM) could not be determined from file name')
        # Read the results
        if model == 'QSAR':
            data = pd.read_csv(file_, sep='\t', index_col=[0,1]).xs(evaluate, level=1)
        else:
            data = pd.read_csv(file_, sep='\t', index_col=[0]).xs(evaluate)
        # Identify the type of split
        if 'random' in file_:
            split = 'random'
        elif 'Year' in file_:
            split = 'year'
        else:
            raise ValueError('data split type could not be determined from file name')
        # Identify the type of protein descriptor
        if 'QSAR' in file_:
            desc = 'QSAR'
        else:
            desc = None
            for prot_desc in prot_desc_list:
                if (prot_desc in file_) and (len(prot_desc.split("_")) <= 7) and (len(os.path.basename(file_).split("_")) <= 13): # Descriptor not in combination
                    desc = prot_desc
                elif (prot_desc in file_) and (len(prot_desc.split("_")) > 7) and (len(os.path.basename(file_).split("_")) > 13): # 3DDPD descriptor in combination
                    desc = prot_desc
            if desc == None:
                # print(f'Protein descriptor not described in list to tabulate, skipping file {file_}')
                continue
                # raise ValueError('protein descriptor could not be determined from file name')
        # Identify the replicate 
        if int(os.path.basename(file_).split('_')[-1].split('.')[0]) in range(0,10):
            seed = os.path.basename(file_).split('_')[-1].split('.')[0]
        else:
            raise ValueError('model replicate could not be determined from file name')                                                
        # Identify the type of model
        if 'regression' in file_:
            metrics = [('r', 'Pearson r'), ('RMSE', 'RMSE')]
        elif 'classification' in file_:
            metrics = [('MCC', 'MCC')]
        else:
            raise ValueError('model type (regressor/classifier) could not be determined from file name')
        
        # Set the values in the right location
        for results_metric, file_metric in metrics:
            if model != 'QSAR':
                if isinstance(data.loc[file_metric], str):
                    results.loc[(model,seed), (split, desc, results_metric)] = float(data.loc[file_metric].strip('[]'))
                else:
                    results.loc[(model,seed), (split, desc, results_metric)] = data.loc[file_metric]
            else:
                results.loc[(model,seed), (split, desc, results_metric)] = data.loc[:, file_metric].mean()
    return results

def list_all_seeds(data: pd.DataFrame):
    """Prepare modelling results for plotting.
    :param data: tabulated results
    :return: results listed over all seeds for QSAR and PCM models
    """
    aggregated = pd.melt(data.reset_index(),
                         id_vars=['level_0', 'level_1']
                         ).rename(columns={'level_0': 'model',
                                           'level_1': 'seed',
                                           'variable_0': 'split',
                                           'variable_1': 'protein_descriptor',
                                           'variable_2': 'variable'})
    
    aggregated = aggregated.dropna()
        
    QSAR = aggregated.loc[(aggregated.model == 'QSAR') & (aggregated.protein_descriptor == 'QSAR')]
    PCM = aggregated.loc[(aggregated.model == 'PCM') & (aggregated.protein_descriptor != 'QSAR')]
    
    return QSAR,PCM,aggregated

def average_over_seeds(data: pd.DataFrame):
    """Compute modelling statistics.
    :param data: tabulated results
    :return: results averaged over seeds for QSAR and PCM models
    """
    QSAR,PCM,aggregated = list_all_seeds(data)

    seed_average_QSAR = QSAR.groupby(by=['model','split','protein_descriptor','variable'])['value'].agg(['mean','std']).reset_index()
    seed_average_PCM = PCM.groupby(by=['model','split','protein_descriptor','variable'])['value'].agg(['mean','std']).reset_index()
    
    return seed_average_QSAR,seed_average_PCM

def t_test(data: pd.DataFrame, desc_1: str, desc_2: str, split: str, variable: str):
    """
    Perform independent T-test of a performance metric of 10 model replicates for two different protein descriptors 
    """
    data_ttest = data.loc[(data.split == split) & (data.variable == variable)]
    data_ttest_1 = data_ttest[data_ttest['protein_descriptor'] == desc_1]['value'].to_numpy(dtype='float')
    data_ttest_2 = data_ttest[data_ttest['protein_descriptor'] == desc_2]['value'].to_numpy(dtype='float')

    ttest = ttest_ind(data_ttest_1, data_ttest_2)
    
    return ttest

def stat_significance_heatmap(data: pd.DataFrame, split: str, variable: str, protein_descriptors: list, desc_set: str='all_descriptors', save: bool=True, output_dir:str='../PCM_modelling/analysis'):
    """
    Create heatmap with statistical significance (independent t-test) between different protein descriptors
    :param data: tabulated results for each model type
    :param split: type of split to analyze results for
    :param variable: variable to analyze results for
    :param protein_descriptors: list of protein descriptors to compute heatmap for
    :param desc_set: alias of the selection of descriptors, for the figure output name
    :save: whetehr to save the output figure 
    """
    pvalue_matrix = []
    stat_matrix = []
    for prot_desc_1 in protein_descriptors:
        pvalue_list = []
        stat_list = []
        for prot_desc_2 in protein_descriptors:
            pvalue_list.append(t_test(data,prot_desc_1,prot_desc_2,split,variable)[1])
            stat_list.append(t_test(data,prot_desc_1,prot_desc_2,split,variable)[0])
            
        pvalue_matrix.append(pvalue_list)
        stat_matrix.append(stat_list)
    
    pvalue_df = pd.DataFrame(pvalue_matrix,columns=protein_descriptors,index=protein_descriptors)
    stat_df = pd.DataFrame(stat_matrix,columns=protein_descriptors,index=protein_descriptors)
    
    # Create annotations based on statistical significance (p-value)
    def stat_significance(x):
        if x == 1:
            return ''
        elif 1 > x >= 0.05:
            return ''
        elif 0.05 > x >= 0.01:
            return '*'
        elif 0.01 > x >= 0.001:
            return '**'
        else:
            return '***'
    
    df_heatmap_annot = pvalue_df.applymap(stat_significance).to_numpy()
    
    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(3000/300,3000/300), dpi=300)
    fig.set_facecolor('w')
    
    cmap = sns.diverging_palette(12, 108, s=99, l=39, as_cmap=True, center='light')
    norm = mcolors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)
    mask = np.triu(np.ones_like(stat_df, dtype=bool)) # show only lower half of the triangle 

    sns.heatmap(stat_df,annot=df_heatmap_annot, cmap=cmap, linewidth=1,linecolor='w',square=True, fmt='', mask=mask, norm=norm,
               cbar_kws={'orientation':'vertical', 'shrink':0.3, 'label':'Independent T-test statistic'})
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='grey', alpha=0.1)
    textstr = '\n'.join((
    '*: p-value < 0.05',
    '**: p-value < 0.01',
    '***: p-value < 0.001'))
        
    # place a text box in upper left in axes coords to explain annotations 
    ax.text(0.75, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.title(f'Statistical significance split {split} for variable {variable}')
    
    # Save figure
    if save:
        output_file = f'stat_significance_{split}_{variable}.svg'
        plt.savefig(os.path.join(output_dir,output_file))
    

def plot_performance_aggregated(data: pd.DataFrame, split: str,  desc_filter: Union[str, list] = None, desc_set: str='all_descriptors', zoom: str = None, save: bool=True, output_dir: str='../PCM_modelling/analysis') -> Figure:
    """Plot modelling results aggregated (MCC, Pearson r and RMSE)
    :param data: tabulated results for each model type
    :split: type of split to analyze results for
    :desc_filter: subtring to match or list of descriptors to filter from data to plot
    :desc_set: alias of the selection of descriptors, for the figure output name
    :zoom: variable to zoom into ('MCC'/'r'/'RMSE')
    :return: matplotlib figure
    """
    # Filter split of interest
    data = data[data['split'] == split]
    # Filter protein descriptors
    if desc_filter is not None:
        if isinstance(desc_filter,str):
            data = data[data['protein_descriptor'].apply(lambda x: desc_filter in x)]
        elif isinstance(desc_filter,list):
            data = data[data['protein_descriptor'].apply(lambda x: x in desc_filter)]
    # Cast input to list
    if not isinstance(data, list) and isinstance(data, pd.DataFrame):
        data = [data]
    elif not isinstance(data, list):
        raise TypeError('data must be a pandas dataframe or a list of dataframes')
    # Make scale of RMSE go from 0 to 1.5 when others go from 0 to 1.0
    scale = 2 / 3.0
    for df in data:
        mask = df.variable.isin(['RMSE'])
        df.loc[mask, 'value'] = df.loc[mask, 'value'] * scale
    
    #Plot data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8.5))
    g = sns.barplot(x='protein_descriptor', y='value', hue='variable',
                    errwidth=1, capsize=0.1, errorbar='sd',
                    data=data[0], ax=ax, palette=sns.color_palette("Set2"))
    ax.set_ylabel('$MCC$ $and$ $r_{Pearson}$')
    ax2 = ax.twinx()
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylabel('$RMSE$')
    ax.set_xlabel('')
    

    # Set limits of plot
    if not zoom:
        ax.set_ylim(0, 1.0)
        ax2.set_ylim(ax.get_ylim())
    else:
        y_min = round(data[0][data[0]['variable'] == zoom]['value'].min(), 2) - 0.005
        y_max = round(data[0][data[0]['variable'] == zoom]['value'].max(), 2) + 0.005
        ax.set_ylim(y_min, y_max)
        ax2.set_ylim(ax.get_ylim())

    ymajorLocator = LinearLocator(11)
    yminorLocator = AutoMinorLocator()
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    _ = ax2.set_yticks(ax.get_yticks())
    _ = ax2.set_yticklabels([f'${x:.2f}$' for x in ax.get_yticks() / scale])
    yminorLocator = LinearLocator(10 * 3 + 1)
    ax2.yaxis.set_minor_locator(yminorLocator)

    sns.move_legend(g, 'upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, title=None, frameon=False)
    # LaTeX-ify the legend
    for t, l in zip(g.get_legend().texts, ['$MCC$', '$Pearson$ $r$', '$RMSE$']):
        t.set_text(l)

    plt.tight_layout()
    
    # Save figure
    if save:
        output_file = f'performance_aggregated_{split}_{desc_set}.svg'
        plt.savefig(os.path.join(output_dir,output_file))


def plot_performance(data: pd.DataFrame, split: str, desc_filter: Union[str, list] = None,
                     desc_names: dict = None, desc_set: str = 'all_descriptors',
                     variable: str = 'r', zoom: bool = False, save: bool = True,
                     output_format: str = 'png', output_dir: str = '../PCM_modelling/analysis') -> Figure:
    """Plot modelling results for one variable of interest
    :param data: tabulated results for each model type
    :param split: type of split to analyze results for
    :param desc_filter: subtring to match or list of descriptors to filter from data to plot
    :param desc_names: dictionary with descriptor identifiers as keys and desired descriptor labels as values
    :param desc_set: alias of the selection of descriptors, for the figure output name
    :param variable: variable plot ('MCC'/'r'/'RMSE')
    :param zoom: whether to zoom to make subplots to later aggregate together
    :param save: whether to save the plot
    :param output_format: format to save the figure. Options are 'png' or 'svg'
    :param output_dir: directory to save the plot
    """
    # Filter split of interest
    data = data[data['split'] == split]
    # Filter variable of interest
    data = data[data['variable'] == variable]
    # Filter protein descriptors
    if desc_filter is not None:
        if isinstance(desc_filter, str):
            data = data[data['protein_descriptor'].apply(lambda x: desc_filter in x)]
            protein_descriptors = data['protein_descriptor'].unique()
        elif isinstance(desc_filter, list):
            data = data[data['protein_descriptor'].apply(lambda x: x in desc_filter)]
            protein_descriptors = desc_filter
        else:
            protein_descriptors = data['protein_descriptor'].unique()
    if desc_names is not None:
        protein_descriptors_labels = [desc_names[desc] for desc in protein_descriptors]
    else:
        protein_descriptors_lables = protein_descriptors

    # Define color palette based on protein descriptor type
    my_pal = {prot_desc: '#33c491' if '3DDPD' in prot_desc else '#2f63c4' for prot_desc in protein_descriptors}
    my_pal['QSAR'] = '#de8a14'
    hatch_color = [color if not ((color == '#33c491') & (len(desc.split('_')) > 7)) else '#2f63c4' for desc, color in
                   my_pal.items()]

    # Define outliers format
    flierprops = dict(markerfacecolor='1', marker='o', markersize=5, linestyle='none')

    # Plot data
    sns.set_style(style='ticks')
    if not zoom:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    else:
        width = len(protein_descriptors) * 0.5
        fig, ax = plt.subplots(1, 1, figsize=(width, 6), dpi=300)
    fig.set_facecolor('w')

    bp = sns.boxplot(x='protein_descriptor', y='value',
                     data=data, ax=ax, palette=my_pal, flierprops=flierprops,
                     order=protein_descriptors)

    # Apply colors
    plt.rcParams["hatch.linewidth"] = 4
    for box, color1, color2 in zip(bp.patches, list(my_pal.values()), hatch_color):
        box.set(facecolor=color2, edgecolor=color1)
        box.set_hatch('//')

    # Set axes
    if zoom:
        ax.set_ylim(0.395, 0.430)
    sns.despine(offset=10, trim=True)
    variable_tag = variable if variable != 'r' else '$r_{Pearson}$'
    ax.set_ylabel(variable_tag)
    ax.set_xticklabels(protein_descriptors_labels, rotation=45, ha='right')
    ax.set_xlabel('')

    plt.tight_layout()
    # Save figure
    if save:
        # Best not to save as SVG because hatching messes up the formatting
        output_file = f'performance_{split}_{desc_set}_{variable}.{output_format}'
        plt.savefig(os.path.join(output_dir, output_file), dpi=300)


def plot_optimization_performance(data: pd.DataFrame, split: str, desc_filter: Union[str, list], desc_set: str,
                                  optimization_strategy: dict, variable: str, plot_significance: bool, save: bool,
                                  output_format: str, output_dir: str):
    """Plot optimization modelling results for one variable of interest
        :param data: tabulated results for each model type
        :param split: type of split to analyze results for
        :param desc_filter: subtring to match or list of descriptors to filter from data to plot
        :param desc_set: alias of the selection of descriptors, for the figure output name
        :param optimization_strategy: dictionary with optimization steps as keys and a list of tuples with descriptor
                                    identifiers and descriptor labels for all descriptors used in that optimization step
        :param variable: variable plot ('MCC'/'r'/'RMSE')
        :param plot_significance: whether to plot statistical significance within each optimization step
        :param save: whether to save the plot
        :param output_format: format to save the figure. Options are 'png' or 'svg'
        :param output_dir: directory to save the plot
        """
    # Filter data of interest
    data = data[(data['split'] == split) & (data['variable'] == variable)]

    # Initialize plot
    n_subplots = len(optimization_strategy.keys())
    descriptors_in_strategy = [len(descriptors) for descriptors in optimization_strategy.values()]
    fig, axes = plt.subplots(1, n_subplots, figsize=(10, 5), gridspec_kw={'width_ratios': descriptors_in_strategy},
                             dpi=300)

    # Define color palette based on protein descriptor type
    my_pal = {prot_desc: '#33c491' if '3DDPD' in prot_desc else '#2f63c4' for prot_desc in
              data.protein_descriptor.unique()}
    my_pal['QSAR'] = '#de8a14'
    hatch_color = [color if not ((color == '#33c491') & (len(desc.split('_')) > 7)) else '#2f63c4' for desc, color in
                   my_pal.items()]

    # Define outliers format
    flierprops = dict(markerfacecolor='1', marker='o', markersize=5, linestyle='none')

    # Get minimum and maximum values
    y_min = data[data['protein_descriptor'].isin(desc_filter)]['value'].min()
    y_max = data[data['protein_descriptor'].isin(desc_filter)]['value'].max()

    # Plot each strategy step in one suplot
    def plot_individual_performance(strategy, axis):
        # Extract data for the strategy of interest
        descriptors = [descriptor[0] for descriptor in optimization_strategy[strategy]]
        descriptor_labels = [descriptor[1] for descriptor in optimization_strategy[strategy]]
        strategy_df_list = []
        for prot_desc in desc_filter:
            df_desc = data[data['protein_descriptor'] == prot_desc]
            if prot_desc in descriptors:
                df_desc['strategy'] = strategy
                strategy_df_list.append(df_desc)
        strategy_df = pd.concat(strategy_df_list)

        # Plot data
        axis.set_xlim(0, len(descriptors) - 1)
        bp = sns.boxplot(x='protein_descriptor', y='value',
                         data=strategy_df, ax=axis, order=descriptors,
                         palette=my_pal, flierprops=flierprops, width=0.8)

        # Apply colors
        plt.rcParams["hatch.linewidth"] = 4
        for box, color1, color2 in zip(bp.patches, list(my_pal.values()), hatch_color):
            box.set(facecolor=color2, edgecolor=color1, hatch=r'//')

        # Set axes and titles
        axis.set_title(strategy)
        axis.set_ylim(y_min, y_max)
        sns.despine(offset=10, trim=True, ax=axis)
        axis.set_xticklabels(descriptor_labels, rotation=45, ha='right')
        axis.set_xlabel('')
        variable_tag = variable if variable != 'r' else '$r_{Pearson}$'
        axis.set_ylabel(variable_tag)

        # Plot statistical significance
        if plot_significance:
            axis.set_ylim(y_min, y_max + 0.001 * max(descriptors_in_strategy))
            import itertools
            from statistics import mean
            def stat_significance(x):
                if x == 1:
                    return 'ns'
                elif 1 > x >= 0.05:
                    return 'ns'
                elif 0.05 > x >= 0.01:
                    return '*'
                elif 0.01 > x >= 0.001:
                    return '**'
                else:
                    return '***'

            for desc1, desc2 in itertools.combinations(descriptors, 2):
                p_value = t_test(strategy_df, desc1, desc2, split, variable)[1]
                significance_label = stat_significance(p_value)
                desc1_x = descriptors.index(desc1)
                desc2_x = descriptors.index(desc2)
                y_pos = y_max + 0.001 * (min(desc1_x, desc2_x) + 1)

                if significance_label != 'ns':
                    axis.plot([min(desc1_x, desc2_x), max(desc1_x, desc2_x)], [y_pos, y_pos], color='black')
                    axis.plot([min(desc1_x, desc2_x), min(desc1_x, desc2_x)], [y_pos - 0.0005, y_pos], color='black')
                    axis.plot([max(desc1_x, desc2_x), max(desc1_x, desc2_x)], [y_pos - 0.0005, y_pos], color='black')
                    axis.text(x=mean([desc1_x, desc2_x]), y=y_pos + 0.000005, s=significance_label, ha='center')

        return fig

    for i, strategy in enumerate(optimization_strategy.keys()):
        plot_individual_performance(strategy, axes[i])
    fig.subplots_adjust(wspace=0.8)
    for i, ax in enumerate(axes.flat):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.set_ylabel('')

    if save:
        output_file = f'performance_{desc_set}.{output_format}'
        plt.savefig(os.path.join(output_dir, output_file), dpi=300)

def read_training_feature_importance(model_dir:str, model_name: str):
    """
    Load pre-trained model and return feature importance for the Full model
    """
    loaded_model = joblib.load(os.path.join(model_dir, model_name))
    forest = loaded_model['Full model']
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=forest.feature_names_in_)
    
    return forest_importances

def plot_feature_importance(model_dir:str, model_type: str, protein_descriptor: str, split_by: str, top=25, save=True, output_dir: str='../PCM_modelling/analysis'):
    
    df_dict = {}
    for seed in ['0','1','2','3','4','5','6','7','8','9']:
        model_name = f'PCM_results_{model_type}_ECFP_{protein_descriptor}_{split_by}_{seed}.joblib.xz'
        feature_importance_seed = read_training_feature_importance(model_dir,model_name)
        df_dict[seed] = feature_importance_seed
    
    df = pd.DataFrame(df_dict)
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    
    df.sort_values(by='mean', ascending=False, inplace=True)
    
    df_top = df.head(top)

    sns.set_style(style='ticks')
    if top <= 25:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=300)
    elif top > 25:
        fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=300)
    fig.set_facecolor('w')

    features = df_top.index.tolist()
    importance = df_top['mean'].tolist()
    importance_sd = df_top['std'].tolist()
    y_pos = np.arange(len(features))
    # color features by feature type
    color_feature =[]
    custom_lines = [Line2D([0], [0], color='#33c491', lw=3),
                    Line2D([0], [0], color='#2f63c4', lw=3),
                    Line2D([0], [0], color='#de8a14', lw=3)]
    custom_labels = ['3DDPD protein descriptor', 'Classical protein descriptor', 'Compound fingerprint']
    for feature in features:
        if 'PC' in feature:
            c = '#33c491' # green; md_3ddpd
            custom_lines.append(Line2D([0], [0], color=c, lw=3))
        # elif 'AA' in feature:
        #     c = '#2f63c4' # blue; classical protein descriptor
        elif 'ECFP6' in feature:
            c = '#de8a14' # light orange; compound fingerprint
        else:
            c = '#2f63c4'  # blue; classical protein descriptor
        color_feature.append(c)

    barlist = ax.barh(y_pos, importance, xerr=importance_sd, align='center')
    for i,bar in enumerate(barlist):
        barlist[i].set_color(color_feature[i])
    ax.set_yticks(y_pos, labels=features)
    ax.invert_yaxis()

    if top > 25:
        plt.tick_params(axis='x', which='major', labelsize=16)
        ax.legend(custom_lines, custom_labels, fontsize=16)
        ax.set_xlabel('Feature importance', fontsize=14)

    else:
        ax.legend(custom_lines, custom_labels)
        ax.set_xlabel('Feature importance')

    sns.despine(offset=10, trim=True)

    plt.tight_layout()

    if save:
        output_file =  f'feature_importance_top{top}_{model_type}_{protein_descriptor}_{split_by}.svg'
        plt.savefig(os.path.join(output_dir, output_file))

