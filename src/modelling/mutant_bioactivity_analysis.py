import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 50)

mutant_dir = 'C:\\Users\gorostiolam\Documents\Gorostiola Gonzalez, Marina\PROJECTS\\3_GPCRS_3DDPDs\DATA\\2_Analysis\manuscript\\3_mutant_bioactivity'


entries_gpcrmd_mutants = '165_aa1r_wt_1,49_aa2ar_wt_1,111_acm2_wt_1,11_adrb2_wt_1,118_ccr5_wt_1,' \
                         '49_aa2ar_S91A_1,49_aa2ar_S277A_1,49_aa2ar_M177A_1,49_aa2ar_N253A_1,49_aa2ar_L85A_1,49_aa2ar_L167A_1,49_aa2ar_N181A_1,49_aa2ar_I66A_1,49_aa2ar_Y271A_1,49_aa2ar_T88D_1,' \
                         '165_aa1r_T277A_1,165_aa1r_R296C_1,165_aa1r_R291C_1,' \
                         '11_adrb2_S204A_1,11_adrb2_N293L_1,11_adrb2_S203A_1,11_adrb2_D130N_1,11_adrb2_D79N_1,11_adrb2_V317A_1,' \
                         '118_ccr5_Y108A_1,' \
                         '111_acm2_D103E_1,111_acm2_D103N_1,111_acm2_V421L_1'

gpcrdb_mut = pd.read_csv(os.path.join(mutant_dir, 'gpcrdb_classA_mutations.csv'))
gpcrdb_mut['mutation'] = gpcrdb_mut.apply(lambda x: f'{x["mutation_from"]}{x["mutation_pos"]}{x["mutation_to"]}', axis=1)
gpcrdb_mut['target'] = gpcrdb_mut['protein'].str.replace('_human','')
gpcrdb_mut['target_mutation'] = gpcrdb_mut.apply(lambda x: f'{x["target"]}_{x["mutation"]}', axis=1)

# Filter mutants in list
gpcrdb_mut_sel = gpcrdb_mut[gpcrdb_mut['target_mutation'].isin(['_'.join(x.split('_')[1:3]) for x in entries_gpcrmd_mutants.split(',')])]
# Drop datapoints with Fold change 0 (this means no available data)
gpcrdb_mut_sel = gpcrdb_mut_sel[gpcrdb_mut_sel['exp_fold_change'] != 0]
gpcrdb_mut_sel_plot = gpcrdb_mut_sel[['protein','target_mutation','ligand_name','ligand_class', 'exp_type','exp_func','exp_fold_change']]

# gpcrdb_mut_sel_plot['exp_fold_change'] = gpcrdb_mut_sel_plot['exp_fold_change'].apply(lambda x: 30.0 if x > 30.0 else x)
# gpcrdb_mut_sel_plot['exp_fold_change'] = gpcrdb_mut_sel_plot['exp_fold_change'].apply(lambda x: -30.0 if x < -30.0 else x)
def discretize_fold_change(x):
    if x >= 30.0:
        return '>= 30'
    elif 2 <= x < 30:
        return '2 - 30'
    elif -2 <= x < 2:
        return 'No change'
    elif -30 <= x < -2:
        return '-30 - -2'
    else:
        return '< -30'
gpcrdb_mut_sel_plot['exp_fold_change_disc'] = gpcrdb_mut_sel_plot['exp_fold_change'].apply(discretize_fold_change)

palette_dict = {'aa1r_human': 'Blues',
                'aa2ar_human': 'Blues',
                'acm2_human': 'YlOrBr',
                'adrb2_human': 'Greys',
                'ccr5_human': sns.set_palette(sns.color_palette(['#ebe4ab'])),
                }
for protein in gpcrdb_mut_sel_plot['protein'].unique().tolist():
    df_protein = gpcrdb_mut_sel_plot[gpcrdb_mut_sel_plot['protein'] == protein]
    df_protein['exp_fold_change_disc'] = pd.Categorical(df_protein['exp_fold_change_disc'], ['< -30', '-30 - -2', 'No change', '2 - 30', '>= 30'])
    sns.histplot(data=df_protein, x='exp_fold_change_disc',  hue='target_mutation', multiple='stack',  palette=palette_dict[protein])
    plt.xlabel('Experimental Fold Change')
    plt.show()