#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:37:11 2023

@author: Yohann
"""
#%%
from scipy.stats import chi2, chi, norm
#from scipy.special import gamma
import autograd.numpy as np
#from autograd import elementwise_grad as egrad
import seaborn as sb
import matplotlib.pyplot as plt
#from matplotlib import cm
#from tqdm import tqdm
import pandas as pd
#from datetime import datetime
import os
#%%

#read the path
file_path = "/Users/decastro/Dropbox/0_Yohann/0_Articles/Article_Locura_Tensor/General case/EXPERIMENTS"
#list all the files from the directory
file_list = os.listdir(file_path)
#file_list.remove('.DS_Store')

#%%
df_append = pd.DataFrame()
#append all files together
for file in file_list:
            df_temp = pd.read_csv(file)
            df_append = df_append.append(df_temp, ignore_index=True)
            #df_append = pd.concat([df_append, df_temp], ignore_index=True)
df_append


#%%
cols = ['spacing_pvalue', 'tspacing_pvalue'] # The columns you want to search for outliers in

# Calculate quantiles and IQR

# Return a boolean array of the rows with (any) non-outlier column values
condition = ~((df_append[cols] < 0) | (df_append[cols] > 1)).any(axis=1)

# Filter our dataframe based on condition
filtered_expe = df_append[condition]


condition2 = ~((filtered_expe['sigma_estimate'] > 30))

#filtered_expe = filtered_expe[condition2]

#%%

condition3 = ((filtered_expe['alpha'] == 1) 
                        | (filtered_expe['alpha'] == 0)
                        | (filtered_expe['alpha'] == 0.5)
                        | (filtered_expe['alpha'] == 1.5)
                        | (filtered_expe['alpha'] == 2)
                        | (filtered_expe['alpha'] == 2.5)
                        | (filtered_expe['alpha'] == 3)
                        #| (filtered_expe['alpha'] == 3.5)
                        | (filtered_expe['alpha'] == 4)
                        | (filtered_expe['alpha'] == 5))

condition3bis = ((filtered_expe['alpha'] == 1) 
                        | (filtered_expe['alpha'] == 0)
                        #| (filtered_expe['alpha'] == 0.2)
                        | (filtered_expe['alpha'] == 0.5)
                        #| (filtered_expe['alpha'] == 0.7)
                        #| (filtered_expe['alpha'] == 1.2)
                        | (filtered_expe['alpha'] == 1.5)
                        | (filtered_expe['alpha'] == 2)
                        | (filtered_expe['alpha'] == 2.5)
                        | (filtered_expe['alpha'] == 3)
                        | (filtered_expe['alpha'] == 3.5)
                        | (filtered_expe['alpha'] == 4)
                        | (filtered_expe['alpha'] == 4.5)
                        | (filtered_expe['alpha'] == 5))

#%%
print(np.sum((df_append['alpha'] == 0)),
      np.sum((df_append['alpha'] == 0.5)),
      np.sum((df_append['alpha'] == 1.5)),
      np.sum((df_append['alpha'] == 2)),
      np.sum((df_append['alpha'] == 2.5)),
      np.sum((df_append['alpha'] == 3)),
      np.sum((df_append['alpha'] == 4)),
      np.sum((df_append['alpha'] == 5)))

print(np.sum((filtered_expe['alpha'] == 0)),
      np.sum((filtered_expe['alpha'] == 0.5)),
      np.sum((filtered_expe['alpha'] == 1.5)),
      np.sum((filtered_expe['alpha'] == 2)),
      np.sum((filtered_expe['alpha'] == 2.5)),
      np.sum((filtered_expe['alpha'] == 3)),
      np.sum((filtered_expe['alpha'] == 4)),
      np.sum((filtered_expe['alpha'] == 5)))

print(np.sum((filtered_expe['alpha'] == 0))/np.sum((df_append['alpha'] == 0)),
      np.sum((filtered_expe['alpha'] == 0.5))/np.sum((df_append['alpha'] == 0.5)),
      np.sum((filtered_expe['alpha'] == 1.5))/np.sum((df_append['alpha'] == 1.5)),
      np.sum((filtered_expe['alpha'] == 2))/np.sum((df_append['alpha'] == 2)),
      np.sum((filtered_expe['alpha'] == 2.5))/np.sum((df_append['alpha'] == 2.5)),
      np.sum((filtered_expe['alpha'] == 3))/np.sum((df_append['alpha'] == 3)),
      np.sum((filtered_expe['alpha'] == 4))/np.sum((df_append['alpha'] == 4)),
      np.sum((filtered_expe['alpha'] == 5))/np.sum((df_append['alpha'] == 5)))

#%%

sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["spacing_pvalue"])
plt.show()
sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["tspacing_pvalue"])
plt.show()
sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["distance_t0t1"])
plt.show()
sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["distance_t1t2"])
plt.show()
sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["sigma_estimate"])
plt.show()
sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["det_R"])
plt.show()
sb.violinplot(x=filtered_expe["alpha"], y=filtered_expe["trace_R"])
plt.show()
sb.scatterplot(x=filtered_expe["spacing_pvalue"], y=filtered_expe["tspacing_pvalue"], hue=filtered_expe['alpha'])
plt.show()

#%%

# df_pvalues = pd.DataFrame({'Spacing': filtered_expe["spacing_pvalue"].values,
#                    't-Spacing': filtered_expe["spacing_pvalue"].values})

# df_pvalues_melt = df_pvalues.melt().assign(x='p-value')

# sb.violinplot(data=df_pvalues_melt, x='x', y='value', 
#                hue='variable', split=True, inner='quart')

#%%

df_long_pvalues = filtered_expe[condition3].melt(id_vars=['alpha'], value_vars=['spacing_pvalue', 'tspacing_pvalue'],
                  var_name=r'$p$-value', value_name='value')

df_long_pvalues.rename(columns={'alpha': r'$\gamma$'}, inplace=True)


sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_context("paper")
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(8, 4))
sb.violinplot(ax=ax,
               data=df_long_pvalues,
               x=r'$\gamma$',
               y='value',
               hue=r'$p$-value',
               palette='summer',
               inner = 'quartile',
               scale='count',
               split=True)
ax.set(xlabel=r'$\gamma$', ylabel=r'$p$-value')
ax.set_ylim([0,1])
ax.legend(handles=ax.legend_.legend_handles, labels=['Spacing', r'$t$-Spacing'])
sb.despine()
plt.tight_layout()
plt.show()

#%%

df_long_distances = filtered_expe[condition3].melt(id_vars=['alpha'], value_vars=['distance_t0t1', 'distance_t1t2'],
                  var_name='distance', value_name='value')

df_long_distances.rename(columns={'alpha': r'$\gamma$'}, inplace=True)

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_context("paper")
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(8, 4))
sb.violinplot(ax=ax,
               data=df_long_distances,
               x=r'$\gamma$',
               y='value',
               hue='distance',
               palette='summer',
               inner = 'quartile',
               scale='count',
               split=True)
ax.set(xlabel=r'$\gamma$', ylabel=r'distance')
ax.set_ylim([0,1])
ax.legend(handles=ax.legend_.legend_handles, labels=[r'$d(t_0,t_1)$', r'$d(t_1,t_2)$'])
sb.despine()
plt.tight_layout()
plt.show()

#%%
df_long_R = filtered_expe[condition3].melt(id_vars=['alpha'], value_vars=['det_R', 'trace_R'],
                  var_name='Hessian', value_name='value')

df_long_R.rename(columns={'alpha': r'$\gamma$'}, inplace=True)

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(8, 4))
sb.violinplot(ax=ax,
               data=df_long_R,
               x=r'$\gamma$',
               y='value',
               hue='Hessian',
               palette='summer',
               inner = 'quartile',
               scale='count',
               split=True)
ax.set(xlabel=r'$\gamma$', ylabel=r'Random part of the Hessian')
ax.set_ylim([-5,5])
ax.legend(handles=ax.legend_.legend_handles, labels=[r'$\det(R)$', r'$\mathrm{Trace}(R)$'])
sb.despine()
plt.tight_layout()
plt.show()

#%%
df_long_lambda = filtered_expe[condition3].melt(id_vars=['alpha'], value_vars=['lambda_1', 'lambda_2'],
                  var_name='lambda', value_name='value')

df_long_lambda.rename(columns={'alpha': r'$\gamma$'}, inplace=True)

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(8, 4))
sb.violinplot(ax=ax,
               data=df_long_lambda,
               x=r'$\gamma$',
               y='value',
               hue='lambda',
               palette='summer',
               inner = 'quartile',
               scale='count',
               split=True)
ax.set(xlabel=r'$\gamma$', ylabel=r'eigenvalues')
ax.set_ylim([0,13])
ax.legend(handles=ax.legend_.legend_handles, labels=[r'$\lambda_1$', r'$\lambda_2$'])
sb.despine()
plt.tight_layout()
plt.show()

#%%


sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_context("paper")
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(4, 4))
sb.ecdfplot(   ax=ax,
               data=df_long_pvalues[(df_long_pvalues[r'$p$-value']== 'spacing_pvalue')],
               x='value',
               hue=r'$\gamma$',
               palette="flare",
               )
ax.set(xlabel=r'$p$-value of spacing test', ylabel=r'proportion')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
sb.despine()
plt.tight_layout()
plt.savefig("pvalues_spacing")
plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_context("paper")
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(4, 4))
sb.ecdfplot(   ax=ax,
               data=df_long_pvalues[(df_long_pvalues[r'$p$-value']== 'tspacing_pvalue')],
               x='value',
               hue=r'$\gamma$',
               palette="flare",
               )
ax.set(xlabel=r'$p$-value of $t$-spacing test', ylabel=r'proportion')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
sb.despine()
plt.tight_layout()
plt.savefig("pvalues_tspacing")
plt.show()

#%%
# sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sb.set_style("darkgrid")
# ax = sb.displot(
#                data=df_long_pvalues[(df_long_pvalues[r'$p$-value']== 'tspacing_pvalue')],
#                x='value',
#                hue=r'$\gamma$',
#                kde=True,
#                palette="flare",
#                height=4,
#                aspect=1
#                )
# ax.set(xlabel=r'$p$-value of $t$-spacing test', ylabel=r'counts')
# sb.despine()
# plt.tight_layout()
# plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_pvalues[(df_long_pvalues[r'$p$-value']== 'tspacing_pvalue')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$p$-value of $t$-spacing test', 
       ylabel=r'density', 
       ylim=(0,5),
       xlim=(0,1))
sb.despine()
plt.tight_layout()
plt.savefig("0_pvalues_tspacing")
plt.show()

#%%
# sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sb.set_style("darkgrid")
# ax = sb.displot(
#                data=df_long_pvalues[(df_long_pvalues[r'$p$-value']== 'spacing_pvalue')],
#                x='value',
#                hue=r'$\gamma$',
#                kde=True,
#                palette="flare",
#                height=4,
#                aspect=1
#                )
# ax.set(xlabel=r'$p$-value of spacing test', ylabel=r'counts')
# sb.despine()
# plt.tight_layout()
# plt.show()

#%%
sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_pvalues[(df_long_pvalues[r'$p$-value']== 'spacing_pvalue')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$p$-value of spacing test', 
       ylabel=r'density', 
       ylim=(0,5),
       xlim=(0,1))
sb.despine()
plt.tight_layout()
plt.savefig("0_pvalues_spacing")
plt.show()

#%%
# sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sb.set_style("darkgrid")
# ax = sb.displot(
#                data=df_long_distances[(df_long_distances['distance']== 'distance_t0t1')],
#                x='value',
#                hue=r'$\gamma$',
#                kde=True,
#                palette="flare",
#                height=4,
#                aspect=1
#                )
# ax.set(xlabel=r'$d(t_0,t_1)$', ylabel=r'counts')
# sb.despine()
# plt.tight_layout()
# plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_distances[(df_long_distances['distance']== 'distance_t0t1')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$d(t_0,t_1)$', 
       ylabel=r'density', 
       ylim=(0,5),
       xlim=(0,1))
sb.despine()
plt.tight_layout()
plt.savefig("0_distance_t0t1")
plt.show()
       
#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_context("paper")
sb.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(6, 4))
sb.ecdfplot(   ax=ax,
               data=df_long_distances[(df_long_distances['distance']== 'distance_t0t1')],
               x='value',
               hue=r'$\gamma$',
               palette="flare",
               )
ax.set(xlabel=r'$d(t_0,t_1)$', ylabel=r'proportion')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
sb.despine()
plt.tight_layout()
plt.savefig("distance_t0t1")
plt.show()

#%%
# sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sb.set_style("darkgrid")
# ax = sb.displot(
#                data=df_long_distances[(df_long_distances['distance']== 'distance_t1t2')],
#                x='value',
#                hue=r'$\gamma$',
#                kde=True,
#                palette="flare",
#                height=4,
#                aspect=1,
#                kind='hist',
#                common_norm=False
#                )
# ax.set(xlabel=r'$d(t_1,t_2)$', ylabel=r'counts')
# sb.despine()
# plt.tight_layout()
# plt.show()

#%%


sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_distances[(df_long_distances['distance']== 'distance_t1t2')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$d(t_1,t_2)$', 
       ylabel=r'density',
       xlim=(0,1))
sb.despine()
plt.tight_layout()
plt.savefig("distance_t1t2")
plt.show()


#%%

df_long_sigma = filtered_expe[condition3].melt(id_vars=['alpha'], value_vars=['sigma_estimate'],
                  var_name='sigma', value_name='estimate')

df_long_sigma.rename(columns={'alpha': r'$\gamma$'}, inplace=True)

# sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sb.set_context("paper")
# sb.set_style("darkgrid")
# fig, ax = plt.subplots(figsize=(8, 4))
# sb.violinplot(ax=ax,
#                data=df_long_sigma,
#                x=r'$\gamma$',
#                y='estimate',
#                hue='sigma',
#                palette='summer',
#                inner = 'quartile',
#                scale='count')
# ax.set(xlabel=r'$\gamma$', ylabel=r'$\hat\sigma$')
# ax.set_ylim([0,5])
# sb.despine()
# plt.tight_layout()
# plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_sigma,
               x='estimate',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
grid_plot = np.linspace(0, 2, 3000)
ax.plot(grid_plot, np.sqrt(7) * chi.pdf(np.sqrt(7) * grid_plot, df=7), 
        'k--', 
        linewidth=2)
ax.set(xlabel=r'$\hat\sigma$', 
       ylabel=r'density',
       xlim=(0,2.5))
sb.despine()
plt.tight_layout()
plt.savefig("sigma")
plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_R[(df_long_R['Hessian']== 'det_R')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$\det(R)$', 
       ylabel=r'density',
       xlim=(-3.5,3.5))
sb.despine()
plt.tight_layout()
plt.savefig("detR")
plt.show()

#%%
sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_R[(df_long_R['Hessian']== 'trace_R')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$\mathrm{Trace}(R)$', 
       ylabel=r'density',
       xlim=(-5,5))
sb.despine()
plt.tight_layout()
plt.savefig("traceR")
plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_lambda[(df_long_lambda['lambda']== 'lambda_1')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
ax.set(xlabel=r'$\lambda_1$', 
       ylabel=r'density',
       xlim=(0,13))
sb.despine()
plt.tight_layout()
plt.savefig("lambda1")
plt.show()

#%%
df_long_lambda1_normalized = df_long_lambda[(df_long_lambda['lambda']== 'lambda_1')]
temp = df_long_lambda1_normalized.loc[:, ('value')]
df_long_lambda1_normalized.loc[:, ('value')] = temp - np.sqrt(3*np.log(3)+3*np.log(np.log(3)))* df_long_lambda1_normalized.loc[:, ('$\gamma$')]

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_lambda1_normalized,
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
grid_plot = np.linspace(-4, 5, 3000)
ax.plot(grid_plot, norm.pdf(grid_plot), 
        'k--', 
        linewidth=2.5)
ax.set(xlabel=r'$\lambda_1-\lambda_0$', 
       ylabel=r'density',
       xlim=(-4,5))
sb.despine()
plt.tight_layout()
plt.savefig("lambda1_centered")
plt.show()

#%%

sb.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sb.set_style("darkgrid")
ax = sb.histplot(
               data=df_long_lambda[(df_long_lambda['lambda']== 'lambda_2')],
               x='value',
               hue=r'$\gamma$',
               stat='density',
               kde=True,
               palette="flare",
               alpha= 0.25,
               common_norm=False
               )
# grid_plot = np.linspace(0, 5, 3000)
# ax.plot(grid_plot, norm.pdf(grid_plot+np.sqrt(3*np.log(3)+3*np.log(np.log(3)))), 
#         'k--', 
#         linewidth=2.5)
ax.set(xlabel=r'$\lambda_2$', 
       ylabel=r'density',
       xlim=(0,5))
sb.despine()
plt.tight_layout()
plt.savefig("lambda2")
plt.show()
    
