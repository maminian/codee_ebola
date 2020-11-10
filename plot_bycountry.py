import json
import pandas
import numpy as np
from matplotlib import pyplot,ticker
import datetime

############

cases_palette = [pyplot.cm.tab20(2*j+1) for j in range(5)]
pred_palette = [pyplot.cm.tab20(2*j) for j in range(5)] # only needed for sierraleone and liberia.

COUNTRIES = ['SierraLeone','Liberia','Guinea','Nigeria','Senegal']
ALPHAS = [1,1,0.4,0.4,0.4]  # lessen visual clutter of countries not being highlighted.

with open('country_timeseries.json', 'r') as ff:
    data_j = json.load(ff)
#


################

headers = list( data_j[0].keys() )

data_raw = [[row[h] for h in headers] for row in data_j]
df = pandas.DataFrame(data=data_raw, columns=headers)

# cast all but the date column to integer.
# impute missing values with 0 for now.
#df = df.replace('','np.nan')
for h in headers:
    if h=='Date':
        df[h] = [datetime.datetime.strptime(hi,'%m/%d/%Y') for hi in df[h]]
    else:
        df[h] = [int(hi) if hi!='' else np.nan for hi in df[h].values]
#        df[h] = np.array( df[h].values, dtype=int)
#

# sort by "Day", which is days since some fixed date.
df = df.sort_values('Day')

#

fig,ax = pyplot.subplots(1,2, figsize=(14,6), constrained_layout=True)

for j,(h,col,alf) in enumerate(zip(['Cases_'+s for s in COUNTRIES], cases_palette, ALPHAS)):
#    ax.plot(df['Day'].values, df[h].values, c=pyplot.cm.Set2(j), lw=2, marker='d',label=h)
    ax[0].plot(df['Date'].values, df[h].values, c=col, lw=2, marker='d',label=h, alpha=alf)
    ax[1].plot(df['Date'].values, df[h].values, c=col, lw=2, marker='d',label=h, alpha=alf)
#


ax[1].set_yscale('log')

ax[0].yaxis.grid()
ax[1].yaxis.grid()


# Attempt to fit Sierra Leone and Liberia.
# June 1 chosen arbitrarily as the starting point when starting the models.
# corresponds to row 32 onwards in the dataframe as processed to this point.
df_sub = df.iloc[32:][['Date', 'Day', 'Cases_SierraLeone', 'Cases_Liberia']]
cases_s = df_sub['Cases_SierraLeone'].values
cases_l = df_sub['Cases_Liberia'].values
t_reset = df_sub['Day'].values
t0 = int(t_reset[0])
t_reset -= t0

#
mask_s = np.logical_not( np.isnan(cases_s) )
coef_s = np.polyfit(t_reset[mask_s], np.log( cases_s[mask_s] ), 1)
# sierra leone :: log(i) = alpha + beta*t; alpha=4.5892, beta=0.0278 -> I_0~=98; doubling time ~ 25 days

mask_l = np.logical_not( np.isnan(cases_l) )
coef_l = np.polyfit(t_reset[mask_l], np.log( cases_l[mask_l] ), 1)
# liberia :: log(i) = alpha + beta*t; alpha=2.8820, beta=0.0494 -> I_0~=18; doubling time ~ 14 days

pred_l = np.exp( np.polyval(coef_l, df['Day'].values - t0) )
pred_s = np.exp( np.polyval(coef_s, df['Day'].values - t0) )

ax[0].plot(df['Date'].values[32:], pred_s[32:], c=pred_palette[0], alpha=1,ls='--', lw=3, label='Predicted_SierraLeone')
ax[1].plot(df['Date'].values[32:], pred_s[32:], c=pred_palette[0], alpha=1,ls='--', lw=3)

ax[0].plot(df['Date'].values[32:], pred_l[32:], c=pred_palette[1], alpha=1,ls='--', lw=3, label='Predicted_Liberia')
ax[1].plot(df['Date'].values[32:], pred_l[32:], c=pred_palette[1], alpha=1,ls='--', lw=3)

ax[0].set_xlabel('Date')
ax[0].set_ylabel('Cases')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Cases')
ax[0].legend(loc='upper left')

# Label curves with text annotation.
for j,(h,bgcol) in enumerate(zip(COUNTRIES,cases_palette)):
#    ax[1].plot(df['Date'].values, df[h].values, c=col, lw=2, marker='d',label=h, alpha=alf)
    xx = df['Date'].values[-1]
    yy = df['Cases_'+h].values[-1]
        
    ax[1].annotate(h, (xx,yy), 
        fontsize=9, 
        ha='left', 
        va='bottom', 
        annotation_clip=False,
        bbox={'color':np.sqrt(np.array(bgcol)), 'alpha':1, 'edgecolor':None, 'boxstyle':'round'}
        )
    
    ax[0].annotate(h, (xx,yy), 
        fontsize=9, 
        ha='left', 
        va='bottom', 
        annotation_clip=False,
        bbox={'color':np.sqrt(np.array(bgcol)), 'alpha':1, 'edgecolor':None, 'boxstyle':'round'}
        )
#


if True:
    fig.savefig('cases_bycountry.png', dpi=120)
#
fig.show()
