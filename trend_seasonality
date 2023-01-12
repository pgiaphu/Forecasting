import pandas as pd
import numpy as np
import statsmodels.api as sm



#trend and seasonal streng
df = pd.read_excel("as.xlsx",index_col=0,parse_dates=True)
def decompotots(data, function): #inser DecompoResult into df column
    return data.apply(lambda x: pd.Series(function(x)))

def unpackdecompo(data, function): #extract DecompoResult then merge it into df
    newdata = pd.DataFrame(function(data), index = data.index)
    duplicol = [col for col in newdata.columns if col in data.columns]
    newdata = newdata.drop(columns = duplicol)
    return data.join(newdata)

dfgb = (df
        .groupby(['Material'])
        .pipe(decompotots, lambda x: dict( stl = sm.tsa.STL(x.Quantity, period = 12).fit()))
        .pipe(unpackdecompo, lambda x: dict( trend_streng = [1 - (stl.resid.var() / (stl.resid + stl.trend).var()) for stl in x.stl],
                                             seasonal_streng = [1 - (stl.resid.var() / (stl.resid + stl.seasonal).var()) for stl in x.stl]))
        .drop(columns = 'stl')
        .reset_index()
        )


dfgb.to_excel('trend.xlsx',sheet_name="Sheet1")
