#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:32:41 2017

@author: jose
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import optimize
sns.set()
sns.set_context("talk")
path="/home/jose/Escritorio/Gender and STEM/"
GDPdata=pd.read_csv(path+"gdp.csv")[["TIME","Country","Value"]]
edu=pd.read_csv(path+"edu.csv")
life=pd.read_csv(path+"lifex.csv")
rent=pd.read_csv(path+"nrent.csv")
muslim=pd.read_csv(path+"muslim.csv")
totaleduc=pd.read_csv(path+"tertiarytotal.csv")
bygender=pd.read_csv(path+"percentages_by_gender.csv")
iqs=pd.read_csv(path+"iqs2.csv",sep=";"   )

anxiety=pd.read_csv(path+"anxiety_2.csv")
mergeon=["Country"]
time=["TIME"]
# http://data.uis.unesco.org
# %%


def grab_latest(x):
    maxt=x.TIME.max()
    return x.query("TIME==TIME.max()")
latestgdp=(GDPdata
#           .groupby("Country")
#           .apply(grab_latest)
           .reset_index(drop=True)
           .rename(columns={"Value":"gdppc"})
           .merge(rent,how="left",on="Country")         
           )[["Country","gdppc","Nrrent"]+time]

# %%
pop=(life[life.iloc[:,0]=='200101']
#             .groupby("Country")
#           .apply(grab_latest)
           .reset_index(drop=True)
           .rename(columns={"Value":"pop"})
)[["Country","pop"]+time]


# %%

life_expectancy=(
        #life.query("DEMO_IND=='SP_DYN_LE00_IN'")
        life.query("Indicator=='Life expectancy at birth, total (years)'")
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"life_exp"})
               
        )[["Country","life_exp"]+time]
mortality=(
        #life.query("DEMO_IND=='SP_DYN_LE00_IN'")
        life.query("Indicator=='Mortality rate, infant (per 1,000 live births)'")
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"mortality"})
               
        )[["Country","mortality"]+time]
gdp_life_pop=(latestgdp
              .merge(pop,how="left",on=["Country"]+time)
              .merge(life_expectancy,how="left",on=["Country"]+time)
              .merge(mortality,how="left",on=["Country"]+time)
              )


# %%
edu_stem=(edu
               .query("EDULIT_IND=='FGP_5T8_F500600700'")
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"f_STEM"})
               )[["Country","f_STEM"]+time]
edu_ict=(edu
               .query("EDULIT_IND=='FGP_5T8_F600'")[["Country","TIME","Value"]]
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"female_ICT"})
               )[["Country","female_ICT"]+time]
edu_engi=(edu
               .query("EDULIT_IND=='FGP_5T8_F700'")[["Country","TIME","Value"]]
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"female_engi"})
               )[["Country","female_engi"]+time]
edu_edu=(edu
               .query("EDULIT_IND=='FGP_5T8_F110'")[["Country","TIME","Value"]]
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"female_edu"})
               )[["Country","female_edu"]+time]
edu_health=(edu
               .query("EDULIT_IND=='FGP_5T8_F900'")[["Country","TIME","Value"]]
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"female_health"})
               )[["Country","female_health"]+time]

edu_ah=(edu
               .query("EDULIT_IND=='FGP_5T8_F200'")[["Country","TIME","Value"]]
#               .groupby("Country")
#               .apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"female_ah"})
               )[["Country","female_ah"]+time]

# %%
#Arab
gdp_life_pop=gdp_life_pop.merge(muslim.rename(columns={"Amount":"muslim"}),
                                 how="left",on=["Country"])
gdp_life_pop["muslim"]=gdp_life_pop["muslim"].fillna(0)
gdp_life_pop["Nrrent"]=gdp_life_pop["Nrrent"].fillna(0)
# %%
#Education total
total_edu=(totaleduc
               .query("EDULIT_IND=='FGP_5T8'")
               #.groupby("Country")
               #.apply(grab_latest)
               .reset_index(drop=True)
               .rename(columns={"Value":"female_uni"})
               [["Country","female_uni"]+time]
               )
gdp_life_pop=(gdp_life_pop.merge(total_edu,
                                 how="left",on=["Country"]+time))
#%%

bygender_edu=(bygender
               .query("EDULIT_IND in ['FOSGP_5T8_F700_M','FOSGP_5T8_F700_F','FOSGP_5T8_F600_M','FOSGP_5T8_F600_F','FOSGP_5T8_F500_F','FOSGP_5T8_F500_M']")
               #ENgineering only
               #.query("EDULIT_IND in ['FOSGP_5T8_F110_M','FOSGP_5T8_F110_F']")
               #.groupby("Country")
               #.apply(grab_latest)
               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F700_M','FOSGP_5T8_F600_M','FOSGP_5T8_F500_M','FOSGP_5T8_F110_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()
               .groupby(["TIME","Country"])
               .apply(lambda x: (x.query("male==False").Value.values/x.query("male==True").Value.values))
               .reset_index()     
               .rename(columns={0:"female_over_male"})
               .assign(female_over_male=lambda x: x.female_over_male.apply(lambda x: x.squeeze()))
               .dropna()
               .pipe(lambda x: x[x.female_over_male.apply(lambda i: type(i)==float)])
               .assign(female_over_male=lambda x: x.female_over_male.astype(float))          
               
               )
bygender_engi=(bygender
               #.query("EDULIT_IND in ['FOSGP_5T8_F700_M','FOSGP_5T8_F700_F','FOSGP_5T8_F600_M','FOSGP_5T8_F600_F','FOSGP_5T8_F500_F','FOSGP_5T8_F500_M']")
               #ENgineering only
               .query("EDULIT_IND in ['FOSGP_5T8_F700_M','FOSGP_5T8_F700_F']")
               #.groupby("Country")
               #.apply(grab_latest)
               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F700_M','FOSGP_5T8_F600_M','FOSGP_5T8_F500_M','FOSGP_5T8_F110_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()
               .groupby(["TIME","Country"])
               .apply(lambda x: (x.query("male==False").Value.values/x.query("male==True").Value.values))
               .reset_index()     
               .rename(columns={0:"female_over_male_engi"})
               .assign(female_over_male_engi=lambda x: x.female_over_male_engi.apply(lambda x: x.squeeze()))
               .dropna()
               .pipe(lambda x: x[x.female_over_male_engi.apply(lambda i: type(i)==float)])
               .assign(female_over_male_engi=lambda x: x.female_over_male_engi.astype(float))          
               
               )
bygender_edu_health=(bygender
               .query("EDULIT_IND in ['FOSGP_5T8_F900_F','FOSGP_5T8_F900_M','FOSGP_5T8_F110_F','FOSGP_5T8_F110_M']")
               #ENgineering only
               #.query("EDULIT_IND in ['FOSGP_5T8_F110_M','FOSGP_5T8_F110_F']")
               #.groupby("Country")
               #.apply(grab_latest)
               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F900_M','FOSGP_5T8_F110_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()
               .groupby(["TIME","Country"])
               .apply(lambda x: (x.query("male==False").Value.values/x.query("male==True").Value.values))
               .reset_index()     
               .rename(columns={0:"female_over_male_edu"})
               .assign(female_over_male_edu=lambda x: x.female_over_male_edu.apply(lambda x: x.squeeze()))
               .dropna()
               .pipe(lambda x: x[x.female_over_male_edu.apply(lambda i: type(i)==float)])
               .assign(female_over_male_edu=lambda x: x.female_over_male_edu.astype(float))          
               
               )
bygender_edu_art=(bygender
               .query("EDULIT_IND in ['FOSGP_5T8_F200_M','FOSGP_5T8_F200_F']")
               #ENgineering only
               #.query("EDULIT_IND in ['FOSGP_5T8_F110_M','FOSGP_5T8_F110_F']")
               #.groupby("Country")
               #.apply(grab_latest)
               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F200_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()
               .groupby(["TIME","Country"])
               .apply(lambda x: (x.query("male==False").Value.values/x.query("male==True").Value.values))
               .reset_index()     
               .rename(columns={0:"female_over_male_art"})
               .assign(female_over_male_art=lambda x: x.female_over_male_art.apply(lambda x: x.squeeze()))
               .dropna()
               .pipe(lambda x: x[x.female_over_male_art.apply(lambda i: type(i)==float)])
               .assign(female_over_male_art=lambda x: x.female_over_male_art.astype(float))          
               
               )

gdp_life_pop=(gdp_life_pop
              .merge(bygender_edu,how="left",on=["Country"]+time)
#.merge(bygender_edu_health,how="left",on=["Country"]+time)
            .merge(bygender_edu_art,how="left",on=["Country"]+time)
            .merge(bygender_engi,how="left",on=["Country"]+time)
)
# %%
#Male and female interest in STEM
bygender_edu_sep=(bygender
               .query("EDULIT_IND in ['FOSGP_5T8_F700_M','FOSGP_5T8_F700_F','FOSGP_5T8_F600_M','FOSGP_5T8_F600_F','FOSGP_5T8_F500_F','FOSGP_5T8_F500_M']")

               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F700_M','FOSGP_5T8_F600_M','FOSGP_5T8_F500_M','FOSGP_5T8_F110_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()
               
         
               
               )
male=bygender_edu_sep.query("male==True").drop("male",axis=1).rename(columns={"Value":"rel_m_stem"})
female=bygender_edu_sep.query("male==False").drop("male",axis=1).rename(columns={"Value":"rel_f_stem"})
bygender_edu_sep=pd.merge(male,female,on=["Country","TIME"])
gdp_life_pop=gdp_life_pop.merge(bygender_edu_sep,
                                 how="left",on=["Country"]+time)
#For education
bygender_eh_sep=(bygender
               #.query("EDULIT_IND in ['FOSGP_5T8_F900_F','FOSGP_5T8_F900_M','FOSGP_5T8_F110_F','FOSGP_5T8_F110_M']")
               .query("EDULIT_IND in ['FOSGP_5T8_F900_F','FOSGP_5T8_F900_M','FOSGP_5T8_F110_F','FOSGP_5T8_F110_M']")
               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F900_M','FOSGP_5T8_F110_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()              
         
               
               )
male=bygender_eh_sep.query("male==True").drop("male",axis=1).rename(columns={"Value":"rel_m_eh"})
female=bygender_eh_sep.query("male==False").drop("male",axis=1).rename(columns={"Value":"rel_f_eh"})
bygender_eh_sep=pd.merge(male,female,on=["Country","TIME"])
gdp_life_pop=gdp_life_pop.merge(bygender_eh_sep,
                                 how="left",on=["Country"]+time)

#For art
bygender_ah_sep=(bygender
               #.query("EDULIT_IND in ['FOSGP_5T8_F900_F','FOSGP_5T8_F900_M','FOSGP_5T8_F110_F','FOSGP_5T8_F110_M']")
                .query("EDULIT_IND in ['FOSGP_5T8_F200_M','FOSGP_5T8_F200_F']")
               .reset_index(drop=True)
               .assign(male=lambda x:  [i in ['FOSGP_5T8_F200_M'] for i in x.EDULIT_IND])
               [["TIME","Country","Indicator","Value","male"]]
               .groupby(["TIME","Country","male"]).sum()
               .reset_index()              
         
               
               )
male=bygender_ah_sep.query("male==True").drop("male",axis=1).rename(columns={"Value":"rel_m_ah"})
female=bygender_ah_sep.query("male==False").drop("male",axis=1).rename(columns={"Value":"rel_f_ah"})
bygender_ah_sep=pd.merge(male,female,on=["Country","TIME"])
gdp_life_pop=gdp_life_pop.merge(bygender_ah_sep,
                                 how="left",on=["Country"]+time)

# %%
# IQ
iqs["Country"]=iqs.Country.str.strip()
gdp_life_pop=gdp_life_pop.merge(iqs,
                                 how="left",on=["Country"])
#Anxiety
anxiety.columns=["Country","SelfEfficacy","SelfConcept","math_anxiety"]
anxiety[["SelfEfficacy","SelfConcept","math_anxiety"]]=anxiety[["SelfEfficacy","SelfConcept","math_anxiety"]].apply(lambda x: x.str.replace(",",".")).astype(float)
anxiety=anxiety.assign(TIME=2014)
gdp_life_pop=gdp_life_pop.merge(anxiety,
                                how="left",on=["Country"]+time)
# %%
#sigi
sigi2014=pd.read_excel(path+"sigi2014.xls")
sigi2014["Country"]=sigi2014["Country"].str.strip()
sigi2014=sigi2014.replace("..",100)
sigi2014=sigi2014.assign(n_comp=lambda x: 1/(5-(round(x.iloc[:,2:].sum(axis=1)/100)).astype(int)))
sigi2014=sigi2014.replace(100,0)
sigi2014=sigi2014.assign(equality_index=lambda x: np.square(x.iloc[:,2:7]).sum(axis=1)*x.n_comp)
sigi2014=sigi2014.assign(TIME=2014)

sigi2012=pd.read_excel(path+"sigi2012.xls")
sigi2012["Country"]=sigi2012["Country"].str.strip()
sigi2012=sigi2012.replace("..",100)
sigi2012=sigi2012.assign(n_comp=lambda x: 1/(5-(round(x.iloc[:,2:].sum(axis=1)/100)).astype(int))).query("n_comp<=0.3")
sigi2012=sigi2012.replace(100,0)
sigi2012=sigi2012.assign(equality_index=lambda x: np.square(x.iloc[:,2:7]).sum(axis=1)*x.n_comp)
sigi2012=sigi2012.assign(TIME=2012)
sigi=pd.concat([sigi2012,sigi2014])
gdp_life_pop=gdp_life_pop.merge(sigi[["Country","equality_index","TIME"]],
                                how="left",on=["Country"]+time)

# %%
target=gdp_life_pop
def get_gap(x):
    return 100*x/(x+1)
var="gdppc"
merged_stem=pd.merge(target,edu_stem,on=["Country"]+time,how="left" )
merged_ict=pd.merge(target,edu_ict,on=["Country"]+time,how="left")
merged_all=pd.merge(target,edu_stem,on=["Country"]+time,how="left" )
merged_all=(merged_all
            .merge(edu_ict,on=["Country"]+time,how="left" )
            .merge(edu_engi,on=["Country"]+time,how="left" )
            .merge(edu_health,on=["Country"]+time,how="left" )
            .merge(edu_ah,on=["Country"]+time,how="left" )
            )
merged_all=merged_all.reset_index(drop=True).assign(loggdppc=lambda x:np.log(x.gdppc))
merged_all=merged_all.assign(hy_f_STEM=lambda x: get_gap(x.female_over_male))
merged_all=merged_all.assign(hy_f_engi=lambda x: get_gap(x.female_over_male_engi))
merged_all["math_anxiety"]=-merged_all.math_anxiety
merged_all["equality_index"]=-merged_all.equality_index
merged_all.to_csv("/home/jose/Escritorio/Gender and STEM/Rdata.csv")
# %%
plt.close()
lowess = sm.nonparametric.lowess

#merged_stem.pipe(lambda x: sns.regplot(data=x,x=var,y="f_STEM",lowess=True  ))
t=merged_stem.reset_index(drop=True).assign(loggdppc=lambda x:np.log(x.gdppc))
t=t.assign(hy_f_STEM=lambda x: get_gap(x.female_over_male)).dropna(subset=["f_STEM","hy_f_STEM"])
t["math_anxiety"]=-t.math_anxiety
t["equality_index"]=-t.equality_index
#t=t.dropna()
my_t=t
# %%
target_x="loggdppc"
t=t.sort_values(by=target_x).query("pop>0000")
target="hypothetical_f_STEM"
###################################
plt.subplot(2,2,1)
(t
#.query(" muslim<20")
.pipe(lambda x:x.plot.scatter(x=target_x,y="female_uni",s=np.sqrt(x["pop"])/5,
                              
                             ax=plt.gca()))
)
z = lowess(t.female_uni.values,t[target_x].values,return_sorted=True,frac=0.6)
plt.plot(t[target_x].values,z[:,1])
plt.xlabel("log(GDP per capita)")
plt.ylabel("% of female tertiary education students")
##########################
plt.subplot(2,2,2)
(t
#.query(" muslim<20")
.pipe(lambda x:x.plot.scatter(x=target_x,y=target,s=np.sqrt(x["pop"])/5,
                             
                             ax=plt.gca()))
)
z = lowess(t[target].values,t[target_x].values,return_sorted=True,frac=0.6)
plt.plot(t[target_x].values,z[:,1])
plt.xlabel("log(GDP per capita)")
plt.ylabel("% of hypothetical female STEM students")
##########################
plt.subplot(2,2,3)
t=t.sort_values(by="pop").query("pop>0000").assign(muslim=lambda x: np.log(x.muslim+0.01))
(t
#.query(" muslim<20")
.pipe(lambda x:x.plot.scatter(x="pop",y=target,
                             
                             ax=plt.gca()))
)
#z = lowess(t.female_over_male.values,t["pop"].values,return_sorted=True,fra)
#plt.plot(t["pop"].values,z[:,1])
plt.axhline(t[target].median())
plt.axhline(t[target].mean())
plt.xlabel("population")
plt.xscale("log")
plt.ylabel("% of female STEM students")

##############
plt.subplot(2,2,4)
t=t.sort_values(by="female_uni").query("pop>0000")
(t
#.query(" muslim<20")
.pipe(lambda x:x.plot.scatter(x="female_uni",y=target,s=np.sqrt(x["pop"])/5,
                             
                             ax=plt.gca()))
)
z = lowess(t[target].values,t.female_uni.values,return_sorted=True,frac=0.6)
plt.plot(t.female_uni.values,z[:,1])
plt.xlabel("% of female tertiary education students")
plt.ylabel("% of female STEM students")
plt.suptitle("Hypothetical % female in STEM\nIf equal gender ratio in tertiary education\nCountries with pop over 4M, GDPpc over 8100$, % muslim<20")
##############
rich_t=t.query("gdppc>exp(10)")
poor_t=t.query("gdppc<exp(9)")
#(t.pipe(lambda x: x.assign(life_exp=np.power(x.life_exp,2)))
#.pipe(lambda x: sns.regplot(data=x,x="muslim",y="f_STEM",lowess=True))
#
#)

#%%
#Rich nonmuslim countries
t=my_t.dropna(subset=["f_STEM","hy_f_STEM","equality_index","IQ"]).copy()

#Positive anxiety: more gender gap against women
#d=t.query("pop>000 &log(gdppc)>0 ").dropna().reset_index()
#d["high_nrent"]=(d.Nrrent>30).astype(int)
d=t.sort_values("pop").groupby("Country").mean()
d=t
#d["Country"]=d.Country.str.replace(" ","")

#tring_fixed="+".join(["Country_"+i for i in d.Country.unique()])
#d=pd.get_dummies(d)
results1=smf.ols("(hy_f_STEM)~(IQ)+Nrrent",data=d,missing="raise").fit()
weights=1/(np.power(results1.resid,2))
#print(results1.summary())
#results=smf.ols("hypothetical_f_STEM~pop",data=d,missing="raise",weights=weights).fit()

d=d.assign(res=results1.resid)
results=smf.ols("hy_f_STEM~equality_index+IQ",
                data=d,    
               # weights=weights,
                missing="raise").fit()
#a,b,c=wls_prediction_std(results)
print(results.summary())
res=d.assign(predicted=results.predict(d))
res=res.assign(diff=lambda x: 100*(x.predicted-x.female_over_male)/x.female_over_male)
#k=d[["loggdppc","Nrrent","muslim","res","math_anxiety"]].corr().pipe(lambda x: np.round(x,3))
# %%
#Pop
#Over time
plt.close()
plt.figure()
names=t.sort_values(by="hypothetical_f_STEM").query("hypothetical_f_STEM>40").Country.unique()
set_special=t.query("Country in @names").sort_values(by="TIME").groupby("Country")
for i in set_special:
    i[1].plot(x="TIME",y="hypothetical_f_STEM",label=i[0],ax=plt.gca(),linestyle="--")
    
plt.ylabel("Hypothetical % female in STEM")
#set_special.agg({"hypothetical_f_STEM":["mean","count"]}).hypothetical_f_STEM.plot.scatter("mean","count")
sns.boxplot(x="hypothetical_f_STEM",y="Country",data=rich,order=rich.groupby("Country").mean()["pop"].sort_values().index)

d=bygender.query("Country=='India'")[["Indicator","TIME","Value"]].dropna().groupby("Indicator").mean()
# %%
sns.heatmap((merged_all[["female_over_male","female_over_male_edu","equality_index","IQ","muslim","loggdppc","Nrrent"]]
.rename(columns={"female_over_male":"fm_STEM","female_over_male_edu":"fm_EDU"})
.corr()),annot=True)
plt.yticks(rotation="horizontal")
plt.title("Correlation matrix")
# %% bRING THE PCA!


d=merged_all.reset_index().assign(mortality=lambda x: np.log(x.mortality))
keep=["loggdppc","equality_index","Nrrent","female_uni","IQ","female_over_male","life_exp"]
keep=["loggdppc","IQ","life_exp","mortality"]
pca_data_o=d.dropna( subset=set(keep+["hy_f_STEM","f_STEM"])).set_index("Country")
countries=pca_data_o.index
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()
pca_data=scaler.fit_transform(pca_data_o[keep])
pca_data=pd.DataFrame(pca_data)
pca_data.columns=keep
pca_data=pca_data.set_index(countries)
pca=PCA(n_components=3)


k=pca.fit_transform(pca_data)
print(pca.fit(pca_data).explained_variance_ratio_)
pca_res=pd.DataFrame(pca.components_)
pca_res.columns=keep
pca_data=pca_data.assign(f1=k[:,0],f2=k[:,1],f3=k[:,2],hy_f_STEM=pca_data_o.hy_f_STEM,f_STEM=pca_data_o.f_STEM)
results=smf.ols("hy_f_STEM~f1+f2+f3",data=pca_data,missing="raise").fit()
print(results.summary())
plt.close()
sns.heatmap(pca_data.corr(),annot=True)
plt.yticks(rotation="horizontal")
# %%
#Humanities, SciEngir,Edu,Health,Business
#https://s3.amazonaws.com/academia.edu.documents/32816810/Parent.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1503161821&Signature=ueziuKH69rqJQxB7%2BUQuOJc0jQo%3D&response-content-disposition=inline%3B%20filename%3DThe_Impact_of_Parental_Occupation_and_So.pdf
ses10F=np.array([0.1283,0.2813,0.2039,0.1372,0.2493])
ses90F=np.array([0.4985,0.0582,0.2263,0.0431,0.1739])

ses10M=np.array([0.2154, 0.2978,0.2475,0.0842, 0.1552])
ses90M=np.array([0.3241,0.3368,0.0748,0.0285, 0.2360])

ses50F=np.array([0.2871,0.1452,0.2439,0.0873,0.2365])
ses50M=np.array([0.2760,0.3308, 0.1421,0.0512,0.1999])
def stemmer(x):
    if x.isin(["SciEng","Edu"]):
        return "STEM"
    if x.isin(["HumSoc","Edu","Health"]):
        return "SEH"
sesdata=pd.DataFrame({
        "gender":["Female"]*10+["Male"]*10+(["Female"]*5+["Male"]*5),
        "ses":   2*(["Low"]*5+["High"]*5)+10*["Med"],
        "field":["HumSoc","SciEng","Edu","Health","Business"]*6,
        "value":np.concatenate([ses10F,ses90F,ses10M,ses90M,ses50F,ses50M])
        })
sesdata["group"]=np.where(sesdata.field.isin(["SciEng"]),"STEM",np.where(sesdata.field.isin(["HumSoc","Edu","Health"]),"SEH","Business"))
sesdata.groupby(["gender","group","ses"]).sum()

#stemgroup=sesdata.groupby(["gender","group","ses"]).sum().reset_index()
(sns.factorplot(x="ses",hue="gender",y="Proportion who chooses\neach career group",
                data=sesdata.rename(columns={"value":"Proportion who chooses\neach career group"}),
                col="field",
                legend_out=False,
                order=["Low","Med","High"],col_wrap=2)
    )
    
    #%%
spanish_data=pd.read_csv(path+"spanish_data.csv")
spanish_data["value"]=spanish_data["value"].str.replace(",",".").astype(float)
(sns.factorplot(x="SES",hue="Gender",y="Proportion who chooses\neach career group",
                data=spanish_data.rename(columns={"value":"Proportion who chooses\neach career group"}),
                col="Field",
                legend_out=False,
                order=["Low","High"],col_wrap=4)
    )
    # %%
plt.close()
target="loggdppc"
merged_all.assign(Engistem=lambda x: x.hy_f_engi/x.hy_f_STEM)
#merged_all.query("Nrrent<20").plot.scatter("loggdppc","Engistem")
(merged_all.groupby("Country").mean().pipe(lambda x: x.plot
 .scatter(target,"Engistem",s=x["pop"].pipe(lambda i: np.sqrt(i)),
          c=x["Nrrent"],cmap="viridis")
 ))
sns.regplot(target,"Engistem",data=merged_all.groupby("Country").mean().query("pop>3000"),lowess=True,scatter=False,ax=plt.gca())
plt.ylabel("% female Engineering / % female STEM")
plt.xlabel("SIGI (0=less legal discrimination)")
# %%
pa="Reds"
pal=sns.palettes.color_palette(pa,len(k))
k=merged_all.query("loggdppc<9.5")
plt.close()
#sns.set_palette(pa,len(k))
sns.set_palette( "deep")
plt.figure()
i=0
jp.joyplot(k,column="hy_f_STEM",by="TIME")
for name,group in k.groupby("TIME"):
    #sns.kdeplot(group.hy_f_STEM,ax=plt.gca(),label=name)
    plt.axvline(group.hy_f_STEM.mean(),c=pal[i],ymax=0.5)
    i+=1
# %%
k=merged_all.assign(rich="No").query("muslim<20")
k.loc[(k.loggdppc>10) & (k.Nrrent<20),"rich"]="Yes"
sns.factorplot(x="TIME",y="hy_f_STEM",data=k,hue="rich",legend_out=False)
plt.ylabel("Hypothetical % female STEM")







   