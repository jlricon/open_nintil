library(data.table)
library(lavaan)
library(semPlot)
library(magrittr)
library(dplyr)
library(car)
dt=fread("/home/jose/Escritorio/Gender and STEM/Rdata.csv") 
dt=dt[,-c("V1")]
#dt=dt[!is.na(hy_f_STEM)]
#dt=setDT(aggregate(dt, list(dt$Country), mean,na.action=na.pass, na.rm=TRUE))
data=dt[TIME<2000,IQ:=NA][TIME<2000,Nrrent:=NA][TIME>=2000][TIME<2000,muslim:=NA]

#data=data[loggdppc<8]

#setnames(data,"female_eduedu","f_edu")
setnames(data,"female_engi","f_engi")
setnames(data,"female_ah","f_ah")
data$loggdppc=log(data$gdppc*(100-data$Nrrent)/100)

data$hy_f_edu=100*data$female_over_male_edu/(data$female_over_male_edu+1)
data$hy_f_engi=100*data$female_over_male_engi/(data$female_over_male_engi+1)

#LOGS
  data$mortality %<>% log()
 # data$life_exp %<>% log()
 # data$muslim %<>% logit(percents=TRUE)
 # data$f_edu %<>% logit(percents=TRUE)
 # data$f_STEM %<>% logit(percents=TRUE)
 # data$f_engi %<>% logit(percents=TRUE)
 # data$female_uni %<>% logit(percents=TRUE)
cols <- c("IQ","muslim","Nrrent","equality_index",
         "female_over_male","female_over_male_art","math_anxiety",
         "female_uni","loggdppc","mortality","life_exp","gdppc","f_ah","f_STEM")
#data[, (cols) := lapply(.SD, scale), .SDcols=cols]

#Factor
k=prcomp(~life_exp+mortality+loggdppc+IQ, data=data, center = TRUE, scale = TRUE, na.action = na.omit)
resul=scale(data[,.(life_exp,mortality,loggdppc,IQ)], k$center, k$scale) %*% k$rotation %>% as.data.frame() 
l=resul["PC1"]
#data$develop=data$life_exp

model="
       develop=~loggdppc+life_exp+mortality+IQ

   
        muslim ~~ develop
        

        equality_index~develop+muslim

        hy_f_engi~ develop+ equality_index
        hy_f_STEM~ develop+ equality_index
        
        female_uni~   develop+equality_index
        
        
 
"
#        female_over_male~    equality_index+develop+muslim
#female_over_male_edu~equality_index+develop+muslim
# model="
# loggdppc~~IQ
# loggdppc~Nrrent
# equality_index~loggdppc+muslim+IQ+Nrrent
# muslim~Nrrent+IQ+loggdppc
# IQ~0*Nrrent"
fit = sem(model,data=data,estimator="MLR",missing="FIML",fit.measures=TRUE)
summary(fit,standardized=TRUE)
 semPaths(fit,curvePivot=TRUE,what = "std",nCharNodes = 0,
          edge.label.cex = 1,label.cex=5,residuals = FALSE,layout="spring",intercepts = FALSE)
 #data$develop=as.data.frame(lavPredict(fit))
