## Religion by county (2010)

Data was obtained from http://www.thearda.com/Archive/Files/Downloads/RCMSCY10_DL2.asp
I got the State file and loaded it into pandas

- Data issues: The file was collected from statisticians from each different religious denomination, based on who, according to the,m, belongs to the religion, so these figures do not match self-reported values, and also lead to the sum of % religious in a County to be over 100, in some cases being over 200%.
- Extracted variables for mainline (`m_protes`), evangelical(`e_protes`), and black protestants(`b_protes`), mormons(`mormon`), southern baptist(`south_baptist`), orthodox jews (`o_jew`), muslims(`muslim`), and catholics(`catholic`).
- Created a religiosity variable (`religiosity`) with the sum of the alleged adherence rates, trimmed to be in [0,1]. However, the source of the data says one should not use this as a measure of atheism/religiosity, as perhaps there are smaller denominations that have not been counted.
- Dropped counties without population

## Census data (2010)

Acquired income in the last 12 months below poverty(`povrate`), population(`pop_census`), income per person(`income_census`), number of non hispanic whites(`nhwrate`), number of blacks(`blkrate`),number of asians(`asianrate`), number of native americans (`nativerate`) number of hispanics/latinos(`hisprate`). Created population-normalised rates for these. Checked that the population numbers here match those of the religion report

- Data issues: When merging it with the previous dataframe, a few (less than 20) counties were not matched (I tried to match them manually by changing their names, e.g. St. Lake **c**ity and St. Lake **C**ity), and were dropped. This also happened in the subsequent merges. Nonetheless we still keep 3126 in the final dataset.

## SAIPE data (2010)

Acquired poverty rates for 2010(`pov_saipe_2010`) and 2017 (`pov_saipe_2017`), merged them into the dataframe

## Mortality from substance abuse, violence (2010)

Acquired data from the Small Area Income and Poverty Estimates survey, extracted just the mortality rates for 2010 for alcohol use disorders, drug use dissorders, self-harm, and interpersonal violence. (`mortality_{alcohol,drugs_us,self_ha,interpe}`

## Statistics

The final dataset has 3126 entries.

- Poverty data: The SAIPE and census poverty columns are highly (r=.92) correlated and a the first component of a PCA explains .97 of their variance. Both will be retained for robustness checks.
- Likewise for population (First component explains .99 of the variance)
- The mortality variables were correlated to diverse degrees, and a single component explains .5 of their variance. I Created a `mortality` variable with it.
- Adding poverty(`pov_saipe_2010`) to the mix allows me to create a generic negative outcomes variable (again, a single factor, .47), this is `outcomes`.
- I create a`state_mormon` variable by aggregating the counties.

| variable          | Correlation with pov_saipe_2010 |
| ----------------- | ------------------------------- |
| pov_saipe_2017    | 0.933304                        |
| povrate           | 0.922399                        |
| outcomes          | 0.720665                        |
| mortality_interpe | 0.656781                        |
| blkrate           | 0.478595                        |
| b_protes          | 0.438711                        |
| mortality         | 0.408112                        |
| south_baptist     | 0.361322                        |
| mortality_drug_us | 0.321659                        |
| e_protes          | 0.276299                        |
| mortality_alcohol | 0.237223                        |
| mortality_self_ha | 0.213052                        |
| nativerate        | 0.204426                        |
| hisprate          | 0.142766                        |
| muslim            | -0.011683                       |
| o_jew             | -0.026969                       |
| religiosity       | -0.042128                       |
| mormon            | -0.050736                       |
| state_mormon      | -0.066232                       |
| pop2010           | -0.070696                       |
| pop_census        | -0.070750                       |
| asianrate         | -0.156543                       |
| catholic          | -0.226741                       |
| m_protes          | -0.307757                       |
| nhwrate           | -0.498856                       |
| income_census     | -0.708682                       |

## Pew data

Data obtained from http://www.pewforum.org/datasets/ (Religious landscape survey, 2014, phone survey)

It does not include data for alcoholism or drug use, but it has income (in brackets), so it may be useful for comparative study as a supplement to the analysis above.
