# OpenNintil

This is a repository of the code used to produce figures and analysis for the [Nintil] blog(www.nintil.com).

The principles behind are:

1. Reproducibility: All figures and results should be easily reproducible from the provided code. This includes making it easy to acquire the data.
2. APIs, not files. As far as possible I will use API calls to source the data, instead of downloading files and placing them in special folders
3. Unified theme: All plots will have the same style, using `seaborn.set_context('talk'); plt.style.use('ggplot')`
4. Don't hide the mess: Keep all intermediate notebooks even if those analysis and explorations didn't lead anywhere.
5. Explained: Notebooks will have comments in case a piece of code is particularly unusual
