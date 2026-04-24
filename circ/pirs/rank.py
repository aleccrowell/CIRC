import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
import math
from tqdm import tqdm

class ranker:
    """
    Ranks and sorts expression profiles from most to least constitutive.


    This class takes a dataset of time series expression profiles.  It optionally performs an anova and removes time series with strong differential expression.  A linear regression is then performed and prediction intervals are calculated.  Ranking scores for an expression profile are produced by comparing the mean expression to the prediction intervals at each timepoint.


    Parameters
    ----------
    filename : str
        Path to the input dataset.
    anova : bool
        Whether or not to perform an ANOVA and remove profiles with strong differential expression.
    
    Attributes
    ----------
    data : dataframe
        This is where the input data is stored.
    anova : bool
        Whether an anova should be performed


    """
    def __init__(self, filename, anova=True):
        """
        Imports data and initializes a ranker object.


        Takes a file  which has one index column and a number of timepoint labeled data columns.

        """
        self.data = pd.read_csv(filename, sep='\t', header=0, index_col=0)
        self.data = self.data[(self.data.T != 0).any()]
        self.anova = anova

    def get_tpoints(self):
        """
        Extracts timepoints from header of data.


        Splits strings in header based on required syntax and generates timepoints.


        Attributes
        ----------
        tpoints : array
            array of timepoints at which samples were collected.

        """

        tpoints = [i.replace('ZT', '').replace('CT', '') for i in self.data.columns.values]
        tpoints = [int(i.split('_')[0]) for i in tpoints]
        #deprecated splitting for alternative header syntax
        #tpoints = [int(i.split('.')[0]) for i in tpoints]
        self.tpoints = np.asarray(tpoints)

    def remove_anova(self, alpha=0.05):
        """
        Performs anova to remove profiles with strong differential expression.


        Uses anova to detect expression profile with high confidence that some blocks of timepoint replicates have differential expression.  Removes these profiles as a filtering step since they definitionally will not be constitutively expressed.

        Parameters
        ----------
        alpha : float
            Significance threshold for anova filtering.

        """
        to_remove = []
        for index, row in self.data.iterrows():
            vals = []
            for i in list(set(self.tpoints)):
                vals.append([row.values[j] for j in range(len(row)) if self.tpoints[j] == i])
            f_val, p_val = stats.f_oneway(*vals)
            if p_val < alpha:
                to_remove.append(index)
        self.data = self.data[~self.data.index.isin(to_remove)]

    def calculate_scores(self, alpha=0.5):
        """
        Calculates prediction intervals and generates ranking scores.

        Performs linear regression and generates corresponding prediction intervals for each time series. Uses these values along with the expression mean to generate a ranking score.

        Parameters
        ----------
        alpha : float
            Significance level for prediction interval calculation.

        Attributes
        ----------
        errors : series
            Prediction interval ranking scores for each expression profile.

        """
        dof = len(np.unique(self.tpoints))
        es = {}
        for index in tqdm(range(len(self.data))):
            regr = linear_model.LinearRegression()
            _ = regr.fit(np.array(self.tpoints)[:, np.newaxis], np.array(self.data.iloc[index]))
            rsq = np.sum((regr.predict(np.array(self.tpoints)[:, np.newaxis]) - np.array(self.data.iloc[index])) ** 2)
            regr_error = math.sqrt(rsq/(dof-2))
            xsq = np.sum((np.array(self.tpoints) - np.mean(np.array(self.tpoints))) ** 2)
            upper = regr.predict(np.arange(min(self.tpoints), max(self.tpoints), 0.1)[:, np.newaxis]) + stats.t.ppf(1.0-alpha/2., dof) * regr_error*np.sqrt(1+1/dof+(((np.arange(min(self.tpoints), max(self.tpoints), 0.1)-np.mean(self.tpoints))**2)/xsq))
            lower = regr.predict(np.arange(min(self.tpoints), max(self.tpoints), 0.1)[:, np.newaxis]) - stats.t.ppf(1.0-alpha/2., dof) * regr_error*np.sqrt(1+1/dof+(((np.arange(min(self.tpoints), max(self.tpoints), 0.1)-np.mean(self.tpoints))**2)/xsq))
            pred = upper + lower
            mean_expr = np.mean(np.array(self.data.iloc[index]))
            denom = mean_expr ** 2 if mean_expr != 0 else 1.0
            error = np.sum([(i - mean_expr) ** 2 for i in pred]) / denom
            es[index] = np.mean(error)
        self.errors = pd.DataFrame.from_dict(es, orient='index')
        self.errors.columns = ['score']
        self.errors.index = self.data.index
        self.errors.sort_values('score', inplace=True)
        return self.errors

    def pirs_sort(self, outname=False):
        """
        Runs analysis pipeline and outputs data sorted by prediction interval ranking score.


        Takes input data, calculates timepoints, runs anova filtering if applicable, calculates PIRS and outputs data.

        Parameters
        ----------
        outname : str
            Path to desired output file.

        Returns
        -------
        sorted_data : dataframe
            Input data sorted by PIRS.

        """
        self.get_tpoints()
        if self.anova:
            self.remove_anova()
        self.calculate_scores()
        sorted_data = self.data.loc[self.errors.index.values]
        if outname:
            self.errors.to_csv(outname, sep='\t')
        return sorted_data

class rsd_ranker:
    """
    Ranks and sorts expression profiles from most to least constitutive using old algorithm for benchmarking.


    This class takes a dataset of time series expression profiles.  It then calculates Relative Standard Deviations (RSD) and ranks on this value.


    Parameters
    ----------
    filename : str
        Path to the input dataset.

    Attributes
    ----------
    data : dataframe
        This is where the input data is stored.


    """
    def __init__(self, filename):
        """
        Imports data and initializes a ranker object.


        Takes a file  which has one index column and a number of timepoint labeled data columns.

        """
        self.data = pd.read_csv(filename, sep='\t', header=0, index_col=0)
        self.data = self.data[(self.data.T != 0).any()]

    def calculate_scores(self):
        """
        Calculates Relative Standard Deviations.

        Calculates a ranking score.

        Attributes
        ----------
        rsd : series
            Relative Standard Deviation for each expression profile.

        """
        rsd = (1 + (1/(4*len(self.data)))) * np.std((self.data.values), axis=1) / np.abs(np.mean((self.data.values), axis=1))
        self.rsd = pd.DataFrame(rsd, index=self.data.index) 
        self.rsd.columns = ['score']
        self.rsd.sort_values('score', inplace=True)
        return self.rsd

    def rsd_sort(self, outname=False):
        """
        Runs analysis pipeline and outputs data sorted by relative standard deviation.


        Takes input data, calculates RSD, sorts and outputs data.

        Parameters
        ----------
        outname : str
            Path to desired output file.

        Returns
        -------
        sorted_data : dataframe
            Input data sorted by RSD.

        """
        self.calculate_scores()
        sorted_data = self.data.loc[self.rsd.index.values]
        if outname:
            self.rsd.to_csv(outname, sep='\t')
        return sorted_data

