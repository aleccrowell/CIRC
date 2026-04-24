import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

##Simulating single expression level, would have to equalize distribution of mean expressins between classes if simulating multiple expression levels

class simulate:
    """
    Generates a properly formulated simulated circadian dataset for testing.


    This class takes user specifications on dataset size along with the number and frequency of batch effects and levels of background noise and outputs a simulated dataset along with a key showing which rows represent truly circadian data.


    Parameters
    ----------
    tpoints : int
        
    nrows : int

    nreps : int

    tpoint_space : int
        
    pcirc : float

    plin : float

    phase_prop : float

    phase_noise : float

    amp_noise : float


    Attributes
    ----------

    simdf : dataframe

    Simulated data without noise.

    simndf : dataframe

    Simulated data with noise.

    """

    def __init__(self, tpoints=24, nrows=1000, nreps=3, tpoint_space=2, pcirc=.4, plin=.4, phase_prop=.5, phase_noise=.05, amp_noise=.35,  rseed=0):
        """
        Simulates circadian, linear and constitutive data and saves as a properly formatted example .csv file.

        

        """

        np.random.seed(rseed)
        self.tpoints = int(tpoints)
        self.nreps = int(nreps)
        self.nrows = int(nrows)
        self.tpoint_space = int(tpoint_space)
        self.pcirc = float(pcirc)
        self.plin = float(plin)
        self.phase_prop = float(phase_prop)
        self.phase_noise = float(phase_noise)
        self.amp_noise = float(amp_noise)

        #procedurally generate column names
        self.cols = []
        for i in range(self.tpoints):
            for j in range(self.nreps):
                self.cols.append(
                    "ZT"+("{0:0=2d}".format(self.tpoint_space*i+self.tpoint_space))+'_'+str(j+1))

        #randomly determine which rows are circadian
        self.const = np.random.binomial(1, (1-self.pcirc-self.plin), self.nrows)
        #generate a base waveform
        base = np.arange(0, (4*np.pi), (4*np.pi/self.tpoints))
        #simulate data
        self.sim = []
        phases = []
        trend_types = np.random.binomial(1, self.pcirc, len(np.where(~self.const)[0]))
        for ind, val in enumerate(self.const):
            if val == 0:
                if trend_types[ind] == 1:
                    temp = []
                    p = np.random.binomial(1, self.phase_prop)
                    phases.append(p)
                    for j in range(self.nreps):
                        temp.append(np.sin(base+np.random.normal(0, self.phase_noise, 1) + np.pi*p)+np.random.normal(1, self.amp_noise, self.tpoints))
                    self.sim.append([item for sublist in zip(*temp) for item in sublist])
                else:
                    phases.append('nan')
                    temp = []
                    for j in range(self.nreps):
                        temp.append(2*np.arange(0, (self.tpoints*self.tpoint_space), self.tpoint_space)/(self.tpoints*self.tpoint_space)+np.random.normal(1, self.amp_noise, self.tpoints))
                    self.sim.append([item for sublist in zip(*temp) for item in sublist])
            else:
                phases.append('nan')
                temp = []
                for j in range(self.nreps):
                    temp.append(np.random.normal(1, self.amp_noise, self.tpoints))
                self.sim.append([item for sublist in zip(*temp) for item in sublist])
        
        self.sim = scale(np.array(self.sim), axis=1, with_mean=False)

    def write_output(self, out_name='simulated_data_with_noise.txt'):
        """

        out_name : str

        output file stem

        """

        self.out_name = str(out_name)
        self.simndf = pd.DataFrame(self.sim, columns=self.cols)
        self.simndf.index.names = ['#']
        self.simndf.to_csv(out_name, sep='\t')

        pd.DataFrame(self.const, columns=['Const'], index=self.simndf.index).to_csv(out_name[:-4]+'_true_classes.txt', sep='\t')

    def write_genorm(self, out_name='simulated_data_with_noise_bc.txt'):
        """

        out_name : str

        output file stem

        """

        self.out_name = str(out_name)
        self.simbcdf = pd.DataFrame(self.sim, columns=self.cols)
        self.simbcdf.index.names = ['#']
        self.simbcdf.reset_index(inplace=True)
        self.simbcdf = pd.melt(self.simbcdf, id_vars=['#'])
        self.simbcdf.columns = ['Detector','Sample','Cq']
        self.simbcdf = self.simbcdf[['Sample','Detector','Cq']]
        self.simbcdf.Sample = self.simbcdf.Sample.apply(lambda x: x.split('_')[0])
        self.simbcdf.Detector = self.simbcdf.Detector.astype(str)
        self.simbcdf.Sample = self.simbcdf.Sample.astype(str)
        self.simbcdf.Sample = self.simbcdf.Sample.apply(lambda x: 'timepoint_' + x)
        self.simbcdf.Detector = self.simbcdf.Detector.apply(lambda x: 'gene_' + x)
        self.simbcdf.Cq = (self.simbcdf.Cq - np.min(self.simbcdf.Cq))/(np.max(self.simbcdf.Cq) - np.min(self.simbcdf.Cq))
        self.simbcdf.to_csv(out_name, sep=' ', index=False)

    def write_normfinder(self, out_name='simulated_data_with_noise_bc.txt'):
        """

        out_name : str

        output file stem

        """

        self.out_name = str(out_name)
        self.simbcdf = pd.DataFrame(self.sim, columns=self.cols)
        self.simbcdf.index.names = ['#']
        self.simbcdf.reset_index(inplace=True)
        self.simbcdf = pd.melt(self.simbcdf, id_vars=['#'])
        self.simbcdf.columns = ['Detector', 'Sample', 'Cq']
        self.simbcdf = self.simbcdf[['Sample', 'Detector', 'Cq']]
        self.simbcdf.Detector = self.simbcdf.Detector.astype(str)
        self.simbcdf.Sample = self.simbcdf.Sample.astype(str)
        self.simbcdf.Sample = self.simbcdf.Sample.apply(
            lambda x: 'timepoint_' + x)
        self.simbcdf.Detector = self.simbcdf.Detector.apply(
            lambda x: 'gene_' + x)
        self.simbcdf.Cq = (self.simbcdf.Cq - np.min(self.simbcdf.Cq))/(np.max(self.simbcdf.Cq) - np.min(self.simbcdf.Cq))
        self.simbcdf.to_csv(out_name, sep=' ', index=False)

class analyze:
    def __init__(self):
        self.true_classes = pd.DataFrame()
        self.merged = pd.DataFrame()

    def add_classes(self, filename_classes, rep=0):
        tc = pd.read_csv(filename_classes, sep='\t')
        tc['rep'] = rep
        self.true_classes = pd.concat([self.true_classes,tc])

    def add_data(self, filename_pirs, tag, rep=0):
        ranks = pd.read_csv(filename_pirs, sep='\t')
        ranks['method'] =  tag
        ranks['rep'] = rep
        ranks['score'] = ranks['score'].fillna(ranks['score'].max())
        if self.merged.empty:
            self.merged = ranks
        else:
            self.merged = pd.concat([self.merged, ranks])

    def generate_pr_curve(self):
        self.curves = pd.DataFrame(columns=['precision','recall','method'])
        #colors = ["light grey", "black"]
        colors = ["windows blue","amber","light grey", "black"]
        #need to pivot out and fillna with max to replace missing values
        self.merged = self.merged.pivot_table(index=['rep', 'method'], columns='#', values='score').fillna(self.merged.score.max()).reset_index().melt(id_vars=['rep', 'method'], value_name='score')
        for rep in self.merged.rep.unique():
            for method in self.merged.method.unique():
                pr = pd.merge(self.true_classes[self.true_classes['rep'] == rep], self.merged[(self.merged.rep == rep) & (self.merged.method == method)], on=['#', 'rep'])
                precision, recall, _ = precision_recall_curve(pr['Const'].values, 1/pr['score'].values, pos_label=1)
                temp = pd.DataFrame()
                temp['precision'] = precision
                temp['recall'] = recall
                temp['method'] = method
                temp['rep'] = rep
                self.curves = pd.concat([self.curves, temp],sort=False)
        ax = sns.lineplot(x='recall', y='precision', hue='method', units='rep', palette=sns.xkcd_palette(colors), estimator=None, data=self.curves)
        ax.set_aspect(aspect=0.5)
        plt.plot([0, 1], [np.mean(self.true_classes['Const']), np.mean(self.true_classes['Const'])], color='r', linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.setp(ax.lines, linewidth=0.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Comparison')
        #plt.legend(loc="center right")
        leg = plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True)
        #leg._legend_box.align = "right"
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        plt.savefig('PR.pdf',dpi=25)
        plt.close()

