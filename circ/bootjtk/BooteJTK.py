#!/usr/bin/env python
"""
Created on Nov 1, 2015
@author: Alan L. Hutchison, alanlhutchison@uchicago.edu, Aaron R. Dinner Group, University of Chicago

This script is a bootstrapped expansion of the eJTK method described in

Hutchison AL, Maienschein-Cline M, and Chiang AH et al. Improved statistical methods enable greater sensitivity in rhythm detection for genome-wide data, PLoS Computational Biology 2015 11(3): e 1004094. doi:10.1371/journal.pcbi.1004094

This script bootstraps time series and provides phase and tau distributions from those bootstraps to allow for measurement of the variance on phase and tau estimates.


Please use ./BooteJTK -h to see the help screen for further instructions on running this script.

"""
VERSION="0.1"

#import cmath
from scipy.stats import circmean as sscircmean
from scipy.stats import circstd as sscircstd
#import scipy.stats as ss
import numpy as np
#from scipy.stats import kendalltau as kt
from scipy.stats import multivariate_normal as mn
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.special import polygamma

import multiprocessing
import pickle
#from operator import itemgetter
import sys
import argparse
import time
import os
import os.path

from .get_stat_probs import get_stat_probs as gsp_get_stat_probs
from .get_stat_probs import get_waveform_list as gsp_get_waveform_list
from .get_stat_probs import make_references as gsp_make_references
from .get_stat_probs import rank_references as gsp_rank_references
from .get_stat_probs import kt

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_files')



_g_triples = None
_g_dref = None
_g_new_header = None
_g_ref_ranks = None

def _init_worker(triples, dref, new_header, ref_ranks):
    global _g_triples, _g_dref, _g_new_header, _g_ref_ranks
    _g_triples, _g_dref, _g_new_header, _g_ref_ranks = triples, dref, new_header, ref_ranks

def _process_gene(args):
    geneID, d_data_item, serie_item, precomputed_dorder, size, fn, waveform = args
    if fn == 'DEFAULT' or serie_item is None:
        mmax = mmin = MAX_AMP = sIQR_FC = smean = sstd = sFC = np.nan
    else:
        mmax, mmin, MAX_AMP = series_char(serie_item)
        sIQR_FC = IQR_FC(serie_item)
        smean = series_mean(serie_item)
        sstd = series_std(serie_item)
        sFC = FC(serie_item)
    s_stats = [smean, sstd, mmax, mmin, MAX_AMP, sFC, sIQR_FC, size]
    if precomputed_dorder is not None:
        dorder = precomputed_dorder
        gene_order_probs = precomputed_dorder
        gene_boots = None
    else:
        d_op, d_b = get_order_prob({geneID: d_data_item}, size)
        if geneID not in d_op:
            return None
        dorder = d_op[geneID]
        gene_order_probs = d_op[geneID]
        gene_boots = d_b[geneID]
    out1, out2, d_taugene, d_pergene, d_phgene, d_nagene = gsp_get_stat_probs(
        dorder, _g_new_header, _g_triples, _g_dref, _g_ref_ranks, size)
    out_line = [str(l) for l in [geneID, waveform] + out1 + s_stats + out2]
    return geneID, out_line, d_taugene, d_phgene, gene_order_probs, gene_boots


def main(args):

    fn = args.filename
    fn_means = args.means
    fn_sds = args.sds
    fn_ns = args.ns

    prefix = args.prefix
    fn_waveform = args.waveform
    fn_period = args.period
    fn_phase = args.phase
    fn_width = args.width
    fn_out = args.output
    fn_out_pkl = args.pickle # This is the output file which could be read in early
    fn_list = args.id_list # This is the the list of ids to go through
    fn_null_list = args.null_list # These are geneIDs to be used to estimate the SD
    size = int(args.size)
    reps = int(args.reps)
    write_pickle = args.write

    
    ### If no list file set id_list to empty
    id_list = read_in_list(fn_list) if fn_list.split('/')[-1]!='DEFAULT' else []
    null_list = read_in_list(fn_null_list) if fn_null_list.split('/')[-1]!='DEFAULT' else []
        
    ### If no pkl out file, modify the option variable
    opt = 'premade' if os.path.isfile(fn_out_pkl) and fn_out_pkl.split('/')[-1]!='DEFAULT' else ''

    ### If not given a new name, name fn_out after the fn file
    if fn_out.split('/')[-1] == "DEFAULT":
        endstr = '_{0}_boot{1}-rep{2}.txt'.format(prefix,int(size),int(reps))
        fn_out = fn.replace('.txt',endstr) if  ".txt" in fn else fn+endstr

    ### If not given a new name, name fn_out_pkl based on the fn file
    if fn_out_pkl.split('/')[-1] == 'DEFAULT':
        endstr = '_{0}_boot{1}-rep{2}_order_probs.pkl'.format(prefix,int(size),int(reps))
        fn_out_pkl = fn.replace('.txt',endstr) if  ".txt" in fn else fn+endstr

    ### Name vars file based on pkl out file
    fn_out_pkl_vars = fn_out_pkl.replace('.pkl','_vars.pkl') 

    assert fn.split('/')[-1]!='DEFAULT' or fn_out_pkl.split('/')[-1]!='DEFAULT'   

    ### If we already have the PKL file, we just need a place to put the header information
    #if fn.split('/')[-1]=='DEFAULT' and fn_out_pkl.split('/')[-1]!='DEFAULT':
    #    d_series = dict(zip([key for key in d_data_master.keys()],[[]*len(d_data_master)]))
    #    fn_out= fn_out_pkl.replace('.pkl','_{0}-bootejtk.txt'.format(int(size)))
    #    new_header = [0,4,8,12,16,20,0,4,8,12,16,20] if fn_out_pkl[:-4]=='2.pkl' else [0,4,8,12,16,20]
    ### If we have the initial data we can get it 
    #elif fn.split('/')[-1]!='DEFAULT':
    ### WE HAVE CHANGED HOW THE DATA GETS PUT IN

    print('Going to read in data now')
    """ Read in the data """
    header,data = read_in(fn)
    d_series = dict_data(data)

    periods = np.array(read_in_list(fn_period),dtype=int)
    period = float(periods[0])
    phases = np.array(read_in_list(fn_phase),dtype=int)
    widths = np.array(read_in_list(fn_width),dtype=int)

    waveform = 'cosine'
    
    if fn_means.split('/')[-1]!='DEFAULT' and fn_sds.split('/')[-1]!='DEFAULT' and fn_ns.split('/')[-1]!='DEFAULT':
        print('Taking Limma/noreps input')
        header2,means = read_in(fn_means)
        _,sds = read_in(fn_sds)
        _,ns = read_in(fn_ns)
        d_data_master,new_header = get_data_multi(header,header2,means,sds,ns,period)
    else:
        d_data_master,new_header = get_data2(header,data,period)
        
        #new_header = list(new_header)*reps
    
        if 'premade' not in opt:
            print('Running internal eBayes')
            D_null = get_series_data(d_data_master,null_list) if null_list!=[] else {}
            d_data_master = eBayes(d_data_master,D_null)
        elif 'premade' in opt:
            d_data_master,d_order_probs = pickle.load(open(fn_out_pkl,'rb'))

        
    def add_on_out(outfile):
        add_on = 1
        while os.path.isfile(outfile):
            print(outfile, "already exists, take evasive action!!!")
            if '.txt' in outfile:
                end = '.txt'
            elif '.pkl' in outfile:
                end = '.pkl'
            origstr = end if add_on==1 else '_{0}'.format(add_on-1)+end
            outfile = outfile.replace(origstr,'_{0}'.format(add_on))+end
            add_on = add_on + 1
        return outfile

    
    fn_out = add_on_out(fn_out)
    fn_out_pkl = add_on_out(fn_out_pkl)
    fn_out_pkl_vars = add_on_out(fn_out_pkl_vars)    


    waveform = 'cosine'

    triples = gsp_get_waveform_list(periods,phases,widths)

    dref = gsp_make_references(new_header, triples, waveform)
    ref_ranks = gsp_rank_references(dref, triples)
    
    d_data_master1 = {}
    d_order_probs_master = {}
    d_boots_master = {}

    Ps = []

    d_tau = {}
    d_ph = {}

    done = []
    remaining = []

    id_list = d_series.keys() if id_list==[] else id_list
    out_lines = []

    d_order_probs_preloaded = d_order_probs if 'premade' in opt else {}
    gene_ids = [geneID for geneID in d_data_master if geneID in id_list]
    gene_args = [
        (geneID, d_data_master[geneID], d_series.get(geneID),
         d_order_probs_preloaded.get(geneID),
         size, fn, waveform)
        for geneID in gene_ids
    ]

    n_workers = args.workers
    pool_size = n_workers if n_workers > 0 else None
    actual_workers = pool_size or multiprocessing.cpu_count()
    print(f'Processing {len(gene_ids)} gene(s) with {actual_workers if n_workers != 1 else 1} worker(s)...')
    t0 = time.time()
    if n_workers == 1:
        _init_worker(triples, dref, new_header, ref_ranks)
        results = [_process_gene(a) for a in gene_args]
    else:
        with multiprocessing.Pool(pool_size, initializer=_init_worker,
                                   initargs=(triples, dref, new_header, ref_ranks)) as pool:
            chunksize = max(1, len(gene_args) // (actual_workers * 4))
            results = pool.map(_process_gene, gene_args, chunksize=chunksize)
    print(f'Done in {time.time() - t0:.1f}s')

    for result in results:
        if result is None:
            continue
        geneID, out_line, d_taugene, d_phgene, gene_order_probs, gene_boots = result
        out_lines.append(out_line)
        done.append(geneID)
        d_tau[geneID] = d_taugene
        d_ph[geneID] = d_phgene
        if write_pickle and 'premade' not in opt:
            d_data_master1[geneID] = d_data_master[geneID]
            d_order_probs_master[geneID] = gene_order_probs
            if gene_boots is not None:
                d_boots_master[geneID] = gene_boots
            
    if write_pickle==True: 
        pickle.dump([d_tau,d_ph],open(fn_out_pkl_vars,'wb'))                    
        pickle.dump([d_data_master1,d_order_probs_master,d_boots_master],open(fn_out_pkl,'wb'))
    else:
        print('Not writing out pickle results')

    taus = [[i,float(out[-2])] for i,out in enumerate(out_lines)]
    taus = sorted(taus,key=lambda x: np.abs(x[1]),reverse=True)

    indexes = np.array([i[0] for i in taus])

    out_lines = np.array(out_lines)[np.array(indexes)]
    g = open(fn_out,'a')
    g.write("ID\tWaveform\tPeriodMean\tPeriodStdDev\tPhaseMean\tPhaseStdDev\tNadirMean\tNadirStdDev\tMean\tStd_Dev\tMax\tMin\tMax_Amp\tFC\tIQR_FC\tNumBoots\tTauMean\tTauStdDev\n")
    for out_line in out_lines:
        g.write("\t".join(out_line)+"\n")
    g.close()

    return fn_out,fn_out_pkl,header
    

    
def read_in_EMdata(fn):
    """Reads in one of the two EM files """
    WT = {}
    with open(fn,'r') as f:
        for line in f:
            words = line.strip('\n').split()
            if words[0]=='#':
                header = words
            else:
                key = words[0]
                m = [float(w) for w in words[1:7]]
                s = [float(s) for s in words[7:]]
                    
                WT[key] = [m,s,np.ones(6)*3]
    return WT


#def get_SD_distr(dseries,reps,size):
#    data = np.zeros(size)
#    for j in xrange(size):
#        ser = []
#        for i in xrange(len(dseries[0])):
#            ser.append(np.random.normal(dseries[0][i],dseries[1][i],size=reps))
#        ser = np.concatenate(ser)
#        SD = np.std(ser)
#        data[j] = SD
#    return data


def append_out(fn_out,line):
    line = [str(l) for l in line]
    with open(fn_out,'a') as g:
        g.write("\t".join(line)+"\n")

def write_out(fn_out,output):
    with open(fn_out,'w') as g:
        for line in output:
            g.write(str(line)+"\n")

def is_number(s):
    try:
        return np.isfinite(float(s))
    except (TypeError, ValueError):
        return False


def read_in_list(fn):
    with open(fn,'r') as f:
        lines = f.read().splitlines()
    return lines
        
def read_in(fn):
    """Read in data to header and data"""
    with open(fn,'r') as f:
        data=[]
        start_right=0
        for line in f:
            words = line.strip().split()
            words = [word.strip() for word in words]
            if words[0] == "#" or words[0]=='ID':
                start_right = 1
                header = words[1:]
            else:
                if start_right == 0:
                    print("Please enter file with header starting with # or ID")
                    exit
                elif start_right == 1:
                    data.append(words)
    return header, data

def dict_data(data):
    d_series = {}
    for dat in data:
        d_series[dat[0]] = dat
    return d_series


def IQR_FC(series):
    qlo = __score_at_percentile__(series, 25)
    qhi = __score_at_percentile__(series, 75)
    if not is_number(qlo) or not is_number(qhi):
        return np.nan
    elif (qhi==0):
        return 0
    elif ( qlo==0):
        return np.nan
    else:
        #print qhi,qlo
        iqr = qhi/qlo
        return iqr

def FC(series):
    series=[float(s) if is_number(s) else 0 for s in series[1:]]
    if series!=[]:
        mmax = max(series)
        mmin = min(series)
        if mmin==0:
            sFC = -10000
        else:
            sFC = mmax / mmin
    else:
        sFC = np.nan
    return sFC


def series_char(series):
    """Uses interquartile range to estimate amplitude of a time series."""
    series=[float(s) for s in series[1:] if is_number(s)]
    if series!=[]:
        mmax = max(series)
        mmin = min(series)
        diff=mmax-mmin
    else:
        mmax = "NA"
        mmin = "NA"
        diff = "NA"
    return mmax,mmin,diff


def series_mean(series):
    """Finds the mean of a timeseries"""
    series = [float(s) for s in series[1:] if is_number(s)]
    return np.mean(series)

def series_std(series):
    """Finds the std dev of a timeseries"""
    series = [float(s) for s in series[1:] if is_number(s)]
    return np.std(series)

def __score_at_percentile__(ser, per):
    ser = [float(se) for se in ser[1:] if is_number(se)]
    if len(ser)<5:
        score ="NA"
        return score
    else: 
        ser = np.sort(ser)
        i = int(per/100. * len(ser))
        if (i % 1 == 0):
            score = ser[i]
        else:
            interpolate = lambda a,b,frac: a + (b - a)*frac
            score = interpolate(ser[int(i)], ser[int(i) + 1], i % 1)
        return float(score)

def generate_mod_series(reference,series):
    """
    Takes the series from generate_base_null, takes the list from data, and makes a null
    for each gene in data or uses the one previously calculated.
    Then it runs Kendall's Tau on the exp. series against the null
    """
    tau,p=kt(series,reference)
    p = p / 2.0
    return tau,p

##################################
### HERE WE INSERT BOOT FUNCTIONS
##################################

def get_data(header,data,period):
    """This function does not use the header information to set the number of replicates """
    new_h = [float(h[2:].split('_')[0])%period if 'ZT' in h or 'CT' in h else float(h.split('_')[0])%period for h in header]
    #print new_h
    length = len(new_h)
    seen = []
    dref = {}
    for i,h in enumerate(new_h):
        if h not in seen:
            seen.append(h)
            dref[h]=[i]
        else:
            dref[h].append(i)
    d_data = {}
    #print dref
    for dat in data:
        name=dat[0]
        series = [float(da) if is_number(da) else np.nan for da in dat[1:]]
        if len(series)==length:
            out = [[],[],[]]
            for i,s in enumerate(seen):
                points = [series[idx] for idx in dref[s]]
                N = len([p for p in points if not np.isnan(p)])
                m = np.nanmean(points)
                std = np.nanstd(points)
                out[0].append(m)
                out[1].append(std)
                out[2].append(N)
            #print name,seen,out
            d_data[name]=out
    #print d_data.keys()
    return d_data,seen


def get_data2(header,data,period):
    """ This function uses the header information to set the number of replicates"""
    new_h = [float(h[2:].split('_')[0])%period if 'ZT' in h or 'CT' in h else float(h.split('_')[0])%period for h in header]
    length = len(new_h)
    seen = []
    dref = {}
    for i,h in enumerate(new_h):
        if h not in seen:
            seen.append(h)
            dref[h]=[i]
        else:
            dref[h].append(i)
    d_data = {}
    for dat in data:
        name=dat[0]
        series = [float(da) if is_number(da) else np.nan for da in dat[1:]]

        out = [[],[],[]]
        for i,s in enumerate(new_h):
            points = [series[idx] for idx in dref[s]]
            N = len([p for p in points if not np.isnan(p)])
            m = np.nanmean(points)
            std = np.nanstd(points)
            out[0].append(m)
            out[1].append(std)
            out[2].append(N)
            #print name,seen,out
        d_data[name]=out
    #print d_data.keys()
    return d_data,new_h



def get_data_multi(header,header2,data,sds,ns,period):
    """ This function takes in several pre-eBayes files to create the d_data dictionary"""
    new_h = [float(h[2:].split('_')[0])%period if 'ZT' in h or 'CT' in h else float(h.split('_')[0])%period for h in header]
    h2 = [float(h[2:].split('_')[0])%period if 'ZT' in h or 'CT' in h else float(h.split('_')[0])%period for h in header2]    
    length = len(new_h)
    seen = []
    dref = {}
    for i,h in enumerate(new_h):
        seen.append(h2.index(h))
    d_data = {}
    for j,dat in enumerate(data):
        g_sds = sds[j]
        g_ns = ns[j]
        name=dat[0]
        g_means = [float(da) if is_number(da) else np.nan for da in dat[1:]]
        g_means = [g_means[i] for i in seen]
        g_sds =   [float(da) if is_number(da) else np.nan for da in g_sds[1:]]
        g_sds = [g_sds[i] for i in seen]
        g_ns =   [float(da) if is_number(da) else np.nan for da in g_ns[1:]]
        g_ns = [g_ns[i] for i in seen]
        out = [g_means,g_sds,g_ns]
        d_data[name]=out

    return d_data,new_h




def get_series_data(d_data_master,id_list):
    d_data = {}
    for key in d_data_master:
        if key in id_list:
            dataset = d_data_master[key]
            #print key,datase
            N = np.sum(dataset[2])
            length = len(dataset[0])
            one = 1./N*np.sum([dataset[2][i]*(dataset[1][i]**2+dataset[0][i]**2) for i in range(length)])
            two = (1./N * np.sum([dataset[2][i]*dataset[0][i] for i in range(length)]))**2
            std = np.sqrt(one+two)
            m = 1./N *np.sum([dataset[2][i]*dataset[0][i] for i in range(length)])
            d_data[key] = [m,std,N]
    return d_data

def eBayes(d_data,D_null={}):
    """
    This is based on Smyth 2004 Stat. App. in Gen. and Mol. Biol. 3(1)3
    It generates a dictionary with eBayes adjusted variance values.
    Ns have been set to 1 to not complicate downstream analyses.
    """
    
    def get_d0_s0(d_data,D_null):
        digamma = lambda x: polygamma(0,x)
        trigamma = lambda x: polygamma(1,x)
        tetragamma = lambda x: polygamma(2,x)

        def solve_trigamma(x):
            """To solve trigamma(y)=x, x>0"""
            y0 = 0.5 + 1./x
            d =1000000.
            y = y0
            while -d/y >1e-8:
                d = trigamma(y)*(1-trigamma(y)/x)/tetragamma(y)
                y = y+d
            return y

        if D_null!={}:
            dg = np.hstack(np.array(list(D_null.values()))[:,2])
            s  = np.hstack(np.array(list(D_null.values()))[:,1])
        else:
            dg = np.hstack(np.array(list(d_data.values()))[:,2])
            s  = np.hstack(np.array(list(d_data.values()))[:,1])
        #print D_null.keys()    
        #print dg,s
        G = len(d_data)
        s2 = np.array([ss for ss in s if ss!=0])
        dg2 = np.array([dg[i] for i,ss in enumerate(s) if ss!=0])
        G = len(dg2)
        z = 2.*np.log(s2)

        e = z - digamma(dg2/2) + np.log(dg2/2)
        emean = np.nanmean(e)
        d0 = 2.* solve_trigamma( np.nanmean( (e-emean)**2 *G/(G-1)-trigamma(dg2/2) )   )
        s0 = np.sqrt(np.exp(emean + digamma(d0/2)- np.log(d0/2.)))

        #print d0,s0
        return d0,s0
    def posterior_s(d0,s0,s,d):
        return np.sqrt( (d0*s0**2 + d*s**2) /(d0+d) )

    d0,s0=get_d0_s0(d_data,D_null)
    
    for key in d_data:
        s_arr = np.array(d_data[key][1], dtype=float)
        d_arr = np.array(d_data[key][2], dtype=float)
        d_data[key][1] = list(np.sqrt((d0 * s0**2 + d_arr * s_arr**2) / (d0 + d_arr)))
        d_data[key][2] = [1] * len(s_arr)
    return d_data

def get_order_prob(d_data,size):
    ### WE WANT TO CHANGE HOW REPS GET INCORPORATED...
    d_order_prob = {}
    d_boots = {}
    for key in d_data:
        res = d_data[key]
        #print 'res is', res
        d_order_prob[key],d_boots[key]=dict_of_orders(res[0],res[1],res[2],size)        
        #d_order_prob[key],d_boots[key]=dict_of_orders(list(res[0])*reps,list(res[1])*reps,list(res[2])*reps,size)
    return d_order_prob,d_boots

def dict_of_orders(M,SDS,NS,size):
    """
    This produces a dictionary of probabilities
    for the different orders given the data
    """
    index = range(len(M))
    dorder,s2 = dict_order_probs(M,SDS,NS,size)
    d = {}
    for idx in dorder:
        d[idx]=dorder[idx]
    SUM = sum(d.values())
    for key in d:
        d[key]=d[key]/SUM
    return d,s2

def dict_order_probs(ms, sds, ns, size=100):
    sds = [sd if is_number(sd) else np.nanmean(sds) for sd in sds]
    ms_arr = np.array(ms, dtype=float)
    sds_arr = np.array(sds, dtype=float)
    s3 = np.random.normal(ms_arr, sds_arr, size=(size, len(ms_arr)))
    d = {}
    for r in rankdata(s3, axis=1):
        key = tuple(map(int, r))
        d[key] = d.get(key, 0) + 1.0 / size
    return d, s3


def __create_parser__():
    p = argparse.ArgumentParser(
        description="Bootstrap empirical JTK_CYCLE (eJTK) for circadian rhythm detection. "
                    "See Hutchison et al. PLoS Comput Biol 2015 11(3):e1004094.",
        epilog="Either -f (raw data) or -k (pre-computed pickle) must be supplied.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--version', '-V', action='version', version='%(prog)s ' + VERSION)

    p.add_argument("-o", "--output",
                   dest="output",
                   action='store',
                   metavar="FILE",
                   type=str,
                   default="DEFAULT",
                   help="Output filename. Defaults to the input filename with a run-specific suffix.")

    p.add_argument("-k", "--pickle",
                   dest="pickle",
                   metavar="FILE",
                   type=str,
                   action='store',
                   default="DEFAULT",
                   help="Path to a pre-computed bootstrap pickle file. "
                        "Skips bootstrap generation when provided.")

    p.add_argument("-l", "--list",
                   dest="id_list",
                   metavar="FILE",
                   type=str,
                   action='store',
                   default="DEFAULT",
                   help="File of gene/series IDs to analyse (one per line). "
                        "Analyses all IDs if omitted.")

    p.add_argument("-n", "--null",
                   dest="null_list",
                   metavar="FILE",
                   type=str,
                   action='store',
                   default="DEFAULT",
                   help="File of non-cycling gene IDs used to estimate variance for eBayes shrinkage. "
                        "Ignored when -k is used.")

    analysis = p.add_argument_group(title="Analysis options")

    analysis.add_argument("-f", "--filename",
                          dest="filename",
                          action='store',
                          metavar="FILE",
                          default='DEFAULT',
                          type=str,
                          help="Tab-delimited input file. Header row must start with # or ID "
                               "followed by timepoints in ZTn or CTn format. "
                               "Use NA for missing values.")

    analysis.add_argument("-F", "--means",
                          dest="means",
                          action='store',
                          metavar="FILE",
                          default='DEFAULT',
                          type=str,
                          help="Pre-computed timepoint means file (Limma/no-reps input). "
                               "Same format as -f.")

    analysis.add_argument("-S", "--sds",
                          dest="sds",
                          action='store',
                          metavar="FILE",
                          default='DEFAULT',
                          type=str,
                          help="Pre-computed timepoint standard deviations file (Limma/no-reps input).")

    analysis.add_argument("-N", "--ns",
                          dest="ns",
                          action='store',
                          metavar="FILE",
                          default='DEFAULT',
                          type=str,
                          help="Pre-computed timepoint replicate-count file (Limma/no-reps input).")

    analysis.add_argument("-W", "--write",
                          dest="write",
                          action='store_true',
                          default=False,
                          help="Write bootstrap order-probability dictionaries to a pickle file.")

    analysis.add_argument('-x', "--prefix",
                          dest="prefix",
                          type=str,
                          metavar="STR",
                          action='store',
                          default="",
                          help="Label inserted into all output filenames for this run.")

    analysis.add_argument('-r', "--reps",
                          dest="reps",
                          type=int,
                          metavar="N",
                          action='store',
                          default=2,
                          help="Replicates per timepoint to bootstrap.")

    analysis.add_argument('-z', "--size",
                          dest="size",
                          type=int,
                          metavar="N",
                          action='store',
                          default=50,
                          help="Number of bootstrap resamples per gene.")

    analysis.add_argument('-j', '--workers',
                          dest='workers',
                          type=int,
                          metavar='N',
                          action='store',
                          default=1,
                          help='Parallel worker processes. 0 = all available CPUs.')

    analysis.add_argument('-w', "--waveform",
                          dest="waveform",
                          type=str,
                          metavar="STR",
                          action='store',
                          default="cosine",
                          choices=["cosine", "trough", "impulse", "step"],
                          help="Reference waveform shape to match against.")

    analysis.add_argument("--width", "-a", "--asymmetry",
                          dest="width",
                          type=str,
                          metavar="FILE",
                          action='store',
                          default=os.path.join(_REF_DIR, 'asymmetries_02-22_by2.txt'),
                          help="File listing asymmetry (width) values to search, one per line.")

    analysis.add_argument('-s', "-ph", "--phase",
                          dest="phase",
                          metavar="FILE",
                          type=str,
                          default=os.path.join(_REF_DIR, 'phases_00-22_by2.txt'),
                          help="File listing phases to search (hours), one per line.")

    analysis.add_argument("-p", "--period",
                          dest="period",
                          metavar="FILE",
                          type=str,
                          action='store',
                          default=os.path.join(_REF_DIR, 'period24.txt'),
                          help="File listing period(s) to search (hours), one per line.")

    distribution = analysis.add_mutually_exclusive_group(required=False)
    distribution.add_argument("-e", "--exact",
                               dest="harding",
                               action='store_true',
                               default=False,
                               help="Use Harding's exact null distribution.")
    distribution.add_argument("-g", "--gaussian", "--normal",
                               dest="normal",
                               action='store_true',
                               default=False,
                               help="Use normal approximation to null distribution.")

    return p




def cli():
    parser = __create_parser__()
    args = parser.parse_args()
    main(args)


if __name__=="__main__":
    cli()
