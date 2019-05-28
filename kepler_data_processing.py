################
## Packages to be used
####

import matplotlib.pyplot as plt
import lightkurve as lk
import pandas as pd
import numpy as np
import time
import os

################
## Methods to be used for data cleaning/processing
####

def open_files(csv_file):
    '''
    The open_files function opens the csv and extracts the needed data.
    Works on entire file at once.
    
    Args: 
        csv_file: Should contain desired TCEs and parameters. 
          Required Paremeter: kepid,
                              av_training_set,
                              tce_plnt_num,
                              tce_period,
                              tce_time0bk.
    
    Returns:
        tce_data: Panda DataFrame that has the parameters listed above.
    '''
    
    ## Opening csv as tce_info using pandas
    ## This contains all metadata for all TCEs
    tce_info = pd.read_csv(csv_file)
    
    ## Removing TCEs with label 'UNK'
    tce_info = tce_info[tce_info.av_training_set != 'UNK']#.reset_index() 
    
    ## Isolate AFP and NTP
    not_pc = tce_info[tce_info['av_training_set'] != 'PC']

    ## Isolate all PCs
    pc_only = tce_info[tce_info['av_training_set'] == 'PC']

    ## Only keep TCE with tce_plnt_num == 1 and reset the index
    pc_only = pc_only[pc_only.tce_plnt_num == 1].reset_index(drop = True)

    ## Add PCs back to full set
    tce_info = pd.concat([pc_only, not_pc], ignore_index=True, sort=False)

    ## Shuffle the dataframe because all PCs are at the start
    tce_info = tce_info.sample(frac=1).reset_index(drop=True)
    

    ## Extracting and combining kepids, periods, epochs into one DataFrame
    tce_data = tce_info[['kepid', 
                         'av_training_set', 
                         'tce_plnt_num', 
                         'tce_period', 
                         'tce_time0bk']]
    
    return tce_data
    
    
def get_metadata(kepid, tce_data):
    '''
    The get_metadata function gathers required metadata.
    Works on one kepid at a time.
    
    Args: 
        kepid: Object of interest.
        tce_data: DataFrame containing needed parameter values.
    
    Returns:
        period: Period given by Kepler pipeline.
        tranmid: Time corresponding to zero phase; used for folding a lc.
    '''
    
    ## Getting period and tranmid
    period = tce_data[tce_data.kepid == kepid]
    if len(period) == 1:
        period = float(period['tce_period'])
    else:
        temp_var_period = 1
        for i in period['tce_period']:
            if (temp_var_period == 1):
                period = i
            i = i+1
            
    tranmid = tce_data[tce_data.kepid == kepid]
    if len(tranmid) == 1:
        tranmid = float(tranmid['tce_time0bk'])
    else:
        temp_var_tranmid = 1
        for i in tranmid['tce_time0bk']:
            if (temp_var_tranmid == 1):
                tranmid = i
            i = i+1
    
    ## Returns period and tranmid
    return period, tranmid
    
    
def get_kepid_files(kepid):
    '''
    The get_kepid_files function gathers list of required files.
    There are about ~12-18 fits files per kepid.
    Works on one kepid at a time.
    
    Args: 
        kepid: Object of interest.
    
    Returns:
        fits_files: List of all fits files corresponding to the kepid.
    '''
    
    ## Pad the kepid with leading zeros to be a str of length 9
    kepid = str(kepid).zfill(9)
    
    ## Get the first four numbers (because of the filesystem)
    kepid_front = str(kepid[0:4])
    
    ## Will hold paths to fits files for the input kepid
    path = ('data' + '/' + kepid_front + '/' + kepid + '/')
    fits_files = []
    
    ## Searches the path for fits files
    ## r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'kplr'+kepid in file:
                fits_files.append(os.path.join(r, file))
    
    # Returns list containing all found files
    return fits_files
    
    
def get_total_flux(kepid,tce_data,window_length=101,binsize='calculated'): 
    '''
    The get_total_flux function stiches all the cleaned fits files 
      and cleans the folded light curve.
    Works on one kepid at a time.
    
    Each LightCurve file contains two fluxes (SAP_FLUX and PDCSAP_FLUX).
    PDCSAP_FLUX will be chosen because it has been through an extra
        flattening process in the Kepler pipeline.
    
    Args: 
        kepid: Object of interest.
        tce_data: DataFrame containing needed parameter values.
        window_length: Used when flattening. Default = 101.
        binsize: Used when binning. If binsize is explicitly given as
            a float, that value will be used. Otherwise, binsize will be 
            calculated so that the output has length 2001.
    
    Returns:
        main_lc: Main light curve corresponding to the given kepid.
            Has been normalized, flattened, folded, binned, and made to
            the correct length (of 2001).
    '''
    
    ## Getting all fits files for this kepid
    paths = get_kepid_files(kepid)
    
    ## Opening the first fits file (will append others onto this one)
    main_lc = lk.search.open(
        paths[0]).PDCSAP_FLUX.flatten(
        window_length=window_length)
    
    ## Append the remaining fits files to the first fits file
    for i in range(len(paths)-1): 
        ## Opening the following fits file
        main_lc = main_lc.append(lk.search.open(
            paths[i+1]).PDCSAP_FLUX.flatten(
            window_length=window_length))
    
    ## Getting kepid's metadata
    period, tranmid = get_metadata(kepid, tce_data)

    ## Calculating binsize to make total length of vector 2001
    if binsize == 'calculated':
        binsize = round((len(main_lc.flux) / 2001.4), 4)
    elif type(binsize) == float:
        binsize = binsize
    
    ## Returning the cleaned main_lc
    return main_lc.fold(
            period=period, 
            t0=tranmid).bin(
            binsize=binsize).normalize()


def print_info(tce_num, tce_data, i):
    '''
    The print_info function will display the progress of the 
        light curve cleaning process, depending on dataset size.
    The printed info will look like this:
        08 Processing:  kepid-001026133 UNK
        (IndexNumber):     (kepid)    (Label)
    
    Args: 
        tce_num: int; number of TCEs in dataset
        tce_data: Panda DataFrame containing TCE metadata
        i: index keeping track of which TCE is being worked on
    '''
    
    ## Checking for small datsets
    if tce_num < 20:
        ## Printing step by step
        ## Only useful for small datasets
        print(str(i).zfill(2)+' Processing:  '+'kepid-'+
            str(tce_data['kepid'][i]).zfill(9)+
            str(tce_data['av_training_set'][i]).rjust(4, ' '))
        if i == (tce_num-1):
            print('Done cleaning and combining flux data.')
    else:
        if (i % 100 == 0):
            ## For large datasets, will show progress every 100 TCEs
            print('Processed '+str(i).rjust(5, ' ')+
                  '/'+str(tce_num)+'..')
        elif (i == tce_num-1):
            print('Processed '+str(tce_num).rjust(5, ' ')+
                  '/'+str(tce_num)+'..')
    
    
################
## Methods to be used for data visualization
####

def visualize_all(data, kepid_labels, every='all'):
    '''
    The visualize_all function plots all the given TCE fluxes. 
    
    Args: 
        data: Numpy array containing flux data. Should have been 
            generated using the main function.
        kepid_labels: Numpy array containing labels. Should have been 
            generated using the main function.
        every: Specify how many light curves to plot. 
            For every 8th light curve, quanity=8.
            Default is 'all'. 
    '''
    
    if every == 'all':
        for i in range(len(data)):
            lc = lk.LightCurve(flux=data[i][:])
            print('Plotting kepid-'+str(kepid_labels[i][0]).zfill(9)+
                  ' Label: '+str(kepid_labels[i][1]))
            lc.scatter()
            plt.ion()
            plt.show()
    
    if (type(every) == int):
        for i in range(len(data)):
            if (i % every) == 0:
                lc = lk.LightCurve(flux=data[i][:])
                print('Plotting kepid-'+str(kepid_labels[i][0]).zfill(9)+
                      ' Label: '+str(kepid_labels[i][1]))
                lc.scatter()
                plt.ion()
                plt.show()              


################
## Main method for data cleaning/processing
####

def main_data_processing(csv_file):
    '''
    The main_data_processing function processes the light curves for the TCEs in
        the given csv file, assuming that the corresponding light curves 
        have already been downloaded.
    
    Args: 
        csv_file: Should contain desired TCEs and parameters. 
    
    Returns:
        flux_data: Np.array containing all TCE flux data. 
            For n TCEs, the list is of size n x 2001.
        flux_labels: Np.array containing all TCE kepids and labels.
            For n TCEs, the list is of size n x 2.
    '''
    
    ## Start counting towards time of completion and display start message
    start = time.time()
    print('Commencing data processing...')
    
    ## Opens the chosen csv file and extracts the needed data
    tce_data = open_files(csv_file)
    
    ## Holds the number of TCEs
    tce_num = len(tce_data)

    ## Will contain all the flux data (tce_num x 2001)
    flux_data = []
    
    ## Will contain all the labels (tce_num x 1)
    flux_labels = []
    
    ## Will contain all the kepids and labels (tce_num x 2)
    flux_kepid_labels = []

    ## Getting total flux for each kepid
    for i in range(tce_num):

        ## Current flux data as list 
        temp_flux_data = get_total_flux(
            int(tce_data['kepid'][i]), tce_data).flux.tolist()

        ## Adding current flux data to flux_data
        flux_data.append(temp_flux_data)
        
        ## Adding current label to flux_labels
        flux_labels.append(tce_data['av_training_set'][i])
        
        ## Adding current kepid/index and label to flux_index_labels
        flux_kepid_labels.append([tce_data['kepid'][i], 
                           tce_data['av_training_set'][i]])

        ## Printing relevant info 
        print_info(tce_num, tce_data, i)
    
    ## Converting from list to np.array
    flux_data = np.asarray(flux_data)
    flux_labels = np.asarray(flux_labels)
    flux_kepid_labels = np.asarray(flux_kepid_labels)
    
    ## Changing labels for tensorflow
    flux_labels[flux_labels == 'PC'] = 1
    flux_labels[flux_labels == 'AFP'] = 0
    flux_labels[flux_labels == 'NTP'] = 0
    
    ## Saving the processed data arrays as npy files
    np.save('flux_data.npy', flux_data)
    np.save('flux_labels.npy', flux_labels)
    np.save('flux_kepid_labels.npy', flux_kepid_labels)
    
    ## Display time of completion
    end = time.time()
    print('Completed data processing in ' + str(round(end-start, 4)) + ' seconds')
    
    ## Return np.arrays containing all kepids and flux data
    return flux_data, flux_labels, flux_kepid_labels
    
                
