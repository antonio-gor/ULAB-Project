# Exoplanet Detection Using Machine Learning <a href="ULAB"><img src="https://ulab.berkeley.edu/static/img/logos/logo_physics.png" align="left" height="48" width="48" title="ULAB-logo" ></a>

The project is split into two main parts.

  - Data processing
  - Model implementation
  
## Table of Contents
* [Libraries Used](#libraries-used)
* [Data Processing](#data-processing)
* [Model Implementation](#model-implementation)
* [Future Work](#future-work)
* [Citations](#citations)

## Libraries Used
The project primarily made use of
* <a href="http://docs.lightkurve.org">Lightkurve</a>
* <a href="https://pandas.pydata.org">Pandas</a>
* <a href="https://www.numpy.org">Numpy</a>
* <a href="https://www.tensorflow.org">Tensorflow</a>
  
## Data Processing
The <a href="https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce">Kepler Q1-Q17 DR24 TCE</a> dataset was used. 

#### TCE: Threshold Crossing Events
These are the stars that the Kepler pipeline flagged as possibly
having a planet transit.

This dataset was chosen because of the human made labels that identify each TCE.
There are four types of labels.

  - PC: Planet Candidate
  - AFP: Astrophysical False Positive  
  - NTP: Nontransiting Phenomenon 
  - UNK: Unkown
    
We will classify PC as "planet", merge AFP & NTP as "non-planet", and ignore UNK.

*More explicitly, 1=planet, 0=non-planet.*

The <a href="http://docs.lightkurve.org">Lightkurve</a> package will be used for most of the data cleaning and processing. The data processing portion can be split into several steps:

  - Acquiring the lightcurves from <a href="https://archive.stsci.edu/kepler/">MAST</a> as FITS files (~15-18 per TCE)
  - Extracting the PDCSAP_FLUX lightcurves from the FITS files
  - Flattening each PDCSAP_FLUX lightcurve
  - Stitching all lightcurves corresponding to a TCE together, generating one lightcurve per TCE
  - Flattening, binning, folding, and normalizing each generated lightcurve
  - Saving the cleaned lightcurves as either TFRecords or Numpy arrays (depending on model)
  
The following animation demonstrates what the process looks like, using KIC 6022556 as an example.

<a href="ULAB"><img src="/img/lightcurve.gif" align="left" title="lightcurve-cleaning" >
  
## Model Implementation

Various small convolutional neural networks (CNNs) using convolution, pooling, and dense layers where implemented to begin with. The final model loosley emulates the models described by Shallue & Vanderburg (2018)<sup>1</sup> and Ansdell et al. (2018)<sup>2</sup>. The biggest difference is that only a single-view was used as input for the CNN, whereas the papers described local and gloabl-views as input. This likely accounts for one of many areas that could be improved on.

In addition to the convolution, pooling, and dense layers, it was found that adding dropout layers helped increase both the accuracy and  the time it took to reach the highest accuracy, which was 81.81% at the end of the project. This maximum accuracy did not emulate the ones in the paper. The step in the process that would most benefit from revising would likely be data processing.

## Future Work
Accuracy of the CNN model could be improved through various modifications to both the data cleaning process and CNN model itself. The period of each TCE (used to fold the flux time series data) was obtained from the metadata provided from the Exoplanet Archive but manually finding each TCEs period through a more rigorous method could improve accuracy. The processing time of each TCE would increase but yield a better folded light curve. Another potentially helpful change would be to remove the data of multiple planet candidate transits from each TCE, as many TCE light curves contain more than one transit. In addition to this, creating another view of the light curves could help find different patterns that are missed by inputting a single light curve view into the model. For every TCE, only one global view light curve was generated, but adding a ‘zoomed in’ local view might help detect otherwise easily missed patterns. One final change would be to add an additional class to the labels, so that the model could be used to distinguish PCs, AFPS, and NTP. The current model makes no distinction between AFPs and NTPs. 


## Citations
<sup>1</sup>Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. The Astronomical Journal, 155(2), 94.

<sup>2</sup>Ansdell, M., Ioannou, Y., Osborn, H. P., Sasdelli, M., Smith, J. C., Caldwell, D., ... & Angerhausen, D. (2018). Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning. The Astrophysical Journal Letters, 869(1), L7.
