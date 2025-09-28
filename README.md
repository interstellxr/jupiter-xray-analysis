# jupiter-xray-analysis
TPIVb project at EPFL with the purpose of analyzing data provided by the INTEGRAL telescope to study hard X-ray emissions from Jupiter.

## Code organization
- 'data/': contains the FITS files queried from the INTEGRAL archive, figures used in the report and other data files.
- 'jupiter-xrays/': contains the main code for the analysis.
    Main notebooks:
    - 'crablongterm.ipynb'/'jupiterlongterm.ipynb': notebook to query the Crab/Jupiter FITS files from a given ScW list.
    - 'stacking.ipynb': notebook to stack images of multiple ScWs to increase the SNR.
    - 'jscws.ipynb': notebook to create a list of ScWs where Jupiter is in the FoV.
    - 'crabplotting.ipynb'/'jupiterplotting.ipynb': notebook to extract data from the Crab/Jupiter FITS files and plot lightcurves.
    Utility scripts:
    - 'utils.py': utility functions used in the notebooks.
    - 'plots.py': functions to plot all kinds of graphs.
    Other notebooks:
    - 'spectra.ipynb': notebook to extract and plot spectra from csv files.
    - 'tests.ipynb': as the name suggests, a notebook to test various things.
    - 'calculations.ipynb': notebook to perform various calculations.
    - 'crab.ipynb': notebook to analyze a single Crab FITS file.
    - 'jupiter.ipynb': notebook to analyze a single Jupiter FITS file.

## Workflow
1. Create a list of ScWs where Jupiter is in the FoV using 'jscws.ipynb'.
2. Query the FITS files for these ScWs using 'jupiterlongterm.ipynb'.
3. Run 'crablongterm.ipynb' to query Crab FITS files for calibration purposes.
4. Run 'crabplotting.ipynb' to extract lightcurves from the Crab FITS files and create conversion factors.
5. Run an automatic statistical analysis on the FITS files to extract lightcurves using 'jupiterplotting.ipynb'.
6. If needed, stack images of multiple ScWs using 'stacking.ipynb' to increase the SNR.
7. 'spectra.ipynb' can be used to extract and plot spectra from csv files, and add upper limits on the plots.

## Requirements
- Python 3.x
- oda-api (install using `pip install oda_api`)
- astroquery (install using `pip install astroquery`)
- astropy (install using `pip install astropy`)

## Some documentation links
API documentation: https://oda-api.readthedocs.io/en/latest/  
MMODA: https://www.astro.unige.ch/mmoda/  
INTEGRAL notes: [https://docs.google.com/document/d/18VBsLnJf_uufGxqk4uSUYD_vNk3Ti_Gu5luSPAe-g2k/edit?tab=t.0](https://docs.google.com/document/d/18VBsLnJf_uufGxqk4uSUYD_vNk3Ti_Gu5luSPAe-g2k/edit?usp=sharing)  
Article summaries: https://docs.google.com/document/d/1Lwd9mXqzKYKDE4VaOx93JynVrsqpzkFtzbT6kAd9tWk/edit?usp=sharing  
ScWs: https://www.isdc.unige.ch/browse/W3Browse/integral-rev2/integral_rev2_scw.html

## Consulted references:  
- Discovery of diffuse hard X-ray emission around Jupiter with Suzaku: https://iopscience.iop.org/article/10.1088/2041-8205/709/2/L178/meta  
- The detection of X rays from Jupiter: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JA088iA10p07731  
- Observation and origin of non-thermal hard X-rays from Jupiter: https://www.nature.com/articles/s41550-021-01594-8  
- Low- to middle-latitude X-ray emission from Jupiter: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JA011792
- A study of Jupiter’s aurorae with XMM-Newton: https://www.aanda.org/articles/aa/pdf/2007/08/aa6406-06.pdf
- X-ray Emissions from the Jovian System: https://arxiv.org/abs/2208.13455
- A pulsating auroral X-ray hot spot on Jupiter: https://www.nature.com/articles/4151000a
- A very good overview/summary of the above articles and others: https://cxc.harvard.edu/newsletters/news_27/article1.html
- Astroquery docs: https://astroquery.readthedocs.io/en/latest/index.html
- INTEGRAL Cross-calibration Status: Crab observations between 3 keV and 1 MeV: https://arxiv.org/pdf/0810.0646v1

