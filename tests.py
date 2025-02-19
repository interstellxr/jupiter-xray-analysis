from oda_api.api import DispatcherAPI # interface to query MMODA server
from oda_api.plot_tools import OdaImage,OdaLightCurve # tools to retrieve images, lightcurves
from oda_api.data_products import BinaryData
import os

disp=DispatcherAPI(url='https://www.astro.unige.ch/mmoda/dispatch-data',instrument='mock')
# 'mock' is a placeholder, not querying a real object yet

instr_list=disp.get_instruments_list() # list of available instruments on MMODA
for i in instr_list:
    print(i)

print(disp.get_instrument_description('isgri')) # ISGRI instrument details
print(disp.get_product_description(instrument='isgri',product_name='isgri_image')) # ISGRI images details


##% IMAGE (ISGRI)

data_collection=disp.get_product(instrument='isgri',
                      product='isgri_image',
                      T1='2003-03-15T23:27:40.0', # start time
                      T2='2003-03-16T00:03:15.0', # end time
                      E1_keV=20.0, # lower energy bound in keV
                      E2_keV=40.0, # upper energy bound in keV
                      osa_version='OSA10.2',
                      RA=257.815417,
                      DEC=-41.593417,
                      detection_threshold=5.0, # detection threshold sigma level
                      radius=15., # search radius in degrees
                      product_type='Real') # real data

data_collection.show() # summary of retrieved data

# access member by name or by position in data list (here the main mosaic image):
data_collection.mosaic_image_0_mosaic
data_collection._p_list[0] 

print(data_collection.dispatcher_catalog_1.table) # associated source catalog

data_collection.mosaic_image_0_mosaic.show() # info on the image
data_collection.mosaic_image_0_mosaic.show_meta()

im=OdaImage(data_collection)
im.show(unit_ID=4)  # display image

##% LIGHT CURVE (ISGRI)

data_collection=disp.get_product(instrument='isgri',
                      product='isgri_lc',
                      T1='2003-03-15T23:27:40.0',
                      T2='2003-03-16T00:03:12.0',
                      time_bin=70,
                      osa_version='OSA10.2',
                      RA=255.986542,
                      DEC=-37.844167,
                      radius=15.,
                      product_type='Real')

data_collection.show() # lists all available light curves
data_collection.isgri_lc_2_GX349p2.show() # select light curve of GX349p2

for ID,s in enumerate(data_collection._p_list): # list object IDs
    print (ID,s.meta_data['src_name'])

lc=data_collection._p_list[0] # first light curve
print(lc.data_unit[1].data)
lc.show()
print(lc.meta_data)

OdaLightCurve(lc).show(unit_ID=1) # plot light curve

##% ISGRI SPECTRUM

from threeML.plugins.OGIPLike import  OGIPLike
from threeML.io.package_data import get_path_of_data_file
from threeML import *
import matplotlib.pylab as plt

data_collection=disp.get_product(instrument='isgri',
                      product='isgri_spectrum',
                      T1='2003-03-15T23:27:40.0',
                      T2='2003-03-16T00:03:12.0',
                      osa_version='OSA10.2',
                      RA=255.986542,
                      DEC=-37.844167,
                      radius=15.,
                      product_type='Real')

data_collection.show()
d=data_collection._p_list[0] # first spectrum
print(d.meta_data)

# Can select all products from same source name: 
data_sel=data_collection.new_from_metadata('src_name','4U 1700-377') # choose source
data_sel.show()
data_sel.save_all_data() # save these products


ogip_data = OGIPLike('ogip',
                     observation='prod_0_4U1700m377_isgri_spectrum.fits', # obtained from data_sel.show()
                     arf_file= 'prod_1_4U1700m377_isgri_arf.fits' ,
                     response= 'prod_2_4U1700m377_isgri_rmf.fits')

ogip_data.set_active_measurements('20-60') # channel ranges

ogip_data.view_count_spectrum() # plot spectrum
plt.ylim(1E-5,10)

# Fit the spectrum: 
fit_function = Cutoff_powerlaw()
point_source = PointSource('ps', 0, 0, spectral_shape=fit_function)
model = Model(point_source)
datalist = DataList(ogip_data)
jl = JointLikelihood(model, datalist)
jl.fit();

display_spectrum_model_counts(jl, step=True) # display model spectrum

plot_point_source_spectra(jl.results, ene_min=20, ene_max=60, num_ene=100,
                          flux_unit='erg / (cm2 s)')

