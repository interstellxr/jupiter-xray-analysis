from oda_api.api import DispatcherAPI
from oda_api.plot_tools import OdaImage,OdaLightCurve
from oda_api.data_products import BinaryData
import os

disp=DispatcherAPI(url='https://www.astro.unige.ch/mmoda/dispatch-data',instrument='mock')

instr_list=disp.get_instruments_list()
for i in instr_list:
    print(i)

disp.get_instrument_description('isgri')

disp.get_product_description(instrument='isgri',product_name='isgri_image')

data_collection=disp.get_product(instrument='isgri',
                      product='isgri_image',
                      T1='2003-03-15T23:27:40.0',
                      T2='2003-03-16T00:03:15.0',
                      E1_keV=20.0,
                      E2_keV=40.0,
                      osa_version='OSA10.2',
                      RA=257.815417,
                      DEC=-41.593417,
                      detection_threshold=5.0,
                      radius=15.,
                      product_type='Real')

data_collection.show()
data_collection.mosaic_image_0_mosaic
data_collection._p_list[0]
data_collection.dispatcher_catalog_1.table

im=OdaImage(data_collection)
im.show(unit_ID=4)