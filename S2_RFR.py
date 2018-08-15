from datetime import datetime, timedelta

def RFR_reference_date(input_date = None):
    """Returns the closest reference date prior to an input_date
    The reference_date is put in a dict with the original input_date
    If no input_date is given then today() is used
    >>> RFR_reference_date(datetime(2018, 1, 1))
    {'input_date': datetime.datetime(2018, 1, 1, 0, 0), 'reference_date': '20171231'}
    >>> RFR_reference_date(datetime(2018, 8, 15))
    {'input_date': datetime.datetime(2018, 8, 15, 0, 0), 'reference_date': '20180731'}
    """
    
    reference_date = input_date
    
    if (reference_date is None) or (reference_date > datetime.today()):
        reference_date = datetime.today()
    
        if (reference_date.day < 5):
            reference_date = reference_date.replace(day = 1) - timedelta(days = 1)
    else:
        reference_date = reference_date + timedelta(days = 1)
    
    # to do : check if end of month
    reference_date = reference_date.replace(day = 1) - timedelta(days = 1)

    cache = {"input_date": input_date, 
             "reference_date": reference_date.strftime('%Y%m%d')}
    
    return cache
   
def RFR_dict(input_date = None):
    """Returns a dict with location and filenames from the EIOPA website based on the input_date
    >>> RFR_dict(datetime(2018,1,1))
    {'input_date': datetime.datetime(2018, 1, 1, 0, 0), 
     'reference_date': '20171231', 
     'location': 'https://eiopa.europa.eu/Publications/Standards/', 
     'zipfile': 'EIOPA_RFR_20171231.zip', 
     'excelfile': 'EIOPA_RFR_20171231_Term_Structures.xlsx'}
    """
    
    cache = RFR_reference_date(input_date)

    reference_date = cache['reference_date']
    cache['location'] = "https://eiopa.europa.eu/Publications/Standards/"
    cache['zipfile'] = "EIOPA_RFR_" + reference_date + ".zip"
    cache['excelfile'] = "EIOPA_RFR_" + reference_date + "_Term_Structures" + ".xlsx"
    
    return cache
    
from urllib.request import urlopen
import zipfile
import os

def download_RFR(input_date = None):
    """Downloads the zipfile from the EIOPA website and extracts the Excel file
    Returns the cache with info
    >>> download_RFR(datetime(2018,1,1))
    {'excelfile': 'EIOPA_RFR_20171231_Term_Structures.xlsx',
     'input_date': datetime.datetime(2018, 1, 1, 0, 0),
     'location': 'https://eiopa.europa.eu/Publications/Standards/',
     'reference_date': '20171231',
     'zipfile': 'EIOPA_RFR_20171231.zip'}
     """
    cache = RFR_dict(input_date)
    
    if not(os.path.isfile(cache["excelfile"])):

        # download file
        request = urlopen(cache["location"] + cache["zipfile"])

        # save zip-file
        output = open(cache["zipfile"], "wb")
        output.write(request.read())
        output.close()

        # extract file from zip-file
        zip_ref = zipfile.ZipFile(cache["zipfile"])
        zip_ref.extract(cache["excelfile"])
        zip_ref.close()

        # remove zip file
        # os.remove(cache["zipfile"])
        
    return cache
    
import pandas as pd

def read_spot(xls, cache):
    """Reads the RFR spot from the Excel file
    Returns the cache with the dataframes
    >>> 
    """
    for name in ["RFR_spot_no_VA", "RFR_spot_with_VA"]:
        
        df = pd.read_excel(io = xls, 
                           sheet_name = name, 
                           index_col = 1)
        
        df = df.drop('Unnamed: 0', axis = 1)
        df.loc["VA"].fillna(0, inplace = True)
        df = df.iloc[8:]
        df.index.names = ['Duration']
        cache[name] = df

    return cache

def read_meta(xls, cache):
    """Reads the RFR metadata from the Excel file
    Returns the cache with the dataframe
    >>> 
    """

    df_meta = pd.read_excel(xls, 
                            sheet_name = "RFR_spot_with_VA", 
                            index_col = 1, 
                            skipfooter = 150)
    
    df_meta = df_meta.drop('Unnamed: 0', axis = 1)
    df_meta = df_meta.iloc[0:8]
    df_meta.index.names = ['metadata']
    df_meta.index = df_meta.index.fillna("Info")

    df_append = pd.DataFrame(index = ['reference date'], 
                             columns = df_meta.columns)
    df_append.loc['reference date'] = cache["reference_date"]
    df_meta = df_meta.append(df_append)
    
    cache['metadata'] = df_meta

    return cache

def dict_RFR(input_date = None):
    
    cache = download_RFR(input_date)
    xls = pd.ExcelFile(cache["excelfile"])
    cache = read_meta(xls, cache)
    cache = read_spot(xls, cache)
    
    return cache