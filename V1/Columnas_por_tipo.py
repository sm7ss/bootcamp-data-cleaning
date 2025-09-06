#Importación de las liberarías necesarias
import polars as pl
from typing import Optional
import logging

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Función para obtener las columnas numéricas y categóricas del DataFrame
def num_cat_col(nombre_archivo: str, 
                df: pl.DataFrame) -> Optional[tuple]: 
    '''
    Función para obtener las columnas numéricas y categóricas de un DataFrame
    
    Parámetros:
    nombre_archivo (str): Nombre del archivo
    df (pl.DataFrame): DataFrame a analizar
    
    Retorna:
    num_col (pl.DataFrame.columns): Columnas numéricas
    cat_col (pl.DataFrame.columns): Columnas categóricas
    '''
    if df.is_empty(): 
        logger.error(f'DataFrame vacío para archivo {nombre_archivo}')
        return None
    
    num_col = df.select(pl.selectors.numeric()).columns
    cat_col = df.select(pl.selectors.string()).columns
    return num_col, cat_col

#Función para obtener las columnas no ID del DataFrame
def columnas_no_id(df: pl.DataFrame) -> Optional[list]:
    '''
    Función para obtener las columnas no ID del DataFrame
    
    Parámetros:
    df (pl.DataFrame): DataFrame a analizar
    
    Retorna:
    lista_no_id (pl.DataFrame.columns): Columnas no ID
    '''
    if df.is_empty(): 
        logger.error(f'DataFrame vacío')
        return None
    
    lista_no_id = []
    
    for id in df.columns: 
        if id == 'user_id' or id == 'id': 
            continue
        else: 
            lista_no_id.append(id)
    return lista_no_id

