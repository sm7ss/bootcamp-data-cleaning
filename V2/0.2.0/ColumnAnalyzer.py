#Importación de las liberarías necesarias
import polars as pl
from typing import Optional
import logging

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Función para obtener las columnas numéricas y categóricas del DataFrame
class ColumnAnalyzer: 
    def __init__(self, df:pl.DataFrame) -> None:
        self.df = df
    
    def columna_categorica(self) -> pl.DataFrame: 
        df_categoria = self.df.select(pl.selectors.string())
        return df_categoria
    
    def columna_numerica(self) -> pl.DataFrame: 
        df_numerico = self.df.select(pl.selectors.numeric())
        return df_numerico
    
    def columna_fecha(self) -> pl.DataFrame: 
        df_fecha = self.df.select(pl.selectors.temporal())
        return df_fecha
