#Importación de las liberarías necesarias
import polars as pl
import logging
from dataclasses import dataclass
from ColumnAnalyzer import ColumnAnalyzer
from enum import Enum
from typing import Optional
from Validations import ValidacionesDinamicas, ValidacionesEstaticas

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame
    
    @property
    def ColumnaCategorica(self) -> pl.DataFrame: 
        return ColumnAnalyzer(df=self.df).columna_categorica()

class RemplazoTexto: 
    def __init__(self, df: pl.DataFrame) -> None:
        self.validaciones = ValidacionesDinamicas(df=df)
        self.validacion_estatica = ValidacionesEstaticas()
        self.validaciones.dataframe_vacio()
        self.df_remplazo = DFWrapper(df=df).ColumnaCategorica
    
    def remplazar_palabras_especificas(self, col: str, patron: str, remplazo: str) -> pl.DataFrame: 
        self.validaciones.columna_categorica(col=col)
        self.validaciones.valor_existente(col=col)
        df_remplazo = self.df_remplazo.with_columns(
            pl.col(col).str.replace_all(patron, remplazo)
        )
        logger.info(f'Se remplazo {patron} por {remplazo} en la columna {col}')
        return df_remplazo
    
    def remplazar_nombres_columnas(self, diccionario: dict) -> pl.DataFrame: 
        self.validaciones.llaves(diccionario=diccionario)
        self.validacion_estatica.diccionario_vacio(diccionario=diccionario)
        df_remplazo = self.df_remplazo.rename(diccionario)
        logger.info(f'Se renombraron las columnas {list(diccionario.keys())} a {list(diccionario.values())}')
        return df_remplazo

#PIPELINE + clase por cada operacion + eliminar columnas 


