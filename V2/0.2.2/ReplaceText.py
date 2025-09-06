#Importación de las liberarías necesarias
import polars as pl
import logging
from dataclasses import dataclass
from typing import Optional
from Validations import ValidacionesDinamicas, ValidacionesEstaticas

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame

class RemplazoTexto: 
    def __init__(self, df: pl.DataFrame) -> None:
        self.validaciones = ValidacionesDinamicas(df=df)
        self.validaciones.dataframe_vacio()
        self.df_remplazo = DFWrapper(df=df).df
    
    def remplazar_palabras_especificas(self, col: str, patron: str, remplazo: str) -> Optional[pl.DataFrame]: 
        self.validaciones.columna_categorica(col=col)
        self.validaciones.valor_existente(col=col, patron=patron)
        df_remplazo = self.df_remplazo.with_columns(
            pl.col(col).str.replace_all(patron, remplazo)
        )
        logger.info(f'Se remplazo {patron} por {remplazo} en la columna {col}')
        return df_remplazo

class NombreColumna: 
    def __init__(self, df: pl.DataFrame) -> None:
        self.validaciones = ValidacionesDinamicas(df=df)
        self.validacion_estatica = ValidacionesEstaticas()
        self.validaciones.dataframe_vacio()
        self.df_remplazo = DFWrapper(df=df).df
    
    def renombrar_columna(self, diccionario: dict) -> Optional[pl.DataFrame]: 
        self.validacion_estatica.diccionario_vacio(diccionario=diccionario)
        self.validaciones.llaves(diccionario=diccionario)
        df_remplazo = self.df_remplazo.rename(diccionario)
        logger.info(f'Se renombraron las columnas {list(diccionario.keys())} a {list(diccionario.values())}')
        return df_remplazo
