#Importacion de librerías 
import polars as pl
import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from Validations import ValidacionesDinamicas

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame

class EstrategiaDuplicados(str, Enum): 
    general = 'general'
    columna = 'columna'
    general_con_reset = 'general con reset'
    columna_con_reset = 'columna con reset'

class LimpiezaDuplicados: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_duplicados = df.df 
    
    def duplicados_generales(self) -> Optional[pl.DataFrame]: 
        df_limpio = self.df_duplicados.unique()
        logger.info(f'Se eliminaron los duplicados del DataFrame. Se eliminaron {self.df_duplicados.height - df_limpio.height}')
        return df_limpio
    
    def duplicados_columna(self, col: str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_existente(col=col)
        df_limpio = self.df_duplicados.unique(subset=[col])
        logger.info(f'Se eliminaron los duplicados en la columna {col}. Se eliminaron {self.df_duplicados.height - df_limpio.height}')
        return df_limpio
    
    def resetear_indice_dup_general(self) -> Optional[pl.DataFrame]: 
        return self.duplicados_generales().with_row_index()
    
    def resetear_indice_dup_columna(self, col:str) -> Optional[pl.DataFrame]: 
        return self.duplicados_columna(col=col).with_row_index()

class PipelineLimpiezaDuplicados: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df)
        self.df_duplicados = LimpiezaDuplicados(df=self.df)
    
    def PipelineDuplicados(self, col: str=None, estrategia: Optional[EstrategiaDuplicados]=None) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiaDuplicados.general: 
                    df_duplicado = self.df_duplicados.duplicados_generales()
                case EstrategiaDuplicados.general_con_reset: 
                    df_duplicado = self.df_duplicados.resetear_indice_dup_general()
                case EstrategiaDuplicados.columna: 
                    df_duplicado = self.df_duplicados.duplicados_columna(col=col)
                case EstrategiaDuplicados.columna_con_reset: 
                    df_duplicado = self.df_duplicados.resetear_indice_dup_columna(col=col)
                case None: 
                    logger.warning('No se seleccionó una estrategia válida para la limpieza de duplicados')
                    return self.df.df
                case _: 
                    logger.error(f'Estrategia {estrategia} no valida')
                    return self.df.df
            return df_duplicado
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer eliminar los datos duplicados: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')
