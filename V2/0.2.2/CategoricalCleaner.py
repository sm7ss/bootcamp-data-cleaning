#Importación de las liberarías necesarias
import polars as pl
import logging
from dataclasses import dataclass
from enum import Enum
from ColumnAnalyzer import  ColumnAnalyzer
from Validations import ValidacionesDinamicas
from typing import Optional, Union

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame

class EstrategiaImputacion(str, Enum): 
    moda = 'moda'
    imputacion_valor = 'imputacion valor'

class EstrategiaEliminacion(str, Enum): 
    filas = 'filas'
    columnas = 'columnas'

class GestionPipelineGeneral(str, Enum): 
    imputacion = 'imputacion'
    eliminacion = 'eliminacion'

class EliminarNulosCategoricos: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_eliminacion_nulos = df.df
    
    def eliminar_columnas_categoricas_nulos(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        df_limpieza = self.df_eliminacion_nulos.drop(col)
        logger.info(f'Columna {col} eliminada')
        return df_limpieza
    
    def eliminar_filas_categoricas_nulos(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        df_limpieza = self.df_eliminacion_nulos.filter(~pl.col(col).is_null())
        logger.info(f'Se eliminaron filas con nulos en la columna {col}')
        return df_limpieza

class ImputacionNulosCategoricos: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_imputacion_nulos = df.df
    
    def imputar_moda_categorica(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        moda = self.df_imputacion_nulos[col].mode()[0]
        df_limpieza = self.df_imputacion_nulos.with_columns(
            pl.col(col).fill_null(moda)
        )
        logger.info(f'Se imputo la moda en la columna {col} para nulos')
        return df_limpieza
    
    def imputacion_valor_categorico(self, col:str, valor: str='Desconocido') -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        df_limpieza = self.df_imputacion_nulos.with_columns(
            pl.col(col).fill_null(valor)
        )
        logger.info(f'Se imputo {valor} en la columna {col} para nulos')
        return df_limpieza

class PipelineEliminarCategoricos: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df)
        self.eliminar_categoricos = EliminarNulosCategoricos(df=self.df)
    
    def PipelineEliminarColumnasFilas(self, col: str, estrategia: Optional[EstrategiaEliminacion]=None) -> Optional[pl.DataFrame]: 
        try: 
            match estrategia: 
                case EstrategiaEliminacion.columnas: 
                    df_limpieza = self.eliminar_categoricos.eliminar_columnas_categoricas_nulos(col=col)
                case EstrategiaEliminacion.filas: 
                    df_limpieza = self.eliminar_categoricos.eliminar_filas_categoricas_nulos(col=col)
                case None: 
                    logger.warning(f'No se introdujo una estrategia')
                    return self.df.df
                case _: 
                    logger.error(f'Estrategia {estrategia} no valida')
                    return self.df.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error en la eliminacion de columnas o filas: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error al querer eliminar las columnas o filas: {e}')

class PipelineImputacionCategorica: 
    def __init__(self, df:pl.DataFrame) -> None:
        self.df = DFWrapper(df=df)
        self.imputacion_categorico = ImputacionNulosCategoricos(df=self.df)
    
    def PipelineImputacionDeCategorias(self, col: str, valor: Optional[str]='Desconocido', estrategia: Optional[EstrategiaImputacion]=None) -> Optional[pl.DataFrame]: 
        try: 
            match estrategia: 
                case EstrategiaImputacion.moda: 
                    df_limpieza = self.imputacion_categorico.imputar_moda_categorica(col=col)
                case EstrategiaImputacion.imputacion_valor: 
                    df_limpieza = self.imputacion_categorico.imputacion_valor_categorico(col=col, valor=valor)
                case None: 
                    logger.warning(f'No se introdujo una estrategia')
                    return self.df.df
                case _: 
                    logger.error(f'Estrategia {estrategia} no valida')
                    return self.df.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error en la imputacion de nulos categoricos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error al querer imputar categorias: {e}')

class PipelineLimpiezaNulosCategoricos: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.pipeline_eliminacion = PipelineEliminarCategoricos(df=df)
        self.pipeline_imputacion = PipelineImputacionCategorica(df=df)
    
    def PipelineGeneral(self, col: str, valor: Optional[str]=None, estrategia: Optional[GestionPipelineGeneral]=None, estrategia_general: Union[EstrategiaEliminacion, EstrategiaImputacion]=None) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case GestionPipelineGeneral.eliminacion: 
                    df_limpieza = self.pipeline_eliminacion.PipelineEliminarColumnasFilas(col=col, estrategia=estrategia_general)
                case GestionPipelineGeneral.imputacion: 
                    df_limpieza = self.pipeline_imputacion.PipelineImputacionDeCategorias(col=col, valor=valor, estrategia=estrategia_general)
                case None:  
                    logger.warning(f'No se introdujo una estrategia')
                    return self.df.df
                case _: 
                    logger.error(f'Estrategia {estrategia} no valida')
                    return self.df.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer usar la estrategia {estrategia}: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')
