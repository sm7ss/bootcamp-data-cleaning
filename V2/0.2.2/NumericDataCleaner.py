#Importación de las liberarías necesarias
import polars as pl
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
from Validations import ValidacionesDinamicas

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame

class EstrategiaImputacion(str, Enum): 
    mediana = 'mediana'
    media = 'media'
    interpolacion = 'interpolacion'

class EstrategiaEliminacion(str, Enum): 
    filas = 'filas'
    columnas = 'columnas'

class GestionPipelineGeneral(str, Enum): 
    imputacion = 'imputacion'
    eliminacion = 'eliminacion'
    interpolacion = 'interpolacion'

class EliminarNulosNumericos: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_eliminacion_nulos = df.df
    
    def eliminar_columnas_numericas_nulos(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        df_limpieza = self.df_eliminacion_nulos.drop(col)
        logger.info(f'Columna {col} eliminada')
        return df_limpieza
    
    def eliminar_filas_numericas_nulos(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        df_limpieza = self.df_eliminacion_nulos.filter(~pl.col(col).is_null())
        logger.info(f'Se eliminaron filas con nulos en la columna {col}')
        return df_limpieza

class ImputacionNulosNumericos: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_imputacion_nulos = df.df
    
    def imputar_mediana_numerica(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        df_limpieza = self.df_imputacion_nulos.with_columns(
        pl.col(col).fill_null(pl.col(col).median())
        )
        logger.info(f'Columna {col} se le imputo la mediana a los valores nulos')
        return df_limpieza
    
    def imputar_media_numerica(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        df_limpieza = self.df_imputacion_nulos.with_columns(
        pl.col(col).fill_null(pl.col(col).mean())
        )
        logger.info(f'Columna {col} se le imputo la media a los valores nulos')
        return df_limpieza

class InterpolacionNulosNumericos: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_interpolacion_nulos = df.df
    
    def interpolacion_nulos(self, col: str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        interpolacion = self.df_interpolacion_nulos.with_columns(pl.col(col).interpolate())
        return interpolacion
    
    def interpolacion_media_nulos_numerica(self, col: str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        interpolacion = self.interpolacion_nulos(col=col)
        df_limpieza = interpolacion.with_columns(pl.col(col).fill_null(pl.col(col).mean()))
        logger.info(f'Se interpolo la columna {col} con la media')
        return df_limpieza
    
    def interpolacion_mediana_nulos_numerica(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        interpolacion = self.interpolacion_nulos(col=col)
        df_limpieza = interpolacion.with_columns(pl.col(col).fill_null(pl.col(col).median()))
        logger.info(f'Se interpolo la columna {col} con la mediana')
        return df_limpieza

class PipelineEliminacionNulosNumerico: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df_nulos = DFWrapper(df=df)
        self.eliminar_numericos = EliminarNulosNumericos(df=self.df_nulos)
    
    def PipelineEliminacionNulosNumerico(self, col:str, estrategia: EstrategiaEliminacion) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiaEliminacion.filas: 
                    df_limpieza = self.eliminar_numericos.eliminar_filas_numericas_nulos(col=col)
                case EstrategiaEliminacion.columnas: 
                    df_limpieza = self.eliminar_numericos.eliminar_columnas_numericas_nulos(col=col)
                case None: 
                    logger.warning(f'No se pudo hacer una eliminacion en la columna {col} porque no se selecciono una estrategia')
                    return self.df_nulos.df
                case _: 
                    logger.error(f'Estrategia no valida')
                    return self.df_nulos.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer eliminar las colulmnas o categorias numericas: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineImputacionNulosNumerico: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df_imputacion = DFWrapper(df=df)
        self.imputacion_numerico = ImputacionNulosNumericos(df=self.df_imputacion)
    
    def PipelineImputacionNulosNumericos(self, col: str, estrategia: EstrategiaImputacion) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiaImputacion.media: 
                    df_limpieza = self.imputacion_numerico.imputar_media_numerica(col=col)
                case EstrategiaImputacion.mediana: 
                    df_limpieza = self.imputacion_numerico.imputar_mediana_numerica(col=col)
                case None: 
                    logger.warning(f'No se pudo hacer una imputacion en la columna {col} porque no se selecciono una estrategia')
                    return self.df_imputacion.df
                case _: 
                    logger.error(f'Estrategia no valida')
                    return self.df_imputacion.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer imputar los datos numericos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineInterpolacionNulosNumerico: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df_interpolacion = DFWrapper(df=df)
        self.interpolacion = InterpolacionNulosNumericos(df=self.df_interpolacion)
    
    def PipelineInterpolacionNulosNumericos(self, col:str, estrategia: EstrategiaImputacion) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiaImputacion.interpolacion: 
                    df_limpieza = self.interpolacion.interpolacion_nulos(col=col)
                    logger.info(f'Se interpolo la columna {col}')
                case EstrategiaImputacion.media: 
                    df_limpieza = self.interpolacion.interpolacion_media_nulos_numerica(col=col)
                case EstrategiaImputacion.mediana: 
                    df_limpieza = self.interpolacion.interpolacion_mediana_nulos_numerica(col=col)
                case None: 
                    logger.warning(f'No se pudo hacer una interpolacion en la columna {col} porque no se selecciono una estrategia')
                    return self.df_interpolacion.df
                case _: 
                    logger.error(f'Estrategia no valida')
                    return self.df_interpolacion.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer interpolar los datos numericos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineLimpiezaNulos: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df_limpieza = df
        self.eliminacion_nulos = PipelineEliminacionNulosNumerico(df=self.df_limpieza)
        self.imputacion_nulos = PipelineImputacionNulosNumerico(df=self.df_limpieza)
        self.interpolacion_nulos = PipelineInterpolacionNulosNumerico(df=self.df_limpieza)
    
    def PipelineGeneralLimpiezaNulos(self, col: str, estrategia: GestionPipelineGeneral, estrategia_general: Union[EstrategiaEliminacion, EstrategiaImputacion]=None) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case GestionPipelineGeneral.eliminacion: 
                    df_limpieza = self.eliminacion_nulos.PipelineEliminacionNulosNumerico(col=col, estrategia=estrategia_general)
                case GestionPipelineGeneral.imputacion: 
                    df_limpieza = self.imputacion_nulos.PipelineImputacionNulosNumericos(col=col, estrategia= estrategia_general)
                case GestionPipelineGeneral.interpolacion: 
                    df_limpieza = self.interpolacion_nulos.PipelineInterpolacionNulosNumericos(col=col, estrategia=estrategia_general)
                case None: 
                    logger.warning(f'No se pudo hacer una eliminación, imputación o interpolación en la columna {col} porque no se seleccionó una estrategia')
                    return self.df_limpieza
                case _: 
                    logger.error(f'Estrategia no valida: {estrategia}')
                    return self.df_limpieza
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer manejar la limpieza numerica de los datos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

