#Importación de las liberarías necesarias
import polars as pl
from typing import Optional
import logging
from dataclasses import dataclass
from enum import Enum
from Validations import ValidacionesDinamicas

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame

class EstrategiasOutlier(str, Enum): 
    winsorizacion = 'winsorizacion'
    eliminacion = 'eliminacion'

class IQR: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_numerico = df.df
    
    def cuartiles(self, col: str, max_q: float=0.75, min_q: float=0.25) -> Optional[tuple]: 
        self.validacion.columna_numerica(col=col)
        q1 = self.df_numerico[col].quantile(min_q)
        q3 = self.df_numerico[col].quantile(max_q)
        return q1, q3
    
    def IQR_calculo(self, col: str, max_q: float=0.75, min_q: float=0.25) -> Optional[dict]:
        self.validacion.columna_numerica(col=col)
        q1, q3 = self.cuartiles(col=col, max_q=max_q, min_q=min_q)
        iqr = q3 - q1
        minimo = q1 - 1.5 * iqr 
        maximo = q3 + 1.5 * iqr
        
        return {
            'minimo' : round(float(minimo), 3), 
            'maximo' : round(float(maximo), 3)
        } 

class OutlierCleaner: 
    def __init__(self, df: DFWrapper) -> None:
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_limpieza = df.df
        self.IQR_object = IQR(df=df)
    
    def winsorizacion(self, col: str, max_q: float=0.75, min_q: float=0.25) -> Optional[pl.DataFrame]:
        self.validacion.columna_numerica(col=col)
        iqr = self.IQR_object.IQR_calculo(col=col, max_q=max_q, min_q=min_q)
        
        max_b = iqr['maximo']
        min_b = iqr['minimo']
        
        df_limpieza = self.df_limpieza.with_columns(
            pl.col(col).clip(lower_bound=min_b, upper_bound=max_b)
            )
        logger.info(f'Se winsorizo la columna {col}')
        
        return df_limpieza
    
    def eliminacion(self, col:str, max_q: float=0.75, min_q: float=0.25) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        filas_antes = self.df_limpieza.height
        
        iqr= self.IQR_object.IQR_calculo(col=col, max_q=max_q, min_q=min_q)
        
        max_b = iqr['maximo']
        min_b = iqr['minimo']
        
        df_limpieza = self.df_limpieza.filter(
            pl.col(col).is_between(min_b, max_b)
        )
        filas_despues = df_limpieza.height
        logger.info(f'Se eliminaron {filas_antes - filas_despues} filas de outliers en la columna {col}')
        
        return df_limpieza

class PipelineLimpiezaOutliers: 
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = DFWrapper(df=df)
        self.estrategia_eliminacion = OutlierCleaner(df=self.df)
    
    def PipelineLimpiezaOutlier(self, col: str, max_q: float=0.75, min_q: float=0.25, estrategia: Optional[EstrategiasOutlier]=None) -> Optional[pl.DataFrame]:
        try:
            match estrategia: 
                case EstrategiasOutlier.winsorizacion: 
                    df_limpieza = self.estrategia_eliminacion.winsorizacion(col=col, max_q=max_q, min_q=min_q)
                case EstrategiasOutlier.eliminacion: 
                    df_limpieza = self.estrategia_eliminacion.eliminacion(col=col, max_q=max_q, min_q=min_q)
                case None: 
                    logger.warning(f'No se seleccionó una estrategia')
                    return self.df.df
                case _: 
                    logger.warning(f'No se selecciono una estrategia valida de imputacion para los outliers: {estrategia}')
                    return self.df.df
            return df_limpieza
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer manejar los outliers: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')
