#Importación de las liberarías necesarias
import polars as pl
from typing import Optional
import logging

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Min - Max IQR
def IQR(df: pl.DataFrame, 
        col: str, 
        max_q: float=0.75, 
        min_q: float=0.25) -> Optional[tuple]:
    """
    Calcula el rango intercuartil (IQR) de una columna de un DataFrame
    
    Args:
        df (pl.DataFrame): DataFrame de polars
        col (str): Nombre de la columna
        max_q (float, optional): Percentil para el cálculo del tercer cuartil. Defaults to 0.75
        min_q (float, optional): Percentil para el cálculo del primer cuartil. Defaults to 0.25
        
    Returns:
        tuple: Valores mínimo y máximo del IQR
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame está vacío')
        return None
    
    df_limpieza = df.clone()
    
    if col not in df_limpieza.columns: 
        logger.error(f'La columna {col} no se encuentra en el DataFrame')
        return None
    
    try: 
        q1 = df_limpieza[col].quantile(min_q)
        q3 = df_limpieza[col].quantile(max_q)
        iqr = q3 - q1
        minimo = q1 - 1.5 * iqr 
        maximo = q3 + 1.5 * iqr
        
        return minimo, maximo
    except Exception as e: 
        logger.error(f'Ocurrio un error para calcular el IQR para la columna {col} : {e}')
        return None

#Manejo de Outliers
def limpieza_de_outlier(nombre_archivo: str, 
                        df: pl.DataFrame, 
                        columnas: list,
                        winsorizacion_col: bool=True, 
                        max_q: float=0.75, 
                        min_q: float=0.25,
                        eliminacion_col: bool=False) -> Optional[pl.DataFrame]: 
    """
    Limpia los outliers de un DataFrame
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de polars
        columnas (list): Lista de columnas a limpiar
        winsorizacion_col (bool, optional): Activar o desactivar la winsorización de outliers. Defaults to True
        max_q (float, optional): Percentil para el cálculo del tercer cuartil. Defaults to 0.75
        min_q (float, optional): Percentil para el cálculo del primer cuartil. Defaults to 0.25
        eliminacion_col (bool, optional): Activar o desactivar la eliminación de outliers. Defaults to False
    
    Returns:
        Optional[pl.DataFrame]: DataFrame sin outliers
    """
    
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    df_limpieza = df.clone()
    
    if winsorizacion_col and eliminacion_col: 
        logger.error(f'No se pueden activar winsorizacion_col y eliminacion_col al mismo tiempo')
        return None
    
    columnas_numericas = df_limpieza.select(pl.col(pl.Int64, pl.Float64))
    validacion = [col for col in columnas if col not in columnas_numericas.columns]
    if validacion: 
        logger.error(f'Las columnas {validacion} no son númericas')
        return None
    
    try: 
        for col in columnas: 
            min_b, max_b = IQR(df=df_limpieza, col=col, max_q=max_q, min_q= min_q)
            
            if min_b is None or max_b is None: 
                logger.warning(f'No se pudieron calcular los limites IQR para {col}')
                continue
            
            if winsorizacion_col: 
                df_limpieza = df_limpieza.with_columns(
                pl.col(col).clip(lower_bound=min_b, upper_bound=max_b)
                )
                logger.info(f'Se winsorizo la columna {col}')
            elif eliminacion_col: 
                filas_antes = df_limpieza.height
                df_limpieza = df_limpieza.filter(
                    pl.col(col).is_between(min_b, max_b)
                )
                filas_despues = df_limpieza.height
                logger.info(f'Se eliminaron {filas_antes - filas_despues} filas de outliers en la columna {col}')
        return df_limpieza
    except Exception as e: 
        logger.error(f'Ocurrio un error al querer limpiar los outliers para el archivo {nombre_archivo} : {e}')
        return df_limpieza

