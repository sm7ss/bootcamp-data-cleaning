#Importación de las liberarías necesarias
import polars as pl
from typing import Optional
import itertools
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest
import logging

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Lectura de columnas y filas
def columnas_y_filas(nombre_archivo: str, 
                    df: pl.DataFrame) -> None: 
    """
    Realiza una lectura de las columnas y filas de un DataFrame de Polars

    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    logger.info('--- Columnas y Filas ---')
    filas, col = df.shape
    logger.info(f'Columnas totales: {col} - Filas totales: {filas}')

#Lectura de nombre de columnas
def nombre_columnas(nombre_archivo: str, 
                    df: pl.DataFrame) -> None: 
    """
    Realiza una lectura de los nombres de las columnas de un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    logger.info('--- Nombre de las Columnas ---')
    logger.info(f'{df.columns}')

#Lectura de vista previa de los datos
def vista_previa_general(nombre_archivo: str,
                        df: pl.DataFrame) -> None:
    """
    Realiza una lectura de una vista previa general de un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    logger.info('--- Vista Previa General ---')
    logger.info(f'{df.head()}')

#Lectura de estadisticas descriptivas
def estadisticas_descriptivas(nombre_archivo: str, 
                            df: pl.DataFrame) -> None:
    """
    Realiza una lectura de las estadísticas descriptivas de un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    logger.info('--- Estadisticas Descriptivas ---')
    logger.info(f'Vista previa general: \n {df.describe()}')

#Agrupaciones para conteo de categorias 
def agrupaciones(nombre_archivo: str, 
                df: pl.DataFrame,  
                metodo: str='count') -> None:
    '''
    Realiza agrupaciones por un valor categórico
    
    Parámetros:
        nombre_archivo (str): Nombre del archivo CSV
        df (pl.DataFrame): DataFrame de Polars que contiene los datos
        metodo (str): Método de agrupación ('count' por defecto)
        
    Retorna:
        None
    '''
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    if metodo not in ['count', 'sum', 'mean', 'min', 'max']:
        logger.error(f'El método {metodo} no es válido')
        return None
    
    logger.info('--- Agrupaciones ---')

    
    for col in df.columns: 
        if metodo == 'count': 
            agrupacion = df.group_by(col).agg(pl.count()).sort(col)
            logger.info(agrupacion.head())
        elif metodo == 'sum': 
            agrupacion = df.group_by(col).agg(pl.sum()).sort(by=col)
            logger.info(agrupacion.head())
        elif metodo == 'mean': 
            agrupacion = df.group_by(col).agg(pl.mean()).sort(by=col)
            logger.info(agrupacion.head())
        elif metodo == 'min': 
            agrupacion = df.group_by(col).agg(pl.min()).sort(by=col)
            logger.info(agrupacion.head())
        elif metodo == 'max': 
            agrupacion = df.group_by(col).agg(pl.max()).sort(by=col)
            logger.info(agrupacion.head())

#Lectura de valores nulos 
def nulos(nombre_archivo: str, 
                df: pl.DataFrame, 
                columnas: bool = False) -> Optional[dict]: 
    """
    Realiza una lectura de los valores nulos en un DataFrame de Polars

    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
        columnas (bool, optional): Determina si se desea verificar por columnas

    Returns:
        Optional[dict]: Un diccionario con los nombres de archivo y columnas con valores nulos, o None si el DataFrame está vacío
    """
    
    if df.is_empty(): 
        logger.error(f'DataFrame de {nombre_archivo} está vacío')
        return None
    
    diccionario_nulos_True = {}
    
    
    logger.info(f'--- Nulos ---')
    
    if columnas: 
        diccionario_nulos_True[nombre_archivo] = {}
        for columna in df.columns: 
            logger.info(f'Total de nulos en la columna {columna} : {df[columna].is_null().sum()}')
            if df[columna].is_null().sum() > 0: 
                logger.info(df.filter(pl.col(columna).is_null()))
                diccionario_nulos_True[nombre_archivo][columna] = True
        return diccionario_nulos_True
    else: 
        logger.info(f'Total nulos para {nombre_archivo} : {df.null_count()}')
        return diccionario_nulos_True

#Lectura de valores únicos
def unicos(nombre_archivo:str, 
        df:pl.DataFrame, 
        diccionario: bool=False) -> Optional[dict]:
    """
    Realiza una lectura de los valores unicos en un DataFrame de Polars

    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
        lista (bool, optional): Determina si se desea obtener una lista de retorno

    Returns:
        Optional[dict]: Un diccionario con los nombres de archivo y columnas con valores unicos
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return {}
    
    diccionario_unicos = {}
    
    if diccionario: 
        diccionario_unicos[nombre_archivo] = {}
    
    logger.info('--- Unicos ---')
    
    for columna in df.columns: 
        logger.info(f'valores unicos en la columna {columna}: \n{df[columna].unique()}')
        if diccionario: 
            diccionario_unicos[nombre_archivo][columna] = df.select(pl.col(columna)).unique().to_series().to_list()
    
    return diccionario_unicos

#Lectura de tipo de datos 
def tipo_de_dato(nombre_archivo: str, 
                df: pl.DataFrame) -> None: 
    """
    Realiza una lectura de los tipos de datos en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return {}
    
    diccionario_tipo_de_dato = {}
    diccionario_tipo_de_dato[nombre_archivo] = {}
    
    logger.info('--- Tipo de Datos ---')
    
    for col in df.columns: 
        diccionario_tipo_de_dato[nombre_archivo][col] = df[col].dtype
    logger.info(f'{diccionario_tipo_de_dato}')

#Comparación de similitud entre pares de valores únicos
def coincidencia_aproximada(nombre_archivo: str, 
                            valores_unicos: list, 
                            umbral_similitud: int=90) -> None: 
    """
    Realiza una comparación de similitud entre pares de valores únicos en un DataFrame de Polars

    Args:
        nombre_archivo (str): Nombre del archivo
        valores_unicos (list): Lista de valores únicos a comparar
        umbral_similitud (int, optional): Umbral de similitud para considerar coincidencias. Defaults to 90
    """
    if not valores_unicos: 
        logger.error(f'Lista vacía para el archivo {nombre_archivo}')
        return None
    
    similitud = []
    
    logger.info('--- Coincidencia aproximada ---')
    
    pares_comparacion = itertools.combinations(valores_unicos, 2)
    for val1, val2 in pares_comparacion: 
        puntaje = fuzz.token_set_ratio(val1, val2)
        
        if puntaje >= umbral_similitud: 
            similitud.append({
                'valor 1' : val1, 
                'valor 2' : val2, 
                'similitud' : puntaje
            })
    
    logger.info(f'Coincidencias encontradas para el archivo {nombre_archivo} : {similitud}')

#Lectura de valores duplicados
def duplicados(nombre_archivo: str, 
            list_col: list, 
            df:pl.DataFrame,
            general: bool=False) -> None: 
    """
    Realiza una vizualización de duplicados en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        list_col (list): Lista de columnas a vizualizar duplicados
        df (pl.DataFrame): DataFrame de Polars a analizar
        general (bool, optional): Determina si se vizualizan los duplicados en todas las columnas
    """
    
    if df.is_empty(): 
        logger.error(f'DataFrame del archivo {nombre_archivo} vacío')
        return None
    
    validacion = [col for col in list_col if col not in df.columns]
    if validacion:
        logger.error(f'Las siguientes columnas no están en el DataFrame del archivo {nombre_archivo}: {validacion}')
        return None
    
    logger.info('--- Duplicados ---')
    
    for col in list_col: 
        duplicados = df.group_by(col).agg(pl.count().alias('conteo')).filter(pl.col('conteo')>1)
        if not (duplicados.is_empty()): 
            logger.info(f'Los duplicados para la columna {col} es de: {duplicados}')
    
    if general: 
        duplicados_general = df.group_by(df.columns).agg(pl.count().alias('Total_Duplicados')).filter(pl.col('Total_Duplicados')>1)
        if not (duplicados_general.is_empty()): 
            logger.info(f'Nombre del archivo: {nombre_archivo}')
            logger.info(f'Filas duplicadas: {duplicados_general}')

#Lectura de cardinalidad
def cardinalidad(nombre_archivo: str, 
                df: pl.DataFrame, 
                list_col: list, 
                general:bool=False, 
                mostrar_cardinalidad_rara:bool=False, 
                umbral: float=0.05) -> None: 
    """
    Realiza una vizualización de cardinalidad en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
        list_col (list): Lista de columnas a vizualizar cardinalidad
        general (bool, optional): Determina si se vizualizan los duplicados en todas las columnas
        mostrar_cardinalidad_rara (bool, optional): Determina si se vizualizan las categorías raras
        umbral (float, optional): Umbral para categorías raras. Defaults to 0.05
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo}, está vacío')
        return None
    
    if not list_col: 
        logger.error(f'La lista pasada está vacía')
        return None
    
    validador = [col for col in list_col if col not in df.columns]
    if validador: 
        logger.error(f'Las siguientes columnas no están en el DataFrame del archivo {nombre_archivo}: {validador}')
        return None
    
    logger.info('--- Cardinalidad ---')
    
    for col in list_col: 
        logger.info(f'Cantidad de valores únicos en {col}: {df[col].n_unique()}') 
        if mostrar_cardinalidad_rara: 
            freq = df[col].value_counts()
            categorias_raras = freq.filter(pl.col('count') / df.shape[0] < umbral)[col]
            if len(categorias_raras) > 0: 
                logger.info(f'Categorías raras: {categorias_raras}')
            else: 
                logger.info(f'La columna {col} no tiene categorías raras')
    
    if general: 
        unicos_numericos_general = df.select(pl.col(df.columns).n_unique())
        logger.info(f'Valores unicos general: {unicos_numericos_general}')

#Detección de Outliers
def outlier(nombre_archivo:str, 
            df: pl.DataFrame, 
            contaminacion: float=0.01, 
            columnas: bool=False) -> None: 
    """
    Realiza una detección de outliers en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a analizar
        contaminacion (float, optional): Valor de contaminación para el modelo de detección de outliers. Defaults to 0.01
        columnas (bool, optional): Determina si se realiza la visualización de outliers por columna. Defaults to False
        retorno_outliers (bool, optional): Determina si se retorna un DataFrame con los outliers de las columnas
    
    Returns:
        Optional[pl.DataFrame]: DataFrame con los outliers detectados si se especifica retorno_outliers
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return None
    
    X_columns = df.select(pl.col(pl.Int64, pl.Float64)).columns
    x = df.select(pl.col(pl.Int64, pl.Float64)).to_numpy()
    
    if not X_columns: 
        logger.error(f'El DataFrame del archivo "{nombre_archivo}" no contiene columnas numéricas (Int64 o Float64) para detectar outliers. Saltando detección de outliers.')
        return None
    
    try: 
        mod = IsolationForest(contamination=contaminacion, random_state=42)
        pred = mod.fit_predict(x)
        
        df = df.with_columns(
            pl.Series('Anomalia', pred)
        )
        
        outliers_observar = df.filter(pl.col('Anomalia')== -1)
        total_outliers_general = outliers_observar.height
        
        if not (outliers_observar.is_empty()): 
            logger.info('--- Outliers ---')
            logger.info(f'{nombre_archivo}')
            logger.info(f'Total de outliers detectados: {total_outliers_general}')
            logger.info(f'Muestro de Outliers: {outliers_observar.head()}')
        
        if columnas: 
            for col in X_columns: 
                X = df[col].to_numpy().reshape(-1, 1)
                modelo = IsolationForest(contamination=contaminacion, random_state=42)
                prediccion = modelo.fit_predict(X)
                outliers = df[col].filter(
                    pl.Series(values=(prediccion == -1))
                    ).to_list()
                if outliers: 
                    logger.info(f'Total de Outliers encontrados para la columna {col} : {len(outliers)}')
                    print(f'Muestreo de Outliers para la columna {col}: {outliers[:10]}')
    except Exception as e: 
        logger.error(f'Aparecio una error para {nombre_archivo} : {e}') 

#Pipeline de gestión rápida
def pipeline(
            #General
            nombre_archivo: str,
            df: pl.DataFrame, 
            #Estadisticas descriptivas 
            df_estaditisticas: pl.DataFrame,
            #duplicados
            list_col_duplicados:list,
            #cardinalidad
            list_col_cardinalidad:list,
            #duplicados
            general_duplicados: bool=False,
            #cardinalidad
            general_cardinalidad: bool=False, 
            mostrar_cardinalidad_rara: bool=False, 
            umbral_cardinalidad: float=0.05,
            #Nulos 
            columnas_nulos: bool=False,
            #Outlier
            contaminacion: float=0.01, 
            columnas_outlier: bool=False, 
            #Agrupaciones 
            metodo_agrupaciones: str='count'
            ) -> None: 
    '''
    Realiza un análisis exploratorio de datos (EDA) en un DataFrame de Polars
        
    Returns:
        Optional[tuple]: Una tupla con los diccionarios de nulos y unicos
    '''
    logger.info(f'--- EDA para {nombre_archivo} ---')
    columnas_y_filas(nombre_archivo=nombre_archivo, df=df)
    nombre_columnas(nombre_archivo=nombre_archivo, df=df)
    vista_previa_general(nombre_archivo=nombre_archivo, df=df)
    estadisticas_descriptivas(nombre_archivo=nombre_archivo, df=df_estaditisticas)
    agrupaciones(nombre_archivo=nombre_archivo, df=df, metodo=metodo_agrupaciones)
    nulos(nombre_archivo=nombre_archivo, df=df, columnas=columnas_nulos)
    tipo_de_dato(nombre_archivo=nombre_archivo, df=df)
    duplicados(nombre_archivo=nombre_archivo, list_col=list_col_duplicados, df=df, general=general_duplicados)
    cardinalidad(nombre_archivo=nombre_archivo, df=df, list_col=list_col_cardinalidad, general=general_cardinalidad, mostrar_cardinalidad_rara=mostrar_cardinalidad_rara, umbral=umbral_cardinalidad)
    outlier(nombre_archivo=nombre_archivo, df=df, contaminacion=contaminacion, columnas=columnas_outlier)
