#Importación de las liberarías necesarias
import polars as pl
import logging
from Outliers import limpieza_de_outlier

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Procesa columnas númericas 
def procesar_nulos_num_col(nombre_archivo: str, 
                        df: pl.DataFrame, 
                        columna: str, 
                        umbral_eliminar_col: float = 0.80, 
                        umbral_imputar: float = 0.05, 
                        estrategia_de_imputación : str= 'media', 
                        interpolacion: str='mediana') -> pl.DataFrame:
    """
    Realiza limpieza de nulos en una columna numérica de un DataFrame de Polars 
    Si el porcentaje de nulos supera el umbral de imputación, se imputa con una estrategia de imputación
    Si el porcentaje de nulos supera el umbral de eliminación, se elimina la columna
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a limpiar
        columna (str): Nombre de la columna numérica a limpiar
        umbral_eliminar_col (float, optional): Umbral de porcentaje de nulos para eliminar la columna. Defaults to 0.80
        umbral_imputar (float, optional): Umbral de porcentaje de nulos para imputar. Defaults to 0.05
        estrategia_de_imputación (str, optional): Estrategia de imputación a utilizar. Defaults to 'media'
        interpolacion (str, optional): Tipo de interpolación a utilizar. Defaults to 'mediana'

    Returns:
        pl.DataFrame: DataFrame de Polars con los nulos imputados o la columna eliminada o el df pasdo en caso de estar vacío o la columna pasada no está en el DataFrame 
    """
    if df.is_empty(): 
        logger.error(f'DataFrame de {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    if columna not in df_limpieza.columns: 
        logger.error(f'{columna} no se encuentra en el archivo {nombre_archivo}')
        return df_limpieza
    
    nulos = df_limpieza[columna].is_null().sum() / df_limpieza.shape[0]

    logger.info(f'Columna{columna} tiene un porcentaje de {nulos:.2%} datos nulos')
    
    estrategias_validas = ['mediana', 'media', 'interpolacion']
    estrategias_validas_interpolacion = ['media', 'mediana']
    
    if nulos >= umbral_eliminar_col: 
        df_limpieza = df_limpieza.drop(columna)
        logger.info(f'Columna {columna} eliminada. Porcentaje de nulos {nulos:.2%}')
        return df_limpieza
    
    elif umbral_imputar<=nulos<umbral_eliminar_col: 
        if estrategia_de_imputación in estrategias_validas: 
            if estrategia_de_imputación == 'media': 
                df_limpieza = df_limpieza.with_columns(pl.col(columna).fill_null(pl.col(columna).mean()))
                logger.info(f'Columna {columna} se le imputo la media a los valores nulos')
                return df_limpieza
            elif estrategia_de_imputación == 'mediana': 
                df_limpieza = df_limpieza.with_columns(pl.col(columna).fill_null(pl.col(columna).median()))
                logger.info(f'Columna {columna} se le imputo la mediana a los valores nulos')
                return df_limpieza
            elif estrategia_de_imputación == 'interpolacion': 
                df_limpieza = df_limpieza.with_columns(pl.col(columna).interpolate())
                if interpolacion in estrategias_validas_interpolacion: 
                    if interpolacion == 'media':
                        df_limpieza = df_limpieza.with_columns(pl.col(columna).fill_null(pl.col(columna).mean()))
                        logger.info(f'Columna {columna} se le imputo interpolación a los valores nulos con la media')
                        return df_limpieza
                    elif interpolacion == 'mediana': 
                        df_limpieza = df_limpieza.with_columns(pl.col(columna).fill_null(pl.col(columna).median()))
                        logger.info(f'Columna {columna} se le imputo interpolación a los valores nulos con la mediana')
                        return df_limpieza
                else: 
                    logger.warning(f'{interpolacion} no disponible')
                    return df_limpieza 
        else: 
            logger.warning(f'Metodo {estrategia_de_imputación} no disponible')
            return df_limpieza
    else: 
        logger.warning(f'Porcentaje de nulos muy bajos: {nulos:.2%}%. No se elimina ni se inputa')
        return df_limpieza

#Procesa columnas categorícas 
def procesar_nulos_cat_col(nombre_archivo:str, 
                        df: pl.DataFrame, 
                        col:str, 
                        umbral_eliminar_col:float=0.80, 
                        umbral_imputar: float=0.05, 
                        estrategia_de_imputación: str='moda', 
                        valor_desconocido:str='desconocido') -> pl.DataFrame:
    """
    Realiza limpieza de nulos en una columna categórica de un DataFrame de Polars 
    Si el porcentaje de nulos supera el umbral de imputación, se imputa con una estrategia de imputación
    Si el porcentaje de nulos supera el umbral de eliminación, se elimina la columna
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a limpiar
        col (str): Nombre de la columna categórica a limpiar
        umbral_eliminar_col (float, optional): Umbral de porcentaje de nulos para eliminar la columna. Defaults to 0.80
        umbral_imputar (float, optional): Umbral de porcentaje de nulos para imputar. Defaults to 0.05
        estrategia_de_imputación (str, optional): Estrategia de imputación a utilizar. Defaults to 'moda'
        valor_desconocido (str, optional): Valor a imputar en los nulos. Defaults to 'desconocido'
    
    Returns:
        pl.DataFrame: DataFrame de Polars con los nulos imputados o la columna eliminada o el df pasdo en caso de estar vacío o la columna pasada no está en el DataFrame 
    """
    if df.is_empty(): 
        logger.error(f'DataFrame de {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    if col not in df_limpieza.columns: 
        logger.error(f'{col} no se encuentra en el archivo {nombre_archivo}')
        return df_limpieza
    
    nulos = df_limpieza[col].is_null().sum() / df_limpieza.shape[0]

    logger.info(f'La columna {col} tiene un total de nulos del {nulos:.2%}%')
    
    estrategia_de_imputación_validas = ['moda', 'desconocido']
    
    if nulos >= umbral_eliminar_col: 
        df_limpieza = df_limpieza.drop(col)
        
        logger.info(f'Eliminamos la columna {col} por un alto porcentaje de valores nulos: {nulos:.2%}%')
        return df_limpieza
    
    elif umbral_imputar<=nulos<umbral_eliminar_col: 
        if estrategia_de_imputación in estrategia_de_imputación_validas: 
            if estrategia_de_imputación == 'moda': 
                val_imputar = df_limpieza[col].mode()
                if not val_imputar.is_empty(): 
                    val_imputar = val_imputar.item()
                    logger.info(f'Columna {col} imputa valor {val_imputar}')
                    df_limpieza = df_limpieza.with_columns(pl.col(col).fill_null(val_imputar))
                    return df_limpieza
                else: 
                    logger.warning(f'No se pudo calcular la moda para la columna {col}')
                    return df_limpieza
            elif estrategia_de_imputación == 'desconocido': 
                df_limpieza = df_limpieza.with_columns(pl.col(col).fill_null(valor_desconocido))
                logger.info(f'Imputamos {valor_desconocido} a los valores nulos en columna {col}')
                return df_limpieza
            else: 
                logger.warning(f'{estrategia_de_imputación} no disponible')
                return df_limpieza
        else: 
            logger.warning(f'{estrategia_de_imputación} no disponible')
            return df_limpieza
    else: 
        logger.warning(f'Porcentaje de nulos muy bajos: {nulos:.2%}%. No se elimina ni se inputa')
        return df_limpieza

#Procesa tipo de datos 
def conversion_de_tipo_de_datos(nombre_archivo: str, 
                                df: pl.DataFrame, 
                                diccionario: dict) -> pl.DataFrame: 
    """
    Realiza la conversión de tipo de datos en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a convertir
        diccionario (dict): Diccionario con las columnas y los tipos de datos a convertir
    
    Returns:
        pl.DataFrame: DataFrame de Polars con los tipos de datos convertidos
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    tipo_de_datos_disponibles = [pl.Int64, pl.Utf8, pl.Float64, pl.Date]
    
    try: 
        for col, tipo_de_dato in diccionario.items(): 
            if tipo_de_dato not in tipo_de_datos_disponibles: 
                logger.error(f'El tipo de dato {tipo_de_dato} introducido no está disponible. Usar alguno de los disponibles: {tipo_de_datos_disponibles}')
                continue
            
            if col not in df_limpieza.columns: 
                logger.error(f'La columna {col} no está en el DataFrame del archivo {nombre_archivo}')
                continue
            
            if tipo_de_dato == pl.Date: 
                df_limpieza = df_limpieza.with_columns(
                    pl.col(col).str.to_date(strict=False)
                )
                logger.info(f'Se conviertieron los datos de la columna {col} a {df_limpieza[col].dtype}') 
            elif tipo_de_dato in [pl.Int64, pl.Utf8, pl.Float64]: 
                df_limpieza = df_limpieza.with_columns(
                    pl.col(col).cast(tipo_de_dato)
                )
                logger.info(f'Se conviertieron los datos de la columna {col} a {df_limpieza[col].dtype}')
    except Exception as e: 
        print(f'Ocurrio un error en la tranformación de datos tipo: {e}')
        return df_limpieza
    return df_limpieza

#Normalización de los datos en el df en columna específica
def normalizacion_texto(nombre_archivo:str, 
                        df:pl.DataFrame, 
                        col:str, 
                        espacios_extra:bool=True, 
                        lowercase:bool=True, 
                        uppercase:bool=False,
                        espacios:bool=True, 
                        car_alfanum: bool=False) -> pl.DataFrame: 
    """
    Realiza la normalización de texto en una columna de un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a normalizar
        col (str): Nombre de la columna a normalizar
        espacios_extra (bool, optional): Elimina los espacios extra. Defaults to True
        lowercase (bool, optional): Convierte a minúsculas. Defaults to True
        uppercase (bool, optional): Convierte a mayúsculas. Defaults to False
        espacios (bool, optional): Elimina los espacios. Defaults to True
        car_alfanum (bool, optional): Elimina los caracteres no alfanuméricos. Defaults to False
    
    Returns:
        pl.DataFrame: DataFrame de Polars con el texto normalizado
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} esá vacío')
        return df
    
    df_limpieza = df.clone()
    
    if col not in df_limpieza.columns: 
        logger.error(f'La columna {col} no se encunetra el DataFrame del archivo {nombre_archivo}')
        return df_limpieza
    
    if car_alfanum: 
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).str.replace(r"[^a-zA-Z0-9\s]", " ", literal=False)
        )
        logger.info(f'Se eliminaron los caracteres alfanumericos para la columna {col}')
    
    if espacios_extra: 
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).str.replace(r'\s+', ' ', literal=False)
        )
        logger.info(f'Se eliminaron los espacios extra para la columna {col}')
    
    if lowercase: 
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).str.to_lowercase()
        )
        logger.info(f'Se convirtieron las palabras a lowercase para la columna {col}')
    
    if uppercase: 
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).str.to_uppercase()
        )
        logger.info(f'Se convirtieron las palabras a uppercase para la columna {col}')
    
    if espacios: 
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).str.strip_chars()
        )
        logger.info(f'Se eliminaron los espacios para la columna {col}')
    return df_limpieza

#Nomalización de los nombres de las columnas
def normalizacion_nom_columnas(nombre_archivo: str, 
                            df: pl.DataFrame,
                            dict_renombrar: dict) -> pl.DataFrame:
    """
    Realiza la normalización de nombres de columnas en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a normalizar
        dict_renombrar (dict): Diccionario con los nombres de las columnas a renombrar
    
    Returns:
        pl.DataFrame: DataFrame de Polars con los nombres de las columnas normalizados
    """
    if df.is_empty(): 
        logger.error(f'DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    if not dict_renombrar: 
        logger.error(f'El diccionario para renombrar las columnas esta vacío')
        return df_limpieza
    
    df_limpieza = df_limpieza.rename(dict_renombrar)
    logger.info(f'Se renombraron las columnas {dict_renombrar.keys()} a {dict_renombrar.values()}')
    
    return df_limpieza

#Remplazo de valores en las columnas del df
def remplazo_dinamico(nombre_archivo: str, 
                    col: str, 
                    df:pl.DataFrame, 
                    diccionario:dict) -> pl.DataFrame: 
    """
    Realiza el remplazo de valores en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        col (str): Nombre de la columna a remplazar
        df (pl.DataFrame): DataFrame de Polars a remplazar
        diccionario (dict): Diccionario con los valores a remplazar
    
    Returns:
        pl.DataFrame: DataFrame de Polars con los valores remplazados
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    if col not in df_limpieza.columns: 
        logger.error(f'La columna {col} no se encuentra en el DataFrame del archivo {nombre_archivo}')
        return df_limpieza
    
    try:
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).replace(diccionario)
        )
        logger.info(f'Se remplazaron los valores de la columna {col}')
    except Exception as e: 
        logger.error(f'Ocurrio una excepcion tipo {e} para la columna {col} del dataframe del archivo {nombre_archivo}')
        return df_limpieza
    return df_limpieza

#Elimina duplicados
def eliminar_duplicados(nombre_archivo: str, 
                        df: pl.DataFrame,
                        eliminar_general: bool=True, 
                        eliminar_columna_dup: bool=False, 
                        columna_eliminar_dup: str=None, 
                        reiniciar_indice: bool=True) -> pl.DataFrame: 
    '''
    Esta función elimina los duplicados de un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a eliminar duplicados
        eliminar_general (bool, optional): Elimina duplicados generales. Defaults to True
        eliminar_columna_dup (bool, optional): Elimina duplicados por columna. Defaults to False
        columna_eliminar_dup (str, optional): Nombre de la columna para eliminar duplicados. Defaults to None
        reiniciar_indice (bool, optional): Reestablece el índice. Defaults to False
    
    Returns:
        pl.DataFrame: DataFrame de Polars sin duplicados o con eliminación de duplicados
    '''
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    if eliminar_general and eliminar_columna_dup: 
        logger.error(f'No se puede eliminar duplicados generales y por columna')
        return df_limpieza
    
    if not eliminar_general and not eliminar_columna_dup: 
        logger.error(f'No se seleccionó ninguna opción de eliminación de duplicados')
        return df_limpieza
    
    if eliminar_general: 
        if reiniciar_indice: 
            df_limpieza = df_limpieza.unique().with_row_index()
            logger.info(f'Se eliminaron los duplicados generales y se reestableció el índice para el archivo {nombre_archivo}')
        else: 
            df_limpieza = df_limpieza.unique()
            logger.info(f'Se eliminaron los duplicados generales')
    
    if eliminar_columna_dup: 
        if reiniciar_indice: 
            if columna_eliminar_dup is not None: 
                df_limpieza = df_limpieza.unique(subset=[columna_eliminar_dup]).with_row_index()
                logger.info(f'Se eliminaron los duplicados de la columna {columna_eliminar_dup} y se reestableció el índice')
            else: 
                logger.error(f'No se seleccionó ninguna columna para eliminar duplicados')
                return df_limpieza
        else: 
            if columna_eliminar_dup is not None: 
                df_limpieza = df_limpieza.unique(subset=[columna_eliminar_dup])
                logger.info(f'Se eliminaron los duplicados de la columna {columna_eliminar_dup}')
            else: 
                logger.error(f'No se seleccionó ninguna columna para eliminar duplicados')
                return df_limpieza
    return df_limpieza

#Reducción de cardinalidad
def reduccion_cardinalidad(nombre_archivo: str, 
                        df: pl.DataFrame, 
                        lista_col: list, 
                        umbral: float=0.05, 
                        remplazo_cardinalidad: str='otro') -> pl.DataFrame: 
    """
    Realiza la reducción de cardinalidad en un DataFrame de Polars
    
    Args:
        nombre_archivo (str): Nombre del archivo
        df (pl.DataFrame): DataFrame de Polars a reducir cardinalidad
        lista_col (list): Lista de columnas a reducir cardinalidad
        umbral (float, optional): Umbral para reducir cardinalidad. Defaults to 0.05
        remplazo_cardinalidad (str, optional): Valor para remplazar cardinalidad baja. Defaults to 'Otro'
    
    Returns:
        pl.DataFrame: DataFrame de Polars con cardinalidad reducida
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    validador= [col for col in lista_col if col not in df_limpieza.columns]
    if validador: 
        logger.error(f'Las columnas {validador} no se encuentran en el DataFrame del archivo {nombre_archivo}')
        return df_limpieza
    
    for col in lista_col: 
        freq = df_limpieza[col].value_counts()
        categorias_raras = freq.filter(pl.col('count') / df_limpieza.shape[0] < umbral)[col]
        
        df_limpieza = df_limpieza.with_columns(
            pl.col(col).replace(categorias_raras, remplazo_cardinalidad)
        )
        logger.info(f'Se redujo la cardinalidad de la columna {col}')
    
    return df_limpieza

#Pipeline General
def pipeline_general(
    nombre_archivo: str, 
    df: pl.DataFrame, 
    #Conversión de tipo de datos
    diccionario_tipo_de_datos: dict
    ) -> pl.DataFrame: 
    """
    Realiza el pipeline de limpieza general de datos para un DataFrame de Polars
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    df_limpieza = conversion_de_tipo_de_datos(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        diccionario=diccionario_tipo_de_datos)
    
    return df_limpieza

#Piepeline limpieza categorica 
def pipeline_limpieza_categorica(
    nombre_archivo: str, 
    df: pl.DataFrame, 
    #Nulos
    columna_nulos: str, 
    #Normalización de texto
    columna_normalizar: str, 
    #Remplazo dinámico 
    columna_remplazo_dinamico: str, 
    diccionario_remplazo_dinamico: dict, 
    #Normalizar nombre de las columnas 
    dict_renom_colulmnas: dict,
    #Reducción de cardinalidad 
    lista_col_cardinalidad: list, 
    umbral_cardinalidad: float=0.05, 
    remplazo_cardinalidad: str='otro',
    #Nulos
    umbral_eliminar_col_nulos: float=0.80, 
    umbral_imputar_nulos: float=0.05, 
    estrategia_de_imputación_nulos: str='moda', 
    valos_desconocido: str='desconocido', 
    #Normalización de texto
    espacios_extra: bool=True,
    lowercase: bool=True, 
    uppercase: bool=False, 
    espacios: bool=True, 
    car_alfanum: bool=False, 
    #Eliminar Duplicados
    eliminar_general_duplicados: bool=True, 
    eliminar_columna_duplicados: bool=False, 
    columna_eliminar_duplicados: str=None, 
    reiniciar_indice_duplicados: bool=True
    ) -> pl.DataFrame: 
    """
    Realiza el pipeline de limpieza de datos para columnas categoricas de un DataFrame de Polars
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    df_limpieza = procesar_nulos_cat_col(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        col=columna_nulos, 
        umbral_eliminar_col=umbral_eliminar_col_nulos, 
        umbral_imputar=umbral_imputar_nulos, 
        estrategia_de_imputación=estrategia_de_imputación_nulos, 
        valor_desconocido=valos_desconocido)
    df_limpieza = normalizacion_texto(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        col=columna_normalizar, 
        espacios_extra=espacios_extra, 
        lowercase=lowercase, 
        uppercase=uppercase, 
        espacios=espacios, 
        car_alfanum=car_alfanum)
    df_limpieza = normalizacion_nom_columnas(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        dict_renom_colulmnas=dict_renom_colulmnas)
    df_limpieza = remplazo_dinamico(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        col=columna_remplazo_dinamico, 
        diccionario_remplazo_dinamico=diccionario_remplazo_dinamico)
    df_limpieza = reduccion_cardinalidad(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        lista_col=lista_col_cardinalidad, 
        umbral=umbral_cardinalidad, 
        remplazo_cardinalidad=remplazo_cardinalidad)
    df_limpieza = eliminar_duplicados(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        eliminar_general_duplicados=eliminar_general_duplicados, 
        eliminar_columna_duplicados=eliminar_columna_duplicados, 
        columna_eliminar_duplicados=columna_eliminar_duplicados, 
        reiniciar_indice_duplicados=reiniciar_indice_duplicados)
    return df_limpieza

#Pipeline limpieza numerica 
def pipeline_limpieza_numerica(
    nombre_archivo: str, 
    df: pl.DataFrame, 
    #Outliers
    columnas_outliers: list,
    #Nulos
    columna_nulos: str,
    umbral_eliminar_col_nulos: float=0.80, 
    umbral_imputar_nulos: float=0.05, 
    estrategia_de_imputación_nulos: str='media', 
    interpolacion_nulos: str='mediana', 
    #Outliers
    winsorizacion_col: bool=True, 
    max_q: float=0.75, 
    min_q: float=0.25,
    eliminacion_col_outliers: bool=False
    ) -> pl.DataFrame: 
    """
    Realiza el pipeline de limpieza de datos para columnas numericas de un DataFrame de Polars
    """
    if df.is_empty(): 
        logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
        return df
    
    df_limpieza = df.clone()
    
    df_limpieza = procesar_nulos_num_col(
        nombre_archivo=nombre_archivo, 
        df=df_limpieza, 
        columna= columna_nulos, 
        umbral_eliminar_col=umbral_eliminar_col_nulos, 
        umbral_imputar=umbral_imputar_nulos, 
        estrategia_de_imputación=estrategia_de_imputación_nulos, 
        interpolacion=interpolacion_nulos
    )
    df_limpieza = limpieza_de_outlier(
        nombre_archivo= nombre_archivo, 
        df= df, 
        columnas= columnas_outliers,
        winsorizacion_col= winsorizacion_col, 
        max_q= max_q, 
        min_q= min_q,
        eliminacion_col= eliminacion_col_outliers
    )
    
    return df_limpieza
