#Importamos las librerías necesarias 
import polars as pl 
from typing import Optional
import logging
from Columnas_por_tipo import num_cat_col

#Configuramos logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelmame)s-%(message)s')
logger = logging.getLogger(__name__)

#Clase para agrupar y unir 
#Clase para agrupar y unir 
class AgruparYUnir: 
    def __init__(self, df: pl.DataFrame, nombre_archivo:str) -> None:
        if df.is_empty(): 
            logger.error(f'El DataFrame del archivo {nombre_archivo} esta vacío')
            return None
        
        self.df = df
        self.nombre_archivo = nombre_archivo
        self.metodos = ['sum', 'count', 'redondear']
        self.metodo_union = ['outer', 'left', 'right']
        self.columnas_numericas = num_cat_col(nombre_archivo=self.nombre_archivo, df=self.df)[0]
    
    #Función para hacer agrupaciones con diferentes opciones de metodos/funciones de conteo 
    def agrupar(self, columna: Optional[str], 
                lista_columnas: Optional[list], 
                columna_numerica: str,
                metodo: str='count') -> Optional[pl.DataFrame]: 
        """
        Realiza agrupaciones y uniones en un DataFrame.
        
        Args:
            columna (Optional[str]): Nombre de la columna por la cual se desea agrupar.
            lista_columnas (Optional[list]): Lista de nombres de columnas por las cuales se desea agrupar.
            columna_numerica (str): Nombre de la columna numérica para realizar agregaciones.
            metodo (str, optional): Método de agrupación ('sum', 'count', 'redondear'). Defaults to 'count'.
        
        Returns:
            Optional[pl.DataFrame]: DataFrame resultante de la agrupación y unión.
        """
        if metodo not in self.metodos: 
            logger.error(f'El metodo {metodo} no está disponible')
            return None
        
        if columna_numerica not in self.columnas_numericas: 
            logger.error(f'La columna {columna_numerica} no se encuentra en el Dataframe del archivo {self.nombre_archivo}')
            return None
        
        if columna is not None: 
            if columna not in self.df.columns: 
                logger.error(f'La columna {columna} no está en el DataFrame del archivo {self.nombre_archivo}')
                return None
            
            if metodo == 'sum': 
                agrupacion = self.df.group_by(columna).agg(
                agrupacion=pl.col(columna_numerica).sum()
            )
                logger.info(f'Se agrupo la columna {columna} con el metodo {metodo}')
                return agrupacion
            elif metodo == 'count': 
                agrupacion = self.df.group_by(columna).agg(
                    agrupacion =pl.col(columna_numerica).count()
                )
                logger.info(f'Se agrupo la columna {columna} con el metodo {metodo}')
                return agrupacion
            elif metodo == 'redondear': 
                agrupacion = self.df.group_by(columna).agg(
                    agrupacion =pl.col(columna_numerica).ceil()
                )
                logger.info(f'Se agrupo la columna {columna} con el metodo {metodo}')
                return agrupacion
        
        if lista_columnas is not None: 
            verificacion = [col for col in lista_columnas if col not in self.df.columns]
            if verificacion: 
                logger.error(f'Las columnas {verificacion} no se encuentran en el archivo {self.nombre_archivo}')
                return None 
            
            if metodo == 'sum': 
                agrupacion = self.df.group_by(lista_columnas).agg(
                agrupacion=pl.col(columna_numerica).sum()
            )
                logger.info(f'Se agrupo la lista de columnas {lista_columnas} con el metodo {metodo}')
                return agrupacion
            elif metodo == 'count': 
                agrupacion = self.df.group_by(lista_columnas).agg(
                    agrupacion =pl.col(columna_numerica).count()
                )
                logger.info(f'Se agrupo la lista de columnas {lista_columnas} con el metodo {metodo}')
                return agrupacion
            elif metodo == 'redondear': 
                agrupacion = self.df.group_by(lista_columnas).agg(
                    agrupacion =pl.col(columna_numerica).ceil()
                )
                logger.info(f'Se agrupo la lista de columnas {lista_columnas} con el metodo {metodo}')
                return agrupacion
    
    def union_df(self, df1: pl.DataFrame,
                df2: pl.DataFrame, 
                lista_col_agrupacion: list, 
                metodo: str='outer') -> Optional[pl.DataFrame]: 
        """
        Realiza una unión de dos DataFrames.
        
        Args:
            df1 (pl.DataFrame): Primer DataFrame.
            df2 (pl.DataFrame): Segundo DataFrame.
            lista_col_agrupacion (list): Lista de nombres de columnas por las cuales se desea agrupar.
            metodo (str, optional): Método de unión ('outer', 'left', 'right'). Defaults to 'outer'.
        
        Returns:
            Optional[pl.DataFrame]: DataFrame resultante de la unión.
        """
        if df1.is_empty() or df2.is_empty(): 
            logger.error(f'El DataFrame df 1 o df 2 no contiene datos')
            return None
        
        verificacion_1 = [col for col in lista_col_agrupacion if col not in df1.columns]
        verificacion_2 = [col for col in lista_col_agrupacion if col not in df2.columns]
        if verificacion_1: 
            logger.error(f'Las columnas {verificacion_1} no se encuentran en el DataFrame (1)')
            return None
        elif verificacion_2: 
            logger.error(f'Las columnas {verificacion_2} no se encuentran en el DataFrame (2)')
            return None
        
        if metodo not in self.metodo_union: 
            logger.error(f'El metodo {metodo} no se encunetra en los metodos de union disponibles')
            return None
        
        union = df1.join(df2, on=lista_col_agrupacion, how=metodo)
        logger.info(f'Se union el df 1 con el df 2 con el metodo {metodo}')
        return union

