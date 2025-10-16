#Importamos las librerías importantes
import polars as pl
import logging 
from typing import Union

#Configuramos el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

class EDAGeneral: 
    def __init__(self, frame: Union[pl.DataFrame, pl.LazyFrame]):
        if isinstance(frame, pl.LazyFrame): 
            frame = frame.slice(0, 1000).collect(engine='streaming')
        
        self.frame = frame
    
    def primera_vista(self) -> None: 
        head = self.frame.head(n=10)
        logger.info(f'Primeras 10 filas del Frame: \n{head} \n\n')
    
    def info_general(self) -> None: 
        filas, columnas_shape = self.frame.shape
        columnas = self.frame.columns
        schema = self.frame.dtypes
        
        logger.info(f'Columnas Totales: {columnas_shape}, Filas Totales: {filas}')
        logger.info(f'Nombre de Columnas: {columnas}')
        logger.info(f'Tipo de Datos: {schema} \n\n')
    
    def estadistica_general(self) -> None: 
        dsc = self.frame.describe()
        logger.info(f'Estadística Básica: \n{dsc} \n\n')
    
    def valores_nulos(self) -> None: 
        nulos_por_columnas = self.frame.null_count()
        logger.info(f'Nulos Totales por Columna: \n{nulos_por_columnas} \n\n')
    
    def EDA_Basico(self) -> None: 
        logger.info(' -- Vistazo al Frame -- ')
        self.primera_vista()
        
        
        logger.info(' -- Información General -- ')
        self.info_general()
        
        
        logger.info(' -- Estadística General -- ')
        self.estadistica_general()
        
        
        logger.info(' -- Valores Nulos --')
        self.valores_nulos()

