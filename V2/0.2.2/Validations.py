#Importando las librerias 
import polars as pl
import logging
from zoneinfo import ZoneInfo, zoneinfo
from Utils import ColumnAnalyzer

#Configuracion del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidacionesEstaticas: 
    @staticmethod
    def zona_valida(zona: str) -> None: 
        try: 
            tz = ZoneInfo(zona)
        except zoneinfo.ZoneInfoNotFoundError as e: 
            logger.error(f'La zona {zona} no es una zona horaria valida')
            raise ValueError(f'La zona {zona} no es una zona horaria valida')
    
    @staticmethod
    def diccionario_vacio(diccionario: dict) -> None: 
        if not diccionario: 
            logger.error('El diccionario está vacio')
            raise ValueError('El diccionario está vacio')

class ValidacionesDinamicas: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = df
        self.columna = ColumnAnalyzer(df=self.df)
    
    def dataframe_vacio(self) -> None: 
        if self.df.is_empty(): 
            logger.error('El DataFrame está vacío')
            raise ValueError('El DataFrame está vacío')
    
    def columna_existente(self, col: str) -> None: 
        self.dataframe_vacio()
        if col not in self.df.columns: 
            logger.error(f'La columa {col} no se encuentra en el DataFrame')
            raise KeyError(f'La columa {col} no se encuentra en el DataFrame')
    
    def columna_numerica(self, col: str) -> None: 
        self.columna_existente(col=col)
        columna_numerica = self.columna.columna_numerica()
        if col not in columna_numerica.columns: 
            logger.error(f'La columna {col} no es una columna numerica')
            raise TypeError(f'La columna {col} no es una columna numerica')
    
    def columna_categorica(self, col: str) -> None: 
        self.columna_existente(col=col)
        columna_categorica = self.columna.columna_categorica()
        if col not in columna_categorica.columns: 
            logger.error(f'La columna {col} no es una columna categorica')
            raise TypeError(f'La columna {col} no es una columna categorica')
    
    def columna_fecha(self, col: str) -> None: 
        self.columna_existente(col=col)
        columna_fecha = self.columna.columna_fecha()
        if col not in columna_fecha.columns: 
            logger.error((f'La columna {col} no es una columna del tipo fecha'))
            raise TypeError(f'La columna {col} no es una columna del tipo fecha')
    
    def nulos(self, col: str) -> None: 
        self.dataframe_vacio()
        self.columna_existente(col=col)
        nulos = self.df[col].is_null().sum()
        if nulos > 0: 
            logger.error(f'La columna {col} tiene nulos')
            raise ValueError(f'La columna {col} tiene nulos')
    
    def valor_existente(self, col: str, patron: str) -> None: 
        self.dataframe_vacio()
        self.columna_existente(col=col)
        self.columna_categorica(col=col)
        valor_existente = self.columna.columna_categorica().filter(pl.col(col) == patron).height
        if valor_existente == 0: 
            logger.error(f'{patron} no se encuntra en la columna {col}')
            raise ValueError(f'{patron} no se encuntra en la columna {col}')
        if patron == None: 
            logger.error(f'El patron no puede ser un tipo None')
            raise ValueError(f'El patron no puede ser un tipo None')
    
    def llaves(self, diccionario: dict) -> None: 
        ValidacionesEstaticas().diccionario_vacio(diccionario=diccionario)
        self.dataframe_vacio()
        columnas_no_existentes = set(diccionario.keys()) - set(self.df.columns)
        if columnas_no_existentes:
            logger.error(f"Las columnas {columnas_no_existentes} no están en el DataFrame")
            raise KeyError(f"Las columnas {columnas_no_existentes} no están en el DataFrame")
        if None in list(diccionario.keys()): 
            logger.error(f"Las llaves no pueden tener valores nulos")
            raise ValueError(f"Las llaves no pueden tener valores nulos")
        if None in list(diccionario.values()): 
            logger.error(f'La clave no puede ser un tipo None')
            raise ValueError(f'La clave no puede ser un tipo None')

