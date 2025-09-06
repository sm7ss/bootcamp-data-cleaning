#Importando las librerias 
import polars as pl
import logging
from zoneinfo import ZoneInfo
from ColumnAnalyzer import ColumnAnalyzer
from typing import Optional

#Configuracion del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidacionesDinamicas: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = df
        self.columna = ColumnAnalyzer(df=self.df)
    
    def dataframe_vacio(self) -> Optional[pl.DataFrame]: 
        if self.df.is_empty(): 
            raise ValueError('El DataFrame está vacío')
    
    def columna_existente(self, col: str) -> Optional[pl.DataFrame]: 
        self.dataframe_vacio()
        if col not in self.df.columns: 
            raise KeyError(f'La columa {col} no se encuentra en el DataFrame')
    
    def columna_numerica(self, col: str) -> Optional[pl.DataFrame]: 
        self.columna_existente(col=col)
        columna_numerica = self.columna.columna_numerica()
        if col not in columna_numerica.columns: 
            raise TypeError(f'La columna {col} no es una columna numerica')
    
    def columna_categorica(self, col: str) -> Optional[pl.DataFrame]: 
        self.columna_existente(col=col)
        columna_categorica = self.columna.columna_categorica()
        if col not in columna_categorica.columns: 
            raise TypeError(f'La columna {col} no es una columna categorica')
    
    def columna_fecha(self, col: str) -> Optional[pl.DataFrame]: 
        self.columna_existente(col=col)
        columna_fecha = self.columna.columna_fecha()
        if col not in columna_fecha.columns: 
            raise TypeError(f'La columna {col} no es una columna del tipo fecha')
    
    def nulos(self, col: str) -> Optional[pl.DataFrame]: 
        self.dataframe_vacio()
        self.columna_existente(col=col)
        nulos = self.df[col].is_null().sum()
        if nulos > 0: 
            raise ValueError(f'La columna {col} tiene nulos')
    
    def valor_existente(self, col: str, patron: str) -> Optional[pl.DataFrame]: 
        self.dataframe_vacio()
        self.columna_existente(col=col)
        self.columna_categorica(col=col)
        valor_existente = self.columna.columna_categorica().filter(pl.col(col) == patron).height
        if valor_existente == 0: 
            raise ValueError(f'{patron} no se encuntra en la columna {col}')

class ValicacionesEstaticas: 
    @staticmethod
    def zona_valida(zona: str) -> None: 
        try: 
            tz = ZoneInfo(zona)
        except zoneinfo.ZoneInfoNotFoundError as e: 
            raise ValueError(f'La zona {zona} no es una zona horaria valida')
    
    @staticmethod
    def diccionario_vacio(diccionario: dict) -> None: 
        if not diccionario: 
            raise ValueError('El diccionario está vacio')

