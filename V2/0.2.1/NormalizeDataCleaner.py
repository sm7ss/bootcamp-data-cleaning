#Librerías importadas 
import polars as pl
import logging
from dataclasses import dataclass
from ColumnAnalyzer import ColumnAnalyzer
from enum import Enum
from typing import Optional
from Validations import ValidacionesDinamicas

#Configuracion del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame
    
    @property
    def ColumnaCategorica(self) -> pl.DataFrame: 
        return ColumnAnalyzer(df=self.df).columna_categorica()

class EstrategiasNormalizacionTexto(str, Enum): 
    caracteres_NO_alfanumericos = 'caracteres no alfanumericos'
    espacios_extra = 'espacio extra'
    minusculas = 'minusculas'
    mayusculas = 'mayusculas'
    espacio_inicial_final = 'espacio inicial final'
    acentos = 'acentos'

class NormalizacionTexto: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df)
        self.validacion.dataframe_vacio()
        self.df_categorico = df.ColumnaCategorica
    
    def caracteres_no_alfanumericos(self, col: str) -> pl.DataFrame: 
        self.validacion.columna_categorica(col=col)
        normalizacion = self.df_categorico.with_columns(
            pl.col(col).str.replace_all(r"[^a-zA-Z0-9\s]", " ", literal=False)
        )
        logger.info(f'Se eliminaron los caracteres no alfanumericos para la columna {col}')
        return normalizacion
    
    def espacios_extra(self, col: str) -> pl.DataFrame: 
        self.validacion.columna_categorica(col=col)
        normalizacion = self.df_categorico.with_columns(
            pl.col(col).str.replace_all(r'\s+', ' ', literal=False)
        )
        logger.info(f'Se eliminaron los espacios extra para la columna {col}')
        return normalizacion
    
    def minusculas(self, col: str) -> pl.DataFrame: 
        self.validacion.columna_categorica(col=col)
        normalizacion = self.df_categorico.with_columns(
            pl.col(col).str.to_lowercase()
        )
        logger.info(f'Se convirtieron las palabras a minusculas para la columna {col}')
        return normalizacion
    
    def mayusculas(self, col: str) -> pl.DataFrame: 
        self.validacion.columna_categorica(col=col)
        normalizacion = self.df_categorico.with_columns(
            pl.col(col).str.to_uppercase()
        )
        logger.info(f'Se convirtieron las palabras a mayusculas para la columna {col}')
        return normalizacion
    
    def espacios(self, col: str) -> pl.DataFrame: 
        self.validacion.columna_categorica(col=col)
        normalizacion = self.df_categorico.with_columns(
            pl.col(col).str.strip_chars()
        )
        logger.info(f'Se eliminaron los espacios al inicio y al final para la columna {col}')
        return normalizacion
    
    def eliminar_acentos(self, col: str) -> pl.DataFrame: 
        self.validacion.columna_categorica()
        normalizacion = self.df_categorico.with_columns(
            pl.col(col)
            .str.normalize("NFKD")
            .str.replace_all(r"\p{M}", "", literal=False)
        )
        logger.info(f'Se eliminaron los acentos en la columna {col}')
        return normalizacion

class PipelineNormalizacionTexto: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df).ColumnaCategorica
        self.normalizacion = NormalizacionTexto(df=self.df)
    
    def PipelineTexto(self, col: str, estrategia: Optional[EstrategiasNormalizacionTexto]=None) -> pl.DataFrame: 
        try:
            match estrategia: 
                case EstrategiasNormalizacionTexto.caracteres_NO_alfanumericos: 
                    normalizacion = self.normalizacion.caracteres_no_alfanumericos(col=col)
                case EstrategiasNormalizacionTexto.espacios_extra: 
                    normalizacion = self.normalizacion.espacios_extra(col=col)
                case EstrategiasNormalizacionTexto.minusculas: 
                    normalizacion = self.normalizacion.minusculas(col=col)
                case EstrategiasNormalizacionTexto.mayusculas: 
                    normalizacion = self.normalizacion.mayusculas(col=col)
                case EstrategiasNormalizacionTexto.espacio_inicial_final: 
                    normalizacion = self.normalizacion.espacios(col=col)
                case EstrategiasNormalizacionTexto.acentos: 
                    normalizacion = self.normalizacion.eliminar_acentos(col=col)
                case None: 
                    logger.warning(f'No se selecciono una estrategia para la columna {col}')
                    return self.df
                case _: 
                    logger.error(f'Estrategia {estrategia} no es valida')
                    return self.df
            return normalizacion
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer normalizar los datos categoricos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')
