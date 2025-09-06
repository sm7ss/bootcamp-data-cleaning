#Importamos las librerías necesarias 
import polars as pl 
from typing import Optional, Union
import logging
from ColumnAnalyzer import ColumnAnalyzer
from enum import Enum
from Validations import ValidacionesDinamicas

#Configuramos logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DFWrapper: 
    df: pl.DataFrame

class EstrategiaDeAgrupacionNumerica(str, Enum): 
    suma = 'suma'
    media = 'media'
    mediana = 'mediana'
    desviacion_estandar = 'desviacion estandar'
    varianza = 'varianza'

class EstrategiaDeAgrupacionGeneral(str, Enum): 
    count = 'count'
    first = 'first'

class AgrupacionNumerica: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_agrupacion = df.df
    
    def suma(self, col_categorica: str, col_numerica: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_numerica(col=col_numerica)
        logger.info(f'Se agrupo la columna {col_categorica} con la columna {col_numerica} con el metodo de suma')
        return self.df_agrupacion.group_by(col_categorica).agg(
            suma_agrupacion=pl.col(col_numerica).sum()
        )
    
    def media(self, col_categorica: str, col_numerica: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_numerica(col=col_numerica)
        logger.info(f'Se agrupo la columna {col_categorica} con la columna {col_numerica} con la media ')
        return self.df_agrupacion.group_by(col_categorica).agg(
            media_agrupacion=pl.col(col_numerica).mean()
        )
    
    def mediana(self, col_categorica: str, col_numerica: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_numerica(col=col_numerica)
        logger.info(f'Se agrupo la columna {col_categorica} con {col_numerica} con la mediana')
        return self.df_agrupacion.group_by(col_categorica).agg(
            mediana_agrupacion=pl.col(col_numerica).median()
        )
    
    def desviacion_estandar(self, col_categorica: str, col_numerica: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_numerica(col=col_numerica)
        logger.info(f'Se agrupo la columna {col_categorica} con {col_numerica} con la desviación estandar')
        return self.df_agrupacion.group_by(col_categorica).agg(
            std_agrupacion=pl.col(col_numerica).std()
        )
    
    def varianza(self, col_categorica: str, col_numerica: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_numerica(col=col_numerica)
        logger.info(f'Se agrupo la columna {col_categorica} con {col_numerica} con la varianza')
        return self.df_agrupacion.group_by(col_categorica).agg(
            varianza_agrupacion=pl.col(col_numerica).var()
        )

class AgrupacionGeneral: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_agrupacion = df.df
    
    def count(self, col_categorica: str, col_agrupacion: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_existente(col=col_agrupacion)
        logger.info(f'Se agrupo la columna {col_categorica} con la columna {col_agrupacion} con el metodo count')
        return self.df_agrupacion.group_by(col_categorica).agg(
            count_agrupacion=pl.col(col_agrupacion).count()
        )
    
    def first(self, col_categorica: str, col_agrupacion: str) -> pl.DataFrame: 
        self.validacion.seleccion_columnas(col_1=col_categorica, col_2=col_numerica)
        self.validacion.columna_categorica(col=col_categorica)
        self.validacion.columna_existente(col=col_agrupacion)
        logger.info(f'Se agrupo la columna {col_categorica} con {col_agrupacion} para ver la primera coincidencia')
        return self.df_agrupacion.group_by(col_categorica).agg(
            first_agrupacion=pl.col(col_agrupacion).first()
        )

class PipelineAgrupacionNumerico: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df)
        self.agrupacion_numerica = AgrupacionNumerica(df=self.df)
    
    def PipelineAgrupacionNumerica(self, col_categorica: str, col_numerica, estrategia: EstrategiaDeAgrupacionNumerica) -> Optional[pl.DataFrame]: 
        try: 
            match estrategia: 
                case EstrategiaDeAgrupacionNumerica.suma: 
                    df_agrupacion = self.agrupacion_numerica.suma(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.media: 
                    df_agrupacion = self.agrupacion_numerica.media(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.mediana: 
                    df_agrupacion = self.agrupacion_numerica.mediana(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.desviacion_estandar: 
                    df_agrupacion = self.agrupacion_numerica.desviacion_estandar(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.varianza: 
                    df_agrupacion = self.agrupacion_numerica.varianza(col_categorica=col_categorica, col_numerica=col_numerica)
                case None: 
                    raise ValueError("Se requiere una estrategia de agrupación")
                case _: 
                    raise ValueError(f"La estrategia '{estrategia}' no es válida")
            return df_agrupacion
        except pl.exceptions.ComputeError as e:
            raise ValueError(f"Error en la operación de agrupación: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error al querer agrupar las columnas: {e}')
            raise 

class PipelineAgrupacionGeneral: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df)
        self.agrupacion_general = AgrupacionGeneral(df=self.df)
    
    def PipelineDeAgrupacionGeneral(self, col_categorica: str, col_agrupacion: str, estrategia: EstrategiaDeAgrupacionGeneral) -> Optional[pl.DataFrame]: 
        try: 
            match estrategia: 
                case EstrategiaDeAgrupacionGeneral.count: 
                    df_agrupacion = self.agrupacion_general.count(col_categorica=col_categorica, col_agrupacion=col_agrupacion)
                case EstrategiaDeAgrupacionGeneral.first: 
                    df_agrupacion = self.agrupacion_general.first(col_categorica=col_categorica, col_agrupacion=col_agrupacion)
                case None: 
                    raise ValueError("Se requiere una estrategia de agrupación")
                case _: 
                    raise ValueError(f"La estrategia '{estrategia}' no es válida")
            return df_agrupacion
        except pl.exceptions.ComputeError as e:
            raise ValueError(f"Error en la operación de agrupación: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error al querer agrupar las columnas: {e}')
            raise 

class PipelineAgrupacion: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df)
        self.agrupacion_numerica = AgrupacionNumerica(df=self.df)
        self.agrupacion_general = AgrupacionGeneral(df=self.df)
    
    def PipelineAgrupacion(self, col_categorica: str, estrategia: Union[EstrategiaDeAgrupacionGeneral, EstrategiaDeAgrupacionNumerica], col_agrupacion: Optional[str]=None, col_numerica: Optional[str]=None) -> Optional[pl.DataFrame]: 
        try: 
            match estrategia: 
                case EstrategiaDeAgrupacionNumerica.suma: 
                    df_agrupacion = self.agrupacion_numerica.suma(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.media: 
                    df_agrupacion = self.agrupacion_numerica.media(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.mediana: 
                    df_agrupacion = self.agrupacion_numerica.mediana(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.desviacion_estandar: 
                    df_agrupacion = self.agrupacion_numerica.desviacion_estandar(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionNumerica.varianza: 
                    df_agrupacion = self.agrupacion_numerica.varianza(col_categorica=col_categorica, col_numerica=col_numerica)
                case EstrategiaDeAgrupacionGeneral.count: 
                    df_agrupacion = self.agrupacion_general.count(col_categorica=col_categorica, col_agrupacion=col_agrupacion)
                case EstrategiaDeAgrupacionGeneral.first: 
                    df_agrupacion = self.agrupacion_general.first(col_categorica=col_categorica, col_agrupacion=col_agrupacion)
                case None: 
                    raise ValueError("Se requiere una estrategia de agrupación")
                case _: 
                    raise ValueError(f"La estrategia '{estrategia}' no es válida")
            return df_agrupacion
        except pl.exceptions.ComputeError as e:
            raise ValueError(f"Error en la operación de agrupación: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error al querer agrupar las columnas: {e}')
            raise 

