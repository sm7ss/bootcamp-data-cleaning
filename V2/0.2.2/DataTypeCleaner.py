import polars as pl
import logging
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
from Validations import ValidacionesDinamicas, ValidacionesEstaticas

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass 
class DFWrapper: 
    df : pl.DataFrame

class EstrategiasDeConversion(str, Enum): 
    entero = 'entero'
    flotante = 'flotante'
    string = 'string'
    fecha = 'fecha'
    fecha_tz = 'fecha para timezone'

class EstrategiasDeFormato(str, Enum): 
    Año_Mes_Dia = 'Año-Mes-Día'
    Dia_Mes_Año = 'Día/Mes/Año'

class EstrategiasDeFormatoTZSinZona(str, Enum): 
    Año_Mes_Dia_Hora_Minuto_Segundo = 'Año-Mes-Día Hora:Minuto:Segundo' 
    utc = 'utc'
    desplazamiento = 'desplazamiento'

class EstrategiasDeFormatoTimezone(str, Enum): 
    localizacion_sin_TZ = 'localizacion sin TZ'
    localizacion_con_TZ = 'localizacion con TZ'

class GestionDeFormatoFecha(str, Enum): 
    estrategia_formato = 'estrategia formato'
    estrategia_formato_TZ_SinZona = 'estrategia formato TZ Sin Zona'
    estrategias_formato_timezone = 'estrategias formato timezone'

class ConversionDeTipoDatos: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_general = df.df
    
    def conversion_a_entero(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_existente(col=col)
        self.validacion.nulos(col=col)
        try: 
            logger.info(f'Se convirtio el tipo de dato {self.df_general[col].dtype} a tipo de entero en la columna {col}')
            return self.df_general.with_columns(
                pl.col(col).cast(pl.Int64)
            )
        except ValueError: 
            logger.error(f'La conversion a entero de la columna {col} no se pudo ejecutar')
    
    def conversion_a_flotante(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_numerica(col=col)
        self.validacion.nulos(col=col)
        logger.info(f'Se convirtio el tipo de dato {self.df_general[col].dtype} a tipo flotante en la columna {col}')
        return self.df_general.with_columns(
            pl.col(col).cast(pl.Float64)
        )
    
    def conversion_a_string(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_existente(col=col)
        self.validacion.nulos(col=col)
        logger.info(f'Se convirtio el tipo de dato {self.df_general[col].dtype} a tipo de string en la columna {col}')
        return self.df_general.with_columns(
            pl.col(col).cast(pl.Utf8)
        )

class ConversionDeTipoDatosFechaSinFormato: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_conversion = df.df
    
    def conversion_a_fecha(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_existente(col=col)
        self.validacion.nulos(col=col)
        try:
            logger.info(f'Se convirtio el tipo de dato {self.df_conversion[col].dtype} a tipo de fecha en la columna {col}')
            return self.df_conversion.with_columns(
                pl.col(col).str.to_date(strict=False)
            )
        except ValueError: 
            logger.error(f'La conversion a fecha de la columna {col} no se pudo ejecutar')
    
    def conversion_a_fecha_timezone(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_existente(col=col)
        self.validacion.nulos(col=col)
        try:
            logger.info(f'Se convirtio el tipo de dato {self.df_conversion[col].dtype} a tipo de fecha timezone en la columna {col}')
            return self.df_conversion.with_columns(
                pl.col(col).str.to_datetime()
            )
        except ValueError: 
            logger.error(f'La conversion a fecha para timezone de la columna {col} no se pudo ejecutar')

class FormatoDeFecha:
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_fecha = df.df
    
    def Año_Mes_Dia(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion.nulos(col=col)
        try:
            df_formateado = self.df_fecha.with_columns(
                pl.col(col).str.strptime(pl.Date, "%Y-%m-%d")
            )
            logger.info(f'La columna {col} ha sido formateada con el formato de fecha: Año-Mes-Día')
            return df_formateado
        except ValueError: 
            logger.error(f'La conversion a formato de fecha de la columna {col} no se pudo ejecutar')
    
    def Dia_Mes_Año(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion.nulos(col=col)
        try:
            df_formateado = self.df_fecha.with_columns(
                pl.col(col).str.strptime(pl.Date, "%d/%m/%Y")
            )
            logger.info(f'La columna {col} ha sido formateada con el formato de fecha: Día/Mes/Año')
            return df_formateado
        except ValueError: 
            logger.error(f'La conversion a formato de fecha de la columna {col} no se pudo ejecutar')

class FormatoDeFechaTZSinZona: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion.dataframe_vacio()
        self.df_fecha_TZ = df.df
    
    def Año_Mes_Dia_Hora_Minuto_Segundo(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion.nulos(col=col)
        try:
            df_formateado = self.df_fecha_TZ.with_columns(
                pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )
            logger.info(f'La columna {col} ha sido formateada con el formato de fecha: Año-Mes-Día Hora:Minuto:Segundo')
            return df_formateado
        except ValueError: 
            logger.error(f'La conversion a formato de fecha sin zona de la columna {col} no se pudo ejecutar')
    
    def UTC(self, col: str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion.nulos(col=col)
        try:
            df_formateado = self.df_fecha_TZ.with_columns(
                pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
            )
            logger.info(f'La columna {col} ha sido formateada a la zona horaria UTC')
            return df_formateado
        except ValueError: 
            logger.error(f'La conversion a formato de fecha UTC de la columna {col} no se pudo ejecutar')
    
    def desplazamiento(self, col:str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion.nulos(col=col)
        try:
            df_formateado = self.df_fecha_TZ.with_columns(
                pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%z")
            )
            logger.info(f'La columna {col} se ha parafraseado a la zona horaria con desplazamiento')
            return df_formateado
        except ValueError: 
            logger.error(f'La conversion a formato de fecha con desplazamiento de la columna {col} no se pudo ejecutar')

class FormatoDeFechaTZConZona: 
    def __init__(self, df: DFWrapper) -> None: 
        self.validacion = ValidacionesDinamicas(df=df.df)
        self.validacion_estatica = ValidacionesEstaticas()
        self.validacion.dataframe_vacio()
        self.df_fecha_TZ = df.df
    
    def localizacion_sin_TZ(self, col:str, zona: str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion_estatica.zona_valida(zona=zona)
        try:
            df_TZ = self.df_fecha_TZ.with_columns(
                pl.col(col).str.to_datetime().dt.tz_localize(zona)
            )
            logger.info(f'La columna {col} ha sido convertida a la zona horaria de {zona} (sin timezone)')
            return df_TZ
        except ValueError: 
            logger.error(f'La conversion a formato de fecha sin timezone de la columna {col} no se pudo ejecutar')
    
    def localizacion_con_TZ(self, col:str, zona: str) -> Optional[pl.DataFrame]: 
        self.validacion.columna_categorica(col=col)
        self.validacion_estatica.zona_valida(zona=zona)
        try:
            df_TZ = self.df_fecha_TZ.with_columns(
                pl.col(col).str.to_datetime().dt.convert_time_zone(zona)
            )
            logger.info(f'La columna {col} ha sido convertida a la zona horaria de {zona} (con timezone)')
            return df_TZ
        except ValueError: 
            logger.error(f'La conversion a formato de fecha con zona y timezone de la columna {col} no se pudo ejecutar')

class PipelineDeLimpiezaDeTipoDeDatos: 
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = DFWrapper(df=df)
        self.conversion = ConversionDeTipoDatos(df=self.df)
        self.conversion_a_fecha = ConversionDeTipoDatosFechaSinFormato(df=self.df)
    
    def PipelineDeConversionDeDatos(self, col:str, estrategia: EstrategiasDeConversion) -> Optional[pl.DataFrame]: 
        try: 
            match estrategia: 
                case EstrategiasDeConversion.entero: 
                    df_limpio = self.conversion.conversion_a_entero(col=col)
                case EstrategiasDeConversion.flotante: 
                    df_limpio = self.conversion.conversion_a_flotante(col=col)
                case EstrategiasDeConversion.string: 
                    df_limpio = self.conversion.conversion_a_string(col=col)
                case EstrategiasDeConversion.fecha: 
                    df_limpio = self.conversion_a_fecha.conversion_a_fecha(col=col)
                case EstrategiasDeConversion.fecha_tz: 
                    df_limpio = self.conversion_a_fecha.conversion_a_fecha_timezone(col=col)
                case None: 
                    logger.warning(f'No se selecciono ninguna estrategia valida: {estrategia}')
                    return self.df_conversion.df
                case _:
                    logger.error(f'Estrategia no válida: {estrategia}')
                    return self.df.df
            return df_limpio
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer tranformar los tipos de datos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineFormatoDeFecha: 
    def __init__(self, df:pl.DataFrame) -> None:
        self.df = DFWrapper(df=df)
        self.df_con_formato = FormatoDeFecha(df=self.df)
    
    def PipelineDeFormatoDeFecha(self, col:str, estrategia: Optional[EstrategiasDeFormato]=None) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiasDeFormato.Año_Mes_Dia: 
                    df_formateado = self.df_con_formato.Año_Mes_Dia(col=col)
                case EstrategiasDeFormato.Dia_Mes_Año: 
                    df_formateado = self.df_con_formato.Dia_Mes_Año(col=col)
                case None: 
                    logger.warning(f'No se selecciono ninguna estrategia valida: {estrategia}')
                    return self.df.df
                case _:
                    logger.error(f'Estrategia no válida: {estrategia}')
                    return self.df.df
            return df_formateado
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer formatear la fecha: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineDeFechaTZSinZona: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df = DFWrapper(df=df)
        self.df_TZ_sinZona = FormatoDeFechaTZSinZona(df=self.df)
    
    def PipelineFechaTZSinZona(self, col: str, estrategia: EstrategiasDeFormatoTZSinZona) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiasDeFormatoTZSinZona.Año_Mes_Dia_Hora_Minuto_Segundo: 
                    df_formateado = self.df_TZ_sinZona.Año_Mes_Dia_Hora_Minuto_Segundo(col= col)
                case EstrategiasDeFormatoTZSinZona.utc: 
                    df_formateado = self.df_TZ_sinZona.UTC(col=col)
                case EstrategiasDeFormatoTZSinZona.desplazamiento: 
                    df_formateado = self.df_TZ_sinZona.desplazamiento(col=col)
                case None: 
                    logger.warning(f'No se selecciono ninguna estrategia valida: {estrategia}')
                    return self.df.df
                case _:
                    logger.error(f'Estrategia no válida: {estrategia}')
                    return self.df.df
            return df_formateado
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer formatear la fecha sin timezone: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineFechaTZConZona: 
    def __init__(self, df: pl.DataFrame) -> pl.DataFrame: 
        self.df = DFWrapper(df=df)
        self.df_fecha_conTZ = FormatoDeFechaTZConZona(df=self.df)
    
    def PipelineFechaConZona(self, col: str, zona: str, estrategia: EstrategiasDeFormatoTimezone) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case EstrategiasDeFormatoTimezone.localizacion_sin_TZ: 
                    df_formateado = self.df_fecha_conTZ.localizacion_sin_TZ(col=col, zona=zona)
                case EstrategiasDeFormatoTimezone.localizacion_con_TZ: 
                    df_formateado = self.df_fecha_conTZ.localizacion_con_TZ(col=col, zona=zona)
                case None: 
                    logger.warning(f'No se selecciono ninguna estrategia valida: {estrategia}')
                    return self.df.df
                case _:
                    logger.error(f'Estrategia no válida: {estrategia}')
                    return self.df.df
            return df_formateado
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer formatear los datos a fecha con timezone y zona: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')

class PipelineFormatoDeFechaGeneral: 
    def __init__(self, df: pl.DataFrame) -> None: 
        self.df_formato_de_fecha = PipelineFormatoDeFecha(df=df)
        self.df_formato_fecha_sin_tz = PipelineDeFechaTZSinZona(df=df)
        self.df_formato_fecha_con_tz = PipelineFechaTZConZona(df=df)
    
    def PipelineFormatoFechaGeneral(self, col=str, zona: Optional[str]=None, estrategia: Optional[GestionDeFormatoFecha]=None, estrategia_formato: Union[EstrategiasDeFormato, EstrategiasDeFormatoTZSinZona, EstrategiasDeFormatoTimezone]=None) -> Optional[pl.DataFrame]: 
        try:
            match estrategia: 
                case GestionDeFormatoFecha.estrategia_formato: 
                    df_formateado = self.df_formato_de_fecha.PipelineDeFormatoDeFecha(col=col, estrategia=estrategia_formato)
                case GestionDeFormatoFecha.estrategia_formato_TZ_SinZona: 
                    df_formateado = self.df_formato_fecha_sin_tz.PipelineFechaTZSinZona(col=col, estrategia=estrategia_formato)
                case GestionDeFormatoFecha.estrategias_formato_timezone: 
                    df_formateado = self.df_formato_fecha_con_tz.PipelineFechaConZona(col=col, zona=zona, estrategia=estrategia_formato)
                case None: 
                    logger.warning(f'No se selecciono ninguna estrategia valida: {estrategia}')
                    return self.df.df
                case _:
                    logger.error(f'Estrategia no válida: {estrategia}')
                    return self.df.df
            return df_formateado
        except pl.exceptions.ComputeError as e:
            logger.error(f"Error al querer formatear los datos: {e}")
        except Exception as e: 
            logger.error(f'Ocurrio un error durante la ejecucion: {e}')
