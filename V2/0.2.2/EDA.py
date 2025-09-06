#Importación de las liberarías necesarias
import polars as pl
from sklearn.ensemble import IsolationForest
import logging
from dataclasses import dataclass
from ColumnAnalyzer import ColumnAnalyzer

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DF: 
    df: pl.DataFrame
    
    @property
    def ColumnasCategoricas(self) -> pl.DataFrame: 
        cat_col = ColumnAnalyzer(df=self.df)
        return cat_col.columna_categorica()
    
    @property
    def ColumnaNumerica(self) -> pl.DataFrame: 
        num_col = ColumnAnalyzer(df=self.df)
        return num_col.columna_numerica()

class GeneralEDA: 
    def __init__(self, df_general: DF) -> None:
        self.df = df_general.df
    
    def columnas_y_filas(self) -> None:
        logger.info(' --- Columnas y Filas ---')
        filas, col = self.df.shape
        logger.info(f'Columnas totales: {col} - Filas totales: {filas}')
    
    def nombre_columnas(self) -> None: 
        logger.info(' --- Nombre de las Columnas ---')
        logger.info(f'\n{self.df.columns}')
    
    def vista_previa_general(self) -> None:
        logger.info(' --- Vista Previa General ---')
        logger.info(f'\n{self.df.head()}')
    
    def nulos(self) -> None: 
        logger.info(' --- Nulos ---')
        logger.info(f'Nulos generales encontrados: \n{self.df.null_count()}')
    
    def duplicados(self) -> None: 
        logger.info(f' --- Duplicados ---')
        duplicados = self.df.group_by(self.df.columns).agg(
            Total_Duplicados= pl.len()
        ).filter(pl.col('Total_Duplicados')>1)
        logger.info(f'Duplicados Generales: \n{duplicados}')

class CategoricalEDA: 
    def __init__(self, df_categoria: DF) -> None:
        self.df_categoria = df_categoria.ColumnasCategoricas
    
    def cardinalidad(self) -> None:
        logger.info(' --- Unicos ---')
        for col in self.df_categoria.columns: 
            logger.info(f'Cardinalidad en la columna {col}:{self.df_categoria[col].n_unique()}')
    
    def unicos(self, max_var: int=5) -> None: 
        logger.info(' --- Cardinalidad ---')
        diccionario_unicos = {}
        for col in self.df_categoria.columns: 
            unicos_val = self.df_categoria[col].unique().sort().head(max_var).to_list()
            diccionario_unicos[col] = unicos_val
        logger.info(f'Valores unicos para los primeros {max_var}:\n{diccionario_unicos}')
    
    def frecuencia(self) -> None: 
        logger.info(' --- Frecuencia ---')
        for col in self.df_categoria.columns: 
            agrupacion = self.df_categoria.group_by(col).agg(
                frecuencia = pl.len()).sort('frecuencia', descending=True)
            logger.info(f'Frecuencia de valores en la columna {col}: \n{agrupacion}')

class NumericalEDA: 
    def __init__(self, df_numerico: DF) -> None:
        self.df_numerico = df_numerico.ColumnaNumerica
    
    def estadisticas_descriptivas(self) -> None:
        logger.info('--- Estadisticas Descriptivas ---')
        logger.info(f'Vista previa general: \n {self.df_numerico.describe()}')
    
    def outlier(self, max_val: int=10, contaminacion: float=0.01) -> None:
        x = self.df_numerico.to_numpy()
        
        try: 
            mod = IsolationForest(contamination=contaminacion, random_state=42)
            pred = mod.fit_predict(x)
            
            df = self.df_numerico.with_columns(
                pl.Series('Anomalia', pred)
            )
            
            outliers_observar = df.filter(pl.col('Anomalia')== -1)
            total_outliers_general = outliers_observar.height
            
            if total_outliers_general > 0: 
                logger.info('--- Outliers General ---')
                logger.info(f'Total de outliers detectados: {total_outliers_general}')
                logger.info(f'Muestro de Outliers: {outliers_observar.head(max_val)}')
            else: 
                logger.info('No se detectaron Outliers generales')
            
            logger.info(' --- Outliers en Columnas ---')
            for col in self.df_numerico.columns: 
                mod_col = IsolationForest(contamination=contaminacion, random_state=42)
                x_col = self.df_numerico[col].to_numpy().reshape(-1, 1)
                pred_col = mod_col.fit_predict(x_col)
                df_num = self.df_numerico.with_columns(
                    pl.Series('Anomalia_Col', pred_col)
                ).filter(pl.col('Anomalia_Col') == -1)
                if not df_num.is_empty(): 
                    logger.info(f'Total de outliers detectados para la columna {col}: {df_num.height}')
                    logger.info(f'Muestreo de Outliers para la columna {col}:\n{df_num.head(max_val)}')
                else: 
                    logger.info(f'No se detectaron outliers en la columna {col}')
        except ValueError as ve: 
            logger.error(f'Error al detectar outliers: {ve}')
        except Exception as e: 
            logger.error(f'Aparecio una error: {e}') 

class PipelineEDA: 
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = DF(df=df)
        self.general = GeneralEDA(df_general=self.df)
        self.categoria = CategoricalEDA(df_categoria=self.df)
        self.numerico = NumericalEDA(df_numerico=self.df)
    
    def pipeline_general(self) -> None: 
        self.general.columnas_y_filas()
        self.general.nombre_columnas()
        self.general.vista_previa_general()
        self.general.nulos()
        self.general.duplicados()
    
    def pipeline_categorico(self) -> None: 
        self.categoria.cardinalidad()
        self.categoria.unicos(max_var=5)
        self.categoria.frecuencia()
    
    def pipeline_numerico(self) -> None: 
        self.numerico.estadisticas_descriptivas()
        self.numerico.outlier(max_val=10, contaminacion=0.01)
    
    def pipeline_completo(self) -> None: 
        self.pipeline_general()
        self.pipeline_categorico()
        self.pipeline_numerico()
