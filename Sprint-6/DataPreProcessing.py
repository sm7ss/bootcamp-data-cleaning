#Importamos las librerías para el proyecto 
import polars as pl
import logging
from pydantic import BaseModel
from typing import Union, List, Tuple
from prefect import task, flow

#Configuracion del logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s-(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

class RenameColumn: 
    def __init__(self, frame: Union[pl.DataFrame, pl.LazyFrame], model: BaseModel):
        self.frame = frame
        self.rename_col = model.Cleaning_Rules.column_rename
    
    def rename_columns(self) -> Union[pl.DataFrame, pl.LazyFrame]: 
        return self.frame.rename(self.rename_col)

class TypeDtype:
    @staticmethod
    def int_dtype(col_int: str) -> pl.Expr: 
        return pl.col(col_int).cast(pl.Int32)
    
    @staticmethod
    def float_dtype(col_float: str) -> pl.Expr: 
        return pl.col(col_float).cast(pl.Float32)
    
    @staticmethod
    def str_dtype(col_str: str) -> pl.Expr: 
        return pl.col(col_str).cast(pl.Utf8)

class DtypeOverride: 
    def __init__(self, frame: Union[pl.DataFrame, pl.LazyFrame], model: BaseModel):
        self.frame = frame
        self.dtype_dict = model.Cleaning_Rules.dtype_override
        self.dtype = TypeDtype()
    
    def type_dtype(self): 
        list_dtype_expr = []
        
        for col, dtype in self.dtype_dict.items(): 
            if dtype == 'Int32': 
                list_dtype_expr.append(self.dtype.int_dtype(col_int=col))
            elif dtype == 'Float32': 
                list_dtype_expr.append(self.dtype.float_dtype(col_float=col))
            else: 
                list_dtype_expr.append(self.dtype.str_dtype(col_str=col))
        
        return list_dtype_expr
    
    def dtype_override(self) -> Union[pl.DataFrame, pl.LazyFrame]: 
        list_expr = self.type_dtype()
        return self.frame.with_columns(list_expr)

class CatNullHandler: 
    def __init__(self, model: BaseModel):
        self.null_cat = model.Cleaning_Rules.null_values.null_imput_cat_operation
    
    def input_cat_data(self, cat_col: str) -> pl.Expr: 
        return pl.col(cat_col).fill_null(self.null_cat)
    
    def input_mode(self, cat_col: str) -> pl.Expr: 
        return pl.col(cat_col).mode().first()
    
    def input_cat_col(self, cat_col: str) -> pl.Expr: 
        if self.null_cat == 'mode': 
            return self.input_mode(cat_col=cat_col)
        else: 
            return self.input_cat_data(cat_col=cat_col)

class NumNullHandler: 
    def __init__(self, model: BaseModel):
        self.null_num = model.Cleaning_Rules.null_values.null_imput_num_operation
    
    def operation_fill(self, col_num: str) -> pl.Expr:
        return (getattr(pl.col(col_num), self.null_num)())
    
    def input_num_data(self, col_num: str) -> pl.Expr: 
        return pl.col(col_num).fill_null(self.operation_fill(col_num=col_num))

class InputData: 
    def __init__(self, frame: Union[pl.LazyFrame, pl.DataFrame], model: BaseModel):
        self.frame = frame
        
        self.num_frame = self.frame.select(pl.selectors.numeric()).columns
        
        self.input_cat = CatNullHandler(model=model)
        self.input_num = NumNullHandler(model=model)
    
    def input_data_op(self, col: str) -> pl.Expr: 
        if col in self.num_frame: 
            operacion = self.input_num.input_num_data(col_num=col)
        else: 
            operacion = self.input_cat.input_cat_col(cat_col=col)
        
        return operacion

class DeleteData: 
    def __init__(self, frame: Union[pl.LazyFrame, pl.DataFrame]):
        self.frame = frame
    
    def delete_column(self, list_col: List[str]) -> pl.Expr: 
        return self.frame.drop(list_col)
    
    def delete_row(self, frame_sucio: Union[pl.DataFrame, pl.LazyFrame]) -> pl.Expr: 
        columna = frame_sucio.get_column('index').to_list()
        return ~pl.col('index').is_in(columna)

class TupleExprNullHanlder: 
    def __init__(self, frame: Union[pl.DataFrame, pl.LazyFrame], model: BaseModel):
        self.frame = frame
        self.model = model
        
        self.delete_data = DeleteData(frame=self.frame)
    
    def row_handler_null(self, filtro_null: Union[pl.DataFrame, pl.LazyFrame], umbral_filas_nulas: float=0.4) -> pl.Expr: 
        umbral_filas = (self.frame.width*umbral_filas_nulas)
        
        filtro_filas_nulas = filtro_null.with_columns(
        suma_nulos=pl.sum_horizontal(pl.col('*').is_null().cast(pl.Int32))
            ).filter(pl.col('suma_nulos') > umbral_filas)
        
        porcentaje_nulos_fila = (filtro_filas_nulas.height/filtro_null.height)*100
        
        if porcentaje_nulos_fila >= 40: 
            return self.delete_data.delete_row(frame_sucio=filtro_filas_nulas)
    
    def col_handler(self, umbral_filas: float=0.4) -> Tuple[List[pl.Expr]]: 
        tamaño_frame = self.frame.height
        frame_index = self.frame.with_row_index()
        
        null_row_col_handler = []
        columnas_a_eliminar = []
        columnas_a_analizar = []
        
        for col in self.frame.columns: 
            filtro_nulos = frame_index.filter(pl.col(col).is_null())
            if filtro_nulos.height > 0: 
                porcentaje_nulos_columna = (filtro_nulos.height/tamaño_frame)*100
                
                if porcentaje_nulos_columna < 65: 
                    nulos = self.row_handler_null(filtro_null=filtro_nulos, umbral_filas_nulas=umbral_filas)
                    if nulos is None: 
                        columnas_a_analizar.append(col)
                    else: 
                        null_row_col_handler.append(nulos)
                else: 
                    columnas_a_eliminar.append(col)
        
        return null_row_col_handler, columnas_a_eliminar, columnas_a_analizar

class AnalysisNullData: 
    def __init__(self, 
        frame: Union[pl.DataFrame, pl.LazyFrame], 
        model: BaseModel,
        columnas_representativas: List[str], 
        columnas_target: List[str]):
        
        self.frame = frame
        self.col_rep = columnas_representativas
        self.col_target = columnas_target
        
        self.input = InputData(frame=self.frame, model=model)
    
    def expr_non_null_count(self) -> List[pl.Expr]: 
        lista_expr = []
        
        for col in self.col_target: 
            non_null_count = pl.col(col).drop_nulls().count().over(self.col_rep)
            total = pl.len().over(self.col_rep)
            porcentaje_no_nulos = ((non_null_count/total)*100).alias(f'porcentaje_no_nulos_{col}')
            lista_expr.append(porcentaje_no_nulos)
        
        return lista_expr
    
    def etiquetar_frame(self, col: str, min_proportion: int=45) -> Union[pl.DataFrame, pl.LazyFrame]: 
        frame_porcentajes = self.frame.with_columns(
            self.expr_non_null_count()
            )
        
        frame_etiquetado = frame_porcentajes.with_columns(
            pl.when(pl.col(f'porcentaje_no_nulos_{col}') < min_proportion)
            .then(pl.lit('eliminar'))
            .otherwise(pl.lit('imputar'))
            .alias(f'estado_{col}')
        )
        
        return frame_etiquetado
    
    def filtrado_frame_imput(self, col: str, min_proportion: int=45) -> Union[pl.DataFrame, pl.LazyFrame]: 
        frame = self.etiquetar_frame(col=col, min_proportion=min_proportion)
        
        frame_imputado = frame.with_columns(
            pl.when(pl.col(f'estado_{col}')=='imputar')
            .then(self.input.input_data_op(col=col).over(self.col_rep))
            .otherwise(pl.col(col))
        ).drop(f'estado_{col}')
        
        return frame_imputado
    
    def analysis_null_data(self, min_proportion: int=45) -> Union[pl.DataFrame, pl.LazyFrame]: 
        for col in self.col_target: 
            self.frame = self.filtrado_frame_imput(col=col, min_proportion=min_proportion)
        
        return self.frame.drop([f'porcentaje_no_nulos_{col}' for col in self.col_target])

class NullHandler: 
    def __init__(self, 
        frame: Union[pl.DataFrame, pl.LazyFrame], 
        model: BaseModel, 
        umbral: float=0.4):
        self.frame = frame
        self.model = model
        
        self.null_row, self.delete_col, self.analyse_null = TupleExprNullHanlder(frame=self.frame, model=self.model).col_handler(umbral_filas=umbral)
    
    def delete_data_row(self, frame: Union[pl.LazyFrame, pl.DataFrame]) -> Union[pl.LazyFrame, pl.DataFrame]: 
        return frame.filter(
            self.null_row
        ).drop('index')
    
    def analysis_data(self, 
        frame: Union[pl.LazyFrame, pl.DataFrame], 
        columnas_representativas: Union[List[str], str],
        min_proportion: int=45) -> Union[pl.LazyFrame, pl.DataFrame]: 
        
        frame_nuevo = AnalysisNullData(
            frame= frame, 
            model= self.model, 
            columnas_representativas= columnas_representativas, 
            columnas_target=self.analyse_null)
        return frame_nuevo.analysis_null_data(min_proportion=min_proportion)
    
    def pipeline_null_handler(self, columnas_representativas: Union[List[str], str], min_proportion: int=45) -> Union[pl.LazyFrame, pl.DataFrame]: 
        frame = self.frame
        
        if self.null_row: 
            frame_con_indice = frame.with_row_index()
            frame = self.delete_data_row(frame=frame_con_indice)
        if self.analyse_null: 
            frame = self.analysis_data(
                frame=frame, 
                columnas_representativas=columnas_representativas, 
                min_proportion=min_proportion
            )
        if self.delete_col: 
            delete_data = DeleteData(frame=frame)
            frame= delete_data.delete_column(list_col=self.delete_col)
        
        tipo_frame = DtypeOverride(frame=frame, model=self.model).dtype_override()
        return tipo_frame

class DataCleaning: 
    def __init__(self, frame: Union[pl.DataFrame, pl.LazyFrame], model: BaseModel):
        self.frame = frame
        self.model = model
    
    @task
    def rename_columns(self, frame: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]: 
        rename = RenameColumn(frame=frame, model=self.model)
        return rename.rename_columns()
    
    @task
    def null_handler(self, 
        frame: Union[pl.DataFrame, pl.LazyFrame], 
        columnas_representativas: List[str],
        umbral: float=0.4,
        min_proportion: int=45) -> Union[pl.DataFrame, pl.LazyFrame]: 
        
        null_data= NullHandler(frame=frame, model=self.model, umbral=umbral)
        return null_data.pipeline_null_handler(
            columnas_representativas=columnas_representativas, 
            min_proportion=min_proportion
        )
    
    @flow(name='Pipeline Limpieza De Datos Básico')
    def pipeline_data_cleaning(self, 
        columas_representativas: List[str],
        umbral: float=0.4,
        min_proportion: int=45) -> Union[pl.DataFrame, pl.LazyFrame]: 
        
        frame_rename = self.rename_columns(frame=self.frame)
        frame_null_hanlder = self.null_handler(
            frame=frame_rename, 
            columnas_representativas=columas_representativas, 
            umbral=umbral, 
            min_proportion=min_proportion
        )
        return frame_null_hanlder

# Agregar un filtro para frame lazy en caso de querer colectar los datos 
"""
    Posiblemente quitar el frame externo y meterlo en el mismo pipeline y dejar unicamente un parametro 
    parapasar el yaml o toml y obtener el model y frame desde dentro 
"""
# Agregar un posible guardado de parquet de los datos limpios 