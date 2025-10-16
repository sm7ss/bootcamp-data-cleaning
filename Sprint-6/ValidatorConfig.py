from pathlib import Path
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Union
from Strategies import StrategyFE, StrategyTest, StrategyDataType, StrategyNullCatImputer, StrategyNullNumImput

class PathConfigValidator(BaseModel): 
    input_file: str
    output_file: str 
    
    @field_validator('input_file')
    def archivo_existente(cls, v): 
        path = Path(v)
        if not path.exists(): 
            raise FileNotFoundError(f'El archivo {path.name} no existe')
        return path
    
    @field_validator('output_file')
    def archivo_salida_existente(cls, v): 
        path = Path(v)
        if path.exists(): 
            raise FileExistsError(f'El archivo {path.name} existe y no se puede sobreescribir')
        if path.suffix not in ['.csv', '.parquet']: 
            raise ValueError(f'El archivo {path.name} debe ser un archivo csv o parquet')
        return v

class NullHanlderValidator(BaseModel): 
    null_val_csv: List[str] = Field(min_length=1)
    null_imput_num_operation: StrategyNullNumImput
    null_imput_cat_operation: Union[StrategyNullCatImputer, str]

class CleaningRulesValidator(BaseModel): 
    column_rename: Optional[Dict[str, str]] = Field(min_length=1)
    dtype_override: Optional[Dict[str, StrategyDataType]] = Field(min_length=1)
    null_values: Optional[NullHanlderValidator]

class FeatureEngineerValidator(BaseModel): 
    sales_column: Optional[List[str]] = Field(default=None, min_length=1)
    operation: Optional[StrategyFE]
    
    @model_validator(mode='after')
    def operacion_valida(self): 
        operacion = self.operation
        list_col = self.sales_column
        
        if list_col is not None: 
            if operacion is None: 
                raise ValueError('Debe de tener una operacion para transformar los datos')
        
        if operacion is not None: 
            if list_col is None: 
                raise ValueError('La operacion a ejecutar en la lista debe de tener una lista')
        
        return self

class AnalysisParams(BaseModel): 
    relevant_year_start: int
    top_platforms_count: int
    regions: List[str] = Field(min_length=1)
    top_regional_count: int
    popular_platform_corr: str

class TestValidator(BaseModel): 
    name: str
    groups: List[str] = Field(min_length=1)
    metric: str

class HypotesisTesting(BaseModel): 
    test_1: TestValidator
    test_2: TestValidator
    test_criteria: StrategyTest
    alpha: float = Field(le=1)

class ValidatorConfig(BaseModel): 
    Paths: PathConfigValidator
    Cleaning_Rules: CleaningRulesValidator
    Feature_Engineer: FeatureEngineerValidator
    Analysis_Params: AnalysisParams
    Hypotesis_Testing: HypotesisTesting
    
    @model_validator(mode='after')
    def columnas_existentes(self): 
        archivo = self.Paths.input_file
        if archivo.suffix == '.csv': 
            schema = pl.scan_csv(archivo).collect_schema()
        else: 
            schema = pl.scan_parquet(archivo).collect_schema()
        
        column_rename = self.Cleaning_Rules.column_rename.keys()
        for col in column_rename: 
            if col not in schema: 
                raise ValueError(f'La columna {col} no se encuentra en el Frame')
        
        dtype = self.Cleaning_Rules.dtype_override.keys()
        col_rename_value = self.Cleaning_Rules.column_rename.values()
        for col in dtype: 
            if col not in col_rename_value: 
                raise ValueError(f'La columna {col} no se ecuentra en el Frame, verificar la renombración de las columnas')
        
        list_sales = self.Feature_Engineer.sales_column
        for col in list_sales: 
            if col not in schema and col not in col_rename_value: 
                raise ValueError(f'La columna {col} no se encuentra en el Frame')
        
        regiones = self.Analysis_Params.regions
        for reg in regiones: 
            if reg not in schema and reg not in col_rename_value: 
                raise ValueError(f'La columna {reg} no se encuentra en el Frame')
        
        metrica_1 = self.Hypotesis_Testing.test_1.metric
        metrica_2 = self.Hypotesis_Testing.test_2.metric
        if metrica_1 not in schema and metrica_1 not in col_rename_value: 
            raise ValueError(f'La columna {metrica_1} no se encuentra en el Frame')
        if metrica_2 not in schema and metrica_2 not in col_rename_value: 
            raise ValueError(f'La columna {metrica_2} no se encuentra en el Frame')
        if metrica_1 != metrica_2: 
            raise ValueError(f'Las metricas para la hipotesis deben ser identicas no diferentes')
        return self
    
    @model_validator(mode='after')
    def valor_existente(self): 
        archivo = self.Paths.input_file
        frame = pl.scan_csv(archivo, null_values=['tbd', 'TBD', 'N/A', 'nan', ''])
        
        sample = frame.slice(0, 1000).collect(engine='streaming')
        pop_platform = self.Analysis_Params.popular_platform_corr
        
        filtro_pop_platform = sample.filter(pl.col('Platform') == pop_platform)
        if filtro_pop_platform.height < 1: 
            raise ValueError(f'La categoría {pop_platform} no existe en el Frame')
        
        grupo_test_1 = self.Hypotesis_Testing.test_1.groups
        grupo_test_2 = self.Hypotesis_Testing.test_2.groups
        
        for cat in grupo_test_1: 
            filtro = sample.filter(pl.col('Platform') == cat)
            if filtro.height < 1: 
                raise ValueError(f'La categoria {cat} no existe en el Frame')
        
        for cat in grupo_test_2: 
            filtro = sample.filter(pl.col('Genre') == cat)
            if filtro.height < 1: 
                raise ValueError(f'La categoria {cat} no existe en el Frame')
        return self

