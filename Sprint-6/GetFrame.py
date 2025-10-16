#Importamos las librerías necesarias
import polars as pl 
import logging 
from typing import Union
import psutil
from pydantic import BaseModel

#Config del logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Config del logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

class FormatFrame: 
    def __init__(self, model: BaseModel):
        self.archivo = model.Paths.input_file
        self.null_data_handler = model.Cleaning_Rules.null_values.null_val_csv
    
    def formato_eager(self) -> pl.DataFrame: 
        if self.archivo.suffix == '.csv': 
            frame = pl.read_csv(self.archivo, null_values=self.null_data_handler)
        elif self.archivo.suffix == '.parquet': 
            frame = pl.read_parquet(self.archivo)
        return frame
    
    def formato_lazy(self) -> pl.LazyFrame: 
        if self.archivo.suffix == '.csv': 
            frame = pl.scan_csv(self.archivo, null_values=self.null_data_handler)
        elif self.archivo.suffix == '.parquet': 
            frame = pl.scan_parquet(self.archivo)
        return frame

class FrameCollector: 
    def __init__(self, model: BaseModel):
        self.archivo = model.Paths.input_file
        self.formato = FormatFrame(model=model)
    
    def decision_frame(self) -> str: 
        tamaño_archivo = self.archivo.stat().st_size
        memoria_disponible = psutil.virtual_memory().available
        
        ratio_eager = memoria_disponible * 0.1 
        ratio_lazy = memoria_disponible * 0.75 
        
        if tamaño_archivo < ratio_eager: 
            return 'eager'
        elif tamaño_archivo < ratio_lazy: 
            return 'lazy'
        else: 
            raise ValueError(f'Archivo {self.archivo.name} demasiado grande, no se puede procesar')
    
    
    def get_frame(self) -> Union[pl.DataFrame, pl.LazyFrame]: 
        decision = self.decision_frame()
        
        if decision == 'eager': 
            frame = self.formato.formato_eager()
            logger.info(f'Se obtuvo el Frame en formato eager correctamente para el archvivo {self.archivo.name}')
        else: 
            frame = self.formato.formato_lazy()
            logger.info(f'Se obtuvo el Frame en formato lazy correctamente para el archvivo {self.archivo.name}')
        
        return frame