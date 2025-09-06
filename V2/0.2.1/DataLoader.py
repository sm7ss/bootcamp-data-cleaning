#Importacion de librerias 
import polars as pl
from pathlib import Path
import logging
from typing import Optional
from dataclasses import dataclass

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class File: 
    archivo: str
    
    @property
    def path(self) -> Path: 
        return Path(self.archivo)

class GetDataFrame:
    @staticmethod
    def load_dataFrame_from_file(archivo: File) -> Optional[pl.DataFrame]:
        file = archivo.path
        extension = file.suffix.lower()
        
        try: 
            match extension: 
                case ".csv": 
                    df = pl.read_csv(file)
                case ".json": 
                    df = pl.read_json(file)
                case ".parquet": 
                    df = pl.read_parquet(file)
                case _: 
                    logger.warning(f'Extension no disponible')
                    return None
        except FileNotFoundError as fne: 
            logger.error(f'Archivo no encontrado: {fne}')
        except Exception as e:
            logger.error(f'Error inesperado: {e}')
        
        logger.info(f'Se convierto el archivo {file} a DataFrame')
        return df
