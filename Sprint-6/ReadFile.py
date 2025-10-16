#Importacion de librerías importantes
import yaml
import tomli
import logging 
from pathlib import Path
from pydantic import BaseModel
from ValidatorConfig import ValidatorConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

class ReadConfig: 
    def __init__(self, archivo: str):
        self.archivo = Path(archivo)
        
        if self.archivo.suffix not in ['.yaml', '.yml', '.toml']: 
            raise ValueError(f'El archivo {self.archivo.name} deber ser un archivo yaml o toml')
    
    def read_yaml(self) -> BaseModel: 
        try: 
            with open(self.archivo, 'r') as f: 
                file = yaml.safe_load(f)
            logger.info(f'Se leyó correctamente el archivo {self.archivo.name}')
            validador = ValidatorConfig(**file)
            logger.info(f'Se valido correctamente el archivo {self.archivo.name}')
            return validador
        except yaml.YAMLError: 
            logger.error(f'El archivo {self.archivo.name} está corrputo')
            raise
        except Exception as e: 
            logger.error(f'Ocurrio un erro al querrer leer el archivo {self.archivo.name}: {str(e)}')
            raise
    
    def read_toml(self) -> BaseModel: 
        try: 
            with open(self.archivo, 'rb') as f: 
                file = tomli.load(f)
            logger.info(f'El archivo {self.archivo.name} se leyó correctamente')
            validador = ValidatorConfig(**file)
            logger.info(f'Se valido el archivo {self.archivo.name} correctamente')
            return validador
        except tomli.TOMLDecodeError: 
            logger.error(f'El archivo {self.archivo.name} está corrupto') 
            raise 
        except Exception as e: 
            logger.error(f'Hubo un problema al querer leer el archivo {self.archivo.name}: {str(e)}')
            raise
    
    def read_config(self) -> BaseModel: 
        if self.archivo.suffix in ['.yaml', '.yml']: 
            return self.read_yaml()
        else: 
            return self.read_toml()


