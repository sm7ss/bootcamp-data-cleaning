#Importación de las liberarías necesarias
import polars as pl
from pathlib import Path
import logging

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Clase para leer archivos
class LeerArchivos: 
    #Función para inicializar la clase
    def __init__(self, lista_archivos: list): 
        if not lista_archivos: 
            logger.warning('Lista de archivos vacía')
            raise ValueError('Lista vacía')
        
        self.lista_archivos_csv = []
        self.lista_archivos_json = []
        self.lista_archivos_excel = []
        self.lista_archivos_parquet = []
        
        for archivo in lista_archivos: 
            file = Path(archivo)
            extension = file.suffix.lower()
            
            if extension == '.csv': 
                self.lista_archivos_csv.append(file)
            elif extension == '.json': 
                self.lista_archivos_json.append(file)
            elif extension == '.parquet':
                self.lista_archivos_parquet.append(file)
            elif extension in ['.xls', '.xlsx']: 
                self.lista_archivos_excel.append(file)
            else: 
                logger.warning(f'Archivo {file} no tiene las extenciones disponible: .csv, .json, .xls y .xlsx')
    
    #Método para leer los archivos de entrada y tener una vista previa de los datos si existe True
    def lectura_tipo_de_archivos(self, preview: bool = False) -> dict: 
        '''
        Función para leer los archivos de entrada y tener una vista previa de los datos si existe True
        
        Params: 
            preview : parametro booleano si queremos o no tener una vista previa de los datos
        
        Return: 
            dict : diccionario con los archivos de entrada por tipo
        '''
        
        #Método para obtener una vista previa de los datos de los archivos de entrada
        def vista_de_datos(file: str, tipo: str, formatos_a_leer: list) -> None: 
            '''
            Función para obtener una vista previa de los datos de los archivos de entrada
            
            Params: 
            file : nombre del archivo
            tipo : tipo de archivo
            formatos_a_leer : lista de formatos de archivos soportados
            '''
            try: 
                if tipo == 'csv': 
                    df = pl.read_csv(file)
                    logger.info(f'Preview {file}: \n{df.head()}')
                elif tipo == 'json': 
                    df = pl.read_json(file)
                    logger.info(f'Preview {file}: \n{df.head()}')
                elif tipo == 'parquet':
                    df = pl.read_parquet(file)
                    logger.info(f'Preview {file}: \n{df.head()}')
                elif tipo == 'excel': 
                    df = pl.read_excel(file)
                    logger.info(f'Preview {file}: \n{df.head()}')
                else: 
                    logger.error(f'Formato {file} no soportado. Usar: {formatos_a_leer}')
            except FileNotFoundError: 
                logger.error(f'{file} no encontrado')
            except Exception as e: 
                logger.error(f'Ocurrio un error tipo {e} para {file}')
            
        formatos_a_leer = {
        'csv' : self.lista_archivos_csv, 
        'json' :  self.lista_archivos_json, 
        'parquet' : self.lista_archivos_parquet,
        'excel' : self.lista_archivos_excel
        }
        resultados = {}
            
        for tipo, archivos in formatos_a_leer.items(): 
            if not archivos: 
                logger.warning(f'No se encontraron archivos tipo .{tipo}')
                continue
            resultados[tipo] = {}
            
            for file in archivos: 
                nombre_archivo = Path(file).stem
                resultados[tipo][nombre_archivo] = file
                if preview: 
                    vista_de_datos(file= file, tipo=tipo, formatos_a_leer=formatos_a_leer.keys())
        return resultados


#Clase para obtener los dataframes de los archivos de entrada
class ObtencionDeDataFrames: 
    def __init__(self, diccionario_archivos: dict):
        if not diccionario_archivos: 
            logger.warning('Diccionario vacío')
            raise ValueError('Diccionario vacío')
        self.diccionario_de_archivos = diccionario_archivos
    
    #Método para obtener los dataframes de los archivos de entrada
    def obtencion_de_dataframes(self) -> dict: 
        '''
        Función para obtener los dataframes de los archivos de entrada
        '''
        if not self.diccionario_de_archivos: 
            logger.warning('El diccionario está vacío')
            return None
        
        diccionario_de_df = {}
        
        for tipo, archivos in self.diccionario_de_archivos.items():
            diccionario_de_df[tipo] = {}  
            
            for nombre_archivo, archivo in archivos.items():            
                file = Path(archivo)
                extension = file.suffix.lower()
                
                if extension == '.csv': 
                    logger.info(f'DataFrame para archivos tipo .{tipo} para {file}')
                    df = pl.read_csv(file)
                    diccionario_de_df[tipo][file] = df
                elif extension == '.json': 
                    logger.info(f'DataFrame para archivos tipo .{tipo} para {file}')
                    df = pl.read_json(file)
                    diccionario_de_df[tipo][file] = df
                elif extension == '.parquet': 
                    logger.info(f'DataFrame para archivos tipo .{tipo} para {file}')
                    df = pl.read_parquet(file)
                    diccionario_de_df[tipo][file] = df
                elif extension in ['.xls', '.xlsx']: 
                    logger.info(f'DataFrame para archivos tipo .{tipo} para {file}')
                    df = pl.read_excel(file)
                    diccionario_de_df[tipo][file] = df
                else: 
                    logger.warning(f'{file} no tiene nuestras extensiones disponibles')
                    continue
        return diccionario_de_df

