#Importación de las librerías necesarias
import polars as pl
import plotly.express as px
from plotly.graph_objects import Figure
from typing import Optional
import logging
from Columnas_por_tipo import num_cat_col

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

#Clase general para graficación
class Graficacion: 
    #Inicialización
    def __init__(self, 
                df: pl.DataFrame, 
                muestra: float=1.0) -> None:
        if df.is_empty(): 
            return None
        
        if not (0 < muestra <= 1.0):
            logger.error(f'La muestra debe ser mayor que 0 y menor o igual a 1.0')
            return None
        
        self.df = self.muestra_df(df, muestra)
        self.columna_categorica = num_cat_col(nombre_archivo=None, df=self.df)[1]
        self.columna_numerica = num_cat_col(nombre_archivo=None, df=self.df)[0]
    
    #Muestreo de dataframe en caso de ser muy grande
    def muestra_df(self, df: pl.DataFrame, muestra: float) -> pl.DataFrame:
        return df.sample(fraction=muestra, shuffle=True) if muestra <= 1.0 else self.df
    
    def grafico_lineas(self, x: str, y: str, 
                        titulo: str, x_title : str, y_title:str, 
                        columna_categorica: Optional[str]=None) -> Optional[Figure]: 
        
        if x not in self.df.columns:
            logger.error(f'La columna {x} no está en el DataFrame')
            return None
        
        if y not in self.df.columns:
            logger.error(f'La columna {y} no está en el DataFrame')
            return None
        
        if columna_categorica is not None: 
            if columna_categorica not in self.df.columns: 
                logger.error(f'La columna {columna_categorica} no está en el DataFrame')
                return None
            
            if columna_categorica not in self.columna_categorica: 
                logger.error(f'La columna {columna_categorica} no es una columna categorica')
                return None
            
            figura = px.line(
                self.df, 
                x= x, 
                y= y, 
                title= titulo,
                color=columna_categorica
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
        else: 
            figura = px.line(
                self.df, 
                x= x, 
                y= y, 
                title= titulo,
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
    
    def grafico_barras(self, x: str, y: str,
                        titulo:str, x_title:str, y_title: str, 
                        columna_categorica: Optional[str]=None) -> Optional[Figure]: 
        if x not in self.df.columns: 
            logger.error(f'La columna {x} no está en el DataFrame')
            return None
        
        if y not in self.df.columns:
            logger.error(f'La columna {y} no está en el DataFrame')
            return None
        
        if columna_categorica is not None: 
            if columna_categorica not in self.df.columns: 
                logger.error(f'La columna {columna_categorica} no está en el DataFrame')
                return None
            
            if columna_categorica not in self.columna_categorica: 
                logger.error(f'La columna {columna_categorica} no es una columna categorica')
                return None
            
            figura = px.bar(
                self.df, 
                x= x, 
                y= y, 
                title = titulo, 
                color=columna_categorica
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
        else: 
            figura = px.bar(
                self.df, 
                x= x, 
                y= y, 
                title = titulo, 
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
    
    def grafica_dispersion(self, x: str, y: str, 
                            size: str, titulo: str, 
                            x_title: str, y_title: str, 
                            columna_categorica: Optional[str]=None) -> Optional[Figure]: 
        if x not in self.columna_numerica:
            logger.error(f'La columna {x} no está en las columnas numericas')
            return None
        
        if y not in self.columna_numerica:
            logger.error(f'La columna {y} no está en las columnas numericas')
            return None
        
        if size not in self.columna_numerica:
            logger.error(f'La columna {size} no está en las columnas numericas')
            return None
        
        if columna_categorica is not None: 
            if columna_categorica not in self.df.columns: 
                logger.error(f'La columna {columna_categorica} no está en el DataFrame')
                return None
            
            if columna_categorica not in self.columna_categorica: 
                logger.error(f'La columna {columna_categorica} no es una columna categorica')
                return None
            
            figura = px.scatter(
                self.df, 
                x= x, 
                y= y, 
                size= size, 
                title= titulo,
                color=columna_categorica, 
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
        else: 
            figura = px.scatter(
                self.df, 
                x= x, 
                y= y, 
                size= size, 
                title= titulo,
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
    
    def boxplot(self, x: str, y: str, 
                titulo: str, x_title:str, y_title:str, 
                columna_categorica:Optional[str]=None) -> Optional[Figure]: 
        if x not in self.df.columns: 
            logger.error(f'La columna {x} no está en el DataFrame')
            return None
        
        if y not in self.df.columns:
            logger.error(f'La columna {y} no está en el DataFrame')
            return None
        
        if columna_categorica is not None: 
            if columna_categorica not in self.df.columns: 
                logger.error(f'La columna {columna_categorica} no está en el DataFrame')
                return None
            
            if columna_categorica not in self.columna_categorica: 
                logger.error(f'La columna {columna_categorica} no es una columna categorica')
                return None
            
            figura = px.box(
                self.df, 
                x= x, 
                y= y, 
                title= titulo,
                color= columna_categorica
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
        else: 
            figura = px.box(
                self.df, 
                x=x, 
                y=y, 
                title=titulo
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
    
    def grafica_histograma(self, x:str, 
                titulo: str, x_title: str, y_title: str, 
                nbins: int=50, columna_categorica: Optional[str] = None) -> Optional[Figure]:
        if x not in self.df.columns: 
            logger.error(f'La columna {x} no está en el DataFrame')
            return None
        
        if columna_categorica is not None: 
            if columna_categorica not in self.df.columns: 
                logger.error(f'La columna {columna_categorica} no está en el DataFrame')
                return None
            
            if columna_categorica not in self.columna_categorica: 
                logger.error(f'La columna {columna_categorica} no es una columna categorica')
                return None
            
            figura = px.histogram(
                self.df, 
                x=x, 
                title=titulo, 
                nbins=nbins, 
                color=columna_categorica
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura
        else: 
            figura = px.histogram(
                self.df, 
                x=x, 
                title=titulo, 
                nbins=nbins
            )
            figura.update_layout(xaxis_title=x_title, yaxis_title=y_title)
            return figura