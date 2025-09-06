#Importación de librerías extra necesariass
import polars as pl
import logging
from typing import Union, Optional
from plotly.graph_objects import Figure
from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, shapiro
#EDA
from EDA import estadisticas_descriptivas
#Grficas
from Graficacion import Graficacion

#Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

class Explorar_y_Formacion_Hipotesis: 
    def __init__(self, nombre_archivo:str, 
                df: pl.DataFrame) -> None:
        if df.is_empty(): 
            logger.error(f'El DataFrame del archivo {nombre_archivo} está vacío')
            return None
        
        self.nombre_archivo = nombre_archivo
        self.df = df
        self.objeto = Graficacion(df=self.df)
        self.alternativa = ['less', 'greater', 'two-sided']
    
    def exploracion_estadistica(self, columna_categorica: str, columna_numerica:str) -> None:
        """
        Realiza una exploración estadística de un DataFrame
        
        Args:
            columna_categorica (str): Nombre de la columna categórica para agrupar los datos
            columna_numerica (str): Nombre de la columna numérica para calcular las estadísticas descriptivas
        
        Returns:
            None
        """
        columnas_numericas = self.df.select(pl.selectors.numeric()).columns
        estadisticas_descriptivas(nombre_archivo=self.nombre_archivo, df=self.df[columnas_numericas])
        
        agrupacion = self.df.group_by(columna_categorica).agg(
        promedio = pl.col(columna_numerica).mean(), 
        estandar = pl.col(columna_numerica).std(), 
        conteo = pl.col(columna_numerica).count(), 
        minimo = pl.col(columna_numerica).min(),
        maximo = pl.col(columna_numerica).max()
        ).sort(by=columna_categorica)
        logger.info(f'{columna_numerica} agrupada por {columna_categorica}: \n{agrupacion}')
    
    def visualizaciones(self, x: str, y: str, titulo: str, 
                        size: Union[str, bool],
                        nbins: Union[int, bool],
                        tipo_graficacion: str='histograma') -> Optional[Figure]:
        """
        Realiza una visualización de un DataFrame
        
        Args:
            x (str): Nombre de la columna para el eje x
            y (str): Nombre de la columna para el eje y
            titulo (str): Título del gráfico
            size (Union[str, bool]): Nombre de la columna para el tamaño de los puntos en el caso de un gráfico de dispersión
            nbins (Union[int, bool]): Número de bins para el histograma
            tipo_graficacion (str, optional): Tipo de gráfico a visualizar. Defaults to 'histograma'
        
        Returns:
            Optional[Figure]: Objeto de figura Plotly con el gráfico
        """
        graficacion_disponible = ['grafico de lineas', 'grafico de barras', 'grafico de dispersion', 'boxplot', 'histograma']
        if tipo_graficacion not in graficacion_disponible: 
            logger.error(f'El tipo de graficación {tipo_graficacion} no está disponible')
            return None
        
        try: 
            if isinstance(size, str) and tipo_graficacion == 'grafico de dispersion': 
                dispersion = self.objeto.grafica_dispersion(x=x, y=y, size=size, titulo=titulo)
                return dispersion
            
            elif isinstance(nbins, int) and tipo_graficacion == 'histograma': 
                histograma = self.objeto.grafica_histograma(x=x, titulo=titulo, nbins=nbins)
                return histograma
            
            elif tipo_graficacion == 'grafico de lineas': 
                lineal = self.objeto.grafico_lineas(x= x,
                                        y= y,
                                        titulo= titulo)
                return lineal
            
            elif tipo_graficacion == 'grafico de barras': 
                barras = self.objeto.grafico_barras(x=x, y=y, titulo=titulo)
                return barras
            
            elif tipo_graficacion == 'boxplot': 
                boxplot = self.objeto.boxplot(x=x, y=y, titulo=titulo)
                return boxplot
        except Exception as e: 
            logger.error(f'Ocurrio un error para el tipo de graficacino {tipo_graficacion} de {x}. Error : {e}')
            return None
    
    def verificacion_normalidad(self, columna_categorica: str, columna_numerica:str, valor_normal:str) -> Optional[dict]: 
        """
        Realiza una verificación de normalidad de una columna numérica en función de una columna categórica
        
        Args:
            columna_categorica (str): Nombre de la columna categórica para filtrar los datos
            columna_numerica (str): Nombre de la columna numérica para verificar la normalidad
            valor_normal (str): Valor específico de la columna categórica para verificar la normalidad
            
        Returns:
            Optional[dict]: Un diccionario que contiene el resultado de la verificación de normalidad, el valor verificado, el estadístico t y el p-value
        """
        columnas_categoricas = self.df.select(pl.selectors.string()).columns
        columnas_numericas = self.df.select(pl.selectors.numeric()).columns
        
        if columna_numerica not in columnas_numericas: 
            logger.error(f'La columna {columna_numerica} no es una columna numerica')
            return None
        
        if columna_categorica not in columnas_categoricas: 
            logger.error(f'La columna {columna_categorica} no es una columna categorica')
            return None
        
        valores = self.df[columna_categorica].unique().to_list()
        
        if valor_normal not in valores: 
            logger.error(f'El valor {valor_normal} no se encuentra en la columna {columna_categorica}')
            return None
        
        normalidad = self.df.filter(pl.col(columna_categorica)==valor_normal)[columna_numerica].to_numpy()
        
        if len(normalidad) < 3: 
            logger.error(f'El valor {valor_normal} no tiene suficientes datos para verificar la normalidad')
            return None
        
        stat, p_value = shapiro(normalidad)
        
        diccionario = {
            'normalidad' : f'La muestra parece ser normal (p >= 0.05)' if p_value >= 0.05 else 'La muestra NO parece ser normal (p < 0.05)',
            'valor verificado' : valor_normal,
            't_stat' : stat, 
            'p_value' : round(p_value, 4)
        }
        
        return diccionario
    
    def promedio_maximo_columna(self, columna_categorica: str, columna_numerica: str) -> Optional[dict]: 
        """
        Realiza un análisis de promedio máximo en una columna numérica agrupada por una columna categórica
        
        Args:
            columna_categorica (str): Nombre de la columna categórica para agrupar los datos
            columna_numerica (str): Nombre de la columna numérica para calcular el promedio
        
        Returns:
            Optional[dict]: Un diccionario que contiene el promedio máximo, la categoría con el promedio más alto y un mensaje descriptivo
        """
        if columna_categorica not in self.df.select(pl.selectors.string()).columns: 
            logger.error(f'La columna {columna_categorica} no es una columna categorica')
            return None
        
        if columna_numerica not in self.df.select(pl.selectors.numeric()).columns: 
            logger.error(f'La columna {columna_numerica} no se encuentra en las columnas numericas')
            return None
        
        agrupacion = self.df.group_by(columna_categorica).agg(promedio=pl.col(columna_numerica).mean()).sort(by='promedio', descending=True)
        categoria_maxima = agrupacion[0, columna_categorica]
        promedio_maximo = agrupacion[0, 'promedio']
        diccionario = {
            'promedio_valor_maximo' : round(promedio_maximo, 3), 
            'mensaje' : f'El promedio en {categoria_maxima} son mayores que en otras categorías'
        }
        return diccionario
    
    def promedio_minimo_columna(self, columna_categorica:str, columna_numerica: str) -> Optional[dict]: 
        """
        Realiza un análisis de promedio mínimo en una columna numérica agrupada por una columna categórica
        
        Args:
            columna_categorica (str): Nombre de la columna categórica para agrupar los datos
            columna_numerica (str): Nombre de la columna numérica para calcular el promedio
        
        Returns:
            Optional[dict]: Un diccionario que contiene el promedio mínimo, la categoría con el promedio más bajo y un mensaje descriptivo
        """
        if columna_categorica not in self.df.select(pl.selectors.string()).columns: 
            logger.error(f'La columna {columna_categorica} no es una columna categorica')
            return None
        
        if columna_numerica not in self.df.select(pl.selectors.numeric()).columns: 
            logger.error(f'La columna {columna_numerica} no se encuentra en las columnas numericas')
            return None
        
        agrupacion = self.df.group_by(columna_categorica).agg(promedio=pl.col(columna_numerica).mean()).sort(by='promedio', descending=False)
        categoria_minima = agrupacion[0, columna_categorica]
        promedio_minima = agrupacion[0, 'promedio']
        diccionario = {
            'promedio_valor_minimo' : round(promedio_minima, 3), 
            'mensaje' : f'El promedio en {categoria_minima} es menor que en otras categorías'
        }
        return diccionario
    
    def correlacion(self, columna_numerica_1: str, columna_numerica_2: str) -> Optional[dict]: 
        """
        Realiza un análisis de correlación entre dos columnas numéricas
        
        Args:
            columna_numerica_1 (str): Nombre de la primera columna numérica para calcular la correlación
            columna_numerica_2 (str): Nombre de la segunda columna numérica para calcular la correlación
        
        Returns:
            Optional[dict]: Un diccionario que contiene el tipo de correlación, el coeficiente de correlación y un mensaje descriptivo
        """
        columnas_numericas = self.df.select(pl.selectors.numeric()).columns
        if columna_numerica_1 not in columnas_numericas:
            logger.error(f"La columna '{columna_numerica_1}' no es una columna numérica.")
            return None
        if columna_numerica_2 not in columnas_numericas:
            logger.error(f"La columna '{columna_numerica_2}' no es una columna numérica.")
            return None
        
        correlacion = self.df.select(pl.corr(columna_numerica_1, columna_numerica_2)).to_numpy()[0][0]
        if abs(correlacion) > 0.5: 
            diccionario = {
                'correlacion' : f'Existe una correlación {"positiva" if correlacion > 0 else "negativa"} significativa entre {columna_numerica_1} y {columna_numerica_2}', 
                'coeficiente' : round(correlacion, 3)
            }
            return diccionario
        else: 
            diccionario = {
                'correlacion' : f'No existe una correlación significativa entre {columna_numerica_1} y {columna_numerica_2}', 
                'coeficiente' : round(correlacion, 3)
            }
            return diccionario
    
    def prueba_t_dos_muestras(self,
                            columna_categorica: str, 
                            columna_numerica: str, 
                            grupo_1: str, 
                            grupo_2: str,
                            alternativa: str='greater') -> Optional[dict]: 
        '''
        Realiza una prueba t de dos muestras para comparar las medias de dos grupos en una columna numérica agrupada por una columna categórica
        
        Args:
            columna_categorica (str): Nombre de la columna categórica para agrupar los datos
            columna_numerica (str): Nombre de la columna numérica para calcular las medias
            grupo_1 (str): Valor de la categoría en la columna categórica para el primer grupo
            grupo_2 (str): Valor de la categoría en la columna categórica para el segundo grupo
            alternativa (str, optional): Tipo de prueba t ('greater' o 'less'). Defaults to 'greater'
        
        Returns:
            Optional[dict]: Un diccionario que contiene la hipótesis, el estadístico t, el valor p y un mensaje descriptivo
        '''
        columnas_numericas = self.df.select(pl.selectors.numeric()).columns
        columnas_categoricas = self.df.select(pl.selectors.string()).columns
        if columna_categorica not in columnas_categoricas: 
            logger.error(f'La columna {columna_categorica} no es una columna categorica')
            return None
        
        if columna_numerica not in columnas_numericas: 
            logger.error(f'La columna {columna_numerica} no se encuentra en las columnas numericas')
            return None
        
        valores_categoricos_unicos = self.df[columna_categorica].unique().to_list()
        if grupo_1 not in valores_categoricos_unicos:
            logger.error(f"El valor '{grupo_1}' no se encontró en la columna categórica '{columna_categorica}'.")
            return None
        if grupo_2 not in valores_categoricos_unicos:
            logger.error(f"El valor '{grupo_2}' no se encontró en la columna categórica '{columna_categorica}'.")
            return None
        
        if alternativa not in self.alternativa: 
            logger.error(f'La hipotesis {alternativa} no está disponible')
            return None
        
        grupo1_datos = self.df.filter(pl.col(columna_categorica) == grupo_1)[columna_numerica].to_numpy()
        grupo2_datos = self.df.filter(pl.col(columna_categorica) == grupo_2)[columna_numerica].to_numpy()
        
        if len(grupo1_datos) < 2 or len(grupo2_datos) < 2: 
            logger.warning('No se puede realizar la prueba t de dos muestras con menos de 2 datos por grupo')
            return None
        
        t_stat, p_value = ttest_ind(grupo1_datos, grupo2_datos, alternative=alternativa, equal_var=False)
        
        if p_value < 0.05: 
            if alternativa == 'greater': 
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: la media de {grupo_1} es significativamente MAYOR que la media de {grupo_2}', 
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
            elif alternativa == 'less': 
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: la media de {grupo_1} es significativamente MENOR que la media de {grupo_2}',
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
            elif alternativa == 'two-sided':
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: las medias de {grupo_1} y {grupo_2} son significativamente DIFERENTES', 
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
        else: 
            diccionario = {
                'hipotesis' : f'No se encontró una diferencia significativa entre las medias de {grupo_1} y {grupo_2} con el nivel de significancia dado', 
                't_stat' : round(t_stat, 4), 
                'p_value' : round(p_value, 4)
            }
            return diccionario
    
    def prueba_t_una_muestra(self, columna_numerica: str, 
                            valor_media: Union[int, float], 
                            alternativa: str='greater') -> Optional[dict]: 
        '''
        Realiza una prueba t de una muestra para comparar la media de una columna numérica con un valor dado
        
        Args:
            columna_numerica (str): Nombre de la columna numérica para calcular la media
            valor_media (Union[int, float]): Valor medio con el que se comparará la media de la columna numérica
            alternativa (str, optional): Tipo de prueba t ('greater' o 'less'). Defaults to 'greater'
        
        Returns:
            Optional[dict]: Un diccionario que contiene la hipótesis, el estadístico t, el valor p y un mensaje descriptivo
        '''
        columnas_numericas = self.df.select(pl.selectors.numeric()).columns
        if columna_numerica not in columnas_numericas: 
            logger.error(f'La columna {columna_numerica} no es una columna numerica')
            return None
        
        if alternativa not in self.alternativa: 
            logger.error(f'La hipotesis {alternativa} no está disponible')
            return None
        
        datos = self.df[columna_numerica].to_numpy()
        if len(datos) < 2: 
            logger.warning('No se puede realizar la prueba t de una muestra con menos de 2 datos')
            return None
        
        t_stat, p_value = ttest_1samp(datos, popmean=valor_media, alternative=alternativa)
        if p_value < 0.05: 
            if alternativa == 'greater': 
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: la media de {columna_numerica} es significativamente MAYOR que la media dada: {valor_media}', 
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
            elif alternativa == 'less': 
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: la media de {columna_numerica} es significativamente MENOR que la media dada: {valor_media}',
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
            elif alternativa == 'two-sided':
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: las medias de {columna_numerica} y la media dada {valor_media} son significativamente DIFERENTES', 
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
        else: 
            diccionario = {
                'hipotesis' : f'No se encontró una diferencia significativa entre las medias de {columna_numerica} y la media dada {valor_media} con el nivel de significancia dado', 
                't_stat' : round(t_stat, 4), 
                'p_value' : round(p_value, 4)
            }
            return diccionario
    
    def prueba_t_pareada(self, col_muestra_antes: str, 
                        col_muestra_despues: str, 
                        alternativa: str='greater') -> Optional[dict]: 
        '''
        Realiza una prueba t de una muestra pareada para comparar las medias de dos muestras relacionadas
        
        Args:
            col_muestra_antes (str): Nombre de la columna numérica que representa la muestra antes de la prueba
            col_muestra_despues (str): Nombre de la columna numérica que representa la muestra después de la prueba
            alternativa (str, optional): Tipo de prueba t ('greater' o 'less'). Defaults to 'greater'
        
        Returns:
            Optional[dict]: Un diccionario que contiene la hipótesis, el estadístico t, el valor p y un mensaje descriptivo
        '''
        if col_muestra_antes not in self.df.columns or col_muestra_despues not in self.df.columns: 
            logger.error(f'La columna {col_muestra_antes} y {col_muestra_despues} no se encuentran en las columnas')
            return None
        
        if alternativa not in self.alternativa:
            logger.error(f'La hipotesis {alternativa} no está disponible')
            return None
        
        antes = self.df[col_muestra_antes].to_numpy()
        despues = self.df[col_muestra_despues].to_numpy()
        
        if len(antes)<2 or len(despues)<2: 
            logger.error('No se puede realizar la prueba t de una muestra pareada con menos de 2 datos')
            return None
        
        t_stat, p_value = ttest_rel(antes, despues, alternative=alternativa)
        if p_value < 0.05: 
            if alternativa == 'greater': 
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: la media de {col_muestra_antes} es significativamente MAYOR que la media de {col_muestra_despues}', 
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
            elif alternativa == 'less': 
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: la media de {col_muestra_antes} es significativamente MENOR que la media de {col_muestra_despues}',
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
            elif alternativa == 'two-sided':
                diccionario = {
                    'hipotesis' : f'Rechazamos la hipótesis nula: las medias de {col_muestra_antes} y {col_muestra_despues} son significativamente DIFERENTES', 
                    't_stat' : round(t_stat, 4), 
                    'p_value' : round(p_value, 4)
                }
                return diccionario
        else: 
            diccionario = {
                'hipotesis' : f'No se encontró una diferencia significativa entre las medias de {col_muestra_antes} y {col_muestra_despues} con el nivel de significancia dado', 
                't_stat' : round(t_stat, 4), 
                'p_value' : round(p_value, 4)
            }
            return diccionario