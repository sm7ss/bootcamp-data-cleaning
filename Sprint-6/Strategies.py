#Importamos librerías
from enum import Enum

class StrategyDataType(str, Enum): 
    FLOAT32 = 'Float32'
    INT32 = 'Int32'
    DATE = 'Date'
    UTF8 = 'Utf8'

class StrategyNullNumImput(str, Enum): 
    MEAN = 'mean'
    MEDIAN = 'median'

class StrategyNullCatImputer(str, Enum): 
    MODE = 'mode'

class StrategyFE(str, Enum): 
    SUM = 'sum'
    MEAN = 'mean'
    MIN = 'min'
    MAX = 'max'

class StrategyTest(str, Enum): 
    T_TEST_IND = 'ttest_ind'