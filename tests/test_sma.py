import pandas as pd
from src.features.indicators import sma
def test_sma_simple():
    df = pd.DataFrame({'close':[1,2,3,4,5]})
    s = sma(df, 3)
    assert s.iloc[2] == (1+2+3)/3