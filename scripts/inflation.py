from fredapi import Fred

fred = Fred(api_key='6827b33b1899d3a7996eaee1ae920bbd')
nigeria_inflation = fred.get_series('FPCPITOTLZGNGA')
us_inflation = fred.get_series('FPCPITOTLZGUSA')
nigeria_inflation.loc['2024-01-01'] = 31.7
us_inflation.loc['2024-01-01'] = 3.0
print(nigeria_inflation)
print(us_inflation)