from fastapi import FastAPI, File,UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
from io import StringIO,BytesIO
import numpy as np
from scipy import stats # Stats calc
import statsmodels.api as sm # Here we will be using StatsModel for ARIMA
import warnings # supress Warnings
from itertools import product
import matplotlib.pyplot as plt
import ujson
from fastapi.responses import JSONResponse



app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post("/uploadfile/")


async def create_upload_file(file: UploadFile =File(...)):

    contents = await file.read()
    contents = str(contents, 'utf-8')
    dataframe = pd.read_csv(StringIO(contents))

    # print(dataframe.columns)
    df = dataframe[["Gross domestic product", "Personal consumption expenditures", "Gross private domestic investment",
               "Imports"]]

    Qs = range(0, 2)
    qs = range(0, 4)
    Ps = range(0, 4)
    ps = range(0, 2)
    D = 1
    d = 1
    parameters = product(ps, qs, Ps, Qs)  # SARIMAX model Parameters:(p,d,q)(sp,sd,sq,s) +exog
    parameterslist = list(parameters)
    len(parameterslist)
    results = []
    bestaic = float("inf")
    warnings.filterwarnings("ignore")
    for param in parameterslist:
        try:
            model = sm.tsa.statespace.SARIMAX(df['Gross domestic product'], order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], 12), enforce_stationarity=False,
                                              enforce_invertibility=False).fit(disp=-1)
        except ValueError:
            continue
        aic = model.aic
        if aic < bestaic:
            model1 = model
            aic1 = aic
            param1 = param
        results.append([param, model.aic])
    result = pd.DataFrame(results)
    result.columns = ['parameters', 'aic']
    # print(result.sort_values(by='aic', ascending=True).head())
    # print(model1.summary())

    by_month = df[['Gross domestic product']]
    upcoming = pd.DataFrame()
    by_month = pd.concat([by_month, upcoming])
    by_month['forecast'] = model1.predict(start=0, end=1000)
    plt.figure(figsize=(40, 50))
    plt.title('GDP')
    plt.xlabel('Years')
    plt.ylabel('Mean GDP')
    by_month['Gross domestic product'].plot()
    by_month.forecast.plot(color='red', ls='--', label='Forecasted Gross domestic product')
    plt.legend()
    # plt.show()

    # figure= plt.savefig('Forecast.jpg')

    # Save image to in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Return image as response
    return StreamingResponse(buffer, media_type="image/png")
    #
    # res_data={'file name': file.filename, 'Forecasting': StreamingResponse(buffer, media_type="image/png")}
    # return JSONResponse(content=ujson.dumps(res_data))
    #



# pip install python-multipart
# pip install uvicorn
# pip install FastAPI
# pip install scipy
# pip install statsmodels
# pip install pandas, matplotlib, ujson

# uvicorn filename.py:app --reload
# the file name has to be changed above
