# Company Valuation Workbench

Interactive company valuation, quantitative research and paper trading.

## Use

Open the deployed app link in a browser. Select **Company valuation 公司估值** to enter company data and valuation assumptions, review forecast and sensitivity results, and download your case JSON. Select **Research & monitoring 量化研究** for historical prices, research, backtesting and paper trading.

The initial company inputs are illustrative. Replace them with the selected company and record the sources. Market-data availability depends on the external provider. Paper trading does not place real orders. Download valuation results before closing the browser. Cloud-generated research files are stored on the server and may be lost when the app restarts.

## Deployment

Streamlit Community Cloud: branch `main`, entry point `app.py`, Python 3.12. Dependencies are in `requirements.txt`. No deployment secret is required for the valuation module. A user may enter their own Tushare token in the research page when that data provider is selected.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Only application code, configuration, tests and illustrative data are included.
