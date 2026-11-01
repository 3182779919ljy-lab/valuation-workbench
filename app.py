"""Default valuation route. Legacy research is imported only when selected."""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title='Company Valuation | 量化工作台', page_icon='◈', layout='wide')
st.logo(str(Path(__file__).parent / 'valuation' / 'wordmark.svg'), size='large')
page = st.navigation({
    'Valuation workspace': [st.Page('valuation_page.py', title='Company valuation 公司估值', icon=':material/finance:', default=True)],
    'Original workbench': [st.Page('research_app.py', title='Research & monitoring 量化研究', icon=':material/monitoring:')],
})
page.run()
