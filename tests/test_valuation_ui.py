from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]

def test_default_route_recalculation_validation_and_reset():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not at.exception
    assert any('Company valuation' in t.value for t in at.title)
    old=at.session_state['valuation_case']['wacc']
    at.number_input(key='v_wacc').set_value(10.).run()
    assert not at.exception
    assert at.session_state['valuation_case']['wacc']==.1 != old
    assert any('30.97' in m.value and 'IMPLIED SHARE PRICE' in m.value for m in at.markdown)
    at.number_input(key='v_wacc').set_value(2.).run()
    assert not at.exception
    assert any('WACC must be greater' in e.value for e in at.error)
    assert not any('IMPLIED SHARE PRICE' in m.value for m in at.markdown)
    next(b for b in at.button if b.label=='Restore demonstration').click().run()
    assert not at.exception
    assert at.session_state['valuation_case']['wacc']==.09
    assert not at.error


def test_custom_forecast_and_company_data_are_accepted():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    at.text_input(key='v_company').set_value('My Group Company').run()
    at.number_input(key='v_shares').set_value(1000.).run()
    at.toggle(key='v_custom').set_value(True).run()
    assert not at.exception
    assert len(at.session_state['valuation_case']['annual_growth'])==5
    assert at.session_state['valuation_case']['company']=='My Group Company'
    editor_key='annual_editor_5_2025_0'
    at.session_state[editor_key]={'edited_rows':{'0':{'Sales growth (%)':8.}},'added_rows':[],'deleted_rows':[]}
    at.run()
    assert at.session_state['valuation_case']['annual_growth'][0]==.08
    at.session_state[editor_key]={'edited_rows':{'0':{'Sales growth (%)':8.},'1':{'Sales growth (%)':9.}},'added_rows':[],'deleted_rows':[]}
    at.run()
    assert at.session_state['valuation_case']['annual_growth'][:2]==[.08,.09]
    at.slider(key='v_forecast_years').set_value(3).run()
    assert not at.exception
    assert len(at.session_state['valuation_case']['annual_growth'])==3


def test_navigation_preserves_valuation_case():
    from unittest.mock import patch
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    at.number_input(key='v_wacc').set_value(11.).run()
    at.text_input(key='v_company').set_value('Navigation check').run()
    with patch('quant_pipeline.realtime.fetch_realtime_snapshot',side_effect=RuntimeError('Offline smoke test')):
        at.switch_page('research_app.py').run()
    assert not at.exception
    assert any('量化研究' in t.value for t in at.title)
    at.switch_page('valuation_page.py').run()
    assert not at.exception
    assert at.number_input(key='v_wacc').value==11.
    assert at.text_input(key='v_company').value=='Navigation check'
