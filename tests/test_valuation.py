from copy import deepcopy
import json
import pytest
from valuation.model import DEFAULT_CASE, ModelError, calculate, dump_case, load_case, sensitivity


def case(**kwargs):
    c=deepcopy(DEFAULT_CASE); c.update(kwargs); return c


def test_level_perpetuity_has_same_value_for_any_horizon():
    # Independent closed form: perpetual after-tax EBIT / discount rate.
    for n in (3,5,10):
        c=case(revenue=1000.,base_nwc=120.,growth=0.,margin=.2,tax_rate=.25,
               capex_ratio=.03,da_ratio=.03,terminal_growth=0.,wacc=.1,forecast_years=n)
        r=calculate(c)
        assert r.enterprise_value == pytest.approx(150/.1)
        assert all(x['fcff']==pytest.approx(150.) for x in r.forecast)


def test_hand_calculated_first_year_and_terminal():
    r=calculate(DEFAULT_CASE)
    first=r.forecast[0]
    assert first['revenue']==10600
    assert first['ebit']==1908
    assert first['taxes']==477
    assert first['delta_nwc']==72
    assert first['fcff']==pytest.approx(1147)
    assert first['pv_fcff']==pytest.approx(1147/1.09)
    terminal_nopat=10000*1.06**5*1.025*.18*.75
    expected_terminal=terminal_nopat*(1-.025/.12)/(.09-.025)
    assert r.terminal['value']==pytest.approx(expected_terminal)
    assert r.pv_terminal==pytest.approx(expected_terminal/1.09**5)


def test_equity_bridge_and_share_units():
    c=case(cash=900.,other_assets=100.,debt=700.,other_claims=50.,shares=250.)
    r=calculate(c)
    assert r.equity_value==pytest.approx(r.enterprise_value+250)
    assert r.share_price==pytest.approx(r.equity_value/250)
    doubled=calculate({**c,'shares':500.})
    assert doubled.share_price==pytest.approx(r.share_price/2)


def test_base_nwc_and_negative_nwc_are_modelled():
    c=case(base_nwc=-1000.,nwc_ratio=-.1)
    r=calculate(c)
    assert r.forecast[0]['delta_nwc']==pytest.approx(-60)
    assert r.forecast[0]['fcff']>r.forecast[0]['nopat']+r.forecast[0]['da']-r.forecast[0]['capex']
    d=calculate({**c,'base_nwc':-900.})
    assert r.enterprise_value-d.enterprise_value==pytest.approx(-100/1.09)


def test_year_specific_forecast_and_loss_tax_treatment():
    r=calculate(case(annual_growth=[.1,.05,.03,.02,.01],annual_margin=[-.1,.05,.1,.15,.18]))
    assert r.forecast[0]['revenue']==11000
    assert r.forecast[1]['revenue']==11550
    assert r.forecast[0]['taxes']==0
    assert r.forecast[0]['nopat']==-1100
    assert any('Loss years' in w for w in r.warnings)


@pytest.mark.parametrize('changes',[{'wacc':.02},{'wacc':.025},{'shares':0.},{'revenue':float('nan')},
    {'wacc':float('inf')},{'forecast_years':4.5},{'annual_growth':[.1]}, {'terminal_roic':.02},
    {'tax_rate':1.},{'margin':-.1},{'shares':True}])
def test_invalid_inputs_are_rejected(changes):
    with pytest.raises(ModelError):calculate(case(**changes))


def test_sensitivity_recalculates_growth_reinvestment_and_masks_invalid():
    c=case()
    cells=sensitivity(c,[.02,.09,.1],[.02,.025])
    assert next(x for x in cells if x['wacc']==.02 and x['terminal_growth']==.025)['share_price'] is None
    centre=next(x for x in cells if x['wacc']==.09 and x['terminal_growth']==.025)
    assert centre['share_price']==calculate(c).share_price
    assert calculate(case(wacc=.1)).share_price<centre['share_price']
    assert calculate(case(terminal_growth=.03)).share_price>centre['share_price']


def test_import_export_and_schema_guard():
    c=case(annual_growth=[.02]*5,annual_margin=[.16]*5,company='Sample 公司')
    assert load_case(dump_case(c))==c
    for invalid in ('[]','{}','not json',json.dumps({**c,'schema_version':True}),json.dumps({**c,'cash':None})):
        with pytest.raises(ModelError):load_case(invalid)


def test_negative_equity_is_not_silently_floored():
    r=calculate(case(debt=1000000.))
    assert r.equity_value<0 and r.share_price<0
    assert any('negative' in w.lower() for w in r.warnings)
