"""Auditable end-of-year FCFF DCF. Monetary amounts and shares are in millions."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import json
import math

DEFAULT_CASE = {
    'schema_version': 1, 'company': 'Illustrative Company', 'ticker': 'DEMO',
    'currency': 'USD', 'data_status': 'Demonstration data', 'base_year': 2025,
    'forecast_years': 5, 'revenue': 10000.0, 'base_nwc': 1200.0,
    'cash': 1000.0, 'debt': 2500.0, 'other_assets': 0.0, 'other_claims': 0.0,
    'shares': 500.0, 'growth': 0.06, 'margin': 0.18, 'tax_rate': 0.25,
    'da_ratio': 0.03, 'capex_ratio': 0.05, 'nwc_ratio': 0.12,
    'wacc': 0.09, 'terminal_growth': 0.025, 'terminal_roic': 0.12,
    'annual_growth': None, 'annual_margin': None,
    'sources': "All values are invented teaching inputs, not financial statements or market data. Replace them with your group's company, financial data, assumptions and citations before submission.",
}

class ModelError(ValueError):
    pass

@dataclass(frozen=True)
class Valuation:
    forecast: list[dict]
    terminal: dict
    pv_forecast: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    share_price: float
    warnings: tuple[str, ...]

def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelError(f'{name} must be a finite number.')
    return float(value)

def validate(case: dict) -> None:
    if not isinstance(case, dict):
        raise ModelError('The case must be a JSON object.')
    missing = DEFAULT_CASE.keys() - case.keys()
    if missing:
        raise ModelError('Missing fields: ' + ', '.join(sorted(missing)))
    if type(case['schema_version']) is not int or case['schema_version'] != 1:
        raise ModelError('Only schema_version 1 is supported.')
    for key in ('company', 'ticker', 'currency', 'data_status', 'sources'):
        if not isinstance(case[key], str) or len(case[key]) > 12000:
            raise ModelError(f'{key} must be text, at most 12,000 characters.')
    if not case['company'].strip():
        raise ModelError('Enter a company name.')
    if case['currency'] not in ('USD', 'HKD', 'CNY', 'EUR', 'GBP', 'JPY'):
        raise ModelError('Unsupported currency code. Choose USD, HKD, CNY, EUR, GBP or JPY.')
    if case['data_status'] not in ('Demonstration data', 'User-supplied data (unverified)'):
        raise ModelError('Choose a valid data status; imported data is not independently verified.')
    for key, lo, hi in [('base_year', 2000, 2100), ('forecast_years', 3, 10)]:
        if type(case[key]) is not int or not lo <= case[key] <= hi:
            raise ModelError(f'{key} must be an integer from {lo} to {hi}.')
    bounds = {
        'revenue': (0.000001, 1e12), 'base_nwc': (-1e12, 1e12),
        'cash': (0, 1e12), 'debt': (0, 1e12), 'other_assets': (0, 1e12),
        'other_claims': (0, 1e12), 'shares': (0.000001, 1e12),
        'growth': (-0.9, 1), 'margin': (-1, 1), 'tax_rate': (0, 1),
        'da_ratio': (0, 1), 'capex_ratio': (0, 1), 'nwc_ratio': (-1, 1),
        'wacc': (0.001, 1), 'terminal_growth': (-0.05, 0.15), 'terminal_roic': (0.001, 1),
    }
    for key, (lo, hi) in bounds.items():
        if not lo <= _number(case[key], key) <= hi:
            raise ModelError(f'{key} must be between {lo:g} and {hi:g} (rates are decimals).')
    if case['wacc'] <= case['terminal_growth']:
        raise ModelError('WACC must be greater than terminal growth. Increase WACC or reduce terminal growth.')
    if case['terminal_growth'] >= case['terminal_roic']:
        raise ModelError('Terminal ROIC must exceed terminal growth so stable cash flow remains positive.')
    for key, lo in [('annual_growth', -0.9), ('annual_margin', -1)]:
        values = case[key]
        if values is not None:
            if not isinstance(values, list) or len(values) != case['forecast_years']:
                raise ModelError(f'{key} must contain one value for each forecast year.')
            for value in values:
                if not lo <= _number(value, key) <= 1:
                    raise ModelError(f'{key} values must be between {lo} and 1.')
    terminal_margin = case['annual_margin'][-1] if case['annual_margin'] is not None else case['margin']
    if terminal_margin <= 0 or case['tax_rate'] == 1:
        raise ModelError('The stable phase needs positive after-tax operating earnings. Set a positive final-year EBIT margin and a tax rate below 100%.')

def calculate(case: dict) -> Valuation:
    validate(case)
    n = case['forecast_years']
    growth = case['annual_growth'] if case['annual_growth'] is not None else [case['growth']] * n
    margins = case['annual_margin'] if case['annual_margin'] is not None else [case['margin']] * n
    prior_revenue, prior_nwc = case['revenue'], case['base_nwc']
    rows = []
    for t, (g, margin) in enumerate(zip(growth, margins), 1):
        revenue = prior_revenue * (1 + g)
        ebit = revenue * margin
        taxes = max(ebit, 0.0) * case['tax_rate']
        nopat = ebit - taxes
        da = revenue * case['da_ratio']
        capex = revenue * case['capex_ratio']
        nwc = revenue * case['nwc_ratio']
        delta_nwc = nwc - prior_nwc
        fcff = nopat + da - capex - delta_nwc
        discount_factor = (1 + case['wacc']) ** -t
        rows.append(dict(year=case['base_year'] + t, period=t, growth=g, margin=margin,
                         revenue=revenue, ebit=ebit, taxes=taxes, nopat=nopat, da=da,
                         capex=capex, nwc=nwc, delta_nwc=delta_nwc, fcff=fcff,
                         discount_factor=discount_factor, pv_fcff=fcff * discount_factor))
        prior_revenue, prior_nwc = revenue, nwc
    g = case['terminal_growth']
    terminal_revenue = rows[-1]['revenue'] * (1 + g)
    terminal_nopat = terminal_revenue * margins[-1] * (1 - case['tax_rate'])
    reinvestment_rate = g / case['terminal_roic']
    terminal_reinvestment = terminal_nopat * reinvestment_rate
    terminal_fcff = terminal_nopat - terminal_reinvestment
    terminal_value = terminal_fcff / (case['wacc'] - g)
    pv_terminal = terminal_value / (1 + case['wacc']) ** n
    pv_forecast = sum(r['pv_fcff'] for r in rows)
    ev = pv_forecast + pv_terminal
    equity = ev + case['cash'] + case['other_assets'] - case['debt'] - case['other_claims']
    terminal = dict(year=case['base_year'] + n + 1, revenue=terminal_revenue,
                    margin=margins[-1], nopat=terminal_nopat, reinvestment_rate=reinvestment_rate,
                    reinvestment=terminal_reinvestment, fcff=terminal_fcff,
                    value=terminal_value, discount_factor=(1 + case['wacc']) ** -n)
    warnings = []
    if case['wacc'] - g < .01:
        warnings.append('WACC and terminal growth differ by less than 1 percentage point; the valuation is highly sensitive.')
    if ev > 0 and pv_terminal / ev > .8:
        warnings.append('More than 80% of enterprise value comes from terminal value. Review stable growth and ROIC.')
    if equity < 0:
        warnings.append('The equity bridge is negative. This is a model residual, not a tradable negative share price; a distressed-equity model may be needed.')
    if any(r['ebit'] < 0 for r in rows):
        warnings.append('Loss years receive no immediate tax benefit. Loss carryforwards are not modelled.')
    if g < 0:
        warnings.append('Negative stable growth implies release of invested capital under g/ROIC. Review that assumption carefully.')
    if not case['sources'].strip():
        warnings.append('Add sources and rationale for company data and assumptions before submission.')
    if not all(math.isfinite(v) for v in (ev, equity, terminal_value, equity / case['shares'])):
        raise ModelError('Inputs produced a non-finite valuation.')
    return Valuation(rows, terminal, pv_forecast, pv_terminal, ev, equity, equity / case['shares'], tuple(warnings))

def sensitivity(case: dict, wacc_values: list[float], growth_values: list[float]) -> list[dict]:
    cells = []
    for w in wacc_values:
        for g in growth_values:
            candidate = deepcopy(case)
            candidate.update(wacc=w, terminal_growth=g)
            try:
                value = calculate(candidate).share_price
                reason = ''
            except ModelError as exc:
                value, reason = None, str(exc)
            cells.append(dict(wacc=w, terminal_growth=g, share_price=value, reason=reason))
    return cells

def load_case(raw: str | bytes) -> dict:
    try:
        if len(raw) > 200_000:
            raise ModelError('Case file must be smaller than 200 KB.')
        case = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ModelError('The file is not a valid JSON case.') from exc
    validate(case)
    return {k: case[k] for k in DEFAULT_CASE}

def dump_case(case: dict) -> str:
    validate(case)
    return json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False)
