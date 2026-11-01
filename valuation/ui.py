from __future__ import annotations
from copy import deepcopy
from html import escape
from pathlib import Path
import json
import altair as alt
import pandas as pd
import streamlit as st
from valuation.model import DEFAULT_CASE, ModelError, calculate, dump_case, load_case, sensitivity

ROOT = Path(__file__).resolve().parents[1]
RATE_FIELDS = {'growth','margin','tax_rate','da_ratio','capex_ratio','nwc_ratio','wacc','terminal_growth','terminal_roic'}


def sync_case(case):
    st.session_state['valuation_case'] = deepcopy(case)
    for key, value in case.items():
        if key not in ('annual_growth','annual_margin','schema_version'):
            st.session_state['v_' + key] = value * 100 if key in RATE_FIELDS else value
    st.session_state['v_custom'] = case['annual_growth'] is not None or case['annual_margin'] is not None
    st.session_state['v_editor_epoch'] = st.session_state.get('v_editor_epoch', 0) + 1


def import_case():
    upload = st.session_state.get('v_upload')
    if upload is None:
        st.session_state['v_import_error'] = 'Choose a JSON case file first.'
        return
    try:
        case = load_case(upload.getvalue())
        sync_case(case)
        st.session_state.pop('v_import_error', None)
        st.session_state['v_import_ok'] = 'Case loaded. Check company data and assumptions.'
    except ModelError as exc:
        st.session_state['v_import_error'] = str(exc)


def reset_case():
    sync_case(DEFAULT_CASE)
    st.session_state.pop('v_import_error', None)
    st.session_state['v_import_ok'] = 'Demonstration restored.'


def apply_wacc(value):
    st.session_state['v_wacc'] = round(value, 4)


def num(label, key, lo, hi, step=1., help=None):
    return st.number_input(label, min_value=float(lo), max_value=float(hi), step=float(step),
                           key='v_' + key, help=help, format='%.4f' if step < .01 else '%.2f')


def money(value):
    return f'{value:,.2f}'


def render():
    saved = st.session_state.get('valuation_case', deepcopy(DEFAULT_CASE))
    for key, value in saved.items():
        if key not in ('annual_growth','annual_margin','schema_version'):
            st.session_state.setdefault('v_' + key, value * 100 if key in RATE_FIELDS else value)
    st.session_state.setdefault('v_custom', saved['annual_growth'] is not None or saved['annual_margin'] is not None)
    st.session_state.setdefault('v_editor_epoch', 0)
    st.markdown('''<style>
      .block-container {padding-top:2.4rem;padding-bottom:3rem;max-width:1450px;}
      h1 {letter-spacing:-.04em;font-weight:650!important;}
      .eyebrow {font-size:11px;letter-spacing:.2em;color:#63d6bd;margin-bottom:8px;}
      .subtitle {color:#a6b7cb;font-size:15px;margin-bottom:24px;}
      .value-card {border:1px solid #2c3c51;background:#111f32;padding:22px 24px;border-radius:10px;min-height:148px;}
      .value-card.primary {border-color:#63d6bd;background:#112c30;}
      .value-label {color:#a9bcd0;font-size:13px;letter-spacing:.04em;}
      .value-number {font-size:clamp(23px,2.6vw,40px);font-weight:650;line-height:1.4;letter-spacing:-.03em;color:#edf5fa;}
      .value-unit {color:#8ea5ba;font-size:12px;}
      div[data-testid="stTabs"] {margin-top:24px;}
      div[data-testid="stMetric"] {border-bottom:1px solid #2c3c51;padding:8px 0;}
      .footer {color:#8ea5ba;font-size:12px;border-top:1px solid #27374b;padding-top:18px;margin-top:28px;}
      @media (max-width:700px) {.block-container {padding-left:1rem;padding-right:1rem;}.value-card {min-height:120px;}}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">FINC4009 · CORPORATE VALUATION</div>', unsafe_allow_html=True)
    st.title('Company valuation')
    st.markdown('<div class="subtitle">公司估值工作台 · Turn operating assumptions into an auditable equity valuation.</div>', unsafe_allow_html=True)
    status_slot = st.empty()
    metrics_slot = st.container()

    case = deepcopy(saved)
    with st.sidebar:
        st.divider()
        st.subheader('Valuation drivers')
        st.caption('修改参数后自动重算 · Rates in %')
        custom = st.session_state['v_custom']
        case['growth'] = st.number_input('Sales growth (%) 销售增长率', min_value=-90., max_value=100., step=.5, key='v_growth', disabled=custom) / 100
        case['margin'] = st.number_input('EBIT margin (%) 营业利润率', min_value=-100., max_value=100., step=.5, key='v_margin', disabled=custom) / 100
        if custom:
            st.caption('Year-specific growth and margins are active in Company data.')
        case['wacc'] = num('WACC (%) 加权平均资本成本', 'wacc', .1, 100, .25) / 100
        case['terminal_growth'] = num('Terminal growth (%) 永续增长率', 'terminal_growth', -5, 15, .25) / 100
        case['terminal_roic'] = num('Terminal ROIC (%) 稳定期投入资本回报率', 'terminal_roic', .1, 100, .5,
                                    'Stable reinvestment = NOPAT × g / ROIC. ROIC must exceed g.') / 100
        case['forecast_years'] = st.slider('Forecast years 预测年限', 3, 10, key='v_forecast_years')
        with st.expander('Load / reset case'):
            st.file_uploader('Import case JSON', type=['json'], key='v_upload', help='Use a case exported by this app or examples/demo_case.json.')
            st.button('Load selected case', on_click=import_case, width='stretch')
            st.button('Restore demonstration', on_click=reset_case, width='stretch')
            if st.session_state.get('v_import_error'):
                st.error(st.session_state['v_import_error'])
            if st.session_state.get('v_import_ok'):
                st.success(st.session_state.pop('v_import_ok'))
        st.caption('No market-data connection is required for this valuation page.')

    overview, forecast_tab, data_tab, about = st.tabs(['Overview 估值概览', 'Forecast & sensitivity 预测与敏感性', 'Company data 公司数据', 'Method & submission 模型与提交'])
    with data_tab:
        st.subheader('Company and source data')
        st.caption('All monetary inputs are in millions of the chosen currency; shares are in millions. Currency selection changes labels only, not exchange rates. 所有金额和股数均以百万为单位。')
        c1,c2,c3 = st.columns([2,1,1])
        with c1:
            case['company'] = st.text_input('Company name 公司名称', key='v_company', max_chars=200)
        with c2:
            case['ticker'] = st.text_input('Ticker 证券代码', key='v_ticker', max_chars=40)
        with c3:
            case['currency'] = st.selectbox('Reporting currency', ['USD','HKD','CNY','EUR','GBP','JPY'], key='v_currency')
        c1,c2 = st.columns([1,2])
        with c1:
            case['base_year'] = st.number_input('Base fiscal year 基准财年', min_value=2000, max_value=2100, step=1, key='v_base_year')
        with c2:
            case['data_status'] = st.selectbox('Data status 数据状态', ['Demonstration data','User-supplied data (unverified)'], key='v_data_status')
        with st.container(border=True):
            st.markdown('**Financial base · 财务基础**')
            c1,c2,c3 = st.columns(3)
            with c1:
                case['revenue'] = num('Base revenue 营业收入', 'revenue', .000001, 1e12, 100)
                case['base_nwc'] = num('Base operating NWC 经营营运资本', 'base_nwc', -1e12, 1e12, 100,
                                        'Operating current assets minus operating current liabilities; exclude cash, debt and financing items. Negative NWC is supported.')
                case['shares'] = num('Diluted shares 稀释后股数', 'shares', .000001, 1e12, 10)
            with c2:
                case['cash'] = num('Excess cash 非经营现金', 'cash', 0, 1e12, 100)
                case['debt'] = num('Debt and debt-like claims 债务', 'debt', 0, 1e12, 100,
                                    'Include lease liabilities if consistent with operating cash flows; do not count claims twice.')
            with c3:
                case['other_assets'] = num('Other non-operating assets 其他资产', 'other_assets', 0, 1e12, 100)
                case['other_claims'] = num('Other claims 其他权益扣减', 'other_claims', 0, 1e12, 100,
                                            'For example preferred equity or non-controlling interests not included in debt.')
        with st.container(border=True):
            st.markdown('**Operating assumptions · 经营假设**')
            cols = st.columns(4)
            for col, key, label, lo in zip(cols, ['tax_rate','da_ratio','capex_ratio','nwc_ratio'],
                    ['Cash tax rate (%) 税率','D&A / revenue (%) 折旧摊销','Capex / revenue (%) 资本开支','NWC / revenue (%) 营运资本'], [0,0,0,-100]):
                with col:
                    case[key] = num(label, key, lo, 100, .5) / 100
            is_custom = st.toggle('Use year-specific growth and EBIT margins 逐年设置', key='v_custom')
            if is_custom:
                n = case['forecast_years']
                gs = saved['annual_growth'] or [case['growth']] * n
                ms = saved['annual_margin'] or [case['margin']] * n
                gs = (gs + [gs[-1]] * n)[:n]
                ms = (ms + [ms[-1]] * n)[:n]
                frame = pd.DataFrame({'Year':[case['base_year']+i+1 for i in range(n)], 'Sales growth (%)':[100*x for x in gs], 'EBIT margin (%)':[100*x for x in ms]})
                editor_key = f"annual_editor_{n}_{case['base_year']}_{st.session_state['v_editor_epoch']}"
                if editor_key not in st.session_state:
                    st.session_state['seed_' + editor_key] = frame
                edited = st.data_editor(st.session_state['seed_' + editor_key], hide_index=True, width='stretch', disabled=['Year'],
                        key=editor_key,
                        column_config={'Sales growth (%)':st.column_config.NumberColumn(min_value=-90,max_value=100,step=.5,required=True),
                                       'EBIT margin (%)':st.column_config.NumberColumn(min_value=-100,max_value=100,step=.5,required=True)})
                case['annual_growth'] = (edited['Sales growth (%)']/100).tolist()
                case['annual_margin'] = (edited['EBIT margin (%)']/100).tolist()
            else:
                case['annual_growth'] = case['annual_margin'] = None
        case['sources'] = st.text_area('Sources and assumption rationale 数据来源与假设依据', key='v_sources', height=150,
                        help='Record annual-report URLs and pages, fiscal periods, units, each input source, and the rationale for forecast rates and WACC.', max_chars=12000)
        with st.expander('WACC calculator · CAPM and market-value weights'):
            st.caption('Optional helper. Its inputs are illustrative; document your chosen estimates in Sources. The case export retains the applied WACC, not these helper inputs.')
            c1,c2,c3 = st.columns(3)
            with c1:
                rf = st.number_input('Risk-free rate (%)', 0., 30., 4., .25)
                beta = st.number_input('Levered beta', 0., 5., 1., .1)
            with c2:
                erp = st.number_input('Equity risk premium (%)', 0., 30., 5., .25)
                kd = st.number_input('Pre-tax cost of debt (%)', 0., 50., 5., .25)
            with c3:
                dw = st.number_input('Debt / (debt + equity) (%)', 0., 100., 20., 5.)
                st.caption('Use market-value weights; debt tax shield uses the operating tax-rate input.')
            helper = (rf + beta * erp) * (1-dw/100) + kd * (1-case['tax_rate']) * dw/100
            st.write(f'Cost of equity: **{rf+beta*erp:.2f}%** · Calculated WACC: **{helper:.2f}%**')
            st.button('Apply calculated WACC', on_click=apply_wacc, args=(helper,), disabled=not .1 <= helper <= 100)

    # Persist non-widget state to survive navigation away from this page.
    st.session_state['valuation_case'] = deepcopy(case)
    with about:
        render_method(case)
    if case['data_status'] != 'Demonstration data':
        status_slot.info('USER-SUPPLIED DATA · 用户输入数据，未经独立核实。Check the Sources field against your group model.')
    try:
        result = calculate(case)
    except ModelError as exc:
        with metrics_slot:
            st.error(str(exc))
            st.caption('Valuation outputs and exports are withheld until the input is corrected. 请修正参数后查看结果。')
        with overview:
            st.info('Correct the input error above to calculate enterprise value, equity value and share price.')
        return
    cur = case['currency']
    with metrics_slot:
        st.caption(f"{case['company']} · {case['ticker']}  |  Valued at end of FY{case['base_year']}  |  {case['forecast_years']}-year FCFF DCF")
        cols = st.columns(3)
        for col,label,value,unit,primary in [
            (cols[0],'ENTERPRISE VALUE · 企业价值',result.enterprise_value,cur+' million',False),
            (cols[1],'EQUITY VALUE · 股权价值',result.equity_value,cur+' million',False),
            (cols[2],'IMPLIED SHARE PRICE · 每股估值',result.share_price,cur+' / share',True)]:
            with col:
                st.markdown(f'<div class="value-card {"primary" if primary else ""}"><div class="value-label">{label}</div><div class="value-number">{money(value)}</div><div class="value-unit">{unit}</div></div>', unsafe_allow_html=True)
        for warning in result.warnings:
            st.warning(warning)
    frame = pd.DataFrame(result.forecast)
    with overview:
        left,right = st.columns([1.8,1], gap='large')
        with left:
            st.subheader('Cash flow outlook')
            st.caption(f'Annual free cash flow to the firm · {cur} million · End-of-year discounting')
            bars = alt.Chart(frame).mark_bar(color='#63d6bd',cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
                x=alt.X('year:O',title=None), y=alt.Y('fcff:Q',title=f'FCFF ({cur} m)'),
                tooltip=[alt.Tooltip('year:O',title='Year'),alt.Tooltip('fcff:Q',format=',.2f',title='FCFF'),alt.Tooltip('pv_fcff:Q',format=',.2f',title='Present value')])
            line = alt.Chart(frame).mark_line(color='#9cacf4',point=True,strokeWidth=2).encode(x='year:O', y='pv_fcff:Q')
            st.altair_chart((bars+line).properties(height=290), width='stretch')
            st.caption('Green bars: forecast FCFF. Lavender line: present value of FCFF.')
        with right:
            st.subheader('From business to equity')
            bridge = pd.DataFrame({
                'Component':['PV of forecast FCFF','PV of terminal value','Enterprise value','+ Excess cash','+ Non-operating assets','− Debt','− Other claims','Equity value'],
                f'{cur} million':[result.pv_forecast,result.pv_terminal,result.enterprise_value,case['cash'],case['other_assets'],-case['debt'],-case['other_claims'],result.equity_value]})
            st.dataframe(bridge, hide_index=True, width='stretch', column_config={f'{cur} million':st.column_config.NumberColumn(format='%.2f')})
            tv_share = result.pv_terminal/result.enterprise_value*100 if result.enterprise_value>0 else None
            st.caption(f'Terminal value / EV: {tv_share:.1f}%' if tv_share is not None else 'Terminal value / EV: not meaningful for non-positive EV.')
        st.divider()
        st.subheader('Save this valuation')
        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button('Download case JSON', dump_case(case), 'valuation_case.json','application/json',width='stretch')
        with c2:
            st.download_button('Download forecast CSV', frame.to_csv(index=False).encode('utf-8-sig'), 'valuation_forecast.csv','text/csv',width='stretch')
        with c3:
            summary = {'company':case['company'],'currency':cur,'data_status':case['data_status'], 'valuation_basis':f"End of FY{case['base_year']}",
                       'enterprise_value_m':result.enterprise_value,'equity_value_m':result.equity_value,'price_per_share':result.share_price,
                       'terminal':result.terminal,'sources':case['sources'],'case':case}
            st.download_button('Download valuation summary',json.dumps(summary,ensure_ascii=False,indent=2),'valuation_summary.json','application/json',width='stretch')
        st.caption('Changes are kept in this browser session. Download the case to preserve your inputs after closing or restarting the app.')

    with forecast_tab:
        st.subheader('Forecast cash flows')
        st.caption(f'{cur} million except growth, margins and discount factors. Positive ΔNWC is a cash outflow.')
        display=frame.drop(columns='period').copy().set_index('year').T
        display.index=['Sales growth','EBIT margin','Revenue','EBIT','Cash taxes','NOPAT','D&A','Capital expenditure','Operating NWC','Change in NWC','FCFF','Discount factor','PV of FCFF']
        st.dataframe(display.style.format('{:,.4f}'),width='stretch')
        st.caption('Growth and margin rows are decimals: 0.06 means 6%. The terminal year uses normalized reinvestment, shown separately below.')
        t=result.terminal
        st.markdown(f"**Stable year FY{t['year']}** · Revenue {money(t['revenue'])} · NOPAT {money(t['nopat'])} · Reinvestment {money(t['reinvestment'])} · FCFF {money(t['fcff'])} ({cur} million)")
        st.caption(f"Stable reinvestment rate = g / ROIC = {t['reinvestment_rate']:.2%}; terminal value at end of FY{case['base_year']+case['forecast_years']} = {money(t['value'])} {cur} million.")
        st.divider()
        st.subheader('WACC × terminal growth')
        st.caption(f'Implied share price · {cur} per share. Each cell recalculates stable reinvestment. — marks invalid combinations.')
        ws=[round(case['wacc']+i*.005,8) for i in [-2,-1,0,1,2]]
        gs=[round(case['terminal_growth']+i*.005,8) for i in [-2,-1,0,1,2]]
        cells=pd.DataFrame(sensitivity(case,ws,gs))
        matrix=cells.pivot(index='wacc',columns='terminal_growth',values='share_price')
        matrix.index=[f'{v:.2%} WACC' for v in matrix.index]
        matrix.columns=[f'{v:.2%} g' for v in matrix.columns]
        st.dataframe(matrix.style.format('{:,.2f}',na_rep='—').background_gradient(cmap='GnBu',axis=None),width='stretch')
        st.caption(f"Centre cell = current case: WACC {case['wacc']:.2%}, g {case['terminal_growth']:.2%}, price {money(result.share_price)}.")
        st.download_button('Download sensitivity CSV',cells.to_csv(index=False).encode('utf-8-sig'),'valuation_sensitivity.csv','text/csv')
    st.markdown('<div class="footer">FINC4009 · Individual valuation module · Existing quantitative research tools remain available in the navigation.</div>',unsafe_allow_html=True)


def render_method(case):
    st.subheader('An auditable FCFF model')
    st.markdown('''This module values operating cash flows at WACC, then adjusts for non-operating assets and claims to arrive at common equity. The valuation date is **the end of the base fiscal year**; forecast cash flows arrive at each subsequent year-end. Use financial data and balance-sheet adjustments consistent with that date.

**1 · Explicit forecast**

- Revenueₜ = Revenueₜ₋₁ × (1 + sales growthₜ)
- EBITₜ = Revenueₜ × EBIT marginₜ
- Cash taxesₜ = max(EBITₜ, 0) × tax rate; NOPATₜ = EBITₜ − cash taxesₜ
- D&Aₜ = Revenueₜ × D&A ratio; Capexₜ = Revenueₜ × Capex ratio
- Operating NWCₜ = Revenueₜ × NWC ratio; ΔNWCₜ = NWCₜ − NWCₜ₋₁
- **FCFFₜ = NOPATₜ + D&Aₜ − Capexₜ − ΔNWCₜ**

Year 1 uses the separately entered base NWC balance. Operating NWC excludes cash and financing balances. Interest is not deducted from FCFF; financing costs are reflected in WACC.

**2 · Stable phase and terminal value**

Revenueₙ₊₁ = Revenueₙ × (1 + g). The final explicit-year EBIT margin and the tax rate continue into the stable phase. Stable NOPAT = Revenueₙ₊₁ × marginₙ × (1 − tax rate).

Stable reinvestment = NOPATₙ₊₁ × g / ROIC. Therefore **FCFFₙ₊₁ = NOPATₙ₊₁ × (1 − g / ROIC)**. This replaces the explicit-period D&A, capex and working-capital ratios with normalized net reinvestment. It does not simply grow final-year FCFF by g.
''')
    st.latex(r'TV_n=\frac{FCFF_{n+1}}{WACC-g},\qquad EV=\sum_{t=1}^{n}\frac{FCFF_t}{(1+WACC)^t}+\frac{TV_n}{(1+WACC)^n}')
    st.markdown('''**3 · Equity bridge and share price**

Equity value = EV + excess cash + other non-operating assets − debt and debt-like claims − other claims.

**Implied share price = equity value / diluted shares.** Both equity value and share count are in millions, so the result is currency per share. Currency selection only changes the unit label; it performs no FX conversion.

**4 · WACC helper**

Cost of equity = risk-free rate + levered beta × equity risk premium. WACC = E/(D+E) × cost of equity + D/(D+E) × cost of debt × (1 − tax rate). Use market-value capital weights and a consistent currency. You may enter your group's WACC directly instead.

**Assumptions and limits**

The model assumes annual end-of-year discounting, constant WACC, constant operating ratios within the explicit period, and stable ROIC after it. Annual growth and margins can be customized. It does not model loss carryforwards, changing debt schedules, dilution over time, FX, or financial-sector balance sheets. Negative equity residuals are displayed with a warning. Terminal growth must be below both WACC and stable ROIC. Input sources are user-provided and are not independently verified.

**Method references**

- [Aswath Damodaran, NYU Stern — Discounted Cashflow Valuation](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/val.html)
- [Aswath Damodaran, NYU Stern — Estimating Terminal Value](https://pages.stern.nyu.edu/~adamodar/pdfiles/valn2ed/ch12.pdf)
''')
    st.divider()
    st.subheader('Submission checklist')
    st.markdown('''1. Replace demonstration inputs with the company and assumptions used in your group project.
2. Record financial-report URLs, page references, fiscal periods and assumptions under **Company data**. Export your case JSON.
3. Check EV, equity value and share price against the group Excel model, including units, timing and terminal reinvestment.
4. Complete the student details in the two-page development report. Update the prompt record if you request more changes.
5. Submit the complete local project and your case JSON, together with the report. The individual assignment deadline is **1 December 2026**.

**Demo walkthrough:** change sales growth, WACC and terminal growth; explain the three value outputs; show the cash-flow table and sensitivity grid; export and reload a case. The research page contains the preserved earlier platform. Its live sources need internet access, while this valuation module works offline after installing Python dependencies.
''')
    for path,label,mime in [(ROOT/'submission'/'Development_Report.docx','Download 2-page development report','application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
                            (ROOT/'submission'/'prompt_log.md','Download prompt record','text/markdown'),
                            (ROOT/'README.md','Download run instructions','text/markdown')]:
        if path.exists(): st.download_button(label,path.read_bytes(),path.name,mime)
