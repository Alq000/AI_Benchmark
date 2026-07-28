import os
import glob
import argparse
import panel as pn
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv

pn.extension('tabulator', sizing_mode="stretch_width")

# --- Parse CLI Arguments ---
parser = argparse.ArgumentParser(description="Launch Benchmark Explorer Dashboard")
parser.add_argument("--file", type=str, default=None, help="Path to compiled parquet/csv file")
parser.add_argument("--port", type=int, default=5006, help="Port to serve dashboard on")
args, _ = parser.parse_known_args()

def discover_available_files():
    files = set()
    for f in glob.glob("*.parquet") + glob.glob("*.csv"):
        files.add(f)
    for f in glob.glob("results/**/compiled_benchmark_data.*", recursive=True):
        files.add(f)
    sorted_files = sorted(list(files))
    return sorted_files if sorted_files else ["EXHAUSTIVE_benchmark_compiled.csv"]

available_files = discover_available_files()
initial_file = args.file if (args.file and os.path.exists(args.file)) else available_files[0]

# --- Widgets ---
file_selector = pn.widgets.Select(name="Dataset Source File", options=available_files, value=initial_file if initial_file in available_files else available_files[0])

preset_select = pn.widgets.Select(
    name="Analysis Presets", 
    options=["Custom", "True vs Estimated k0", "Judge Score vs Error", "Performance vs Compute"], 
    value="Custom"
)

custom_query = pn.widgets.TextInput(name="Arbitrary Filter (Pandas Query)", placeholder="e.g., judge_final_weighted_score > 0.3 and num_experiment_calls <= 3")

model_select = pn.widgets.MultiSelect(name='Filter: Agent Model', options=[], value=[])
status_select = pn.widgets.MultiSelect(name='Filter: Trial Status', options=[], value=[])

x_axis = pn.widgets.Select(name='X-Axis Metric', options=[], value=None)
y_axis = pn.widgets.Select(name='Y-Axis Metric', options=[], value=None)
color_by = pn.widgets.Select(name='Group / Color By', options=[], value=None)

plot_type = pn.widgets.RadioButtonGroup(name='Plot Type', options=['Scatter', 'KDE / Histogram', 'BoxPlot'], value='Scatter')
log_x = pn.widgets.Checkbox(name='Log X-Axis', value=False)
log_y = pn.widgets.Checkbox(name='Log Y-Axis', value=False)

show_ref_line = pn.widgets.Checkbox(name='Show Reference Line (y=mx+b)', value=False)
ref_slope = pn.widgets.FloatInput(name='Slope (m)', value=1.0, width=120)
ref_intercept = pn.widgets.FloatInput(name='Intercept (b)', value=0.0, width=120)

show_error_bands = pn.widgets.Checkbox(name='Show Error Bands (Grouped by X-Axis)', value=False)
band_style = pn.widgets.RadioButtonGroup(name='Band Style', options=['Straight Line', 'Tip-to-Tip'], value='Straight Line')

tabulator_table = pn.widgets.Tabulator(pagination='remote', page_size=15, sizing_mode="stretch_width")

def load_data(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        if filepath.endswith(".parquet"):
            return pd.read_parquet(filepath)
        return pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame()

def update_widget_options(df):
    if df.empty:
        return

    cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    num_cols = list(df.select_dtypes(include=[np.number]).columns)

    if 'agent_model' in df.columns:
        models = list(df['agent_model'].dropna().unique())
        model_select.options = models
        model_select.value = models
        model_select.disabled = False
    else:
        model_select.options = []
        model_select.value = []
        model_select.disabled = True

    if 'trial_status' in df.columns:
        statuses = list(df['trial_status'].dropna().unique())
        status_select.options = statuses
        status_select.value = statuses
        status_select.disabled = False
    else:
        status_select.options = []
        status_select.value = []
        status_select.disabled = True

    current_x = x_axis.value
    current_y = y_axis.value
    x_axis.options = num_cols
    y_axis.options = num_cols

    if num_cols:
        x_axis.value = current_x if current_x in num_cols else num_cols[0]
        y_axis.value = current_y if current_y in num_cols else (num_cols[1] if len(num_cols) > 1 else num_cols[0])

    group_options = ['None'] + cat_cols + num_cols
    current_color = color_by.value
    color_by.options = group_options
    color_by.value = current_color if current_color in group_options else ('agent_model' if 'agent_model' in group_options else 'None')

@pn.depends(preset_select, watch=True)
def apply_preset(preset):
    if preset == "Custom": return
    cols = x_axis.options
    
    if preset == "True vs Estimated k0":
        x_cand = [c for c in cols if c == 'true_k_0' or c == 'true_k0' or c == 'summary_true_k0']
        y_cand = [c for c in cols if 'estimated_k0' in c]
        if x_cand: x_axis.value = x_cand[0]
        if y_cand: y_axis.value = y_cand[0]
        
    elif preset == "Judge Score vs Error":
        x_cand = [c for c in cols if 'judge' in c and ('score' in c or 'weighted' in c)]
        y_cand = [c for c in cols if 'error' in c]
        if x_cand: x_axis.value = x_cand[0]
        if y_cand: y_axis.value = y_cand[0]
        
    elif preset == "Performance vs Compute":
        x_cand = [c for c in cols if 'num_experiment_calls' in c or 'num_assistant_messages' in c]
        y_cand = [c for c in cols if 'error' in c or 'judge' in c]
        if x_cand: x_axis.value = x_cand[0]
        if y_cand: y_axis.value = y_cand[0]

@pn.depends(file_selector, watch=True)
def on_file_change(filepath):
    df = load_data(filepath)
    update_widget_options(df)

@pn.depends(
    file_selector, preset_select, custom_query, model_select, status_select, 
    x_axis, y_axis, color_by, plot_type, log_x, log_y, 
    show_ref_line, ref_slope, ref_intercept, show_error_bands, band_style # Added band_style
)
def update_view(filepath, preset, query_str, models, statuses, x_col, y_col, group_col, p_type, is_log_x, is_log_y, s_ref, r_slope, r_int, s_err, s_band_style):
    df = load_data(filepath)
    if df.empty:
        return pn.Column(hv.Text(0.5, 0.5, "Selected file is empty or missing.").opts(text_font_size='14pt'))

    filtered = df.copy()

    # 1. Base Dropdown Filters
    if 'agent_model' in filtered.columns and models:
        filtered = filtered[filtered['agent_model'].isin(models)]
    if 'trial_status' in filtered.columns and statuses:
        filtered = filtered[filtered['trial_status'].isin(statuses)]

    # 2. Arbitrary Complex Query
    if query_str.strip():
        try:
            filtered = filtered.query(query_str)
        except Exception as e:
            return pn.Column(
                hv.Text(0.5, 0.5, "Invalid Pandas Query").opts(text_font_size='14pt'),
                pn.pane.Alert(f"Query Error: {str(e)}", alert_type="danger")
            )

    tabulator_table.value = filtered

    if filtered.empty:
        return pn.Column(hv.Text(0.5, 0.5, "No data matching filter criteria").opts(text_font_size='14pt'))

    if not x_col or not y_col or x_col not in filtered.columns or y_col not in filtered.columns:
        return pn.Column(hv.Text(0.5, 0.5, "Please select valid X and Y columns.").opts(text_font_size='14pt'))

    hover_cols = [c for c in ['trial_id', 'run_id', 'agent_model', 'trial_status'] if c in filtered.columns]
    by_param = group_col if (group_col and group_col != 'None' and group_col in filtered.columns) else None

    # 3. Generating the Plot
    try:
        kwargs = {"height": 550, "grid": True, "title": f"{y_col} vs {x_col}"}
        if by_param: kwargs["by"] = by_param

        if p_type == 'Scatter':
            plot = filtered.hvplot.scatter(x=x_col, y=y_col, hover_cols=hover_cols, logx=is_log_x, logy=is_log_y, **kwargs)
            
            # Feature: Reference Line
            if s_ref:
                ref_line = hv.Slope(r_slope, r_int).opts(color='black', line_dash='solid', alpha=0.7, line_width=2)
                plot = plot * ref_line
                
            # Feature: Error Bands & Error Bars
            if s_err:
                # Dynamically group discrete values, or bin continuous values
                unique_vals = filtered[x_col].nunique()
                if unique_vals <= 30 or unique_vals <= len(filtered) * 0.2:
                    grp = filtered.groupby(x_col)[y_col].agg(['mean', 'std', 'count']).dropna().reset_index()
                else:
                    bins = pd.cut(filtered[x_col], bins=min(15, max(5, len(filtered)//5)))
                    grp = filtered.groupby(bins, observed=True).agg({x_col: 'mean', y_col: ['mean', 'std', 'count']}).dropna()
                    grp.columns = [x_col, 'mean', 'std', 'count']
                    grp = grp.reset_index(drop=True)
                
                if not grp.empty:
                    grp['std'] = grp['std'].fillna(0.0)
                    grp['upper'] = grp['mean'] + grp['std']
                    grp['lower'] = grp['mean'] - grp['std']
                    grp = grp.sort_values(by=x_col)
                    
                    x_vals = grp[x_col].values
                    
                    if s_band_style == 'Straight Line' and len(grp) > 1:
                        # Regress the boundaries for a straight-line fit
                        m_mean, b_mean = np.polyfit(x_vals, grp['mean'], 1)
                        m_upper, b_upper = np.polyfit(x_vals, grp['upper'], 1)
                        m_lower, b_lower = np.polyfit(x_vals, grp['lower'], 1)
                        
                        grp['fit_mean'] = m_mean * x_vals + b_mean
                        grp['fit_upper'] = m_upper * x_vals + b_upper
                        grp['fit_lower'] = m_lower * x_vals + b_lower
                        
                        band = hv.Area((x_vals, grp['fit_lower'], grp['fit_upper']), vdims=['y', 'y2']).opts(alpha=0.15, color='gray')
                        mean_curve = hv.Curve((x_vals, grp['fit_mean'])).opts(color='blue', line_dash='dashed')
                    else:
                        # Tip-to-tip connection
                        band = hv.Area((x_vals, grp['lower'], grp['upper']), vdims=['y', 'y2']).opts(alpha=0.15, color='gray')
                        mean_curve = hv.Curve((x_vals, grp['mean'])).opts(color='blue')
                    
                    error_bars = hv.ErrorBars((x_vals, grp['mean'], grp['std'])).opts(color='red', line_width=1.5)
                    
                    plot = band * mean_curve * error_bars * plot
        
        elif p_type == 'KDE / Histogram':
            plot = filtered.hvplot.kde(y=y_col, logy=is_log_y, **kwargs)

        else: # BoxPlot
            plot = filtered.hvplot.box(y=y_col, logy=is_log_y, **kwargs)

    except Exception as e:
        plot = hv.Text(0.5, 0.5, f"Could not generate plot:\n{str(e)}").opts(text_font_size='12pt')

    # 4. Means & Covariance Matrix Calculation
    stats_markdown = "### Statistical Summary (Current Filtered Subset)\nNot enough numeric data for covariance."
    if len(filtered) > 1:
        stats_df = filtered[[x_col, y_col]].dropna()
        if not stats_df.empty:
            mean_x, mean_y = stats_df.mean()
            cov_mat = stats_df.cov()
            corr_mat = stats_df.corr()
            
            stats_markdown = (
                f"### Statistical Summary (Current Filtered Subset)\n"
                f"* **Count:** {len(stats_df)} valid points\n"
                f"* **Mean {x_col}:** {mean_x:.5g}\n"
                f"* **Mean {y_col}:** {mean_y:.5g}\n\n"
                f"**Covariance Matrix:**\n"
                f"```text\n{cov_mat.to_string()}\n```\n"
                f"**Correlation (Pearson):** {corr_mat.iloc[0,1]:.4f}"
            )

    return pn.Column(
        pn.pane.HoloViews(plot, sizing_mode="stretch_width"),
        pn.layout.Divider(),
        pn.pane.Markdown(stats_markdown)
    )

on_file_change(file_selector.value)

sidebar = pn.Column(
    "### File & Presets",
    file_selector,
    preset_select,
    "---",
    "### Data Filters",
    custom_query,
    model_select,
    status_select,
    "---",
    "### Plot Controls",
    x_axis,
    y_axis,
    color_by,
    plot_type,
    pn.Row(log_x, log_y),
    "---",
    "### Advanced Overlays",
    show_ref_line,
    pn.Row(ref_slope, ref_intercept),
    show_error_bands
)

dashboard = pn.template.BootstrapTemplate(
    title="Benchmark Explorer Dashboard",
    sidebar=[sidebar],
    main=[
        pn.Column(
            "## Data Visualization & Exploration",
            pn.panel(update_view, sizing_mode="stretch_width"),
            "---",
            "### Trial Data Table",
            tabulator_table
        )
    ]
)

if __name__ == "__main__":
    pn.serve(dashboard, port=args.port, show=True)
