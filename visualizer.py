"""
Portfolio Visualizer Module
============================
Creates interactive (Plotly) and static (Matplotlib) visualizations for
portfolio risk analysis, including Monte Carlo paths, return distributions,
VaR comparisons, correlation heatmaps, and efficient frontier plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union


class PortfolioVisualizer:
    """Creates interactive and static visualizations for portfolio risk analysis.

    Supports two rendering backends:
        - Interactive (Plotly): For web dashboards and Streamlit apps.
        - Static (Matplotlib/Seaborn): For reports and PDF export.

    Args:
        style: Plotly template name (default: 'plotly_white').
    """

    def __init__(self, style: str = 'plotly_white') -> None:
        self.style = style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_price_paths(
        self,
        price_paths: np.ndarray,
        asset_names: list[str],
        n_paths_to_show: int = 100,
        interactive: bool = True
    ) -> Union[go.Figure, plt.Figure]:
        """Plot Monte Carlo simulated price paths for each asset.

        Shows a random subset of simulation paths with the mean path
        overlaid in red for each asset in a stacked subplot layout.

        Args:
            price_paths: 3D array (n_simulations, n_days+1, n_assets).
            asset_names: List of asset name strings.
            n_paths_to_show: Number of paths to render (for readability).
            interactive: If True, returns Plotly figure; else Matplotlib.

        Returns:
            Plotly or Matplotlib figure object.
        """
        n_sims, n_days, n_assets = price_paths.shape

        if interactive:
            fig = make_subplots(
                rows=n_assets, cols=1,
                subplot_titles=[f"{name} Price Paths" for name in asset_names],
                vertical_spacing=0.08
            )
            colors = px.colors.qualitative.Set1

            for asset_idx, asset_name in enumerate(asset_names):
                paths_to_show = min(n_paths_to_show, n_sims)
                selected_paths = np.random.choice(n_sims, paths_to_show, replace=False)

                for i, path_idx in enumerate(selected_paths):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(n_days + 1)),
                            y=price_paths[path_idx, :, asset_idx],
                            mode='lines',
                            line=dict(width=0.5, color=colors[asset_idx % len(colors)]),
                            opacity=0.3,
                            showlegend=True if i == 0 else False,
                            name=f"{asset_name} Paths",
                            hovertemplate=f"<b>{asset_name}</b><br>Day: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>"
                        ),
                        row=asset_idx + 1, col=1
                    )

                mean_path = np.mean(price_paths[:, :, asset_idx], axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_days + 1)),
                        y=mean_path,
                        mode='lines',
                        line=dict(width=3, color='red'),
                        name=f"{asset_name} Mean",
                        hovertemplate=f"<b>{asset_name} Mean</b><br>Day: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>"
                    ),
                    row=asset_idx + 1, col=1
                )

            fig.update_layout(
                title="Monte Carlo Price Path Simulations",
                height=300 * n_assets,
                showlegend=True,
                template=self.style
            )
            fig.update_xaxes(title_text="Days")
            fig.update_yaxes(title_text="Price ($)")
            return fig

        else:
            fig, axes = plt.subplots(n_assets, 1, figsize=(12, 4 * n_assets))
            if n_assets == 1:
                axes = [axes]

            for asset_idx, asset_name in enumerate(asset_names):
                ax = axes[asset_idx]
                paths_to_show = min(n_paths_to_show, n_sims)
                for i in range(paths_to_show):
                    ax.plot(price_paths[i, :, asset_idx], alpha=0.1, color='blue', linewidth=0.5)
                mean_path = np.mean(price_paths[:, :, asset_idx], axis=0)
                ax.plot(mean_path, color='red', linewidth=2, label='Mean Path')
                ax.set_title(f"{asset_name} Price Paths")
                ax.set_xlabel("Days")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

    def plot_portfolio_distribution(
        self,
        portfolio_returns: np.ndarray,
        var_results: dict[str, dict],
        interactive: bool = True
    ) -> Union[go.Figure, plt.Figure]:
        """Plot portfolio return distribution with VaR thresholds.

        Renders a histogram of simulated returns with vertical dashed lines
        marking the VaR levels at each confidence level.

        Args:
            portfolio_returns: Array of simulated portfolio returns.
            var_results: VaR results dictionary.
            interactive: If True, returns Plotly figure; else Matplotlib.

        Returns:
            Plotly or Matplotlib figure object.
        """
        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=portfolio_returns.flatten(),
                nbinsx=100,
                name="Portfolio Returns",
                opacity=0.7,
                marker_color='lightblue'
            ))
            colors = ['red', 'darkred', 'orange', 'darkorange']
            color_idx = 0
            for key, var_data in var_results.items():
                if var_data['horizon_days'] == 1:
                    var_value = var_data['var_percentage']
                    confidence = var_data['confidence_level']
                    fig.add_vline(
                        x=var_value,
                        line_dash="dash",
                        line_color=colors[color_idx % len(colors)],
                        annotation_text=f"VaR {confidence:.0%}: {abs(var_value):.2%}",
                        annotation_position="top"
                    )
                    color_idx += 1
            fig.update_layout(
                title="Portfolio Return Distribution with VaR Estimates",
                xaxis_title="Portfolio Returns",
                yaxis_title="Frequency",
                template=self.style,
                showlegend=True
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(portfolio_returns.flatten(), bins=100, alpha=0.7, color='lightblue', edgecolor='black')
            colors = ['red', 'darkred', 'orange', 'darkorange']
            color_idx = 0
            for key, var_data in var_results.items():
                if var_data['horizon_days'] == 1:
                    var_value = var_data['var_percentage']
                    confidence = var_data['confidence_level']
                    ax.axvline(var_value, color=colors[color_idx % len(colors)],
                              linestyle='--', linewidth=2,
                              label=f"VaR {confidence:.0%}: {abs(var_value):.2%}")
                    color_idx += 1
            ax.set_title("Portfolio Return Distribution with VaR Estimates")
            ax.set_xlabel("Portfolio Returns")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
            return fig

    def plot_var_comparison(
        self, var_results: dict[str, dict], interactive: bool = True
    ) -> Union[go.Figure, plt.Figure]:
        """Create side-by-side VaR comparison bar charts.

        Shows VaR in both dollar and percentage terms for each
        confidence level and time horizon combination.

        Args:
            var_results: VaR results dictionary.
            interactive: If True, returns Plotly figure; else Matplotlib.

        Returns:
            Plotly or Matplotlib figure object.
        """
        var_data = []
        for key, data in var_results.items():
            var_data.append({
                'Key': key,
                'Horizon': f"{data['horizon_days']} Day{'s' if data['horizon_days'] > 1 else ''}",
                'Confidence': f"{data['confidence_level']:.0%}",
                'VaR_Dollar': abs(data['var_dollar']),
                'VaR_Percentage': abs(data['var_percentage']) * 100,
                'Expected_Shortfall': abs(data['expected_shortfall'])
            })
        df = pd.DataFrame(var_data)

        if interactive:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['VaR by Dollar Amount', 'VaR by Percentage'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            fig.add_trace(
                go.Bar(
                    x=df['Key'],
                    y=df['VaR_Dollar'],
                    name='VaR ($)',
                    marker_color='lightcoral',
                    text=[f"${x:,.0f}" for x in df['VaR_Dollar']],
                    textposition='auto'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=df['Key'],
                    y=df['VaR_Percentage'],
                    name='VaR (%)',
                    marker_color='lightblue',
                    text=[f"{x:.1f}%" for x in df['VaR_Percentage']],
                    textposition='auto'
                ),
                row=1, col=2
            )
            fig.update_layout(
                title="Value-at-Risk Comparison",
                template=self.style,
                height=500
            )
            fig.update_xaxes(title_text="VaR Scenario", row=1, col=1)
            fig.update_xaxes(title_text="VaR Scenario", row=1, col=2)
            fig.update_yaxes(title_text="VaR ($)", row=1, col=1)
            fig.update_yaxes(title_text="VaR (%)", row=1, col=2)
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            bars1 = ax1.bar(df['Key'], df['VaR_Dollar'], color='lightcoral', alpha=0.7)
            ax1.set_title('VaR by Dollar Amount')
            ax1.set_xlabel('VaR Scenario')
            ax1.set_ylabel('VaR ($)')
            ax1.tick_params(axis='x', rotation=45)
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}', ha='center', va='bottom')
            bars2 = ax2.bar(df['Key'], df['VaR_Percentage'], color='lightblue', alpha=0.7)
            ax2.set_title('VaR by Percentage')
            ax2.set_xlabel('VaR Scenario')
            ax2.set_ylabel('VaR (%)')
            ax2.tick_params(axis='x', rotation=45)
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            plt.tight_layout()
            return fig

    def plot_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        asset_names: list[str],
        interactive: bool = True
    ) -> Union[go.Figure, plt.Figure]:
        """Plot asset correlation matrix as a heatmap.

        Args:
            correlation_matrix: 2D correlation matrix.
            asset_names: List of asset name labels.
            interactive: If True, returns Plotly figure; else Matplotlib.

        Returns:
            Plotly or Matplotlib figure object.
        """
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=asset_names,
                y=asset_names,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False,
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
            ))
            fig.update_layout(
                title="Asset Correlation Matrix",
                template=self.style,
                height=500,
                width=500
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                       center=0, square=True, linewidths=0.5,
                       xticklabels=asset_names, yticklabels=asset_names, ax=ax)
            ax.set_title('Asset Correlation Matrix')
            return fig

    def plot_efficient_frontier(
        self,
        returns_data: pd.DataFrame,
        n_portfolios: int = 10000,
        interactive: bool = True
    ) -> Union[go.Figure, plt.Figure]:
        """Plot the efficient frontier via random portfolio sampling.

        Generates random weight allocations, computes annualized risk-return
        for each, and plots the cloud with Sharpe ratio color coding.
        Points on the efficient frontier (non-dominated) are highlighted.

        Args:
            returns_data: DataFrame of asset log returns.
            n_portfolios: Number of random portfolios to sample.
            interactive: If True, returns Plotly figure; else Matplotlib.

        Returns:
            Plotly or Matplotlib figure object.
        """
        n_assets = len(returns_data.columns)
        weights = np.random.random((n_portfolios, n_assets))
        weights = weights / weights.sum(axis=1)[:, np.newaxis]

        mean_returns = returns_data.mean()
        cov_matrix = returns_data.cov()

        # Vectorized portfolio return/risk calculation
        portfolio_returns = np.dot(weights, mean_returns) * 252
        portfolio_risks = np.array([
            np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
            for w in weights
        ])

        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_risks * 100,
                y=portfolio_returns * 100,
                mode='markers',
                marker=dict(
                    size=3,
                    color=portfolio_returns / portfolio_risks,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Random Portfolios',
                hovertemplate="Risk: %{x:.1f}%<br>Return: %{y:.1f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>"
            ))

            # Identify efficient frontier points (non-dominated)
            efficient_idx = []
            for i in range(len(portfolio_returns)):
                is_efficient = True
                for j in range(len(portfolio_returns)):
                    if (portfolio_returns[j] >= portfolio_returns[i] and
                        portfolio_risks[j] < portfolio_risks[i]):
                        is_efficient = False
                        break
                if is_efficient:
                    efficient_idx.append(i)

            if efficient_idx:
                fig.add_trace(go.Scatter(
                    x=portfolio_risks[efficient_idx] * 100,
                    y=portfolio_returns[efficient_idx] * 100,
                    mode='markers',
                    marker=dict(size=6, color='red'),
                    name='Efficient Frontier',
                    hovertemplate="Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"
                ))

            fig.update_layout(
                title="Portfolio Efficient Frontier",
                xaxis_title="Risk (Annual Volatility %)",
                yaxis_title="Return (Annual %)",
                template=self.style
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sharpe_ratios = portfolio_returns / portfolio_risks
            scatter = ax.scatter(portfolio_risks * 100, portfolio_returns * 100,
                               c=sharpe_ratios, cmap='viridis', alpha=0.6)
            ax.set_xlabel('Risk (Annual Volatility %)')
            ax.set_ylabel('Return (Annual %)')
            ax.set_title('Portfolio Efficient Frontier')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Sharpe Ratio')
            return fig

    def create_dashboard_summary(
        self,
        var_results: dict[str, dict],
        risk_metrics: dict[str, float],
        portfolio_value: float
    ) -> dict:
        """Compile dashboard-level summary of key risk indicators.

        Args:
            var_results: VaR calculation results.
            risk_metrics: Portfolio risk metrics.
            portfolio_value: Current portfolio value.

        Returns:
            Dictionary with key metrics for dashboard display.
        """
        one_day_95_var = None
        five_day_95_var = None
        for key, data in var_results.items():
            if "1d_95%" in key:
                one_day_95_var = abs(data['var_dollar'])
            elif "5d_95%" in key:
                five_day_95_var = abs(data['var_dollar'])

        summary = {
            'portfolio_value': portfolio_value,
            'annual_return': risk_metrics['annual_return'],
            'annual_volatility': risk_metrics['annual_volatility'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'max_drawdown': risk_metrics['max_drawdown'],
            'skewness': risk_metrics['skewness'],
            'kurtosis': risk_metrics['kurtosis'],
            'one_day_var_95': one_day_95_var,
            'five_day_var_95': five_day_95_var,
            'var_percentage_of_portfolio': (one_day_95_var / portfolio_value * 100) if one_day_95_var else 0
        }
        return summary