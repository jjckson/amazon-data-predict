"""Prompt builder for trading agent market state."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

PROMPT_TEMPLATE_HEADER = "🧭 SYSTEM STATE"
PROMPT_MARKET_STATE_TITLE = "⚙️ CURRENT MARKET STATE"
PROMPT_ACCOUNT_TITLE = "💰 ACCOUNT INFORMATION"
PROMPT_PERFORMANCE_TITLE = "📊 PERFORMANCE METRICS"


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        formatted = f"{value:.8f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return str(value)


def _format_series(series: Iterable[Any]) -> str:
    items = [item for item in series]
    if not items:
        return "[]"
    return "[" + ", ".join(_format_value(item) for item in items) + "]"


def build_trading_prompt(
    system_state: Mapping[str, Any],
    market_states: Sequence[Mapping[str, Any]],
    account_info: Mapping[str, Any],
    *,
    performance_metrics: Mapping[str, Any] | None = None,
) -> str:
    """Render a textual prompt describing the trading environment."""

    performance_metrics = performance_metrics or {}
    lines: list[str] = []

    lines.append(PROMPT_TEMPLATE_HEADER)
    minutes = _format_value(system_state.get("minutes_since_start"))
    current_time = _format_value(system_state.get("current_time"))
    invocation_count = _format_value(system_state.get("invocation_count"))
    lines.append(
        f"It has been {minutes} minutes since you started trading."
    )
    lines.append(
        f"The current time is {current_time} and you've been invoked {invocation_count} times."
    )
    lines.append("Below you are provided with price data, indicators, and account info for decision making.")
    lines.append("")

    lines.append(PROMPT_MARKET_STATE_TITLE)
    lines.append("For each coin:")
    for state in market_states:
        symbol = state.get("symbol", "UNKNOWN")
        lines.append(f"SYMBOL: {symbol}")
        lines.append(
            "current_price = {price}, current_ema20 = {ema}, current_macd = {macd}, current_rsi(7) = {rsi7}".format(
                price=_format_value(state.get("current_price")),
                ema=_format_value(state.get("current_ema20")),
                macd=_format_value(state.get("current_macd")),
                rsi7=_format_value(state.get("current_rsi_7")),
            )
        )
        lines.append("")
        lines.append(
            "Open Interest: Latest={latest}, Average={avg}".format(
                latest=_format_value(state.get("open_interest_latest")),
                avg=_format_value(state.get("open_interest_avg")),
            )
        )
        lines.append(f"Funding Rate: {_format_value(state.get('funding_rate'))}")
        lines.append("")

        short_term = state.get("short_term", {})
        short_interval = short_term.get("interval", "-")
        lines.append(f"Intraday Series ({short_interval} intervals, oldest → latest):")
        lines.append(f"Mid Prices: {_format_series(short_term.get('mid_prices', []))}")
        lines.append(f"EMA(20): {_format_series(short_term.get('ema_series_20', []))}")
        lines.append(f"MACD: {_format_series(short_term.get('macd_series', []))}")
        lines.append(f"RSI(7): {_format_series(short_term.get('rsi_series_7', []))}")
        lines.append(f"RSI(14): {_format_series(short_term.get('rsi_series_14', []))}")
        lines.append("")

        long_term = state.get("long_term", {})
        long_interval = long_term.get("interval", "-")
        lines.append(f"Longer-term ({long_interval} timeframe):")
        lines.append(
            "EMA(20): {ema20}, EMA(50): {ema50}".format(
                ema20=_format_value(long_term.get("ema_20_4h")),
                ema50=_format_value(long_term.get("ema_50_4h")),
            )
        )
        lines.append(
            "ATR(3): {atr3}, ATR(14): {atr14}".format(
                atr3=_format_value(long_term.get("atr_3_4h")),
                atr14=_format_value(long_term.get("atr_14_4h")),
            )
        )
        lines.append(
            "Volume: Current={current}, Average={avg}".format(
                current=_format_value(long_term.get("current_volume")),
                avg=_format_value(long_term.get("avg_volume")),
            )
        )
        lines.append(f"MACD(4h): {_format_series(long_term.get('macd_series_4h', []))}")
        lines.append(f"RSI(14,4h): {_format_series(long_term.get('rsi_series_14_4h', []))}")
        lines.append("")

    lines.append(PROMPT_ACCOUNT_TITLE)
    lines.append(
        f"Total Return: {_format_value(account_info.get('total_return_percent'))}%"
    )
    lines.append(f"Available Cash: {_format_value(account_info.get('available_cash'))}")
    lines.append(f"Account Value: {_format_value(account_info.get('account_value'))}")
    lines.append("")

    lines.append("Positions:")
    positions = account_info.get("positions", [])
    if positions:
        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            lines.append(
                "- {symbol}: qty={qty}, entry={entry}, current={current}, PnL={pnl}, leverage={leverage}".format(
                    symbol=symbol,
                    qty=_format_value(position.get("quantity")),
                    entry=_format_value(position.get("entry_price")),
                    current=_format_value(position.get("current_price")),
                    pnl=_format_value(position.get("unrealized_pnl")),
                    leverage=_format_value(position.get("leverage")),
                )
            )
            exit_plan = position.get("exit_plan", {})
            lines.append(
                "  TP={tp}, SL={sl}, invalidation={invalid}".format(
                    tp=_format_value(exit_plan.get("profit_target")),
                    sl=_format_value(exit_plan.get("stop_loss")),
                    invalid=_format_value(exit_plan.get("invalidation_condition")),
                )
            )
            lines.append(
                "  confidence={confidence}, risk={risk}, notional={notional}".format(
                    confidence=_format_value(position.get("confidence")),
                    risk=_format_value(position.get("risk_usd")),
                    notional=_format_value(position.get("notional_usd")),
                )
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append(PROMPT_PERFORMANCE_TITLE)
    lines.append(
        f"Sharpe Ratio: {_format_value(performance_metrics.get('sharpe_ratio'))}"
    )

    return "\n".join(lines)
