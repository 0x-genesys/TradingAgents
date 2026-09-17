import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    enforce_exact_tool_ticker,
    get_indicators,
    get_analyst_context,
    get_language_instruction,
    get_stock_data,
    sanitize_agent_output,
)
from tradingagents.dataflows.config import get_config


def malformed_market_report(report: str) -> bool:
    """Tool-call text is not a completed technical analysis."""
    return not report.strip() or bool(re.search(
        r"<tool_code>|<tool_call>|```(?:python|json)|"
        r"\b(?:print\s*\(|get_stock_data\s*\(|get_indicators\s*\()",
        report, re.IGNORECASE,
    ))


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        snapshot = state.get("sentiment_source_snapshot") or {}
        instrument_context = build_instrument_context(
            state["company_of_interest"],
            asset_type,
            canonical_name=snapshot.get("company_name"),
            lstm_context_available=False,
        )

        tools = [
            get_stock_data,
            get_indicators,
        ]

        ctx = get_analyst_context(state)
        ctx_line = f"\n\n---\nIMPORTANT CONTEXT — Trade parameters: {ctx}\nFrame ALL analysis for THIS specific trade horizon and fixed entry, target, and stop. Assess whether the target is reachable before the stop. Valuation multiples (P/E, EV/EBITDA) are largely irrelevant for short-term trades." if ctx else ""

        system_message = (
            ctx_line
            + """You are a technical market analyst. Produce evidence only. Do not recommend BUY, HOLD, or SELL, and do not make transaction proposals. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions. Explain current trend, pullback, confirmation, invalidation, volatility, and volume from observed market data only. Do not invent prior model runs or outcomes."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        tags = list(state.get("data_quality_tags") or [])
        if enforce_exact_tool_ticker(result, state["company_of_interest"]):
            tags.append("CORRECTED_TOOL_TICKER")

        if len(result.tool_calls) == 0:
            report = str(result.content or "").strip()
            if malformed_market_report(report):
                synthesis_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "The market analyst returned an empty or malformed final response. "
                            "Tool-code text is not an executed tool call. Using only the prior tool outputs, write the "
                            "complete technical market report now. Do not call tools and "
                            "do not invent unavailable values.\n{system_message}\n"
                            "{instrument_context}",
                        ),
                        MessagesPlaceholder(variable_name="messages"),
                        (
                            "human",
                            "Synthesize the final market report with a Markdown summary table.",
                        ),
                    ]
                ).partial(
                    system_message=system_message,
                    instrument_context=instrument_context,
                )
                repaired = (synthesis_prompt | llm).invoke(state["messages"])
                repaired_report = str(repaired.content or "").strip()
                if not repaired.tool_calls and not malformed_market_report(repaired_report):
                    report = repaired_report
                    result = repaired
                    tags.append("REPAIRED_MARKET_REPORT")
                else:
                    raise RuntimeError(
                        f"INVALID_MARKET_REPORT for {state['company_of_interest']}: "
                        "empty or tool-code response after one synthesis repair"
                    )

        report, output_tags = sanitize_agent_output(report, state)
        tags.extend(output_tags)

        return {
            "messages": [result],
            "market_report": report,
            "data_quality_tags": sorted(set(tags)),
        }

    return market_analyst_node
