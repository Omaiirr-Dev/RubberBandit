"""
AI Trading Brain — GPT-4o-mini powered micro-scalping assistant.
Called every ~60 seconds to analyze chart data and return BUY/SELL/HOLD.
Reads chart as structured text (not images) to keep costs minimal.
"""

import json
import time

import openai


SYSTEM_PROMPT = """You are an aggressive micro-scalping trading assistant for NVDA stock. You analyze price data and make BUY, SELL, or HOLD decisions.

STRATEGY:
- You are targeting $2-3 profit per trade on a ~$10,000 position (~74 shares at ~$135)
- That means you need only a $0.03-0.04/share price move in your favor
- Trade FREQUENTLY — you should be recommending BUY most of the time when not in a position
- A flat/stable market with small moves is IDEAL for scalping — low risk, easy to grab $2
- Buy at local lows/dips, even tiny ones. ANY slight dip is a buy opportunity.
- If price is flat or range-bound, just BUY — it will likely bounce enough for $2
- Sell quickly once you see $2+ profit, or if momentum fades
- If losing money, recommend SELL if loss exceeds $5

READING THE CHART:
- Focus on the last 5-10 minutes (recent action matters most)
- If price is stable/flat within a few cents: BUY — this is low-risk scalping territory
- If price just dipped even 1-2 cents: BUY — that's your entry
- If price is at the low end of its recent range: BUY
- Only HOLD if price is crashing hard (5+ consecutive drops, each > $0.02)

RULES:
- Respond ONLY with valid JSON: {"action": "BUY" or "SELL" or "HOLD", "reason": "brief 10 words max", "confidence": 0.0 to 1.0}
- You need at least 2 minutes of data before your first BUY recommendation
- When NOT in a position: recommend BUY unless there's a strong reason not to
- When IN a position: only recommend SELL or HOLD
- DEFAULT TO BUY when not in a position. HOLD should be rare.
- Only avoid BUY during a clear crash (5+ consecutive large drops)
- If we have $2+ unrealized profit, SELL immediately to lock it in"""


class AIBrain:
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.last_scan_time = 0.0
        self.last_recommendation = "HOLD"
        self.last_reason = "Initializing..."
        self.last_confidence = 0.0
        self.scan_count = 0
        self.first_scan_time = 0.0

    async def analyze(self, chart_data: dict) -> dict:
        """
        Analyze chart data and return AI recommendation.

        chart_data keys:
            prices_30min: list of (time_str, price) tuples
            current_price, support_floor, resistance_ceiling, vwap
            ema_fast, ema_slow, trend
            in_position, entry_price, unrealized_pnl, position_value
            cash, minutes_scanning
        """
        try:
            prompt = self._format_prompt(chart_data)

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )

            text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(text)

            action = result.get("action", "HOLD").upper()
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"

            self.last_recommendation = action
            self.last_reason = result.get("reason", "")[:80]
            self.last_confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            self.last_scan_time = time.time()
            self.scan_count += 1

            return {
                "action": action,
                "reason": self.last_reason,
                "confidence": self.last_confidence,
            }

        except Exception as e:
            print(f"[AI Brain] Error: {e}")
            self.last_reason = f"Error: {str(e)[:50]}"
            self.last_scan_time = time.time()
            return {"action": "HOLD", "reason": self.last_reason, "confidence": 0.0}

    def _format_prompt(self, data: dict) -> str:
        """Format chart data as structured text for the AI."""
        lines = []
        lines.append(f"=== NVDA Micro-Scalp Analysis ===")
        lines.append(f"Current price: ${data['current_price']:.2f}")
        lines.append(f"Support: ${data['support_floor']:.2f} | Resistance: ${data['resistance_ceiling']:.2f}")
        lines.append(f"VWAP: ${data['vwap']:.2f}")
        lines.append(f"EMA fast: ${data['ema_fast']:.2f} | EMA slow: ${data['ema_slow']:.2f}")
        lines.append(f"Trend: {data['trend'].upper()}")
        lines.append(f"Scanning for: {data['minutes_scanning']:.0f} minutes")
        lines.append("")

        if data["in_position"]:
            lines.append(f">>> IN POSITION: bought at ${data['entry_price']:.2f}")
            lines.append(f">>> Unrealized P&L: ${data['unrealized_pnl']:.2f}")
            lines.append(f">>> Position value: ${data['position_value']:.2f}")
        else:
            lines.append(f">>> NO POSITION | Cash: ${data['cash']:.2f}")
        lines.append("")

        prices = data.get("prices_30min", [])
        if prices:
            lines.append(f"Last {len(prices)} price samples:")
            for time_str, price in prices:
                lines.append(f"  {time_str}  ${price:.2f}")

        return "\n".join(lines)

    def get_state(self) -> dict:
        """Return current AI state for frontend display."""
        return {
            "ai_recommendation": self.last_recommendation,
            "ai_reason": self.last_reason,
            "ai_confidence": round(self.last_confidence, 2),
            "ai_last_scan": self.last_scan_time,
            "ai_scan_count": self.scan_count,
        }
