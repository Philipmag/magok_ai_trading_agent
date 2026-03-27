"""
AI Trading Agent - Main Module.

Event-driven trading system that:
1. Ingests data (shipping, market, news)
2. Detects events
3. Builds features
4. Makes ML-based predictions
5. Selects and executes strategies
6. Manages risk
7. Logs all decisions
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, get_log_path, system_settings
from config.assets import ACTIVE_SYMBOLS
from data.ingest import get_data_ingestion, CombinedDataSnapshot
from events.detector import get_event_detector, DetectedEvent
from events.rules import get_rule_engine
from features.builder import get_feature_builder, FeatureVector
from ml.predict import get_prediction_engine, TradingDecision
from strategies.momentum import get_momentum_strategy
from strategies.latency import get_latency_strategy
from strategies.mean_reversion import get_mean_reversion_strategy
from risk.manager import get_risk_manager, TradeRequest
from execution.broker import get_broker
from execution.orders import OrderSide


class TradingLogger:
    """Structured JSON logging."""
    
    def __init__(self, log_file: str = "trading.log"):
        self.log_file = get_log_path(log_file)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger."""
        logger = logging.getLogger("AITradingAgent")
        logger.setLevel(getattr(logging, settings.log_level))
        
        # File handler
        if settings.log_to_file:
            fh = logging.FileHandler(self.log_file)
            fh.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(fh)
        
        # Console handler
        if settings.log_to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
        
        return logger
    
    def log_event(self, level: str, event_type: str, data: Dict):
        """Log an event in structured format."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        getattr(self.logger, level.lower())(json.dumps(log_entry))
    
    def log_data(self, data: CombinedDataSnapshot):
        """Log data ingestion."""
        self.log_event("INFO", "data_ingestion", {
            "market_sentiment": data.market_sentiment,
            "shipping_index": data.shipping_data.global_shipping_index,
            "active_events": len(data.active_events)
        })
    
    def log_events_detected(self, events: List[DetectedEvent]):
        """Log detected events."""
        for event in events:
            self.log_event("INFO", "event_detected", {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity,
                "symbol": event.symbol
            })
    
    def log_decision(self, decision: TradingDecision):
        """Log trading decision."""
        self.log_event("INFO", "trading_decision", {
            "symbol": decision.symbol,
            "strategy": decision.recommended_strategy,
            "direction": decision.direction,
            "confidence": decision.confidence,
            "should_trade": decision.should_trade,
            "reasons": decision.reasons
        })
    
    def log_trade(self, trade_result: Dict):
        """Log executed trade."""
        self.log_event("INFO", "trade_executed", trade_result)
    
    def log_risk_metrics(self, metrics):
        """Log risk metrics."""
        self.log_event("INFO", "risk_metrics", {
            "capital": metrics.total_capital,
            "daily_pnl": metrics.daily_pnl,
            "open_positions": metrics.open_positions,
            "risk_level": metrics.risk_level.value
        })


class AITradingAgent:
    """
    Main AI Trading Agent.
    
    Orchestrates the complete trading pipeline:
    Data -> Events -> Features -> ML -> Strategy -> Risk -> Execution
    """
    
    def __init__(self, symbols: List[str] = None, use_mock: bool = True):
        self.symbols = symbols or ACTIVE_SYMBOLS
        self.use_mock = use_mock
        
        # Initialize components
        self.logger = TradingLogger()
        self.data_ingestion = get_data_ingestion(self.symbols, use_mock)
        self.event_detector = get_event_detector(self.symbols)
        self.rule_engine = get_rule_engine()
        self.feature_builder = get_feature_builder(self.symbols)
        self.prediction_engine = get_prediction_engine(settings.model_confidence_threshold)
        
        # Strategies
        self.momentum_strategy = get_momentum_strategy()
        self.latency_strategy = get_latency_strategy()
        self.mean_reversion_strategy = get_mean_reversion_strategy()
        
        # Risk and execution
        self.risk_manager = get_risk_manager(settings.initial_capital)
        self.broker = get_broker(settings.initial_capital)
        
        # State
        self.running = False
        self.loop_count = 0
        
        self.logger.logger.info("AI Trading Agent initialized")
    
    def start(self, duration_seconds: int = None):
        """Start the trading loop."""
        self.running = True
        start_time = time.time()
        
        self.logger.logger.info(f"Starting trading loop (duration: {duration_seconds or 'indefinite'}s)")
        
        try:
            while self.running:
                self.loop_count += 1
                loop_start = time.time()
                
                # Run one iteration
                self._run_iteration()
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    self.logger.logger.info("Duration limit reached, stopping")
                    break
                
                # Sleep for interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, system_settings.loop_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.logger.info("Received interrupt, stopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading loop."""
        self.running = False
        
        # Log final state
        metrics = self.risk_manager.get_metrics()
        self.logger.log_risk_metrics(metrics)
        
        self.logger.logger.info("Trading Agent stopped")
    
    def _run_iteration(self):
        """Run one iteration of the trading loop."""
        iteration_start = time.time()
        
        # 1. Fetch data
        data = self.data_ingestion.fetch_all()
        self.logger.log_data(data)
        
        # 2. Detect events
        raw_events = self.event_detector.detect_all(data)
        filtered_events = self.rule_engine.apply_rules(raw_events)
        self.logger.log_events_detected(filtered_events)
        
        # 3. Build features
        features = self.feature_builder.build_features(data)
        
        # 4. Make predictions
        decisions = self.prediction_engine.make_decisions(features)
        
        # 5. Select and execute strategies
        for symbol, decision in decisions.items():
            self.logger.log_decision(decision)
            
            if decision.should_trade:
                self._execute_strategy(symbol, decision, features[symbol])
        
        # 6. Check existing positions for exits
        self._check_position_exits(data)
        
        # 7. Log metrics
        if self.loop_count % 10 == 0:  # Every 10 iterations
            metrics = self.risk_manager.get_metrics()
            self.logger.log_risk_metrics(metrics)
        
        iteration_time = time.time() - iteration_start
        self.logger.logger.debug(f"Iteration {self.loop_count} completed in {iteration_time:.2f}s")
    
    def _execute_strategy(self, symbol: str, decision: TradingDecision, features: FeatureVector):
        """Execute a strategy based on decision."""
        # Get strategy signal
        signal = None
        strategy_name = decision.recommended_strategy
        
        if strategy_name == "momentum":
            signal = self.momentum_strategy.analyze(features)
        elif strategy_name == "latency":
            signal = self.latency_strategy.analyze(features)
        elif strategy_name == "mean_reversion":
            signal = self.mean_reversion_strategy.analyze(features)
        
        if signal is None:
            return
        
        # Create trade request
        direction = "long" if decision.direction == "long" else "short"
        quantity = signal.position_size / signal.entry_price
        
        request = TradeRequest(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=strategy_name,
            confidence=signal.confidence
        )
        
        # Risk check
        response = self.risk_manager.can_open_position(request)
        
        if not response.approved:
            self.logger.logger.info(f"Trade rejected for {symbol}: {response.reason}")
            return
        
        # Execute via broker
        order_side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        
        result = self.broker.place_market_order(
            symbol=symbol,
            side=order_side,
            quantity=response.adjusted_quantity,
            strategy=strategy_name,
            stop_loss=response.adjusted_stop_loss,
            take_profit=response.adjusted_take_profit
        )
        
        if result.success:
            # Record position in risk manager
            self.risk_manager.open_position(response, request)
            
            self.logger.log_trade({
                "order_id": result.order_id,
                "symbol": symbol,
                "direction": direction,
                "quantity": result.quantity,
                "price": result.executed_price,
                "strategy": strategy_name,
                "confidence": signal.confidence
            })
    
    def _check_position_exits(self, data: CombinedDataSnapshot):
        """Check if any positions should be exited."""
        positions = self.risk_manager.get_open_positions()
        
        for position in positions:
            quote = data.quotes.get(position.symbol)
            if not quote:
                continue
            
            current_price = quote.close
            
            # Check exit conditions
            exit_reason = self.risk_manager.check_position_exits(
                position.symbol,
                current_price,
                position.stop_loss,
                position.take_profit
            )
            
            if exit_reason:
                # Close position
                trade_result = self.risk_manager.close_position(position.symbol, exit_reason)
                
                if trade_result:
                    self.logger.log_trade({
                        "symbol": position.symbol,
                        "exit_reason": exit_reason,
                        "pnl": trade_result['pnl'],
                        "pnl_pct": trade_result['pnl_pct']
                    })


def run_demo(duration_seconds: int = 60):
    """Run a demo of the trading agent."""
    print("=" * 60)
    print("AI Trading Agent - Demo Mode")
    print("=" * 60)
    print()
    
    agent = AITradingAgent(use_mock=True)
    
    print(f"Initial Capital: ${settings.initial_capital:,.2f}")
    print(f"Max Trades: {settings.max_concurrent_trades}")
    print(f"Loss Cap: {settings.daily_loss_cap:.1%}")
    print(f"Loop Interval: {system_settings.loop_interval}s")
    print()
    
    agent.start(duration_seconds=duration_seconds)
    
    # Final report
    print()
    print("=" * 60)
    print("Final Report")
    print("=" * 60)
    
    metrics = agent.risk_manager.get_metrics()
    print(f"Final Capital: ${metrics.total_capital:,.2f}")
    print(f"Total P&L: ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl_pct:.2%})")
    print(f"Total Trades: {len(agent.risk_manager.trade_history)}")
    print(f"Open Positions: {metrics.open_positions}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")


if __name__ == "__main__":
    # Run demo for 60 seconds
    run_demo(duration_seconds=60)
