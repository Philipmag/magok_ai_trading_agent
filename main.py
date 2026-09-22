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

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, get_log_path, system_settings
from config.assets import ACTIVE_SYMBOLS
from data.ingest import get_data_ingestion, CombinedDataSnapshot
from events.detector import get_event_detector, DetectedEvent
from events.rules import get_rule_engine
from features.builder import get_feature_builder, FeatureVector
from ml.predict import get_prediction_engine, TradingDecision
from ml.trainer import get_model_trainer, ModelTrainer
from strategies.momentum import get_momentum_strategy
from strategies.latency import get_latency_strategy
from strategies.mean_reversion import get_mean_reversion_strategy
from risk.manager import get_risk_manager, TradeRequest, Position
from execution.broker import get_broker
from execution.orders import OrderSide
from db.database import get_position_database


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
        
        # ML Trainer for weekly retraining
        self.ml_trainer = get_model_trainer(lookback_days=90)
        
        # Strategies
        self.momentum_strategy = get_momentum_strategy()
        self.latency_strategy = get_latency_strategy()
        self.mean_reversion_strategy = get_mean_reversion_strategy()
        
        # Risk and execution
        self.risk_manager = get_risk_manager(settings.initial_capital)
        self.broker = get_broker(settings.initial_capital)
        
        # Database for state persistence
        self.db = get_position_database()
        
        # State
        self.running = False
        self.loop_count = 0
        self.last_train_time = datetime.now()  # Track last training time
        
        # Recover state from database on restart
        self._recover_state()
        
        self.logger.logger.info("AI Trading Agent initialized")
    
    def _recover_state(self):
        """Recover open positions and account state from database."""
        try:
            # Load account state
            account_state = self.db.load_account_state()
            if account_state:
                self.risk_manager.capital = account_state.get('capital', self.risk_manager.initial_capital)
                self.risk_manager.daily_start_capital = account_state.get('daily_start_capital', self.risk_manager.capital)
                if account_state.get('daily_start_time'):
                    self.risk_manager.daily_start_time = account_state['daily_start_time']
                self.risk_manager.peak_capital = account_state.get('peak_capital', self.risk_manager.capital)
                self.risk_manager.max_drawdown = account_state.get('max_drawdown', 0.0)
                self.logger.logger.info(f"Recovered account state: capital=${self.risk_manager.capital:,.2f}")
            
            # Recover open positions from database
            open_positions = self.db.get_open_positions()
            for pos_dict in open_positions:
                # Reconstruct Position object
                position = Position(
                    position_id=pos_dict['position_id'],
                    symbol=pos_dict['symbol'],
                    direction=pos_dict['direction'],
                    entry_price=pos_dict['entry_price'],
                    quantity=pos_dict['quantity'],
                    current_price=pos_dict.get('current_price', pos_dict['entry_price']),
                    stop_loss=pos_dict['stop_loss'],
                    take_profit=pos_dict['take_profit'],
                    entry_time=pos_dict['entry_time'] if isinstance(pos_dict['entry_time'], datetime) else datetime.fromisoformat(pos_dict['entry_time']),
                    strategy=pos_dict['strategy']
                )
                # Add to risk manager's open positions
                self.risk_manager.open_positions[pos_dict['symbol']] = position
                self.logger.logger.info(f"Recovered position: {pos_dict['symbol']} ({pos_dict['direction']})")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to recover state: {e}")
    
    def start(self, duration_seconds: int = None):
        """Start the trading loop using asyncio."""
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        self.logger.logger.info(f"Starting trading loop (duration: {duration_seconds or 'indefinite'}s)")
        
        # Run async loop
        asyncio.run(self._run_async_loop(duration_seconds))
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on SIGTERM/SIGINT."""
        self.logger.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def _run_async_loop(self, duration_seconds: int = None):
        """Async trading loop with concurrent data stream tasks."""
        import time
        
        start_time = time.time()
        
        try:
            while self.running:
                self.loop_count += 1
                loop_start = time.time()
                
                # Run one iteration
                await self._run_iteration_async()
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    self.logger.logger.info("Duration limit reached, stopping")
                    break
                
                # Sleep for interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, system_settings.loop_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        except asyncio.CancelledError:
            self.logger.logger.info("Loop cancelled, stopping...")
        finally:
            await self._stop_async()
    
    async def _stop_async(self):
        """Stop the trading loop asynchronously."""
        self.running = False
        
        # Persist final state to database
        self._persist_state()
        
        # Log final state
        metrics = self.risk_manager.get_metrics()
        self.logger.log_risk_metrics(metrics)
        
        self.logger.logger.info("Trading Agent stopped")
    
    def _persist_state(self):
        """Persist current state to database."""
        try:
            self.db.save_account_state(
                capital=self.risk_manager.capital,
                daily_start_capital=self.risk_manager.daily_start_capital,
                daily_start_time=self.risk_manager.daily_start_time,
                peak_capital=self.risk_manager.peak_capital,
                max_drawdown=self.risk_manager.max_drawdown
            )
        except Exception as e:
            self.logger.logger.error(f"Failed to persist state: {e}")
    
    def _check_and_retrain_model(self):
        """Check if weekly retraining is due and run training cycle."""
        from datetime import timedelta
        
        # Check if 7 days have passed since last training
        if datetime.now() - self.last_train_time >= timedelta(days=7):
            self.logger.logger.info("Starting weekly model retraining...")
            model_path = self.ml_trainer.run_training_cycle(self.db)
            if model_path:
                self.last_train_time = datetime.now()
                self.logger.logger.info(f"Model retrained successfully: {model_path}")
                
                # Reload prediction engine with new model
                self.prediction_engine.load_model(model_path)
            else:
                self.logger.logger.warning("Model retraining failed or skipped (insufficient data)")
    
    async def _run_iteration_async(self):
        """Run one iteration of the trading loop asynchronously."""
        import time
        
        iteration_start = time.time()
        
        # 1. Fetch data (with error handling and exponential backoff)
        try:
            data = await self.data_ingestion.fetch_all()
            self.logger.log_data(data)
        except Exception as e:
            self.logger.logger.error(f"Data fetch failed after retries: {e}")
            return
        
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
                await self._execute_strategy_async(symbol, decision, features[symbol])
        
        # 6. Check existing positions for exits
        await self._check_position_exits_async(data)
        
        # 7. Log metrics
        if self.loop_count % 10 == 0:  # Every 10 iterations
            metrics = self.risk_manager.get_metrics()
            self.logger.log_risk_metrics(metrics)
        
        # 8. Persist state to database
        self._persist_state()
        
        iteration_time = time.time() - iteration_start
        self.logger.logger.debug(f"Iteration {self.loop_count} completed in {iteration_time:.2f}s")
    
    async def _execute_strategy_async(self, symbol: str, decision: TradingDecision, features: FeatureVector):
        """Execute a strategy based on decision (async version)."""
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
        
        # Create trade request with FIXED quantity rounding
        direction = "long" if decision.direction == "long" else "short"
        quantity = max(1, int(signal.position_size / signal.entry_price))  # FIX #1: Quantity rounding
        
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
            position = self.risk_manager.open_position(response, request)
            
            # Persist position to database
            self.db.save_position({
                'position_id': position.position_id,
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'current_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'entry_time': position.entry_time,
                'strategy': position.strategy
            })
            
            self.logger.log_trade({
                "order_id": result.order_id,
                "symbol": symbol,
                "direction": direction,
                "quantity": result.quantity,
                "price": result.executed_price,
                "strategy": strategy_name,
                "confidence": signal.confidence
            })
    
    async def _check_position_exits_async(self, data: CombinedDataSnapshot):
        """Check if any positions should be exited (async version)."""
        positions = self.risk_manager.get_open_positions()
        
        for position in positions:
            quote = data.quotes.get(position.symbol)
            if not quote:
                continue
            
            current_price = quote.close
            
            # Check exit conditions WITH SLIPPAGE (FIX #6)
            exit_reason = self.risk_manager.check_position_exits(
                position.symbol,
                current_price,
                position.stop_loss,
                position.take_profit,
                slippage_bps=5.0  # 5 basis points slippage
            )
            
            if exit_reason:
                # Close position
                trade_result = self.risk_manager.close_position(position.symbol, exit_reason)
                
                if trade_result:
                    # Update position in database
                    self.db.close_position(
                        position_id=position.position_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl=trade_result['pnl'],
                        pnl_pct=trade_result['pnl_pct']
                    )
                    
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
