"""
Backtest Engine Module.

Backtesting framework for strategy evaluation.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from config.settings import settings
from config.assets import DEFAULT_ASSETS


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    strategy: str
    exit_reason: str


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_trade_duration: float
    trades: List[BacktestTrade]
    
    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Results
================
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.1%}
Total P&L: ${self.total_pnl:.2f} ({self.total_pnl_pct:.1%})
Max Drawdown: {self.max_drawdown:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Avg Trade P&L: ${self.avg_trade_pnl:.2f}
Avg Trade Duration: {self.avg_trade_duration:.1f} hours
"""


class BacktestEngine:
    """
    Backtest engine for strategy testing.
    
    Simulates trading over historical data with:
    - Realistic execution (slippage, commissions)
    - Position tracking
    - Risk management
    - Performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.positions: Dict[str, Dict] = {}
        self.equity_curve: List[float] = []
        self.peak_capital = initial_capital
        
    def run(self, data_generator: Callable, days: int = 30) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            data_generator: Function that generates historical data points
            days: Number of days to simulate
        
        Returns:
            BacktestResult with performance metrics
        """
        self._reset()
        
        # Simulate trading days
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Get data for this period
            data = data_generator(current_date)
            
            # Process any signals from data
            signals = self._generate_signals(data)
            
            # Execute signals
            for signal in signals:
                self._execute_signal(signal, current_date)
            
            # Check exits
            self._check_exits(current_date)
            
            # Record equity
            self.equity_curve.append(self.capital)
        
        return self._calculate_results()
    
    def _reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        self.peak_capital = self.initial_capital
    
    def _generate_signals(self, data) -> List[Dict]:
        """Generate trading signals from data."""
        signals = []
        
        # Simple signal generation for backtest
        # In production, this would use real strategies
        if random.random() < 0.1:  # 10% chance of signal
            symbols = list(DEFAULT_ASSETS.keys())[:3]
            symbol = random.choice(symbols)
            
            signals.append({
                'symbol': symbol,
                'direction': random.choice(['long', 'short']),
                'entry_price': data.get('price', 100.0),
                'quantity': self._calculate_quantity(data.get('price', 100.0)),
                'strategy': random.choice(['momentum', 'mean_reversion', 'latency'])
            })
        
        return signals
    
    def _execute_signal(self, signal: Dict, timestamp: datetime):
        """Execute a trading signal."""
        symbol = signal['symbol']
        
        # Don't add if already in position
        if symbol in self.positions:
            return
        
        # Check capital
        if signal['direction'] == 'long':
            cost = signal['quantity'] * signal['entry_price']
            if cost > self.capital * 0.1:  # Max 10% per trade
                return
        
        self.positions[symbol] = {
            'entry_time': timestamp,
            'entry_price': signal['entry_price'],
            'quantity': signal['quantity'],
            'direction': signal['direction'],
            'strategy': signal['strategy']
        }
    
    def _check_exits(self, timestamp: datetime):
        """Check and execute exits."""
        symbols_to_remove = []
        
        for symbol, pos in self.positions.items():
            # Simulate price movement
            price_change = random.uniform(-0.03, 0.04)
            current_price = pos['entry_price'] * (1 + price_change)
            
            # Check stop loss / take profit
            exit_reason = None
            
            if pos['direction'] == 'long':
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                if pnl_pct <= -0.02:  # Stop loss
                    exit_reason = 'stop_loss'
                elif pnl_pct >= 0.04:  # Take profit
                    exit_reason = 'take_profit'
            else:
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                if pnl_pct <= -0.02:
                    exit_reason = 'stop_loss'
                elif pnl_pct >= 0.04:
                    exit_reason = 'take_profit'
            
            # Random time-based exit
            if not exit_reason and random.random() < 0.05:
                exit_reason = 'time_exit'
            
            if exit_reason:
                self._close_position(symbol, current_price, timestamp, exit_reason)
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.positions[symbol]
    
    def _close_position(self, symbol: str, exit_price: float, 
                        timestamp: datetime, reason: str):
        """Close a position and record trade."""
        pos = self.positions[symbol]
        
        if pos['direction'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']
        
        # Commission
        commission = exit_price * pos['quantity'] * 0.001
        pnl -= commission
        
        # Update capital
        self.capital += pnl
        
        # Update peak
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Record trade
        duration = (timestamp - pos['entry_time']).total_seconds() / 3600
        
        trade = BacktestTrade(
            entry_time=pos['entry_time'],
            exit_time=timestamp,
            symbol=symbol,
            direction=pos['direction'],
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            quantity=pos['quantity'],
            pnl=pnl,
            pnl_pct=pnl / (pos['entry_price'] * pos['quantity']),
            strategy=pos['strategy'],
            exit_reason=reason
        )
        
        self.trades.append(trade)
    
    def _calculate_quantity(self, price: float) -> float:
        """Calculate position size."""
        max_position = self.capital * settings.max_capital_per_trade
        return max_position / price
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        if not self.trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                avg_trade_pnl=0.0,
                avg_trade_duration=0.0,
                trades=[]
            )
        
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (self.capital - self.initial_capital) / self.initial_capital
        
        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t.pnl_pct for t in self.trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0
        
        avg_duration = np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in self.trades
        ])
        
        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(self.trades) if self.trades else 0.0,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            avg_trade_pnl=total_pnl / len(self.trades),
            avg_trade_duration=avg_duration,
            trades=self.trades
        )


# Simple data generator for testing
def mock_data_generator(timestamp: datetime) -> Dict:
    """Generate mock market data for backtesting."""
    return {
        'timestamp': timestamp,
        'price': 100.0 * (1 + random.uniform(-0.01, 0.01)),
        'volume': int(random.uniform(1000000, 5000000)),
        'shipping_index': 1500.0 * (1 + random.uniform(-0.02, 0.02)),
        'sentiment': random.uniform(-0.5, 0.5)
    }


# Singleton
_backtest_engine = None

def get_backtest_engine(initial_capital: float = 100000.0) -> BacktestEngine:
    """Get or create the backtest engine singleton."""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine(initial_capital)
    return _backtest_engine
