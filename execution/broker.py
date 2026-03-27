"""
Broker Execution Module.

Paper trading broker simulation.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from config.settings import settings
from .orders import Order, OrderType, OrderSide, OrderStatus, get_order_manager


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    executed_price: float
    executed_time: datetime
    commission: float
    message: str


@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    cash: float
    equity: float
    buying_power: float
    day_trades: int = 0


class PaperBroker:
    """
    Paper trading broker.
    
    Simulates order execution with:
    - Realistic fill prices (with slippage)
    - Commission calculation
    - Market simulation
    - Position tracking
    """
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or settings.initial_capital
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position data
        self.order_manager = get_order_manager()
        
        # Market simulation state
        self._last_prices: Dict[str, float] = {}
        
        # Account info
        self.account_id = f"PAPER-{datetime.now().strftime('%Y%m%d')}"
    
    def get_account_info(self) -> AccountInfo:
        """Get current account information."""
        total_equity = self.cash + self._calculate_positions_value()
        
        return AccountInfo(
            account_id=self.account_id,
            cash=round(self.cash, 2),
            equity=round(total_equity, 2),
            buying_power=round(self.cash, 2)  # In paper trading, buying power = cash
        )
    
    def _calculate_positions_value(self) -> float:
        """Calculate total value of open positions."""
        total = 0.0
        for symbol, pos in self.positions.items():
            current_price = self._last_prices.get(symbol, pos['entry_price'])
            if pos['direction'] == 'long':
                total += pos['quantity'] * current_price
            else:
                # Short position: profit when price decreases
                total += pos['quantity'] * (pos['entry_price'] - current_price) + pos['quantity'] * pos['entry_price']
        return total
    
    def place_market_order(self, symbol: str, side: OrderSide, quantity: float,
                          strategy: str = "", stop_loss: float = 0.0,
                          take_profit: float = 0.0) -> ExecutionResult:
        """Place a market order."""
        # Get current price (simulated)
        current_price = self._get_simulated_price(symbol)
        
        # Create order
        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Simulate order execution
        return self._execute_order(order, current_price)
    
    def place_limit_order(self, symbol: str, side: OrderSide, quantity: float,
                         limit_price: float, strategy: str = "",
                         stop_loss: float = 0.0, take_profit: float = 0.0) -> ExecutionResult:
        """Place a limit order."""
        current_price = self._get_simulated_price(symbol)
        
        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=limit_price,
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Check if limit price is favorable
        if side == OrderSide.BUY and limit_price >= current_price:
            return self._execute_order(order, current_price)
        elif side == OrderSide.SELL and limit_price <= current_price:
            return self._execute_order(order, current_price)
        
        # Order not filled (limit not reached)
        self.order_manager.submit_order(order)
        
        return ExecutionResult(
            success=True,
            order_id=order.order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            executed_price=0.0,
            executed_time=datetime.now(),
            commission=0.0,
            message="Limit order placed, waiting for fill"
        )
    
    def _execute_order(self, order: Order, market_price: float) -> ExecutionResult:
        """Execute an order with simulated slippage."""
        # Calculate execution price with slippage
        slippage = market_price * random.uniform(-0.0005, 0.0005)  # 0.05% slippage
        
        if order.side == OrderSide.BUY:
            execution_price = market_price + abs(slippage)
        else:
            execution_price = market_price - abs(slippage)
        
        # Calculate commission (mock)
        commission = order.quantity * execution_price * 0.001  # 0.1% commission
        
        # Check if we have enough cash (for buys)
        total_cost = order.quantity * execution_price + commission
        
        if order.side == OrderSide.BUY and total_cost > self.cash:
            order.reject("Insufficient funds")
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                executed_price=0.0,
                executed_time=datetime.now(),
                commission=0.0,
                message="Insufficient funds"
            )
        
        # Execute the order
        order.fill(order.quantity, execution_price)
        order.commission = commission
        self.order_manager.submit_order(order)
        
        # Update cash and positions
        if order.side == OrderSide.BUY:
            self.cash -= (order.quantity * execution_price + commission)
            
            # Add to position
            if order.symbol in self.positions:
                old_pos = self.positions[order.symbol]
                new_qty = old_pos['quantity'] + order.quantity
                avg_price = (old_pos['entry_price'] * old_pos['quantity'] + 
                           execution_price * order.quantity) / new_qty
                self.positions[order.symbol] = {
                    'direction': 'long',
                    'quantity': new_qty,
                    'entry_price': avg_price
                }
            else:
                self.positions[order.symbol] = {
                    'direction': 'long',
                    'quantity': order.quantity,
                    'entry_price': execution_price
                }
        else:  # SELL
            self.cash += (order.quantity * execution_price - commission)
            
            # Update or close position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                if pos['quantity'] <= order.quantity:
                    del self.positions[order.symbol]
                else:
                    pos['quantity'] -= order.quantity
        
        self._last_prices[order.symbol] = execution_price
        
        return ExecutionResult(
            success=True,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            executed_price=round(execution_price, 2),
            executed_time=datetime.now(),
            commission=round(commission, 2),
            message="Order filled"
        )
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated market price for a symbol."""
        if symbol in self._last_prices:
            # Add small random movement
            last = self._last_prices[symbol]
            change = last * random.uniform(-0.001, 0.001)
            return last + change
        else:
            # Default prices for initialization
            default_prices = {
                "FDX": 250.0, "UPS": 180.0, "CHRW": 140.0,
                "DAL": 55.0, "IYT": 220.0
            }
            return default_prices.get(symbol, 100.0)
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        return self.positions.copy()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        return self._get_simulated_price(symbol)
    
    def close_position(self, symbol: str, quantity: float = None) -> ExecutionResult:
        """Close a position (market order)."""
        if symbol not in self.positions:
            return ExecutionResult(
                success=False,
                order_id="",
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=0,
                executed_price=0.0,
                executed_time=datetime.now(),
                commission=0.0,
                message="No position to close"
            )
        
        pos = self.positions[symbol]
        close_qty = quantity or pos['quantity']
        
        # Determine side to close
        if pos['direction'] == 'long':
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY
        
        current_price = self._get_simulated_price(symbol)
        return self.place_market_order(symbol, side, close_qty)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update simulated prices from market data."""
        for symbol, price in prices.items():
            self._last_prices[symbol] = price


# Singleton
_broker = None

def get_broker(initial_capital: float = None) -> PaperBroker:
    """Get or create the paper broker singleton."""
    global _broker
    if _broker is None:
        _broker = PaperBroker(initial_capital)
    return _broker
