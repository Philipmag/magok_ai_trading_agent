"""
Risk Management Module.

Manages risk controls and position sizing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, date
from enum import Enum

from config.settings import settings


class RiskLevel(Enum):
    """Risk level assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class Position:
    """Represents an open position."""
    position_id: str
    symbol: str
    direction: str  # "long", "short"
    entry_price: float
    quantity: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    def update(self, current_price: float):
        """Update position with current price."""
        self.current_price = current_price
        
        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    total_capital: float
    available_capital: float
    used_capital: float
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown: float
    open_positions: int
    risk_level: RiskLevel
    exposure: float  # % of capital in positions


@dataclass
class TradeRequest:
    """Request to open a new position."""
    symbol: str
    direction: str
    quantity: float
    price: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float


@dataclass
class TradeResponse:
    """Response to a trade request."""
    approved: bool
    reason: str
    adjusted_quantity: float = 0.0
    adjusted_stop_loss: float = 0.0
    adjusted_take_profit: float = 0.0


class RiskManager:
    """
    Manages risk controls for trading.
    
    Enforces:
    - Max 1% capital per trade
    - Max 3 concurrent trades
    - Daily loss cap (3%)
    - Position sizing rules
    - Stop loss and take profit requirements
    """
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or settings.initial_capital
        self.capital = self.initial_capital
        self.open_positions: Dict[str, Position] = {}
        self.daily_start_capital = self.capital
        self.daily_start_time = datetime.now()
        self.max_drawdown = 0.0
        self.peak_capital = self.capital
        self.trade_history: List[Dict] = []
        
    def can_open_position(self, request: TradeRequest) -> TradeResponse:
        """Check if a new position can be opened."""
        # Check 1: Max concurrent trades
        if len(self.open_positions) >= settings.max_concurrent_trades:
            return TradeResponse(
                approved=False,
                reason=f"Maximum concurrent positions ({settings.max_concurrent_trades}) reached"
            )
        
        # Check 2: Daily loss cap
        daily_pnl = self.capital - self.daily_start_capital
        daily_pnl_pct = daily_pnl / self.daily_start_capital if self.daily_start_capital else 0
        
        if daily_pnl_pct <= -settings.daily_loss_cap:
            return TradeResponse(
                approved=False,
                reason=f"Daily loss cap ({settings.daily_loss_cap:.1%}) reached"
            )
        
        # Check 3: Capital available
        position_value = request.quantity * request.price
        max_capital_for_trade = self.capital * settings.max_capital_per_trade
        
        if position_value > max_capital_for_trade:
            # Adjust quantity to fit capital limits
            max_qty = max_capital_for_trade / request.price
            if max_qty < request.quantity * 0.5:
                return TradeResponse(
                    approved=False,
                    reason="Insufficient capital for minimum position size"
                )
            return TradeResponse(
                approved=True,
                reason="Quantity adjusted to meet capital limits",
                adjusted_quantity=max_qty,
                adjusted_stop_loss=request.stop_loss,
                adjusted_take_profit=request.take_profit
            )
        
        # Check 4: Symbol not already traded
        if request.symbol in self.open_positions:
            return TradeResponse(
                approved=False,
                reason=f"Position already open for {request.symbol}"
            )
        
        # Check 5: Risk per trade
        risk_amount = abs(request.entry_price - request.stop_loss) * request.quantity
        risk_pct = risk_amount / self.capital
        
        if risk_pct > settings.max_capital_per_trade * 2:
            return TradeResponse(
                approved=False,
                reason=f"Risk per trade ({risk_pct:.2%}) exceeds limit"
            )
        
        # All checks passed
        return TradeResponse(
            approved=True,
            reason="Trade approved",
            adjusted_quantity=request.quantity,
            adjusted_stop_loss=request.stop_loss,
            adjusted_take_profit=request.take_profit
        )
    
    def open_position(self, response: TradeResponse, request: TradeRequest) -> Position:
        """Open a new position."""
        position_id = f"POS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{request.symbol}"
        
        position = Position(
            position_id=position_id,
            symbol=request.symbol,
            direction=request.direction,
            entry_price=request.price,
            quantity=response.adjusted_quantity,
            current_price=request.price,
            stop_loss=response.adjusted_stop_loss,
            take_profit=response.adjusted_take_profit,
            entry_time=datetime.now(),
            strategy=request.strategy
        )
        
        self.open_positions[request.symbol] = position
        
        # Deduct capital
        position_value = position.quantity * position.entry_price
        self.capital -= position_value
        
        return position
    
    def close_position(self, symbol: str, reason: str = "manual") -> Optional[Dict]:
        """Close an existing position."""
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        
        # Calculate P&L
        pnl = position.unrealized_pnl
        pnl_pct = position.unrealized_pnl_pct
        
        # Record trade
        trade_record = {
            "position_id": position.position_id,
            "symbol": symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": position.current_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "strategy": position.strategy,
            "close_reason": reason
        }
        
        self.trade_history.append(trade_record)
        
        # Return capital
        position_value = position.quantity * position.current_price
        self.capital += position_value + pnl
        
        # Update peak capital and drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Remove position
        del self.open_positions[symbol]
        
        return trade_record
    
    def check_position_exits(self, symbol: str, current_price: float, 
                            stop_loss: float, take_profit: float,
                            slippage_bps: float = 5.0) -> Optional[str]:
        """
        Check if position should be exited.
        
        Args:
            symbol: Symbol to check
            current_price: Current market price
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            slippage_bps: Slippage in basis points (default 5 bps = 0.05%)
        
        Returns exit reason if should exit, None otherwise.
        """
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        
        # Apply slippage to fill price calculation
        slippage_factor = slippage_bps / 10000.0  # Convert bps to decimal
        
        # For stop loss exits, slippage works against us
        # For long positions: stop loss filled at lower price
        # For short positions: stop loss filled at higher price
        if position.direction == "long":
            effective_stop = stop_loss * (1 - slippage_factor)
            effective_target = take_profit * (1 - slippage_factor)  # Take profit also has slippage
            if current_price <= effective_stop:
                return "stop_loss"
            if current_price >= effective_target:
                return "take_profit"
        else:  # short
            effective_stop = stop_loss * (1 + slippage_factor)
            effective_target = take_profit * (1 + slippage_factor)
            if current_price >= effective_stop:
                return "stop_loss"
            if current_price <= effective_target:
                return "take_profit"
        
        return None
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        daily_pnl = self.capital - self.daily_start_capital
        daily_pnl_pct = daily_pnl / self.daily_start_capital if self.daily_start_capital else 0
        
        used_capital = sum(
            p.quantity * p.current_price 
            for p in self.open_positions.values()
        )
        
        exposure = used_capital / self.capital if self.capital > 0 else 0
        
        # Determine risk level
        if len(self.open_positions) >= settings.max_concurrent_trades:
            risk_level = RiskLevel.HIGH
        elif daily_pnl_pct < -settings.daily_loss_cap * 0.5:
            risk_level = RiskLevel.MEDIUM
        elif daily_pnl_pct < 0:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.LOW
        
        return RiskMetrics(
            total_capital=self.capital,
            available_capital=self.capital - used_capital,
            used_capital=used_capital,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            max_drawdown=self.max_drawdown,
            open_positions=len(self.open_positions),
            risk_level=risk_level,
            exposure=exposure
        )
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.open_positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.open_positions.get(symbol)
    
    def reset_daily(self):
        """Reset daily tracking (called at start of new trading day)."""
        self.daily_start_capital = self.capital
        self.daily_start_time = datetime.now()
    
    def get_trade_history(self, days: int = 30) -> List[Dict]:
        """Get trade history."""
        return self.trade_history[-days * 100:]  # Approximate


# Singleton
_risk_manager = None

def get_risk_manager(initial_capital: float = None) -> RiskManager:
    """Get or create the risk manager singleton."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(initial_capital)
    return _risk_manager
