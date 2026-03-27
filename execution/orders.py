"""
Order Management Module.

Order types and order management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float  # Limit price or market price
    stop_price: float = 0.0  # For stop orders
    
    # Order metadata
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    filled_time: Optional[datetime] = None
    
    # Strategy info
    strategy: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Execution info
    commission: float = 0.0
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    def fill(self, quantity: float, price: float):
        """Fill portion of the order."""
        self.filled_quantity += quantity
        self.average_fill_price = (
            (self.average_fill_price * (self.filled_quantity - quantity) + price * quantity)
            / self.filled_quantity if self.filled_quantity > 0 else price
        )
        self.updated_time = datetime.now()
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_time = datetime.now()
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self):
        """Cancel the order."""
        if self.is_active:
            self.status = OrderStatus.CANCELLED
            self.updated_time = datetime.now()
    
    def reject(self, reason: str = ""):
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.updated_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "created_time": self.created_time.isoformat(),
            "filled_time": self.filled_time.isoformat() if self.filled_time else None,
            "strategy": self.strategy,
            "commission": self.commission
        }


class OrderManager:
    """
    Manages orders and tracks order lifecycle.
    """
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    quantity: float, price: float, stop_price: float = 0.0,
                    strategy: str = "", stop_loss: float = 0.0, 
                    take_profit: float = 0.0) -> Order:
        """Create a new order."""
        order = Order(
            order_id=f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.orders[order.order_id] = order
        return order
    
    def submit_order(self, order: Order) -> bool:
        """Submit order for execution."""
        if order.status != OrderStatus.PENDING:
            return False
        
        order.status = OrderStatus.SUBMITTED
        order.updated_time = datetime.now()
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        active = [o for o in self.orders.values() if o.is_active]
        
        if symbol:
            active = [o for o in active if o.symbol == symbol]
        
        return active
    
    def get_orders_for_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [o for o in self.orders.values() if o.symbol == symbol]
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        order.cancel()
        self.order_history.append(order)
        return True
    
    def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel all active orders."""
        count = 0
        for order in list(self.orders.values()):
            if order.is_active:
                if symbol is None or order.symbol == symbol:
                    order.cancel()
                    self.order_history.append(order)
                    count += 1
        return count
    
    def archive_order(self, order: Order):
        """Archive completed order."""
        if order.order_id in self.orders:
            del self.orders[order.order_id]
        self.order_history.append(order)
    
    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Get order history."""
        return self.order_history[-limit:]
    
    def get_total_commission(self) -> float:
        """Get total commission paid."""
        return sum(o.commission for o in self.order_history)


# Singleton
_order_manager = None

def get_order_manager() -> OrderManager:
    """Get or create the order manager singleton."""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
