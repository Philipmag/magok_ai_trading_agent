"""
Database Module for State Persistence.

Handles SQLite database operations for position state persistence and recovery.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class PositionDatabase:
    """
    SQLite database for persisting trading positions.
    
    Provides:
    - Position storage on open/close
    - Recovery of open positions on restart
    - Trade history logging
    """
    
    def __init__(self, db_path: str = "db/positions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                current_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                entry_time TEXT NOT NULL,
                strategy TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                exit_price REAL,
                exit_time TEXT,
                exit_reason TEXT,
                pnl REAL,
                pnl_pct REAL
            )
        """)
        
        # Trade history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                strategy TEXT NOT NULL,
                close_reason TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Account state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                capital REAL NOT NULL,
                daily_start_capital REAL NOT NULL,
                daily_start_time TEXT NOT NULL,
                peak_capital REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_position(self, position: Dict) -> bool:
        """Save or update a position."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO positions 
                (position_id, symbol, direction, entry_price, quantity, 
                 current_price, stop_loss, take_profit, entry_time, strategy, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position['position_id'],
                position['symbol'],
                position['direction'],
                position['entry_price'],
                position['quantity'],
                position.get('current_price', position['entry_price']),
                position['stop_loss'],
                position['take_profit'],
                position['entry_time'].isoformat() if isinstance(position['entry_time'], datetime) else position['entry_time'],
                position['strategy'],
                'open'
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving position: {e}")
            return False
        finally:
            conn.close()
    
    def get_open_positions(self) -> List[Dict]:
        """Retrieve all open positions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM positions WHERE status = 'open'")
            rows = cursor.fetchall()
            
            positions = []
            for row in rows:
                pos_dict = dict(row)
                # Parse datetime fields
                if pos_dict.get('entry_time'):
                    pos_dict['entry_time'] = datetime.fromisoformat(pos_dict['entry_time'])
                positions.append(pos_dict)
            
            return positions
        except Exception as e:
            print(f"Error retrieving positions: {e}")
            return []
        finally:
            conn.close()
    
    def close_position(self, position_id: str, exit_price: float, 
                       exit_reason: str, pnl: float, pnl_pct: float) -> bool:
        """Mark a position as closed and log to history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            exit_time = datetime.now().isoformat()
            
            # Update position status
            cursor.execute("""
                UPDATE positions 
                SET status = 'closed',
                    exit_price = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    pnl = ?,
                    pnl_pct = ?
                WHERE position_id = ?
            """, (exit_price, exit_time, exit_reason, pnl, pnl_pct, position_id))
            
            # Get position details for history
            cursor.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,))
            row = cursor.fetchone()
            
            if row:
                pos_dict = dict(row)
                
                # Insert into trade history
                cursor.execute("""
                    INSERT INTO trade_history 
                    (position_id, symbol, direction, entry_price, exit_price, 
                     quantity, pnl, pnl_pct, entry_time, exit_time, strategy, close_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position_id,
                    pos_dict['symbol'],
                    pos_dict['direction'],
                    pos_dict['entry_price'],
                    exit_price,
                    pos_dict['quantity'],
                    pnl,
                    pnl_pct,
                    pos_dict['entry_time'],
                    exit_time,
                    pos_dict['strategy'],
                    exit_reason
                ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error closing position: {e}")
            return False
        finally:
            conn.close()
    
    def save_account_state(self, capital: float, daily_start_capital: float,
                          daily_start_time: datetime, peak_capital: float,
                          max_drawdown: float) -> bool:
        """Save account state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO account_state 
                (id, capital, daily_start_capital, daily_start_time, peak_capital, max_drawdown, last_updated)
                VALUES (1, ?, ?, ?, ?, ?, ?)
            """, (
                capital,
                daily_start_capital,
                daily_start_time.isoformat() if isinstance(daily_start_time, datetime) else daily_start_time,
                peak_capital,
                max_drawdown,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving account state: {e}")
            return False
        finally:
            conn.close()
    
    def load_account_state(self) -> Optional[Dict]:
        """Load account state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM account_state WHERE id = 1")
            row = cursor.fetchone()
            
            if row:
                state_dict = dict(row)
                if state_dict.get('daily_start_time'):
                    state_dict['daily_start_time'] = datetime.fromisoformat(state_dict['daily_start_time'])
                return state_dict
            return None
        except Exception as e:
            print(f"Error loading account state: {e}")
            return None
        finally:
            conn.close()
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM trade_history 
                ORDER BY exit_time DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error retrieving trade history: {e}")
            return []
        finally:
            conn.close()
    
    def clear_all_positions(self):
        """Clear all positions (for testing)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM positions")
            cursor.execute("DELETE FROM trade_history")
            cursor.execute("DELETE FROM account_state")
            conn.commit()
        finally:
            conn.close()


# Singleton instance
_db_instance = None


def get_position_database(db_path: str = "db/positions.db") -> PositionDatabase:
    """Get or create the position database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = PositionDatabase(db_path)
    return _db_instance
