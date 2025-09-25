import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ========= BALANCED ORB STRATEGY PARAMETERS =========
# Based on research with balanced approach for better performance

# Opening Range Parameters (BALANCED)
ORB_START_TIME = "09:30"  # Market open
ORB_END_TIME = "09:45"    # 15-minute opening range (research shows 15-30 min optimal)
ORB_EXTENSION_MINUTES = 15  # Additional minutes to extend range if needed

# Breakout Parameters (BALANCED)
BREAKOUT_CONFIRMATION_BARS = 1  # Number of bars to confirm breakout
BREAKOUT_FILTER_PERCENT = 0.0   # Minimum percentage above/below range for valid breakout
MIN_BREAKOUT_DISTANCE = 0.5     # Reduced minimum points for valid breakout

# Risk Management (BALANCED)
STOP_LOSS_ATR_MULTIPLIER = 1.8  # Balanced ATR multiplier (research shows 1.2-1.8 optimal)
TAKE_PROFIT_RATIO = 2.2         # Balanced risk:reward ratio (research shows 2:1 minimum)
POSITION_SIZE = 1000            # Position size in dollars
MAX_DAILY_TRADES = 2            # Reduced from 3 (research shows fewer trades = better quality)
MAX_CONCURRENT_POSITIONS = 1    # Maximum concurrent positions

# Time Filters (BALANCED)
TRADING_START_TIME = "14:30"    # Start trading time (9:30 AM EST = 14:30 UTC)
TRADING_END_TIME = "21:00"      # End trading time (4:00 PM EST = 21:00 UTC)
AVOID_FIRST_MINUTES = 5         # Reduced to 5 minutes after ORB

# Backtest Parameters
INITIAL_CAPITAL = 100000        # Starting capital
COMMISSION_PER_TRADE = 2.50     # Commission per trade
SLIPPAGE_POINTS = 0.5           # Slippage in points

# Data Parameters
DATA_FILE = "oanda_NAS100_USD_M1_2023-01-01_to_2025-12-31.csv"
SYMBOL = "NAS100_USD"

# Additional Filters (BALANCED)
USE_VOLUME_FILTER = False       # Disable volume filter for now
MIN_VOLUME_MULTIPLIER = 1.1     # Minimum volume must be 1.1x average
USE_TREND_FILTER = False        # Disable trend filter for now
TREND_MA_PERIOD = 20            # Moving average period for trend filter
MIN_TREND_STRENGTH = 0.05       # Minimum trend strength (0.05 = 5% above/below MA)

# ========= BALANCED ORB STRATEGY CLASSES =========

class BalancedORBStrategy:
    """Balanced Opening Range Breakout Strategy Implementation"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.positions = []
        self.trades = []
        self.daily_stats = {}
        self.current_position = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        
    def calculate_opening_range(self, df: pd.DataFrame, date: str) -> Tuple[float, float, int, int]:
        """
        Calculate opening range high and low for a specific date
        
        Args:
            df: DataFrame with OHLCV data
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Tuple of (range_high, range_low, start_idx, end_idx)
        """
        # Filter data for the specific date
        date_data = df[df.index.date == pd.to_datetime(date).date()]
        
        if date_data.empty:
            return None, None, None, None
            
        # Get opening range time window
        start_time = pd.to_datetime(f"{date} {self.params['orb_start_time']}").time()
        end_time = pd.to_datetime(f"{date} {self.params['orb_end_time']}").time()
        
        # Find opening range period
        orb_mask = (date_data.index.time >= start_time) & (date_data.index.time <= end_time)
        orb_data = date_data[orb_mask]
        
        if orb_data.empty:
            return None, None, None, None
            
        range_high = orb_data['high'].max()
        range_low = orb_data['low'].min()
        start_idx = orb_data.index[0]
        end_idx = orb_data.index[-1]
        
        return range_high, range_low, start_idx, end_idx
    
    def check_breakout(self, df: pd.DataFrame, current_idx: int, range_high: float, range_low: float) -> str:
        """
        Check if current bar represents a valid breakout
        
        Args:
            df: DataFrame with OHLCV data
            current_idx: Current bar index
            range_high: Opening range high
            range_low: Opening range low
            
        Returns:
            'long', 'short', or 'none'
        """
        current_bar = df.iloc[current_idx]
        
        # Check for long breakout (above range high)
        if current_bar['close'] > range_high:
            # Apply breakout filter
            breakout_distance = current_bar['close'] - range_high
            if breakout_distance >= self.params['min_breakout_distance']:
                return 'long'
        
        # Check for short breakout (below range low)
        elif current_bar['close'] < range_low:
            # Apply breakout filter
            breakout_distance = range_low - current_bar['close']
            if breakout_distance >= self.params['min_breakout_distance']:
                return 'short'
        
        return 'none'
    
    def calculate_atr(self, df: pd.DataFrame, current_idx: int, period: int = 14) -> float:
        """Calculate Average True Range"""
        if current_idx < period:
            return 0.0
            
        recent_data = df.iloc[current_idx-period+1:current_idx+1]
        
        high_low = recent_data['high'] - recent_data['low']
        high_close = np.abs(recent_data['high'] - recent_data['close'].shift(1))
        low_close = np.abs(recent_data['low'] - recent_data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.mean()
        
        return atr
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        risk_per_trade = self.params['position_size']
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
            
        position_size = int(risk_per_trade / risk_per_share)
        return max(1, min(position_size, 1000))  # Cap at 1000 shares
    
    def enter_position(self, direction: str, entry_price: float, entry_time: pd.Timestamp, 
                      range_high: float, range_low: float, atr: float) -> bool:
        """Enter a new position"""
        if self.current_position is not None:
            return False  # Already in a position
            
        if self.daily_trade_count >= self.params['max_daily_trades']:
            return False  # Daily trade limit reached
            
        # Calculate stop loss and take profit
        if direction == 'long':
            stop_loss = range_low - (atr * self.params['stop_loss_atr_multiplier'])
            take_profit = entry_price + (abs(entry_price - stop_loss) * self.params['take_profit_ratio'])
        else:  # short
            stop_loss = range_high + (atr * self.params['stop_loss_atr_multiplier'])
            take_profit = entry_price - (abs(entry_price - stop_loss) * self.params['take_profit_ratio'])
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        if position_size == 0:
            return False
            
        # Create position
        self.current_position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'range_high': range_high,
            'range_low': range_low
        }
        
        return True
    
    def check_exit_conditions(self, current_bar: pd.Series, current_time: pd.Timestamp) -> Optional[str]:
        """Check if current position should be exited"""
        if self.current_position is None:
            return None
            
        direction = self.current_position['direction']
        stop_loss = self.current_position['stop_loss']
        take_profit = self.current_position['take_profit']
        
        # Check stop loss
        if direction == 'long' and current_bar['low'] <= stop_loss:
            return 'stop_loss'
        elif direction == 'short' and current_bar['high'] >= stop_loss:
            return 'stop_loss'
        
        # Check take profit
        if direction == 'long' and current_bar['high'] >= take_profit:
            return 'take_profit'
        elif direction == 'short' and current_bar['low'] <= take_profit:
            return 'take_profit'
        
        # Check end of day exit
        end_time = pd.to_datetime(f"{current_time.date()} {self.params['trading_end_time']}").time()
        if current_time.time() >= end_time:
            return 'end_of_day'
        
        return None
    
    def exit_position(self, exit_price: float, exit_time: pd.Timestamp, exit_reason: str) -> Dict:
        """Exit current position and record trade"""
        if self.current_position is None:
            return None
            
        position = self.current_position
        direction = position['direction']
        entry_price = position['entry_price']
        position_size = position['position_size']
        
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - exit_price) * position_size
        
        # Subtract commission and slippage
        total_costs = self.params['commission_per_trade'] + (self.params['slippage_points'] * position_size)
        pnl -= total_costs
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'range_high': position['range_high'],
            'range_low': position['range_low'],
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit']
        }
        
        self.trades.append(trade)
        self.current_position = None
        self.daily_trade_count += 1
        
        return trade


class BalancedORBBacktester:
    """Balanced backtesting engine for ORB strategy"""
    
    def __init__(self, strategy: BalancedORBStrategy):
        self.strategy = strategy
        self.results = {}
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the balanced backtest on historical data"""
        print("Starting Balanced ORB Backtest...")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Total bars: {len(df)}")
        
        # Get unique trading days
        trading_days = sorted(pd.Series(df.index.date).unique())
        print(f"Trading days: {len(trading_days)}")
        
        # Process each trading day
        for day in trading_days:
            day_str = day.strftime('%Y-%m-%d')
            
            # Reset daily trade count
            if self.strategy.last_trade_date != day:
                self.strategy.daily_trade_count = 0
                self.strategy.last_trade_date = day
            
            # Calculate opening range for this day
            range_high, range_low, start_idx, end_idx = self.strategy.calculate_opening_range(df, day_str)
            
            if range_high is None or range_low is None:
                continue
                
            # Get data for this day after opening range
            day_data = df[df.index.date == day]
            trading_start_time = pd.to_datetime(f"{day_str} {self.strategy.params['trading_start_time']}").time()
            trading_end_time = pd.to_datetime(f"{day_str} {self.strategy.params['trading_end_time']}").time()
            
            # Filter for trading hours
            trading_mask = (day_data.index.time >= trading_start_time) & (day_data.index.time <= trading_end_time)
            trading_data = day_data[trading_mask]
            
            # Skip first few minutes after opening range
            avoid_start_time = pd.to_datetime(f"{day_str} {self.strategy.params['orb_end_time']}").time()
            avoid_end_time = (pd.to_datetime(f"{day_str} {self.strategy.params['orb_end_time']}") + 
                            pd.Timedelta(minutes=self.strategy.params['avoid_first_minutes'])).time()
            
            # Process each bar after opening range
            for idx in trading_data.index:
                if idx <= end_idx:
                    continue  # Skip opening range bars
                    
                # Skip first few minutes after opening range
                if avoid_start_time <= idx.time() <= avoid_end_time:
                    continue
                
                current_bar = df.loc[idx]
                
                # Check for exit conditions if in position
                if self.strategy.current_position is not None:
                    exit_reason = self.strategy.check_exit_conditions(current_bar, idx)
                    if exit_reason:
                        self.strategy.exit_position(current_bar['close'], idx, exit_reason)
                
                # Check for new entry if not in position
                if self.strategy.current_position is None:
                    breakout_direction = self.strategy.check_breakout(df, df.index.get_loc(idx), range_high, range_low)
                    if breakout_direction != 'none':
                        atr = self.strategy.calculate_atr(df, df.index.get_loc(idx))
                        success = self.strategy.enter_position(
                            breakout_direction, current_bar['close'], idx, 
                            range_high, range_low, atr
                        )
                        if success:
                            print(f"Entered {breakout_direction} position at {idx}: {current_bar['close']:.2f}")
        
        # Calculate final results
        self.calculate_results()
        return self.results
    
    def calculate_results(self):
        """Calculate backtest performance metrics"""
        if not self.strategy.trades:
            self.results = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            return
        
        trades_df = pd.DataFrame(self.strategy.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio (simplified)
        if len(trades_df) > 1:
            sharpe_ratio = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252) if trades_df['pnl'].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        self.results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades_df': trades_df
        }
    
    def print_results(self):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BALANCED ORB STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        if self.results['total_trades'] == 0:
            print("No trades executed during backtest period.")
            return
        
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Total P&L: ${self.results['total_pnl']:.2f}")
        print(f"Average Win: ${self.results['avg_win']:.2f}")
        print(f"Average Loss: ${self.results['avg_loss']:.2f}")
        print(f"Profit Factor: {self.results['profit_factor']:.2f}")
        print(f"Max Drawdown: ${self.results['max_drawdown']:.2f}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        
        # Show recent trades
        if 'trades_df' in self.results and not self.results['trades_df'].empty:
            print("\nRecent Trades:")
            recent_trades = self.results['trades_df'].tail(10)
            for _, trade in recent_trades.iterrows():
                print(f"{trade['entry_time']} | {trade['direction']} | "
                      f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | "
                      f"P&L: ${trade['pnl']:.2f} | Reason: {trade['exit_reason']}")


def main():
    """Main execution function"""
    print("Loading data...")
    
    # Load data
    try:
        df = pd.read_csv(DATA_FILE, index_col='time', parse_dates=True)
        print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    except FileNotFoundError:
        print(f"Data file {DATA_FILE} not found. Please ensure the file exists.")
        return
    
    # Create balanced strategy parameters dictionary
    strategy_params = {
        'orb_start_time': ORB_START_TIME,
        'orb_end_time': ORB_END_TIME,
        'orb_extension_minutes': ORB_EXTENSION_MINUTES,
        'breakout_confirmation_bars': BREAKOUT_CONFIRMATION_BARS,
        'breakout_filter_percent': BREAKOUT_FILTER_PERCENT,
        'min_breakout_distance': MIN_BREAKOUT_DISTANCE,
        'stop_loss_atr_multiplier': STOP_LOSS_ATR_MULTIPLIER,
        'take_profit_ratio': TAKE_PROFIT_RATIO,
        'position_size': POSITION_SIZE,
        'max_daily_trades': MAX_DAILY_TRADES,
        'max_concurrent_positions': MAX_CONCURRENT_POSITIONS,
        'trading_start_time': TRADING_START_TIME,
        'trading_end_time': TRADING_END_TIME,
        'avoid_first_minutes': AVOID_FIRST_MINUTES,
        'commission_per_trade': COMMISSION_PER_TRADE,
        'slippage_points': SLIPPAGE_POINTS,
        'use_volume_filter': USE_VOLUME_FILTER,
        'min_volume_multiplier': MIN_VOLUME_MULTIPLIER,
        'use_trend_filter': USE_TREND_FILTER,
        'trend_ma_period': TREND_MA_PERIOD,
        'min_trend_strength': MIN_TREND_STRENGTH
    }
    
    # Create strategy and backtester
    strategy = BalancedORBStrategy(strategy_params)
    backtester = BalancedORBBacktester(strategy)
    
    # Run backtest
    results = backtester.run_backtest(df)
    
    # Print results
    backtester.print_results()
    
    return results


if __name__ == "__main__":
    results = main()

