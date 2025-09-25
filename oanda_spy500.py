import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
import pandas as pd

# ========= Account & API Setup =========
ACCOUNT_ID = "101-003-29297410-001"
ACCESS_TOKEN = "e489cd7ce9f304a660ac932fa5083807-29050e51018a56c3f829751f6e6c87e9"
timeframe = "M5"
symbol = "SPY500"
start_date = "2023-01-01"
end_date = "2025-09-24"

# OANDA API Configuration
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"  # Practice account
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}


class CandlestickData:
    """Model class for candlestick data structure"""
    
    def __init__(self, time: str, open: float, high: float, low: float, close: float, volume: int = 0):
        self.time = time
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def to_dict(self) -> Dict:
        return {
            'time': self.time,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
    
    def __repr__(self):
        return f"CandlestickData(time={self.time}, open={self.open}, high={self.high}, low={self.low}, close={self.close})"


class OandaDataFetcher:
    """Business logic class for fetching OANDA data"""
    
    def __init__(self, account_id: str, access_token: str, base_url: str = OANDA_BASE_URL):
        self.account_id = account_id
        self.access_token = access_token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def convert_to_rfc3339(self, date_str: str, is_end_date: bool = False) -> str:
        """
        Convert date string to RFC3339 format required by OANDA API
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            is_end_date: If True, sets time to 23:59:59, otherwise 00:00:00
        
        Returns:
            RFC3339 formatted date string
        """
        try:
            if is_end_date:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            else:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
            
            return dt.isoformat().replace('+00:00', 'Z')
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD format. Error: {e}")
    
    def generate_date_chunks(self, start_date: str, end_date: str, chunk_days: int = 1) -> List[tuple]:
        """
        Generate date chunks for fetching large amounts of historical data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            chunk_days: Number of days per chunk (1 day for M1 data)
        
        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        from datetime import timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        chunks = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            ))
            current = chunk_end + timedelta(days=1)
        
        return chunks
    
    def get_candlestick_data(self, 
                           instrument: str, 
                           granularity: str = "H1", 
                           count: int = 100,
                           from_time: Optional[str] = None,
                           to_time: Optional[str] = None) -> List[CandlestickData]:
        """
        Fetch candlestick data from OANDA API
        
        Args:
            instrument: Trading instrument (e.g., "SPX500_USD")
            granularity: Time granularity (e.g., "H1", "M1", "D")
            count: Number of candles to fetch
            from_time: Start time in RFC3339 format
            to_time: End time in RFC3339 format
        
        Returns:
            List of CandlestickData objects
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "price": "M"  # M for midpoints, B for bid, A for ask
        }
        
        # OANDA API doesn't allow both count and date range parameters together
        if from_time and to_time:
            # Use date range when both from and to are provided
            params["from"] = from_time
            params["to"] = to_time
        else:
            # Use count when no date range is provided
            params["count"] = count
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            # Print detailed error information for debugging
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                print(f"Request URL: {response.url}")
                print(f"Request params: {params}")
                return []
            
            data = response.json()
            candles = []
            
            for candle_data in data.get('candles', []):
                if candle_data.get('complete', False):  # Only include complete candles
                    candle = CandlestickData(
                        time=candle_data['time'],
                        open=float(candle_data['mid']['o']),
                        high=float(candle_data['mid']['h']),
                        low=float(candle_data['mid']['l']),
                        close=float(candle_data['mid']['c']),
                        volume=candle_data.get('volume', 0)
                    )
                    candles.append(candle)
            
            return candles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
        except KeyError as e:
            print(f"Error parsing response: {e}")
            return []
    
    def get_available_instruments(self) -> List[Dict]:
        """Get list of available trading instruments"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json().get('instruments', [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching instruments: {e}")
            return []
    
    def validate_instrument(self, instrument: str) -> bool:
        """Validate if instrument is available for trading"""
        instruments = self.get_available_instruments()
        instrument_names = [inst['name'] for inst in instruments]
        return instrument in instrument_names


class OandaDataProcessor:
    """Business logic class for processing candlestick data"""
    
    @staticmethod
    def to_dataframe(candles: List[CandlestickData]) -> pd.DataFrame:
        """Convert candlestick data to pandas DataFrame"""
        data = [candle.to_dict() for candle in candles]
        df = pd.DataFrame(data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        return df
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV file"""
        df.to_csv(filename)
        print(f"Data saved to {filename}")


def main():
    """Main execution function"""
    # Initialize data fetcher
    fetcher = OandaDataFetcher(ACCOUNT_ID, ACCESS_TOKEN)
    
    # Check if SPY500 is available (try different naming conventions)
    spy500_variants = ["SPX500_USD", "SPX500", "SP500", "SP500_USD", "US500", "US500_USD"]
    valid_instrument = None
    
    for variant in spy500_variants:
        if fetcher.validate_instrument(variant):
            valid_instrument = variant
            print(f"Found valid instrument: {variant}")
            break
    
    if not valid_instrument:
        print("SPY500 not found. Available instruments:")
        instruments = fetcher.get_available_instruments()
        for inst in instruments[:10]:  # Show first 10
            print(f"- {inst['name']}: {inst['displayName']}")
        return
    
    # Determine if chunking is needed based on timeframe and date range
    def needs_chunking(timeframe: str, start_date: str, end_date: str) -> bool:
        """Determine if data fetching needs chunking based on timeframe and date range"""
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days_diff = (end - start).days
        
        # Chunking thresholds based on timeframe
        chunking_thresholds = {
            "M1": 1,    # 1 day for M1 data
            "M5": 7,    # 7 days for M5 data  
            "M15": 30,  # 30 days for M15 data
            "M30": 60,  # 60 days for M30 data
            "H1": 90,   # 90 days for H1 data
            "H4": 180,  # 180 days for H4 data
            "D": 365    # 365 days for daily data
        }
        
        threshold = chunking_thresholds.get(timeframe, 365)
        return days_diff > threshold
    
    # Check if chunking is needed
    if needs_chunking(timeframe, start_date, end_date):
        print(f"Date range: {start_date} to {end_date}")
        print(f"{timeframe} data requires chunking due to API limits. Processing in chunks...")
        
        # Determine chunk size based on timeframe
        chunk_sizes = {
            "M1": 1,    # 1 day chunks for M1
            "M5": 7,    # 7 day chunks for M5
            "M15": 30,  # 30 day chunks for M15
            "M30": 60,  # 60 day chunks for M30
            "H1": 90,   # 90 day chunks for H1
            "H4": 180,  # 180 day chunks for H4
            "D": 365    # 365 day chunks for daily
        }
        
        chunk_days = chunk_sizes.get(timeframe, 30)
        date_chunks = fetcher.generate_date_chunks(start_date, end_date, chunk_days=chunk_days)
        print(f"Generated {len(date_chunks)} chunks of {chunk_days} days each")
        
        all_candles = []
        successful_chunks = 0
        failed_chunks = 0
        
        for i, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            print(f"\n--- Processing chunk {i}/{len(date_chunks)}: {chunk_start} to {chunk_end} ---")
            
            try:
                from_time = fetcher.convert_to_rfc3339(chunk_start, is_end_date=False)
                to_time = fetcher.convert_to_rfc3339(chunk_end, is_end_date=True)
                
                chunk_candles = fetcher.get_candlestick_data(
                    instrument=valid_instrument,
                    granularity=timeframe,
                    from_time=from_time,
                    to_time=to_time
                )
                
                if chunk_candles:
                    all_candles.extend(chunk_candles)
                    successful_chunks += 1
                    print(f"✓ Successfully fetched {len(chunk_candles)} candles for chunk {i}")
                else:
                    failed_chunks += 1
                    print(f"✗ No data for chunk {i}")
                    
            except ValueError as e:
                print(f"✗ Date conversion error for chunk {i}: {e}")
                failed_chunks += 1
        
        candles = all_candles
        print(f"\nChunking complete: {successful_chunks} successful, {failed_chunks} failed")
        
    else:
        # For other timeframes, use the full date range
        try:
            from_time = fetcher.convert_to_rfc3339(start_date, is_end_date=False)
            to_time = fetcher.convert_to_rfc3339(end_date, is_end_date=True)
            print(f"Date range: {start_date} to {end_date}")
            print(f"RFC3339 format: {from_time} to {to_time}")
        except ValueError as e:
            print(f"Date conversion error: {e}")
            return
        
        # Fetch historical candlestick data
        print(f"Fetching {timeframe} historical candlestick data for {valid_instrument}...")
        
        candles = fetcher.get_candlestick_data(
            instrument=valid_instrument,
            granularity=timeframe,
            from_time=from_time,
            to_time=to_time
        )
    
    if not candles:
        print("No data retrieved. Check your API credentials, instrument name, and date range.")
        return
    
    print(f"Retrieved {len(candles)} total candles")
    
    # Process data
    processor = OandaDataProcessor()
    df = processor.to_dataframe(candles)
    df_with_indicators = processor.calculate_technical_indicators(df)
    
    # Display sample data
    print("\nSample data:")
    print(df_with_indicators.head())
    print(f"\nData range: {df_with_indicators.index.min()} to {df_with_indicators.index.max()}")
    
    # Save to CSV
    filename = f"oanda_{valid_instrument}_{timeframe}_{start_date}_to_{end_date}.csv"
    processor.save_to_csv(df_with_indicators, filename)
    
    return df_with_indicators


if __name__ == "__main__":
    data = main()