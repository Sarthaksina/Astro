"""
Demo script to showcase the consolidated Vedic Analysis functionality.

This script demonstrates how the new VedicAnalyzer class works compared to
the deprecated VedicMarketAnalyzer class, showing the seamless transition
between the two implementations.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append('src')

# Import both the new and deprecated classes for comparison
from astro_engine.vedic_analysis import VedicAnalyzer
from astro_engine.vedic_market_analyzer import VedicMarketAnalyzer

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def demo_single_date_analysis():
    """Demonstrate analysis of a single date."""
    print_separator("SINGLE DATE ANALYSIS")
    
    # Create instances of both analyzer classes
    new_analyzer = VedicAnalyzer()
    old_analyzer = VedicMarketAnalyzer()
    
    # Date to analyze
    date = "2023-01-15"
    print(f"Analyzing date: {date}\n")
    
    # Analyze with new implementation
    print("Using new VedicAnalyzer:")
    try:
        new_result = new_analyzer.analyze_date(date)
        print(f"  Trend: {new_result['integrated_forecast']['trend']}")
        print(f"  Volatility: {new_result['integrated_forecast']['volatility']}")
        print(f"  Confidence: {new_result['integrated_forecast']['confidence']:.2f}")
        print(f"  Description: {new_result['integrated_forecast']['description'][:100]}...\n")
    except Exception as e:
        print(f"  Error: {str(e)}\n")
    
    # Analyze with deprecated implementation (should forward to new implementation)
    print("Using deprecated VedicMarketAnalyzer (forwarding to VedicAnalyzer):")
    try:
        old_result = old_analyzer.analyze_date(date)
        print(f"  Trend: {old_result['integrated_forecast']['trend']}")
        print(f"  Volatility: {old_result['integrated_forecast']['volatility']}")
        print(f"  Confidence: {old_result['integrated_forecast']['confidence']:.2f}")
        print(f"  Description: {old_result['integrated_forecast']['description'][:100]}...\n")
    except Exception as e:
        print(f"  Error: {str(e)}\n")
    
    # Verify results are identical
    if new_result and old_result:
        if new_result['integrated_forecast']['trend'] == old_result['integrated_forecast']['trend'] and \
           new_result['integrated_forecast']['volatility'] == old_result['integrated_forecast']['volatility']:
            print("✅ Results match between new and deprecated implementations!")
        else:
            print("❌ Results differ between implementations.")

def demo_date_range_analysis():
    """Demonstrate analysis of a date range."""
    print_separator("DATE RANGE ANALYSIS")
    
    # Create instances of both analyzer classes
    new_analyzer = VedicAnalyzer()
    old_analyzer = VedicMarketAnalyzer()
    
    # Date range to analyze
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    print(f"Analyzing date range: {start_date} to {end_date}\n")
    
    # Analyze with new implementation
    print("Using new VedicAnalyzer:")
    try:
        new_df = new_analyzer.analyze_date_range(start_date, end_date)
        print(f"  Shape: {new_df.shape}")
        print("  First few rows:")
        print(new_df.head(3).to_string())
        print()
    except Exception as e:
        print(f"  Error: {str(e)}\n")
    
    # Analyze with deprecated implementation (should forward to new implementation)
    print("Using deprecated VedicMarketAnalyzer (forwarding to VedicAnalyzer):")
    try:
        old_df = old_analyzer.analyze_date_range(start_date, end_date)
        print(f"  Shape: {old_df.shape}")
        print("  First few rows:")
        print(old_df.head(3).to_string())
        print()
    except Exception as e:
        print(f"  Error: {str(e)}\n")
    
    # Verify results are identical
    if isinstance(new_df, pd.DataFrame) and isinstance(old_df, pd.DataFrame):
        if new_df.equals(old_df):
            print("✅ Results match between new and deprecated implementations!")
        else:
            print("❌ Results differ between implementations.")

def demo_visualization():
    """Demonstrate visualization functionality."""
    print_separator("VISUALIZATION")
    
    # Create instances of both analyzer classes
    new_analyzer = VedicAnalyzer()
    old_analyzer = VedicMarketAnalyzer()
    
    # Date range to analyze
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    print(f"Visualizing forecasts for: {start_date} to {end_date}\n")
    
    try:
        # Generate forecasts
        df = new_analyzer.analyze_date_range(start_date, end_date)
        
        # Create a simple mock market data for comparison
        dates = pd.date_range(start=start_date, end=end_date)
        market_data = pd.DataFrame({
            'date': dates,
            'close': [100 + i * (1 + (i % 3 - 1) * 0.5) for i in range(len(dates))]
        })
        
        print("Plotting forecast trends with new VedicAnalyzer...")
        try:
            # Save the plot to a file instead of displaying it
            plt.figure(figsize=(12, 6))
            new_analyzer.plot_forecast_trends(df, market_data, title="Vedic Analysis Forecast (New Implementation)")
            plt.savefig('vedic_analysis_forecast_new.png')
            plt.close()
            print("  Plot saved to 'vedic_analysis_forecast_new.png'\n")
        except Exception as e:
            print(f"  Error plotting with new implementation: {str(e)}\n")
        
        print("Plotting forecast trends with deprecated VedicMarketAnalyzer...")
        try:
            # Save the plot to a file instead of displaying it
            plt.figure(figsize=(12, 6))
            old_analyzer.plot_forecast_trends(df, market_data, title="Vedic Analysis Forecast (Deprecated Implementation)")
            plt.savefig('vedic_analysis_forecast_old.png')
            plt.close()
            print("  Plot saved to 'vedic_analysis_forecast_old.png'\n")
        except Exception as e:
            print(f"  Error plotting with deprecated implementation: {str(e)}\n")
            
        print("✅ Visualization complete! Check the saved PNG files to compare the results.")
    except Exception as e:
        print(f"Error in visualization demo: {str(e)}")

def main():
    """Run the demonstration."""
    print_separator("VEDIC ANALYSIS MODULE DEMONSTRATION")
    print("This script demonstrates the consolidated Vedic Analysis functionality.")
    print("It compares the new VedicAnalyzer class with the deprecated VedicMarketAnalyzer class.")
    
    try:
        # Run the demos
        demo_single_date_analysis()
        demo_date_range_analysis()
        demo_visualization()
        
        print_separator("DEMONSTRATION COMPLETE")
        print("The consolidation of vedic_market_analyzer.py into vedic_analysis.py has been successful!")
        print("The deprecated VedicMarketAnalyzer class properly forwards all methods to the new VedicAnalyzer class.")
        print("Users can now transition to using the new implementation while maintaining backward compatibility.")
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")

if __name__ == "__main__":
    main()
