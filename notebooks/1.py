# %% [markdown]
# # Player Stats Extraction

# %%
# Library Imports
import pandas as pd
import time
import numpy as np
from datetime import datetime

# Web scraping libraries
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Error handling
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# Data handling
import json
import os

# %%
# Load player data and setup browser for stats extraction
print("Loading NBA player data...")

# Load the complete player database
df_players = pd.read_csv('../data/pages/all_nba_players.csv')

print(f"Total players loaded: {len(df_players):,}")
print(f"Sample of loaded data:")
print(df_players.head())

# %%

# Setup Chrome browser for stats extraction
print("\nSetting up Chrome browser...")
options = Options()
options.add_argument('--headless')  # Run in background
options.add_argument('--no-sandbox')  # Bypass OS security
options.add_argument('--disable-dev-shm-usage')  # Use regular storage instead of in-memory storage

service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

print("Browser setup complete")
print(f"Ready to extract stats from {len(df_players)} player pages")

# %%
# Find the row for Kawhi Leonard
kawhi_row = df_players[df_players['player_name'] == 'Kawhi Leonard']

# Access the 'player_url' from that row
# .values[0] is used to get the first value from the column
kawhi_url = kawhi_row['player_url'].values[0]

print(kawhi_url)

# %%
# Navigate to Kawhi's page
driver.get(kawhi_url)

# Wait for the advanced stats table to load
try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "advanced"))
    )
    print("Advanced stats table loaded successfully")

    # Get page source and parse with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the advanced stats table
    advanced_table = soup.find('table', id='advanced')

    if advanced_table:
        print("Advanced stats table found")

        # Get table headers to understand structure
        headers = advanced_table.find('thead').find_all('th')
        header_names = [th.get('data-stat', th.text.strip()) for th in headers]

        print(f"Table headers found: {len(header_names)}")
        print(f"Header names: {header_names[:10]}...")  # Show first 10

    else:
        print("Advanced stats table not found")

except TimeoutException:
    print("Timeout waiting for advanced stats table")
    print(f"Current URL: {driver.current_url}")

# %%
# Extract Kawhi's season-by-season stats
print("Extracting Kawhi's season stats...")

# Get all data rows from the advanced table
rows = advanced_table.find('tbody').find_all('tr')

# Filter out header rows and totals rows
data_rows = []
for row in rows:
    # Skip rows that are headers or don't have season data
    season_cell = row.find('th', {'data-stat': 'year_id'})
    if season_cell and season_cell.text.strip() and not season_cell.text.strip().startswith('Career'):
        data_rows.append(row)

print(f"Found {len(data_rows)} seasons of data")

# Extract the specific columns we need
kawhi_seasons = []

target_stats = ['year_id', 'age', 'team_name_abbr', 'games', 'mp', 'per', 'bpm', 'vorp', 'usg_pct', 'ws', 'ws_per_48']

for row in data_rows:
    season_data = {}
    
    # Extract each target stat
    for stat in target_stats:
        cell = row.find(['th', 'td'], {'data-stat': stat})
        if cell:
            value = cell.text.strip()
            season_data[stat] = value if value else None
        else:
            season_data[stat] = None
    
    kawhi_seasons.append(season_data)

# Convert to DataFrame for easy viewing
kawhi_df = pd.DataFrame(kawhi_seasons)

print("\nKawhi Leonard's extracted stats:")
print(kawhi_df)

# Check for any missing key stats
print(f"\nData completeness check:")
for stat in ['per', 'bpm', 'ws', 'ws_per_48']:
    missing = kawhi_df[stat].isna().sum()
    total = len(kawhi_df)
    print(f"{stat.upper()}: {total - missing}/{total} seasons available")

# %%
# Extract stats from 20 random players to test our pipeline
print("Testing extraction pipeline with 20 random players...")

# Select 20 random players from our database
random_players = df_players.sample(n=20, random_state=42)
print(f"Selected {len(random_players)} random players for testing")

# Define our target stats
target_stats = ['year_id', 'age', 'team_name_abbr', 'games', 'mp', 'per', 'bpm', 'vorp', 'ws', 'ws_per_48']

# Store all extracted data
all_player_stats = []
successful_extractions = 0
failed_extractions = 0

for idx, (_, player) in enumerate(random_players.iterrows(), 1):
    player_name = player['player_name']
    player_url = player['player_url']
    
    print(f"\n[{idx}/20] Extracting: {player_name}")
    
    try:
        # Navigate to player page
        driver.get(player_url)
        
        # Wait for advanced stats table
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "advanced"))
        )
        
        # Parse the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        advanced_table = soup.find('table', id='advanced')
        
        if advanced_table:
            # Get data rows (exclude headers and totals)
            rows = advanced_table.find('tbody').find_all('tr')
            data_rows = []
            
            for row in rows:
                season_cell = row.find('th', {'data-stat': 'year_id'})
                if season_cell and season_cell.text.strip() and not season_cell.text.strip().startswith('Career'):
                    data_rows.append(row)
            
            # Extract stats for each season
            for row in data_rows:
                season_data = {'player_name': player_name}
                
                for stat in target_stats:
                    cell = row.find(['th', 'td'], {'data-stat': stat})
                    if cell:
                        value = cell.text.strip()
                        season_data[stat] = value if value else None
                    else:
                        season_data[stat] = None
                
                all_player_stats.append(season_data)
            
            print(f"  ✓ Extracted {len(data_rows)} seasons")
            successful_extractions += 1
            
        else:
            print(f"  ✗ No advanced stats table found")
            failed_extractions += 1
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        failed_extractions += 1
    
    # Rate limiting - 3 second delay
    time.sleep(3)

# Convert to DataFrame
df_extracted = pd.DataFrame(all_player_stats)

print(f"\n=== EXTRACTION SUMMARY ===")
print(f"Successful players: {successful_extractions}/20")
print(f"Failed players: {failed_extractions}/20")
print(f"Total seasons extracted: {len(df_extracted)}")

if len(df_extracted) > 0:
    print(f"\nSample of extracted data:")
    print(df_extracted.head())
    
    # Save test data
    df_extracted.to_csv('../data/pages/test_player_stats_20.csv', index=False)
    print(f"\nSaved test data to '../data/pages/test_player_stats_20.csv'")
else:
    print("\nNo data extracted - check for errors above")

# %%
# Full dataset extraction - All NBA players
print("Starting full dataset extraction...")
print(f"Total players to process: {len(df_players)}")

# Estimate time based on actual performance (20 players = 3m 54.6s)
actual_rate = 234.6 / 20  # seconds per player from test run
estimated_seconds = len(df_players) * actual_rate
estimated_minutes = estimated_seconds / 60
estimated_hours = estimated_minutes / 60
print(f"Estimated time: {estimated_minutes:.0f} minutes ({estimated_hours:.1f} hours)")
print(f"Based on test run: 20 players in 3m 54.6s = {actual_rate:.1f}s per player")

# Define target stats
target_stats = ['year_id', 'age', 'team_name_abbr', 'games', 'mp', 'per', 'bpm', 'vorp', 'ws', 'ws_per_48']

# Store all extracted data
all_player_stats = []
successful_extractions = 0
failed_extractions = 0
failed_players = []

# Track progress
start_time = datetime.now()
save_interval = 100  # Save progress every 100 players

print(f"\nStarting extraction at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("Progress will be saved every 100 players...")

for idx, (_, player) in enumerate(df_players.iterrows(), 1):
    player_name = player['player_name']
    player_url = player['player_url']

    # Progress indicator
    if idx % 50 == 0 or idx <= 10:
        elapsed = datetime.now() - start_time
        rate = idx / elapsed.total_seconds() * 60  # players per minute
        remaining = (len(df_players) - idx) / rate if rate > 0 else 0
        print(f"[{idx}/{len(df_players)}] {player_name} | Rate: {rate:.1f}/min | ETA: {remaining:.0f}min")

    try:
        # Navigate to player page
        driver.get(player_url)

        # Wait for advanced stats table
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "advanced"))
        )

        # Parse the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        advanced_table = soup.find('table', id='advanced')

        if advanced_table:
            # Get data rows (exclude headers and totals)
            rows = advanced_table.find('tbody').find_all('tr')
            data_rows = []

            for row in rows:
                season_cell = row.find('th', {'data-stat': 'year_id'})
                if season_cell and season_cell.text.strip() and not season_cell.text.strip().startswith('Career'):
                    data_rows.append(row)

            # Extract stats for each season
            for row in data_rows:
                season_data = {'player_name': player_name}

                for stat in target_stats:
                    cell = row.find(['th', 'td'], {'data-stat': stat})
                    if cell:
                        value = cell.text.strip()
                        season_data[stat] = value if value else None
                    else:
                        season_data[stat] = None

                all_player_stats.append(season_data)

            successful_extractions += 1

        else:
            failed_extractions += 1
            failed_players.append({'player': player_name, 'reason': 'No advanced table'})

    except Exception as e:
        failed_extractions += 1
        failed_players.append({'player': player_name, 'reason': str(e)})

    # Save progress every 100 players
    if idx % save_interval == 0:
        df_temp = pd.DataFrame(all_player_stats)
        df_temp.to_csv(f'../data/pages/nba_stats_progress_{idx}.csv', index=False)
        print(f"  -> Progress saved: {len(all_player_stats)} seasons extracted")

    # Rate limiting - 3 second delay
    time.sleep(3)

# Final save
df_final = pd.DataFrame(all_player_stats)
df_final.to_csv('../data/pages/nba_all_player_stats.csv', index=False)

# Save failed players log
if failed_players:
    df_failed = pd.DataFrame(failed_players)
    df_failed.to_csv('../data/pages/failed_extractions.csv', index=False)

# Final summary
end_time = datetime.now()
total_time = end_time - start_time

print(f"\n=== EXTRACTION COMPLETE ===")
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {total_time}")
print(f"Successful players: {successful_extractions}/{len(df_players)} ({successful_extractions/len(df_players)*100:.1f}%)")
print(f"Failed players: {failed_extractions}")
print(f"Total seasons extracted: {len(df_final)}")
print(f"Average seasons per player: {len(df_final)/successful_extractions:.1f}")
print(f"\nFinal dataset saved to: '../data/pages/nba_all_player_stats.csv'")

# Close browser
driver.quit()
print("Browser closed.")

# %%
# Force fresh ChromeDriver download and setup
import shutil

print("Forcing fresh ChromeDriver download...")

# Remove the cached ChromeDriver
cache_path = "/Users/jankasen/.wdm/drivers/chromedriver"
try:
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print("✓ Removed cached ChromeDriver")
    else:
        print("No cached ChromeDriver found")
except Exception as e:
    print(f"Could not remove cache: {e}")

# Try different approach - let ChromeDriverManager download fresh version
print("\nDownloading fresh ChromeDriver...")

try:
    # Force fresh download
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.core.os_manager import ChromeType
    
    # Get fresh driver path
    driver_path = ChromeDriverManager().install()
    print(f"✓ Fresh ChromeDriver downloaded to: {driver_path}")
    
    # Test if the driver executable works
    import subprocess
    result = subprocess.run([driver_path, "--version"], 
                          capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print(f"✓ ChromeDriver version: {result.stdout.strip()}")
        
        # Now try to create browser
        print("\nCreating browser with fresh driver...")
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        service = ChromeService(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        
        # Test browser
        driver.get("https://www.basketball-reference.com")
        print(f"✓ Browser working: {driver.title}")
        
    else:
        print(f"✗ ChromeDriver test failed: {result.stderr}")
        
except Exception as e:
    print(f"✗ Setup failed: {e}")
    
    # Last resort - try system Chrome if available
    print("\nTrying system Chrome as last resort...")
    try:
        # On Mac, try using system Chrome
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.basketball-reference.com")
        print(f"✓ System Chrome working: {driver.title}")
        
    except Exception as e2:
        print(f"✗ System Chrome also failed: {e2}")
        print("\nTroubleshooting suggestions:")
        print("1. Restart Jupyter notebook completely")
        print("2. Update Chrome browser: Chrome → About Google Chrome")
        print("3. Try: pip install --upgrade selenium webdriver-manager")
        print("4. Restart your computer if all else fails")

# %%
# Recovery extraction for remaining players
print("Starting recovery extraction...")

# Load original player database
print("Loading original player database...")
df_players = pd.read_csv('../data/01-pages/all_nba_players.csv')
print(f"Loaded {len(df_players)} total players")

# Load existing data to identify what's already extracted
try:
    # Load from the most recent progress file that has data
    df_extracted = pd.read_csv('../data/02-player-stats-extraction/extraction_run_1/nba_stats_progress_1400.csv')
    extracted_players = set(df_extracted['player_name'].unique())
    print(f"Found {len(extracted_players)} already extracted players")
except FileNotFoundError:
    extracted_players = set()
    print("No existing extraction file found")

# Identify remaining players
remaining_players = df_players[~df_players['player_name'].isin(extracted_players)]
print(f"Players still needed: {len(remaining_players)}")

if len(remaining_players) == 0:
    print("All players already extracted!")
else:
    # Show remaining letters distribution
    print("\nRemaining players by letter:")
    print(remaining_players['letter'].value_counts().sort_index())

    # Setup for extraction with browser restarts
    target_stats = ['year_id', 'age', 'team_name_abbr', 'games', 'mp', 'per', 'bpm', 'vorp', 'ws', 'ws_per_48']
    all_new_stats = []
    successful_extractions = 0
    failed_extractions = 0
    failed_players = []
    
    # Browser restart interval
    restart_interval = 200
    save_interval = 50
    
    def setup_browser():
        """Setup fresh browser instance with simpler configuration"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)
    
    # Check if we already have a working driver from previous cells
    try:
        # Test if existing driver is still working
        driver.current_url
        print("Using existing browser instance...")
    except:
        # Create new browser if needed
        print("Setting up new browser instance...")
        driver = setup_browser()
    start_time = datetime.now()
    
    print(f"\nStarting recovery at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        for idx, (_, player) in enumerate(remaining_players.iterrows(), 1):
            player_name = player['player_name']
            player_url = player['player_url']
            
            # Restart browser every 200 players to prevent memory issues
            if idx % restart_interval == 0:
                print(f"\n  Restarting browser at player {idx}...")
                driver.quit()
                time.sleep(5)  # Brief pause
                driver = setup_browser()
            
            # Progress indicator
            if idx % 25 == 0 or idx <= 10:
                elapsed = datetime.now() - start_time
                rate = idx / elapsed.total_seconds() * 60 if elapsed.total_seconds() > 0 else 0
                remaining_time = (len(remaining_players) - idx) / rate if rate > 0 else 0
                print(f"[{idx}/{len(remaining_players)}] {player_name} | Rate: {rate:.1f}/min | ETA: {remaining_time:.0f}min")
            
            try:
                # Navigate to player page
                driver.get(player_url)
                
                # Wait for advanced stats table with longer timeout
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "advanced"))
                )
                
                # Parse the page
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                advanced_table = soup.find('table', id='advanced')
                
                if advanced_table:
                    # Get data rows
                    rows = advanced_table.find('tbody').find_all('tr')
                    data_rows = []
                    
                    for row in rows:
                        season_cell = row.find('th', {'data-stat': 'year_id'})
                        if season_cell and season_cell.text.strip() and not season_cell.text.strip().startswith('Career'):
                            data_rows.append(row)
                    
                    # Extract stats for each season
                    for row in data_rows:
                        season_data = {'player_name': player_name}
                        
                        for stat in target_stats:
                            cell = row.find(['th', 'td'], {'data-stat': stat})
                            if cell:
                                value = cell.text.strip()
                                season_data[stat] = value if value else None
                            else:
                                season_data[stat] = None
                        
                        all_new_stats.append(season_data)
                    
                    successful_extractions += 1
                    
                else:
                    failed_extractions += 1
                    failed_players.append({'player': player_name, 'reason': 'No advanced table'})
            
            except TimeoutException:
                failed_extractions += 1
                failed_players.append({'player': player_name, 'reason': 'Timeout waiting for table'})
            except WebDriverException as e:
                failed_extractions += 1
                failed_players.append({'player': player_name, 'reason': f'WebDriver error: {str(e)[:100]}'})
            except Exception as e:
                failed_extractions += 1
                failed_players.append({'player': player_name, 'reason': f'Unknown error: {str(e)[:100]}'})
            
            # Save progress more frequently
            if idx % save_interval == 0:
                if all_new_stats:
                    df_new = pd.DataFrame(all_new_stats)
                    df_new.to_csv(f'../data/02-player-stats-extraction/extraction_run_2/recovery_progress_{idx}.csv', index=False)
                    print(f"  -> Recovery progress saved: {len(all_new_stats)} new seasons")
            
            # Rate limiting
            time.sleep(3)
    
    finally:
        driver.quit()
    
    # Combine with existing data and save
    if all_new_stats:
        df_new = pd.DataFrame(all_new_stats)
        
        # Combine with existing data
        if 'df_extracted' in locals():
            df_combined = pd.concat([df_extracted, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # Save combined dataset
        df_combined.to_csv('../data/02-player-stats-extraction/nba_all_player_stats_complete.csv', index=False)
        
        # Save new failed players
        if failed_players:
            df_failed_new = pd.DataFrame(failed_players)
            df_failed_new.to_csv('../data/02-player-stats-extraction/extraction_run_2/recovery_failed_extractions.csv', index=False)
    
    # Final summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n=== RECOVERY COMPLETE ===")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time}")
    print(f"New successful extractions: {successful_extractions}/{len(remaining_players)}")
    print(f"New failed extractions: {failed_extractions}")
    print(f"New seasons extracted: {len(all_new_stats)}")
    
    if 'df_combined' in locals():
        total_unique_players = len(df_combined['player_name'].unique())
        total_seasons = len(df_combined)
        print(f"\nFinal dataset:")
        print(f"Total unique players: {total_unique_players}")
        print(f"Total seasons: {total_seasons}")
        print(f"Average seasons per player: {total_seasons/total_unique_players:.1f}")
        print(f"Saved to: '../data/02-player-stats-extraction/nba_all_player_stats_complete.csv'")

# %%
# Slow recovery extraction with extended delays to avoid rate limiting
print("Starting slow recovery extraction with extended delays...")

# Load original player database
print("Loading original player database...")
df_players = pd.read_csv('../data/01-pages/all_nba_players.csv')
print(f"Loaded {len(df_players)} total players")

# Load existing data from BOTH extraction runs
extracted_players = set()

# Load from run 1
try:
    df_run1 = pd.read_csv('../data/02-player-stats-extraction/extraction_run_1/nba_stats_progress_1400.csv')
    extracted_players.update(df_run1['player_name'].unique())
    print(f"Found {len(df_run1['player_name'].unique())} players from extraction run 1")
except FileNotFoundError:
    print("No run 1 data found")

# Load from run 2 (latest progress file)
try:
    df_run2 = pd.read_csv('../data/02-player-stats-extraction/extraction_run_2/recovery_progress_1850.csv')
    extracted_players.update(df_run2['player_name'].unique())
    print(f"Found {len(df_run2['player_name'].unique())} players from extraction run 2")

    # Combine both runs for final dataset
    all_existing_data = pd.concat([df_run1, df_run2], ignore_index=True) if 'df_run1' in locals() else df_run2
    print(f"Total existing seasons: {len(all_existing_data)}")
except FileNotFoundError:
    print("No run 2 data found")
    all_existing_data = df_run1 if 'df_run1' in locals() else pd.DataFrame()

print(f"Total unique players already extracted: {len(extracted_players)}")

# Identify remaining players
remaining_players = df_players[~df_players['player_name'].isin(extracted_players)]
print(f"Players still needed: {len(remaining_players)}")

if len(remaining_players) == 0:
    print("All players already extracted!")
else:
    print(f"\nRemaining players by letter:")
    print(remaining_players['letter'].value_counts().sort_index())

    # Setup for slow extraction
    target_stats = ['year_id', 'age', 'team_name_abbr', 'games', 'mp', 'per', 'bpm', 'vorp', 'ws', 'ws_per_48']
    all_new_stats = []
    successful_extractions = 0
    failed_extractions = 0
    failed_players = []

    # MUCH MORE CONSERVATIVE SETTINGS
    restart_interval = 100  # Restart browser more frequently
    save_interval = 25      # Save progress more frequently
    base_delay = 8          # Longer base delay between requests
    long_break_interval = 50 # Take longer breaks more often

    def setup_browser():
        """Setup browser with more conservative options"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        # Add user agent to appear more like regular browser
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        service = ChromeService(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    # Setup fresh browser
    print("Setting up conservative browser...")
    driver = setup_browser()
    start_time = datetime.now()

    print(f"\nStarting slow extraction at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using {base_delay}s delays, {long_break_interval}-player break intervals")

    try:
        for idx, (_, player) in enumerate(remaining_players.iterrows(), 1):
            player_name = player['player_name']
            player_url = player['player_url']

            # Restart browser every 100 players
            if idx % restart_interval == 0:
                print(f"\n  Restarting browser at player {idx}...")
                driver.quit()
                time.sleep(10)  # Longer pause between restarts
                driver = setup_browser()

            # Take longer breaks every 50 players
            if idx % long_break_interval == 0:
                print(f"\n  Taking 2-minute break at player {idx}...")
                time.sleep(120)  # 2 minute break

            # Progress indicator (less frequent to reduce noise)
            if idx % 10 == 0 or idx <= 5:
                elapsed = datetime.now() - start_time
                rate = idx / elapsed.total_seconds() * 60 if elapsed.total_seconds() > 0 else 0
                remaining_time = (len(remaining_players) - idx) / rate if rate > 0 else 0
                print(f"[{idx}/{len(remaining_players)}] {player_name} | Rate: {rate:.1f}/min | ETA: {remaining_time:.0f}min")

            try:
                # Navigate to player page
                driver.get(player_url)

                # Wait longer for advanced stats table
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "advanced"))
                )

                # Parse the page
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                advanced_table = soup.find('table', id='advanced')

                if advanced_table:
                    # Get data rows
                    rows = advanced_table.find('tbody').find_all('tr')
                    data_rows = []

                    for row in rows:
                        season_cell = row.find('th', {'data-stat': 'year_id'})
                        if season_cell and season_cell.text.strip() and not season_cell.text.strip().startswith('Career'):
                            data_rows.append(row)

                    # Extract stats for each season
                    for row in data_rows:
                        season_data = {'player_name': player_name}

                        for stat in target_stats:
                            cell = row.find(['th', 'td'], {'data-stat': stat})
                            if cell:
                                value = cell.text.strip()
                                season_data[stat] = value if value else None
                            else:
                                season_data[stat] = None

                        all_new_stats.append(season_data)

                    successful_extractions += 1

                else:
                    failed_extractions += 1
                    failed_players.append({'player': player_name, 'reason': 'No advanced table'})

            except TimeoutException:
                failed_extractions += 1
                failed_players.append({'player': player_name, 'reason': 'Timeout waiting for table'})
                print(f"  ✗ Timeout: {player_name}")
            except WebDriverException as e:
                failed_extractions += 1
                failed_players.append({'player': player_name, 'reason': f'WebDriver error: {str(e)[:100]}'})
                print(f"  ✗ WebDriver error: {player_name}")
            except Exception as e:
                failed_extractions += 1
                failed_players.append({'player': player_name, 'reason': f'Unknown error: {str(e)[:100]}'})
                print(f"  ✗ Unknown error: {player_name}")

            # Save progress more frequently
            if idx % save_interval == 0:
                if all_new_stats:
                    df_new = pd.DataFrame(all_new_stats)
                    df_new.to_csv(f'../data/02-player-stats-extraction/extraction_run_3/slow_recovery_progress_{idx}.csv', index=False)
                    print(f"  -> Slow recovery progress saved: {len(all_new_stats)} new seasons")

            # LONGER rate limiting delay
            time.sleep(base_delay)

    finally:
        driver.quit()

    # Combine all data and save final dataset
    if all_new_stats:
        df_new_slow = pd.DataFrame(all_new_stats)

        # Combine with all existing data
        df_final_complete = pd.concat([all_existing_data, df_new_slow], ignore_index=True)

        # Save complete dataset
        df_final_complete.to_csv('../data/02-player-stats-extraction/nba_complete_player_stats.csv', index=False)

        # Save new failed players
        if failed_players:
            df_failed_slow = pd.DataFrame(failed_players)
            df_failed_slow.to_csv('../data/02-player-stats-extraction/extraction_run_3/slow_recovery_failed.csv', index=False)

    # Final summary
    end_time = datetime.now()
    total_time = end_time - start_time

    print(f"\n=== SLOW RECOVERY COMPLETE ===")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time}")
    print(f"New successful extractions: {successful_extractions}/{len(remaining_players)}")
    print(f"New failed extractions: {failed_extractions}")
    print(f"New seasons extracted: {len(all_new_stats) if all_new_stats else 0}")
    
    if 'df_final_complete' in locals():
        total_unique_players = len(df_final_complete['player_name'].unique())
        total_seasons = len(df_final_complete)
        print(f"\nFinal complete dataset:")
        print(f"Total unique players: {total_unique_players}")
        print(f"Total seasons: {total_seasons}")
        print(f"Average seasons per player: {total_seasons/total_unique_players:.1f}")
        print(f"Saved to: '../data/02-player-stats-extraction/nba_complete_player_stats.csv'")


