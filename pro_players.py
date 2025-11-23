import cloudscraper
from bs4 import BeautifulSoup
import csv
import time
import random
from fake_useragent import UserAgent
from tqdm import tqdm
import config

# Initialize the fake user agent
ua = UserAgent()

# Create a CloudScraper session
scraper = cloudscraper.create_scraper()  # This handles Cloudflare bypassing

# Function to get a random user-agent
def get_random_user_agent():
    return ua.random

# Function to fetch player profile URLs
def fetch_player_profile_links():
    url = "https://www.hltv.org/stats/players?csVersion=CS2&__cf_chl_tk=6jQ.0uDat3qD.2hslXyLFQFOPpHOW.zKaRUT.TFU1mo-1763482381-1.0.1.1-eVQKzZJsG5Uuc_PkhWHuNphF9.uvlB8esFLNnpgAa2M"
    
    # Set a random user agent for each request to mimic a real user
    headers = {
        'User-Agent': get_random_user_agent()
    }

    # Use cloudscraper to make the request
    response = scraper.get(url, headers=headers)

    # Parse the HTML response
    soup = BeautifulSoup(response.text, 'html.parser')
    
    player_links = []
    rows = soup.select(".player-ratings-table tbody tr")
    
    for row in rows:
        anchor = row.select_one("a")
        if anchor:
            href = anchor['href']
            playerIdMatch = href.split('/')[3]  # Extract player ID from the href
            playerUsername = anchor.text.strip()  # Get the player's username (innerText)
            playerProfile = f"https://www.hltv.org/player/{playerIdMatch}/{playerUsername}"

            # Append the player profile URL to the list
            player_links.append(playerProfile)
    
    return player_links

# Function to fetch Faceit account link from a player profile
def fetch_faceit_account(player_url):
    # Set a random user agent for each request to mimic a real user
    headers = {
        'User-Agent': get_random_user_agent()
    }

    # Use cloudscraper to make the request
    response = scraper.get(player_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check if the player has a team (check for <img> under .playerTeam .listRight)
    team_img = soup.select_one(".playerTeam .listRight img")
    
    if team_img:
        # Find all Faceit links
        faceit_links = soup.select(".gtSmartphone-only")
        for link in faceit_links:
            href = link.get('href')
            if href and "www.faceit.com" in href:
                return href  # Return the Faceit account link
    return None  # No Faceit account found or no team

# Main function to gather all player data and save it to CSV
def save_player_data_to_csv():
    print("Scraping HLTV players...")
    player_links = fetch_player_profile_links()
    player_data = []

    for player_url in tqdm(player_links, unit="player", ncols=80):
        faceit_account = fetch_faceit_account(player_url)
        
        # If no Faceit account is found, skip this player
        if faceit_account is None:
            continue

        # Get the player's nickname from the player profile URL
        player_username = player_url.split('/')[-1]
        player_data.append([player_username, faceit_account])

        # Introduce a random delay between requests to mimic human browsing behavior
        time.sleep(random.uniform(1, 3))  # Random delay between 1 and 3 seconds

    # Save the data to a CSV file
    with open(config.PRO_DATA_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Player Nickname", "Faceit Account Link"])
        writer.writerows(player_data)

    print(f"✅ Data saved to '{config.PRO_DATA_CSV}' with {len(player_data)} entries.")

# Run the script
if __name__ == "__main__":
    save_player_data_to_csv()

# Scraping HLTV players...
# 100%|█████████████████████████████████████| 831/831 [31:39<00:00,  2.29s/player]