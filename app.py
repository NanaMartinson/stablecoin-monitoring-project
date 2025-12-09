# ... existing code ...
import requests
import time

# --- Page Configuration ---
st.set_page_config(
# ... existing code ...
# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_crypto_data(coin='USDT', days=365): 
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    all_data = []
    current_time = int(datetime.now().timestamp())
    hours_needed = days * 24
    limit = 2000
    num_requests = (hours_needed // limit) + 1
    
    # --- API KEY HANDLING (SECURE) ---
    # We look for the key in Streamlit secrets, but we don't crash if it's missing.
    api_key = st.secrets.get("CRYPTO_API_KEY", None)
    headers = {}
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    # ---------------------------------

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for batch in range(num_requests):
            status_text.text(f"Fetching {coin} data batch {batch+1}/{num_requests}...")
            
            params = {
                'fsym': coin, 
                'tsym': 'USD', 
                'limit': min(limit, hours_needed - len(all_data)), 
                'toTs': current_time
            }
            
            # Pass headers securely here
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
# ... existing code ...
