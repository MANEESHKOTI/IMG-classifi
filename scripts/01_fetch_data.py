import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

TAXON_ID = os.environ.get('TAXON_ID', '47157')
NUM_SPECIES = int(os.environ.get('NUM_SPECIES', '10'))
MIN_OBSERVATIONS = int(os.environ.get('MIN_OBSERVATIONS', '200'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

MAX_RETRIES = 5
RETRY_BACKOFF = 3  # seconds


def get_with_retry(url, retries=MAX_RETRIES):
    """HTTP GET with exponential backoff retries to handle transient network errors."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"  [Retry {attempt+1}/{retries}] Network error: {e}. Waiting {wait}s...")
            time.sleep(wait)
    raise ConnectionError(f"Failed to connect after {retries} attempts: {url}")


def fetch_data():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"Fetching species for taxon {TAXON_ID}...")
    url = (
        f"https://api.inaturalist.org/v1/observations/species_counts"
        f"?taxon_id={TAXON_ID}&quality_grade=research&photos=true&per_page=50"
    )

    response = get_with_retry(url)
    if response.status_code != 200:
        print(f"Error fetching species counts: {response.status_code}")
        return

    data = response.json()
    results = data.get('results', [])

    species_found = 0
    for result in results:
        if species_found >= NUM_SPECIES:
            break

        taxon = result.get('taxon', {})
        count = result.get('count', 0)
        species_name = taxon.get('name')
        species_id = taxon.get('id')

        if not (count >= MIN_OBSERVATIONS and species_name and species_id):
            continue

        species_dir = os.path.join(RAW_DATA_DIR, species_name.replace(' ', '_'))
        os.makedirs(species_dir, exist_ok=True)

        # --- Resume support: count already-downloaded images ---
        already_downloaded = len([f for f in os.listdir(species_dir) if f.endswith('.jpg')])
        if already_downloaded >= MIN_OBSERVATIONS:
            print(f"Skipping {species_name}: already have {already_downloaded} images.")
            species_found += 1
            continue

        print(f"Found species: {species_name} ({count} obs). Downloading {MIN_OBSERVATIONS}...")

        downloaded = already_downloaded
        page = 1
        while downloaded < MIN_OBSERVATIONS:
            obs_url = (
                f"https://api.inaturalist.org/v1/observations"
                f"?taxon_id={species_id}&quality_grade=research&photos=true"
                f"&per_page=100&page={page}"
            )
            try:
                obs_response = get_with_retry(obs_url)
            except ConnectionError as e:
                print(f"  Giving up on page {page} for {species_name}: {e}")
                break

            if obs_response.status_code != 200:
                print(f"  HTTP {obs_response.status_code} for {species_name} page {page}. Stopping.")
                break

            obs_data = obs_response.json()
            obs_results = obs_data.get('results', [])

            if not obs_results:
                break

            for obs in obs_results:
                if downloaded >= MIN_OBSERVATIONS:
                    break

                obs_id = obs.get('id')
                photos = obs.get('photos', [])
                if not photos:
                    continue

                img_url = photos[0].get('url', '').replace('square', 'medium')
                if not img_url:
                    continue

                img_path = os.path.join(species_dir, f"{obs_id}.jpg")
                if os.path.exists(img_path):
                    downloaded += 1
                    continue

                try:
                    img_data = get_with_retry(img_url).content
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    downloaded += 1
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  Error downloading image {img_url}: {e}")

            page += 1
            time.sleep(1)

        if downloaded >= MIN_OBSERVATIONS:
            species_found += 1
            print(f"Finished downloading {downloaded} images for {species_name}")
        else:
            print(f"Only got {downloaded}/{MIN_OBSERVATIONS} images for {species_name}. Skipping.")

    print(f"\nDone. {species_found}/{NUM_SPECIES} species acquired.")


if __name__ == '__main__':
    fetch_data()
