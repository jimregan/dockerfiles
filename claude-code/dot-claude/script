import requests
from bs4 import BeautifulSoup
import sys
import os
import re
from urllib.parse import urljoin

# --- Configuration ---
BASE_GUTENBERG_URL = "https://www.gutenberg.org/"

def extract_book_id(gutenberg_url: str) -> str | None:
    """
    Extracts the book ID from a full Gutenberg URL.
    Expected format: https://www.gutenberg.org/ebooks/{book_id}
    """
    # Check for the base pattern first
    if "gutenberg.org/ebooks/" in gutenberg_url:
        # Example: .../ebooks/514
        parts = gutenberg_url.split('/')
        # The book ID should be the last element
        book_id = parts[-1]
        if book_id.isdigit():
            return book_id
    
    print(f"Error: Could not find a clear book ID in the URL: {gutenberg_url}")
    return None

def scrape_relative_link(ebook_page_url: str) -> str | None:
    """
    Fetches the main e-book page and scrapes the relative link to the full HTML file.
    """
    print(f"\n[+] Step 2/3: Fetching and scraping the main ebook page...")
    try:
        response = requests.get(ebook_page_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the link tag with the specific class
        link_tag = soup.find('a', {'class': 'link read_html'})
        
        if link_tag and 'href' in link_tag.attrs:
            relative_link = link_tag['href']
            print(f"[SUCCESS] Found relative link: {relative_link}")
            return relative_link
        else:
            print("[ERROR] Could not find the anchor tag with class 'link read_html'. The page structure might have changed.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[CRITICAL ERROR] Failed to fetch or parse the e-book landing page: {e}")
        return None

def download_and_save_html(relative_link: str, book_id: str, original_url: str) -> None:
    """
    Constructs the full URL and downloads the final content.
    """
    # Construct the full download URL using the base site domain
    download_url = urljoin(BASE_GUTENBERG_URL, relative_link)
    
    print(f"\n[+] Step 3/3: Downloading final content from: {download_url}")
    
    try:
        # Download the actual HTML content
        response = requests.get(download_url, timeout=20)
        response.raise_for_status()
        
        # Determine a safe filename
        # We'll use the book ID and the first part of the title for the filename
        title_match = re.search(r'title="([^"]+)"', str(response.content), re.IGNORECASE)
        if title_match:
            title = title_match.group(1)
            safe_filename = re.sub(r'[\\/*?%<>:]', '', title).strip()
        else:
            safe_filename = f"gutenberg_book_{book_id}"
            
        final_filename = f"{safe_filename}.html"
        
        # Save the content
        with open(final_filename, 'wb') as f:
            f.write(response.content)
        
        print("\n================================================================")
        print("✅ SUCCESS!")
        print(f"Successfully scraped and saved the e-book content to: {final_filename}")
        print("================================================================")
        
    except requests.exceptions.RequestException as e:
        print(f"[CRITICAL ERROR] Failed to download the final content from {download_url}: {e}")
        print("Please check the network connection or if the book is still available online.")
    except IOError as e:
        print(f"[CRITICAL ERROR] Failed to write the file {final_filename}: {e}")


def main():
    """Main function to orchestrate the scraping process."""
    if len(sys.argv) != 2:
        print("Usage: python3 gutenberg_scraper.py <FULL_GUITENBERG_URL>")
        sys.exit(1)

    full_url = sys.argv[1]
    print(f"--- Starting Gutenberg Ebook Scraper ---")
    print(f"Processing URL: {full_url}")

    # Step 1: Extract Book ID
    book_id = extract_book_id(full_url)
    if not book_id:
        sys.exit(1)
    print(f"[+] Step 1/3: Successfully extracted Book ID: {book_id}")

    # Step 2 & 3: Scrape and Download
    # We combine steps 2 and 3 here as they are dependent.
    relative_link = scrape_relative_link(f"{BASE_GUTENBERG_URL}/ebooks/{book_id}")
    
    if relative_link:
        download_and_save_html(relative_link, book_id, full_url)
    else:
        print("\n--- PROCESS ABORTED ---")
        print("Could not retrieve the necessary relative link. Review error messages above.")


if __name__ == "__main__":
    main()

