import requests
import sys
from bs4 import BeautifulSoup

def scrape(url, max_chars):

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all tags that typically contain text (p, div, span, h1, h2, h3, h4, h5, h6, article, section, etc.)
    text_content = []
    for tag in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section']):
        if tag.get_text(strip=True):
            text_content.append(tag.get_text(strip=True))

    # Combine all text from selected tags
    full_text = '\n'.join(text_content)

    # Return the required amount of text based on max_chars
    return(full_text[:max_chars])

if __name__ == '__main__':
    # Take URL and max characters as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python simple-scrapper.py <URL> <max_characters>")
        sys.exit(1)

    url = sys.argv[1]
    if not url.startswith("http"):
        sys.exit(1)
        
    try:
        max_chars = int(sys.argv[2])
    except ValueError:
        print("Please enter a valid integer for max_characters.")
        sys.exit(1)

    web_content = scrape(url, max_chars)
    print(web_content)