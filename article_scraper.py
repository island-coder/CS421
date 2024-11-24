import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
import urllib.robotparser
import time
from helper import append_rows_to_csv,check_file_exists,initialize_csv_with_headers,read_rows_from_csv

def check_robots(target_url, user_agent='*'):
    robots_url=target_url+'/robots.txt'
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.read()
    return rp.can_fetch(user_agent, target_url),rp.crawl_delay(user_agent)

def scrape_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Target the main content area
        main_content = soup.find('article') or soup.find('main') or soup.find('div', {'id': 'main-content'})

        if not main_content:
            print("Main content not found")
            return None

        # Extract title
        title = main_content.find('h1').text if main_content.find('h1') else None

        # Extract text content
        paragraphs = main_content.find_all('p')
        content = " ".join([para.text for para in paragraphs])

        # Extract images and captions from the main content area only, excluding <aside> elements
        images = []
        for img_tag in main_content.find_all('img'):
            if img_tag.find_parent('aside'):
                continue
            img_url = img_tag.get('src')
            caption = img_tag.get('alt') if img_tag.get('alt') else ""
            images.append({'url': img_url, 'caption': caption})

        # Extract metadata
        metadata = {}
        for meta_tag in soup.find_all('meta'):
            if 'name' in meta_tag.attrs:
                metadata[meta_tag.attrs['name']] = meta_tag.attrs.get('content', '')
            elif 'property' in meta_tag.attrs:
                metadata[meta_tag.attrs['property']] = meta_tag.attrs.get('content', '')
        return {'id':url,'title': title, 'content': content, 'images': images, 'metadata': metadata}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def store_data(data, file_name):
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)



urls = [
    "https://www.bbc.com/news/articles/cy79y0v2rn0o",
    "https://www.bbc.com/news/articles/cv2g0wxqxx4o",
    "https://www.bbc.com/news/articles/c7205nx33yyo",
    "https://www.bbc.com/news/articles/cjjw8yxd10go",
    "https://www.bbc.com/news/articles/ce98l84mrmqo",
    "https://www.bbc.com/news/articles/cx72ez4ejp9o",
    "https://www.bbc.com/news/articles/c3gwyn2vqnpo",
    "https://www.bbc.com/news/articles/c10lq4znvejo",
    "https://www.bbc.com/news/articles/cjjwpxw8p51o",
    "https://www.bbc.com/news/articles/c8vde94dzq5o",
    "https://www.bbc.com/news/articles/cv2g0jnx8vdo",
    "https://www.bbc.com/travel/article/20240730-how-climate-change-can-affect-your-summer-vacation",
    "https://www.bbc.com/news/articles/cnl0w8rqk29o",
    "https://www.bbc.com/news/articles/c3g9404p30po",
    "https://www.bbc.com/news/articles/cz5rjjvvzlzo",
    "https://www.bbc.com/news/articles/cjjwln34pzno",
    "https://www.bbc.com/news/articles/ckdg754zr1po",
    "https://www.bbc.com/news/articles/cd1r0xvn0z5o",
    "https://www.bbc.com/news/articles/cjjw149pn0yo",
    "https://www.bbc.com/news/articles/czvx9q1wn47o",
    "https://www.bbc.com/news/articles/clwyj33jznpo",
    "https://www.bbc.com/news/articles/cervy9jkjr7o",
    "https://www.bbc.com/news/articles/cley43870l1o",
    "https://www.bbc.com/news/articles/c6p26qjgjjwo",
    "https://www.bbc.com/news/articles/clmy1nkm51jo",
    "https://www.bbc.com/news/articles/cw9yz2ve27zo",
    "https://www.bbc.com/news/articles/cql85nv8xqyo",
    "https://www.bbc.com/news/articles/c6p2yez5d3yo",
    "https://www.bbc.com/news/articles/ck7glyyl0kwo",
    "https://www.bbc.com/news/articles/c80xxeqel5po",
    "https://www.bbc.com/news/articles/c4nggvg1yggo",
    "https://www.bbc.com/news/articles/cyr7rv3nep2o",
    "https://www.bbc.com/news/articles/cp68rz0zrrwo",
    "https://www.bbc.com/news/articles/c6p2mrr49mno",
    "https://www.bbc.com/news/articles/c7292ye5z7wo",
    "https://www.bbc.com/news/articles/cd168081wxvo",
    "https://www.bbc.com/news/articles/cg33vw9939yo",
    "https://www.bbc.com/news/articles/c0jqjqxl3dyo",
    "https://www.bbc.com/news/articles/ce58p0048r0o",
    "https://www.bbc.com/news/articles/cw0y6wwn9yyo",
    "https://www.bbc.com/news/articles/cv2g9x47441o",
    "https://www.bbc.com/news/articles/ck5gp185n6ro",
    "https://www.bbc.com/news/articles/cv2gz7ky2weo",
    "https://www.bbc.com/news/articles/c0jqjwdyl1ko",
    "https://www.bbc.com/news/articles/cq5xye1d285o",
    "https://www.bbc.com/news/articles/ck7g9kjdrpmo"
]

# Remove duplicates by converting to a set and back to a list
unique_urls = list(set(urls))

def parse_article(
    target_url="https://www.bbc.com/news/articles/crgkk0rle75o"):
    robots_allowed,crawl_delay = check_robots(target_url)
    if robots_allowed:
        print("Scraping allowed")
        if crawl_delay:
            print(f"Crawl delay is {crawl_delay} seconds. Waiting...")
            time.sleep(crawl_delay)
        article_data = scrape_article(target_url)
        # if article_data:
        #     append_rows_to_csv(file_path,[[article_data['id'],article_data['title'], 
        #     article_data['content'], article_data['images'],
        #     article_data['metadata'],None,
        #     None,None,
        #     None]])
        return article_data
    else:
        print("Scraping not allowed")

