import os
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

class ScrapeInput(BaseModel):
    requestId: str
    url: HttpUrl
    timestamp: int

config = {
    'type': 'event',
    'name': 'ScrapeArticle',
    'description': 'Scrapes article content using Firecrawl',
    'subscribes': ['scrape-article'],
    'emits': ['generate-content'],
    'input': ScrapeInput.model_json_schema(),
    'flows': ['content-generation']
}

async def handler(input, context):
    from firecrawl import FirecrawlApp

    context.logger.info(f"🕷️ Scraping article: {input['url']}")

    if not FIRECRAWL_API_KEY:
        raise ValueError('FIRECRAWL_API_KEY is required for article scraping')
    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

    scrapeResult = app.scrape_url(input['url'])

    content = scrapeResult.get('markdown')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Firecrawl returned no article markdown')
    title = scrapeResult.get('metadata', {}).get('title', 'Untitled Article')

    context.logger.info(f"✅ Successfully scraped: {title} ({len(content) if content else 0} characters)")

    await context.emit({
        'topic': 'generate-content',
        'data': {
            'requestId': input['requestId'],
            'url': input['url'],
            'title': title,
            'content': content,
            'timestamp': input['timestamp']
        }
    })