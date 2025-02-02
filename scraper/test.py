import scrapy
import json
from scrapy.crawler import CrawlerProcess

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    start_urls = [
        'https://quotes.toscrape.com/page/1/',
        'https://quotes.toscrape.com/page/2/',
    ]

    def parse(self, response):
        quotes_data = []

        for quote in response.css('div.quote'):
            quote_dict = {
                'text': quote.css('span.text::text').get(),
                'author': quote.css('small.author::text').get(),
                'tags': quote.css('div.tags a.tag::text').getall(),
            }
            quotes_data.append(quote_dict)

        # Save data to JSON
        with open('/content/quotes.json', 'w', encoding='utf-8') as f:
            json.dump(quotes_data, f, ensure_ascii=False, indent=4)

        self.log("Saved file: quotes.json")

# Run Scrapy inside Colab
process = CrawlerProcess()
process.crawl(QuotesSpider)
process.start()
