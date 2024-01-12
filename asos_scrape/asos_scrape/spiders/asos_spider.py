import scrapy
import numpy as np
# item class included here 
url = "https://www.asos.com/women/jumpers-cardigans/cat/?cid=2637"


class AsosItem(scrapy.Item):
    title = scrapy.Field()
    # colour = scrapy.Field()
    price = scrapy.Field()
    material = scrapy.Field()

class AsosSpiderSpider(scrapy.Spider):
    name = "asos_spider"
    allowed_domains = ["www.asos.com"]
    start_urls = [url]


    def parse(self,response):
        NUMBER_OF_CLOTHES = 2759
        CLOTHES_PER_PAGE = 72
        NUMBER_OF_PAGES = np.ceil(NUMBER_OF_CLOTHES/CLOTHES_PER_PAGE)

        for page_number in range(int(NUMBER_OF_PAGES)):
            print(page_number)
            string_extension = "&page=" + str(page_number)
            
            yield scrapy.Request(url + string_extension,callback=self.parse_page)


    def parse_page(self, response):
        item = AsosItem()

        products = response.xpath('//*[@class="productTile_U0clN"]')

        for product in products:
            # id = product.attrib["id"]
            price = product.css("span *::text").extract()[0]
            url = product.css("a").attrib["href"]
            print(url)
            item["price"] = price
            yield scrapy.Request('https://www.asos.com'+ url,callback=self.parse_product, meta={'item': item})


    def parse_product(self,response):
        item = response.meta['item']

        material = response.xpath('//*[@id="productDescriptionAboutMe"]/div/div/text()[2]').extract()
        title = response.xpath('//*[@id="pdp-react-critical-app"]/span[1]/h1/text()').extract()


        item["title"] = title
        item["material"] = material

        yield item