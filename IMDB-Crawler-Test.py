# coding=utf-8
import requests
from bs4 import BeautifulSoup
import xlwt
import re
import time
from dateutil.parser import parse  # 导入解析日期的库
import xlrd
from xlutils.copy import copy

def save_excel(sheet, row_data, row_index):
    for i, data in enumerate(row_data):
        sheet.write(row_index, i, data)
    return row_index + 1

def save_to_file(f, filename):
    f.save(filename)

def scrape_reviews(url, start_row, max_reviews_per_url, filename):
    cnt = start_row
    reviews_fetched = 0
    base_url = "https://www.imdb.com/"
    movie_title = ""
    movie_id_global = "null"
    while url and reviews_fetched < max_reviews_per_url:
        print("Scraping URL:", url)
        res = requests.get(url)
        res.encoding = 'utf-8'

        if res.status_code != 200:
            print(f"Error fetching page: Status code {res.status_code}")
            continue

        soup = BeautifulSoup(res.text, "lxml")

        if not movie_title:
            movie_title = soup.select_one("h3[itemprop='name'] a").text.strip() if soup.select_one(
                "h3[itemprop='name'] a") else "Unknown Movie"

        for item in soup.select(".lister-item-content"):
            if reviews_fetched >= max_reviews_per_url:
                break

            title = item.select_one(".title").text.strip() if item.select_one(".title") else ""
            author = item.select_one(".display-name-link").text.strip() if item.select_one(".display-name-link") else ""
            date = item.select_one(".review-date").text.strip() if item.select_one(".review-date") else ""
            votetext = item.select_one(".actions.text-muted").text.strip() if item.select_one(".actions.text-muted") else ""
            votes = re.findall(r"\d+", votetext)
            upvote = votes[0] if votes else '0'
            totalvote = votes[1] if len(votes) > 1 else '0'
            rating = item.select_one("span.rating-other-user-rating > span").text.strip() if item.select_one("span.rating-other-user-rating > span") else ""
            review = item.select_one(".text.show-more__control").text.strip() if item.select_one(".text.show-more__control") else ""

            if not rating:
                continue

            try:
                parsed_date = parse(date).strftime('%Y-%m-%d')
            except ValueError:
                parsed_date = date

            # Read the existing file, update it and save
            rb = xlrd.open_workbook(filename, formatting_info=True)
            wb = copy(rb)
            sheet = wb.get_sheet(0)
            row_data = [movie_title, title, author, parsed_date, upvote, totalvote, rating, review]
            cnt = save_excel(sheet, row_data, cnt)
            wb.save(filename)

            reviews_fetched += 1

        load_more = soup.select_one(".load-more-data")
        if load_more and 'data-key' in load_more.attrs:
            key = load_more['data-key']
            movie_id = url.split("/")[4]
            url = f"{base_url}title/{movie_id}/reviews/_ajax?ref_=undefined&paginationKey={key}"
            movie_id_global=movie_id
        else:
            break

        time.sleep(1)  # 适当调整这个值

    print(f"已保存{base_url}title/{movie_id_global}的{reviews_fetched}条数据")
    return cnt

# 初始化Excel文件和表头
filename = 'TenThousands-IMDB_Reviews.xlsx'
f = xlwt.Workbook()
sheet1 = f.add_sheet('Movie Reviews', cell_overwrite_ok=True)
headers = ["Movie Name", "Title", "Author", "Date", "Up Vote", "Total Vote", "Rating", "Review"]
for i, header in enumerate(headers):
    sheet1.write(0, i, header)
save_to_file(f, filename)

# 定义要爬取的URL和每个URL的评论爬取上限
reviewNum = 10000
urls = [
    ('https://www.imdb.com/title/tt0317248/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0816692/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt5090568/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0068646/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0468569/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0167260/reviews/?ref_=tt_ql_2'),

    # 添加更多URL
]

cnt = 1
for url in urls:
    cnt = scrape_reviews(url, cnt, reviewNum, filename)

print(f"{cnt - 1} reviews saved.")

