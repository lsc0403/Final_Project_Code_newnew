# coding=utf-8
import requests
from bs4 import BeautifulSoup
import xlwt
import re
import time
import os
from dateutil.parser import parse


def scrape_reviews(url, max_reviews_per_url):
    cnt = 1  # 每个文件的行数从1开始（0行用于表头）
    reviews_fetched = 0
    base_url = "https://www.imdb.com/"
    movie_title = ""

    while url and reviews_fetched < max_reviews_per_url:
        print(f"Scraping URL:{reviews_fetched}", url)
        res = requests.get(url)
        res.encoding = 'utf-8'

        if res.status_code != 200:
            print(f"Error fetching page: Status code {res.status_code}")
            break

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
            votetext = item.select_one(".actions.text-muted").text.strip() if item.select_one(
                ".actions.text-muted") else ""
            votes = re.findall(r"\d+", votetext)
            upvote = votes[0] if votes else '0'
            totalvote = votes[1] if len(votes) > 1 else '0'
            rating = item.select_one("span.rating-other-user-rating > span").text.strip() if item.select_one(
                "span.rating-other-user-rating > span") else ""
            review = item.select_one(".text.show-more__control").text.strip() if item.select_one(
                ".text.show-more__control") else ""

            if not rating:  # 如果没有评分，则跳过
                continue

            try:
                parsed_date = parse(date).strftime('%Y-%m-%d')
            except ValueError:
                parsed_date = date  # 如果日期解析失败，使用原始字符串

            # 创建Excel文件并写入数据
            if cnt == 1:  # 对于每个URL，只在第一次循环时初始化Excel
                f = xlwt.Workbook()
                sheet = f.add_sheet('Movie Reviews', cell_overwrite_ok=True)
                headers = ["Movie Name", "Title", "Author", "Date", "Up Vote", "Total Vote", "Rating", "Review"]
                for i, header in enumerate(headers):
                    sheet.write(0, i, header)

            row_data = [movie_title, title, author, parsed_date, upvote, totalvote, rating, review]
            for i, data in enumerate(row_data):
                sheet.write(cnt, i, data)
            cnt += 1
            reviews_fetched += 1

        load_more = soup.select_one(".load-more-data")
        if load_more and 'data-key' in load_more.attrs:
            key = load_more['data-key']
            movie_id = url.split("/")[4]
            url = f"{base_url}title/{movie_id}/reviews/_ajax?ref_=undefined&paginationKey={key}"
        else:
            url = None

        time.sleep(1)  # 适当的延时

    if reviews_fetched > 0:  # 确保有数据时才保存文件
        # 检查并创建ReviewTotal文件夹
        folder_path = "ReviewTotal"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 生成安全的文件名并保存Excel文件
        safe_movie_title = "".join([c if c.isalnum() else "_" for c in movie_title])
        filename = os.path.join(folder_path, f'{safe_movie_title}_IMDB_Reviews.xlsx')
        f.save(filename)
        print(f"{reviews_fetched} reviews saved to {filename}.")


# 定义要爬取的URL和每个URL的评论爬取上限
reviewNum = 10000
urls = [
    # ('https://www.imdb.com/title/tt0317248/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0816692/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt5090568/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0068646/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0468569/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0167260/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0109830/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt1375666/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0133093/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0118799/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0120689/reviews/?ref_=tt_ql_2'),
    # ('https://www.imdb.com/title/tt0103064/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0076759/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0088763/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0245429/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0938283/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0120201/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt15398776/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt7131622/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt15239678/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt1201607/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt1877830/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt1856101/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt6710474/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt9603212/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt5537002/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt12747748/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0903747/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt1520211/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt4574334/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0108778/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0898266/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt2356777/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt4154796/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0848228/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt4154756/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt10872600/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0499549/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0120338/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0369610/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0241527/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt2488496/reviews/?ref_=tt_ql_2'),
    # 添加更多URL
]

total_reviews_saved = 0
for url in urls:
    reviews_saved = scrape_reviews(url, reviewNum)
    # total_reviews_saved += reviews_saved

print(f"Total {total_reviews_saved} reviews saved across all files.")
