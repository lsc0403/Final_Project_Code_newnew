# coding=utf-8
import requests
from bs4 import BeautifulSoup
import xlwt
import re


def scrape_reviews(url, sheet, start_row, max_reviews_per_url):
    print("Scraping URL:", url)
    res = requests.get(url)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, "lxml")

    # 抓取电影名称，根据实际页面结构调整选择器
    movie_title = soup.select_one("h3[itemprop='name'] a").text.strip() if soup.select_one(
        "h3[itemprop='name'] a") else "Unknown Movie"

    cnt = start_row
    reviews_fetched = 0

    for item in soup.select(".lister-item-content"):
        if reviews_fetched >= max_reviews_per_url:
            break

        title = item.select(".title")[0].text.strip()
        author = item.select(".display-name-link")[0].text.strip()
        date = item.select(".review-date")[0].text.strip()
        votetext = item.select_one(".actions.text-muted").text.strip()
        votes = re.findall(r"\d+", votetext)
        upvote = votes[0] if len(votes) > 0 else '0'
        totalvote = votes[1] if len(votes) > 1 else '0'
        rating = item.select_one("span.rating-other-user-rating > span")
        rating = rating.text.strip() if rating else ""
        review = item.select_one(".text.show-more__control").text.strip() if item.select_one(
            ".text.show-more__control") else ""

        # 添加电影名称到行数据的开头
        row_data = [movie_title, title, author, date, upvote, totalvote, rating, review]
        for i, data in enumerate(row_data):
            sheet.write(cnt, i, data)
        cnt += 1
        reviews_fetched += 1

    return cnt


# 初始化Excel文件和表头
f = xlwt.Workbook()
sheet1 = f.add_sheet('Movie Reviews', cell_overwrite_ok=True)
headers = ["Movie Name", "Title", "Author", "Date", "Up Vote", "Total Vote", "Rating", "Review"]  # 更新表头以包含电影名称
for i, header in enumerate(headers):
    sheet1.write(0, i, header)

# 定义URLs和爬取上限
urls = [
    ('https://www.imdb.com/title/tt0120731/reviews/?ref_=tt_ql_3', 200),
    ('https://www.imdb.com/title/tt0468569/reviews/?ref_=tt_ql_3', 200),
    ('https://www.imdb.com/title/tt5090568/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt0068646/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt0468569/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt0108052/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt0167260/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt0109830/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt1375666/reviews/?ref_=tt_ql_2', 200),
    ('https://www.imdb.com/title/tt0133093/reviews/?ref_=tt_ql_2', 200),

    # 添加更多URL
]

# 爬取评论
cnt = 1
for url, max_reviews in urls:
    cnt = scrape_reviews(url, sheet1, cnt, max_reviews)

# 保存Excel文件
f.save('Review.xls')
print(f"{cnt - 1} reviews saved.")
