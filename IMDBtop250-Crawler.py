import requests
from bs4 import BeautifulSoup
import pandas as pd

# IMDb Top 250 URL
url = 'https://www.imdb.com/chart/top/'

# 发送请求获取HTML内容
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 找到电影列表
movies = soup.select('td.titleColumn')
links = [a.attrs.get('href') for a in soup.select('td.titleColumn a')]
full_links = ['https://www.imdb.com' + link for link in links]

# 解析电影名称和链接
movie_names = [movie.select('a')[0].get_text() for movie in movies]
movie_links = full_links

# 保存到DataFrame
df = pd.DataFrame({
    'Movie Name': movie_names,
    'URL': movie_links
})

# 保存为Excel文件
df.to_excel('imdb_top_250_movies.xlsx', index=False)
