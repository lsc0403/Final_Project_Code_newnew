urls = [
    ('https://www.imdb.com/title/tt0317248/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0816692/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt5090568/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0068646/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0468569/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0167260/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0109830/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt1375666/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0133093/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0118799/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0120689/reviews/?ref_=tt_ql_2'),
    ('https://www.imdb.com/title/tt0103064/reviews/?ref_=tt_ql_2'),
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

# 使用字典来计数每个 URL 出现的次数
url_counts = {}

for url in urls:
    if url in url_counts:
        url_counts[url] += 1
    else:
        url_counts[url] = 1

# 找出并输出重复的 URL
print("重复的URL有：")
for url, count in url_counts.items():
    if count > 1:
        print(url)