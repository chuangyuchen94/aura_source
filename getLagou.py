import requests

request_url = """https://www.lagou.com/wn/zhaopin?fromSearch=true&kd=%25E4%25BA%25BA%25E5%25B7%25A5%25E6%2599%25BA%25E8%2583%25BD&pn=3"""

param = {
    "fromSearch": True,
    "kd": "%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD",
    "pn": 3
}

res = requests.get(url=request_url, data=param)

print(res.status_code)
print(res.text)