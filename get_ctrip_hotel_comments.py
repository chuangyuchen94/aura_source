import requests
from bs4 import BeautifulSoup
import certifi
import time
import hashlib

comment_url = "https://m.ctrip.com/restapi/soa2/21881/json/GetReviewList?testab=f3fabf791917eb95c0a19ec410cbf39b400cee21eabfb5cae3dc86e47d78756d"
request_json = {
    "PageNo":1,
    "PageSize":10,
    "MasterHotelId":96096861,
    "NeedFilter":True,
    "UnUsefulPageNo":1,
    "UnUsefulPageSize":5,
    "isHasFold":False,
    "genk":True,
    "genKeyParam":{
        "a":96096861,
        "d":"zh-cn",
        "e":2},
    "ssr":False,
    "head":{
        "Locale":"zh-CN",
        "Currency":"CNY",
        "Device":"PC",
        "UserIP":"223.74.76.104",
        "Group":"ctrip",
        "ReferenceID":"",
        "UserRegion":"CN",
        "AID":"4899",
        "SID":"135371",
        "Ticket":"",
        "UID":"",
        "IsQuickBooking":"",
        "ClientID":"09031045417084539175",
        "OUID":"",
        "TimeZone":"8",
        "P":"70196881372",
        "PageID":"102003",
        "Version":"",
        "HotelExtension":{
            "WebpSupport":True,
            "group":"CTRIP",
            "Qid":None,
            "hasAidInUrl":False,
            "hotelUuidKey":"zTFI0HrkTi4zIPMjNke7qw0ax1gy73eUBrcYgURcXYmDWAsithWFQiAzYUneZYh3IQEnGK0teptE4hjhpW5oiF1WdwHY3MYhBw8tyGFyzswhovgAjkJkzjG5rZpvkUj4Jnljb6wlqvUOjGJQnvTtY1dysajTcv7OebTYgsja3y5XymsEznEfYQsj1LvApWptyZpjQcvkcEs4v90WBlj3BY68i1JhY08yAMIT4rHmY76ip1wlQROdEdLWoargZEMLi7YG1R0meHgWQ7rZFEb8wsY7tR6fKshxUPxAgY6GELYs7il5xshYoMvsGYaByUojMdvPNeQoY4bj9zylJ57vMqY0BylSj14vlPe3hYbdjH1y8JckYQtvsbWMAW3zRgme5SvtYDsy1EcUigmr9LEMSwMnxQhYXTjndj87WnPrhZjM8vnYzGJ8DrsnYsfEltK9fKTYQ9jbvUGrFv5awA0w8YMGKqcjkcxmzYXZi7fiHpipdjhbKU5R8twzY1pJMMKhovU3y43Ef3YB8wGzv0QJzaYSGwDcRTFYpsIdY7gWzdyzbKmqRPMyD0EzXWkfE1dWNtytByQZvTqJ4nWtkEZGwOkit7vdy4YUNifLw4y4qRtqrFdW5YGfwOHE5nv1MR47y6UE0lW1mY5tvGORpcEQhWZAw57YU6WHzYkXiLSwn6JbYLSeLMjG3IP9js0w4NvNnjMhRZ6igQylYporTSYXSWGORTmwUbW4SWMOeFovPFWUmW5jnMJFqWtYnOyDLImgjlAR8UwHkWAUWTFekAjozynPJ9OjUhrsnKoYZOYNZwDPJp0RAGyZ8EtFWaTEo1WmpRTgJGkiosysTEHsR1sJA3JzayPcRmYLFeo4ElgISTEhojMFWaHWdbWhfY57YOMYH6R43YaFWMpYn7YhDYBHjbze7PE8MWAsegBwS7eF9jbXYG7ydgEl4jScEG4rg8jzawU7ydgW56E0fIQY8kRlpWFaWlsWXMWDnYHYLaEbmRc0jUkvMFEzqWXtySZjbJTov3NESsWAmyBUjHfvMnKh4wlYpjfpeM7vsFEk8EpnEXtR7OEgheNXKDByQYM9RLFwh6wTbY8HEf9EklY0BY4ZYcLY77KF5YaP"
        },
        "Frontend":{
            "vid":"1740291406202.6e28ZSCVpg4i",
            "sessionID":"3",
            "pvid":"1"
        }
    },
    "ServerData":""
}

headers = {
    "user-agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "cookie": "ibulanguage=CN; ibulocale=zh_cn; cookiePricesDisplayed=CNY; GUID=09031045417084539175; UBT_VID=1740291406202.6e28ZSCVpg4i; MKT_CKID=1740291406331.mh0p2.3n14; _RF1=223.74.76.104; _RSG=8UabtOIFxPDMkNu8zdMcRA; _RDG=28b93b88f3c3532dad369abac85106b5b1; _RGUID=d67b3d26-3796-4788-a6e6-5a7f48e17b96; nfes_isSupportWebP=1; cticket=B002257EB872A80FF51E630B2368091FC57FF77AFA6D43C2617DA3970A8D32C8; login_type=0; login_uid=A3E1243D3F86D44E4349862E14FFAB38; DUID=u=52807A9FD0BE550381D4E13410AEB8D8&v=0; IsNonUser=F; AHeadUserInfo=VipGrade=0&VipGradeName=%C6%D5%CD%A8%BB%E1%D4%B1&UserName=&NoReadMessageCount=0; MKT_Pagesource=PC; Session=smartlinkcode=U135371&smartlinklanguage=zh&SmartLinkKeyWord=&SmartLinkQuary=&SmartLinkHost=; Union=AllianceID=4899&SID=135371&OUID=&createtime=1740324050&Expires=1740928849885; intl_ht1=h4=1_96096861,43_114792723,32_5792241; librauuid=; _bfa=1.1740291406202.6e28ZSCVpg4i.1.1740401537764.1740410195461.4.1.102003; _jzqco=%7C%7C%7C%7C%7C1.902813300.1740291406332.1740401537918.1740410195714.1740401537918.1740410195714.0.0.0.9.9",
    "origin": "https://hotels.ctrip.com",
    "referer": "https://hotels.ctrip.com/",
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "x-requested-with": "XMLHttpRequest",
    "sec-ch-ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "macOS"
}

def generate_signature(hotel_id):
    """模拟加密生成ServerData（需根据实际JS逆向完善）"""
    timestamp = int(time.time() * 1000)
    raw = f"{hotel_id}_{timestamp}_CTRIP_SALT_KEY"
    return f"{timestamp}|{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

# 动态更新关键参数
hotel_id = 96096861
request_json.update({
    "ServerData": generate_signature(hotel_id),
    "head": {
        "HotelExtension": {
            "hotelUuidKey": get_hotel_uuid()  # 需实现动态获取逻辑
        },
        "Frontend": {
            "pvid": str(int(time.time()))  # 模拟递增pvid
        }
    }
})

for page_no in range(1, 2):
    request_json["PageNo"] = page_no
    print("当前请求参数结构：", request_json)

    with requests.Session() as s:
        s.headers.update(headers)
        # 添加请求重试逻辑
        s.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))

        res = s.post(
            url=comment_url,
            json=request_json,
            allow_redirects=True,  # 允许重定向
            timeout=15,  # 设置超时
            verify=certifi.where()  # 使用权威CA证书
        )

        print(f"status:{res.status_code}")
        print(f"text{res.text}")
        print(f"json={res.json}")
