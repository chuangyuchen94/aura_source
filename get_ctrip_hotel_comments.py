import requests
from bs4 import BeautifulSoup
import certifi
import time
import hashlib
import hmac


def get_hotel_uuid(hotel_id):
    timestamp = int(time.time() * 1000)
    salt_key = "CTRIP_SALT_KEY"
    secret_key = "CTRIP_SECRET_KEY"

    # 拼接原始字符串
    raw = f"{timestamp}_{hotel_id}"

    # HMAC-SHA256加密
    signature = hmac.new(
        secret_key.encode('utf-8'),
        raw.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()[:32]  # 取前32位

    return f"{timestamp}|{signature}"

comment_url = "https://m.ctrip.com/restapi/soa2/21881/json/GetReviewList?testab=eb1742fb5e5628a815aabca07ce5ef13c4ad09017de021bc4a27122c716453c8"

headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "cookie": """UBT_VID=1740642305374.32c9aiJea17A; GUID=09031046117371225046; Session=smartlinkcode=U135371&smartlinklanguage=zh&SmartLinkKeyWord=&SmartLinkQuary=&SmartLinkHost=; Union=AllianceID=4899&SID=135371&OUID=&createtime=1740642306&Expires=1741247106145; MKT_CKID=1740642306239.ulhlw.ozx5; _ga=GA1.1.745851767.1740642306; _RSG=r3qTyh3biA84FFDQRdiRuB; _RDG=28edec21ef08b22ad02e6752c1c3d33de3; _RGUID=82ac943a-fc77-40f5-be1b-b276da34f3c4; MKT_Pagesource=PC; nfes_isSupportWebP=1; cticket=B002257EB872A80FF51E630B2368091FFEEB3D43F0BFA8FC36C876A6A0327D83; login_type=0; login_uid=A3E1243D3F86D44E4349862E14FFAB38; DUID=u=52807A9FD0BE550381D4E13410AEB8D8&v=0; IsNonUser=F; AHeadUserInfo=VipGrade=0&VipGradeName=%C6%D5%CD%A8%BB%E1%D4%B1&UserName=&NoReadMessageCount=0; ibulanguage=CN; ibulocale=zh_cn; cookiePricesDisplayed=CNY; _ga_5DVRDQD429=GS1.1.1740642306.1.1.1740642423.0.0.0; _ga_B77BES1Z8Z=GS1.1.1740642306.1.1.1740642423.24.0.0; _ga_9BZF483VNQ=GS1.1.1740642306.1.1.1740642423.0.0.0; librauuid=; intl_ht1=h4=2_109169530; _bfa=1.1740642305374.32c9aiJea17A.1.1740642387630.1740642425778.1.4.102003; _jzqco=%7C%7C%7C%7C%7C1.186842550.1740642306244.1740642387877.1740642426248.1740642387877.1740642426248.0.0.0.4.4; _RF1=112.96.176.89""",
    "origin": "https://hotels.ctrip.com",
    "referer": "https://hotels.ctrip.com/",
}


def generate_signature(hotel_id):
    """模拟加密生成ServerData（需根据实际JS逆向完善）"""
    timestamp = int(time.time() * 1000)
    raw = f"{hotel_id}_{timestamp}_CTRIP_SALT_KEY"
    return f"{timestamp}|{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


def get_request_json(hotel_id, page_no):
    # 动态更新关键参数
    request_json = {
        "PageNo": page_no,
        "PageSize": 10,
        "MasterHotelId": hotel_id,
        "NeedFilter": True,
        "UnUsefulPageNo": 1,
        "UnUsefulPageSize": 5,
        "isHasFold": False,
        "genk": False,
        "genKeyParam":
            {"a": hotel_id,
             "d": "zh-cn",
             "e": 2},
        "ssr": False,
        "head": {
            "Locale": "zh-CN",
            "Currency": "CNY",
            "Device": "PC",
            "UserIP": "2408:8456:f108:a4e4:841b:d075:3040:f267",
            "Group": "ctrip",
            "ReferenceID": "",
            "UserRegion": "CN",
            "AID": "4899",
            "SID": "135371",
            "Ticket": "",
            "UID": "",
            "IsQuickBooking": "", "ClientID": "09031046117371225046", "OUID": "", "TimeZone": "8", "P": 60450136241,
            "PageID": "102003", "Version": "",
            "HotelExtension": {"WebpSupport": True, "group": "CTRIP", "Qid": None, "hasAidInUrl": False,
                               "hotelUuidKey": "fOkWk8E18KGQvq0YsXeqcyODJGpiOZRSoYnYUDrThwZSeFpYd0E5keoSvoMJkYzkiZkEtmjo8eaDEnpj5fWkLyqSy64YkY54iLoizytkjQgwMGWOtjXJHZRTZwNLvQljsJGXeXZwmPvM7jnJFcv4cEnPyb5jOPvhaeM6YGOjDQy4crLojOZisYZdr95eBFynNy5mjg6vZHEqTv41WDqjPOxXbxnMelYkhrcOvsFEo7YBOia4wfHRBaE4MWcFJdlj5OifYMfikgEhNJMkjqEHtrTYPFEfwQaykQeq4i10EQY4sRMpIQMESMvszY9fyLaj8Hv1meGhY9qjqpyoJ9Qv5LYpsyazjZ4vBsenTYkfjthyoJFgY3kvotWd1W97edhROoYkYqgiLsvBkvtMvSbeG0Y86i7dYaoE1Py9NrpY15ifhWcni3MyN5jMdK8YPLJ31JG5Il1e3FJpojXY19xmXJL8WlMY1Gih0iFhi30jqNw1Sw05iGY30xQlIZbym3RksJm3YOQwohEcpv71YaQwbhrdDrFGROY55x45Y7BrdbRg4yaGEzAWpZwosE6lYHbEQMj3sJMpwstEMkJzbEaXIAjOYlcyzqrgAY3EPHWgbEcY4DrOQeMhRq0Rd0yzdE14W5PwzhEgzYZniUmWDdJQNi65yhbE3bKNajmqrcYZBwfmWMUr46j7DwBXvTPjDFvNkYz0R8YMZicQYnTiUlRM0EhHEmlWhneUXvB9wqSWt8vGLWZ0JnYHEGHic6R7nRs6ENnEcaWmBeDZvltY4lWgpIBfJ9hesY1NJQ7eqqKadRtmy9kE8GWPAwqcEfTY3OEkqY9kE75R9fwAtYU4yUUxm9E9Y0tEqhj5oJzqEfzjohWp4WQTWSAYBlY9lYLMRN7YSLW3AYXqYZkYq1jUne1zE7HW4SelgwT4ehtj1kYqPyzMEcGjAoEMtrcTjaDwSAyqsxLUR7fYfYQ1RUkWt1WAgWlXW6sYdY90vkgj95rM6v4oEDNWN5ycLjfJfdvFzEDHWU8yPfjPwF4EdvhY0gI1UYObrU0EFZElPEanRodEkavbkeGqyFY4fjUNjg7IPGY1kEamEOHYNcYFqYAZYskwodWzf"},
            "Frontend": {"vid": "1740642305374.32c9aiJea17A", "sessionID": "1", "pvid": "4"}}, "ServerData": ""}

    return request_json


if __name__ == '__main__':
    host_id = "109169530"
    request_json = get_request_json(host_id, 5)

    res = requests.post(url=comment_url, json=request_json, headers=headers)
    res.encoding = res.apparent_encoding
    print(res.status_code)
    print(res.text)
    print(res.json())
