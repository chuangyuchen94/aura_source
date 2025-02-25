import requests
from bs4 import BeautifulSoup

def get_url(page_no):
    request_url = f"""https://www.lagou.com/wn/zhaopin?fromSearch=true&kd=%25E4%25BA%25BA%25E5%25B7%25A5%25E6%2599%25BA%25E8%2583%25BD&pn={page_no}"""

    return request_url

def get_params(page_no):
    param = {
        "fromSearch": True,
        "kd": "%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD",
        "pn": page_no
    }

    return param

def get_headers(page_no):
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "host": "www.lagou.com",
        "referer": f"https://www.lagou.com/wn/zhaopin?fromSearch=true&kd=%25E4%25BA%25BA%25E5%25B7%25A5%25E6%2599%25BA%25E8%2583%25BD&pn={page_no-1}",
        "cookie": """index_location_city=%E5%85%A8%E5%9B%BD; RECOMMEND_TIP=1; JSESSIONID=ABAABJAABAJACEJ7CF173DD21E5C6D072846A49E9C5B120; WEBTJ-ID=20250223225942-19533523d0992-0ed8590a8f5ac5-1c525636-1440000-19533523d0a19ce; sensorsdata2015session=%7B%7D; user_trace_token=20250223225949-8b7149ed-504d-4610-a00a-1360cb15db5a; LGUID=20250223225949-e365e8be-348a-415b-9fe2-eb1faf2f33e1; _ga=GA1.2.158572295.1740322789; LGRID=20250223230202-bb00f215-2ae3-491e-92fc-369c45abd68c; _ga_DDLTLJDLHH=GS1.2.1740322790.1.1.1740322922.43.0.0; gate_login_token=v1####d1522b0f665a797c9234afd8040e68eb43ae38de967271d000601421235b8951; LG_HAS_LOGIN=1; _putrc=576A121222766EE5123F89F2B170EADC; login=true; hasDeliver=0; privacyPolicyPopup=false; __lg_stoken__=62a94fa51dc8491ba41f3a0642284ceaa1b2d6029648df44683a4bfbded2b33b9c5b67f187a6c09bb9bacfaae0e962c2fe2d66899ca0766bcbbb9bdb2355b88c35d96a5d03b6; unick=%E7%94%A8%E6%88%B75134; showExpriedIndex=1; showExpriedCompanyHome=1; showExpriedMyPublish=1; X_HTTP_TOKEN=62bb8932f865e8c37453940471063cec905923f5e8; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2219533523dad1c3-0730b3fa3249fa-1c525636-1440000-19533523dae2af9%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_utm_source%22%3A%22PC_SEARCH%22%2C%22%24os%22%3A%22MacOS%22%2C%22%24browser%22%3A%22Chrome%22%2C%22%24browser_version%22%3A%22133.0.0.0%22%7D%2C%22%24device_id%22%3A%2219533523dad1c3-0730b3fa3249fa-1c525636-1440000-19533523dae2af9%22%7D""",
    }

    return headers


def get_job_info(page_no):
    jost_details = []

    res = requests.get(url=get_url(page_no), data=get_params(page_no), headers=get_headers(page_no))
    res.encoding = res.apparent_encoding

    print(f"第{page_no}页返回结果：{res.status_code}")

    bs = BeautifulSoup(markup=res.text, features="html.parser")
    job_list = bs.find(name="div", attrs={"class": "list__YibNq"})
    jobs = job_list.find_all(name="div", attrs={"class": "item__10RTO"})

    for job in jobs:
        position_name, city = job.find(name="a").text[:-1].split("[")
        origin, distinct = city.split("·")
        money = job.find(name="span", attrs={"class": "money__3Lkgq"}).text
        experience_text = job.find(name="span", attrs={"class": "money__3Lkgq"}).find_next_sibling(string=True).strip()
        print(f"{position_name}\t{money}\t{origin}\t{distinct}\t{experience_text}")
        jost_details.append(
             {"position_name": position_name,
            "money": money,
            "origin": origin,
            "distinct": distinct,
            "experience_text": experience_text,
            }
        )

    return jost_details

if __name__ == '__main__':
    jost_details = get_job_info(1)
    print(jost_details)
