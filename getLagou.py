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
        "cookie": """index_location_city=%E5%85%A8%E5%9B%BD; RECOMMEND_TIP=1; _ga=GA1.2.201396242.1740444374; user_trace_token=20250225084619-e57426b2-f845-42c9-a6da-19ee0059fe6b; LGUID=20250225084619-645f68bf-d09f-40dc-804c-9445ed042b77; _ga_DDLTLJDLHH=GS1.2.1740444385.1.0.1740444385.60.0.0; gate_login_token=v1####d1522b0f665a797c9234afd8040e68eb43ae38de967271d000601421235b8951; LG_HAS_LOGIN=1; showExpriedIndex=1; showExpriedCompanyHome=1; showExpriedMyPublish=1; hasDeliver=0; privacyPolicyPopup=false; JSESSIONID=ABAABJAACBBAAHG420C6ACDA3B30D68AF4ACEDCF09AAC34; WEBTJ-ID=02262025%2C084858-1953fba712119-0a5a6999686681-26011a51-1382400-1953fba712211a1; sensorsdata2015session=%7B%7D; LG_LOGIN_USER_ID=v1####4f68d128b18bc548ccbfbcce920bb01ef4a619ad978a6e9bd5d05554534573b7; _putrc=576A121222766EE5123F89F2B170EADC; login=true; unick=%E9%99%88%E6%A0%91%E5%BD%A6; X_HTTP_TOKEN=c4bd1fdfd8d3d9ae86903504715fbce1d9940cb566; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%221953a917a292e8-03963bac7d5d67-26011a51-1382400-1953a917a2a32d%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E8%87%AA%E7%84%B6%E6%90%9C%E7%B4%A2%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC%22%2C%22%24latest_referrer%22%3A%22https%3A%2F%2Fwww.google.com%2F%22%2C%22%24os%22%3A%22Windows%22%2C%22%24browser%22%3A%22Chrome%22%2C%22%24browser_version%22%3A%22133.0.0.0%22%7D%2C%22%24device_id%22%3A%221953a917a292e8-03963bac7d5d67-26011a51-1382400-1953a917a2a32d%22%7D2""",
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

def get_all_job_info(max_page=0):
    with open("job_info.txt", mode="w", encoding="utf8") as job_info_file:
        page_no = 1

        while True:
            if max_page != 0 and page_no > max_page:
                break

            job_infos = get_job_info(page_no)
            if len(job_infos) == 0:
                break

            page_no += 1

            for job_detail in job_infos:
                job_info_file.write(f"{job_detail.get("position_name")}\t{job_detail.get("money")}\t{job_detail.get("origin")}\t{job_detail.get("distinct")}\t{job_detail.get("experience_text")}\n")

if __name__ == '__main__':
    get_all_job_info(max_page=2)
