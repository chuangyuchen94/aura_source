import requests
import you_get
from bs4 import BeautifulSoup
import os

def get_page_url(url, page_no):
    video_page_url = ""

    if page_no == 1:
        video_page_url = url
    else:
        video_page_url = url + f"&page={page_no}"

    print(video_page_url)
    return video_page_url

def get_video_url(bili_video_card_wraps, video_set):
    bili_video_card_url = ""

    for bili_video_card_wrap in bili_video_card_wraps:
        link = bili_video_card_wrap.get("href")
        if "space" in link:
            continue

        if link in video_set:
            continue

        bili_video_card_url = link if link.startswith("https://") else f"https:{link}"
        video_set.add(link)

    return bili_video_card_url, video_set

def get_video(video_page_url, directory):
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    }

    res = requests.get(video_page_url, headers=headers)
    res.encoding = res.apparent_encoding
    bs = BeautifulSoup(markup=res.text, features="html.parser")

    bili_video_card_div = bs.find(name="div", attrs={"class": "video-list row"}).find_all(name="div", attrs={
        "class": "bili-video-card"})
    print(len(bili_video_card_div))
    print("\n" * 3)

    video_set = set()
    for video_num, bili_video_card in enumerate(bili_video_card_div):
        bili_video_card_wraps = bili_video_card.find(name="div", attrs={"class": "bili-video-card__wrap"}).find_all(
            name="a")
        print(f"第{video_num}个: {len(bili_video_card_wraps)}")

        bili_video_card_url, video_set = get_video_url(bili_video_card_wraps, video_set)
        print(f"{bili_video_card_url}")
        print("\n" * 2)

        os.system(f"you-get -o {directory} {bili_video_card_url}")

    print(f"set:{video_set}")

def download_video(video_url, max_page, video_directory):
    for page_no in range(1, max_page + 1):
        video_page_url = get_page_url(video_url, page_no)
        get_video(video_page_url, video_directory)


if __name__ == "__main__":
    url = "https://search.bilibili.com/all?keyword=deepseek&from_source=webtop_search&spm_id_from=333.788&search_source=5"
    download_video(url, max_page=1, video_directory="./video")
