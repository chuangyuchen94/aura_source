import requests
import uuid
import hashlib
import os

def get_unique_file_name(file_type):
    raw = uuid.uuid4().hex
    return hashlib.sha256(raw.encode()).hexdigest() + "." + file_type.replace(".", "")

def save_image_to_file(url, type, directory):
    res = requests.get(url)
    if res.status_code != 200:
        return

    image_data = res.content
    image_file_name = get_unique_file_name(type)
    image_path = os.path.join(str(directory), image_file_name)

    with open(image_path, mode="wb") as image_file:
        image_file.write(image_data)

def get_headers():
    headers = {
        "user-agent": """Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36""",
        "host": "image.baidu.com",
        "referer": "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=index&fr=&hs=0&xthttps=111110&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E7%BE%8E%E5%A5%B3",
        "cookie": """BDqhfp=%E7%BE%8E%E5%A5%B3%26%260-10undefined%26%262040%26%265; BAIDUID=9D16D022FA304A22BEF4FB395AC72033:FG=1; BAIDUID_BFESS=9D16D022FA304A22BEF4FB395AC72033:FG=1; BDUSS=h1N0lSRHZ3eFRZam9QMjN3a3ZzWU1EZXFGK0liSzh6ZVdmbFdnaUhIbThPYnhuQUFBQUFBPT0AAAAAAAAAAADm6xhw7CQhY2hlbmNoMDA3AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALyslGdnRC-8VG; BDUSS_BFESS=h1N0lSRHZ3eFRZam9QMjN3a3ZzWU1EZXFGK0liSzh6ZVdmbFdnaUhIbThPYnhuQUFBQUFBPT0AAAAAAAAAAADm6xhw7CQhY2hlbmNoMDA3AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALyslGdnRC-8VG; H_WISE_SIDS=62037_62113_62177_62185_62187_62183_62197_62284_62325; arialoadData=false; BIDUPSID=9D16D022FA304A22BEF4FB395AC72033; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=null; ab_sr=1.0.1_MmQxODU4NzkwZDY0ZDkwMGQwZGU2M2JiZjYxYzNjM2ZhMGQyMDdhZjBkZTYzN2UzMWVmMzZjNTdhMjE5Y2MyOTllNGE0YjhhYWM3ZTg4MDlkNmU5NzgyZjBjMzIyODRhODUxZjE4NDk2MmRhZDczYzgwOGUzZTc1NGE0YzQxMGRhOGFmODUxODU5NTVlODcxYjk0NjdjMzQ5MmJjZWQ3OQ==; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm"""
    }

    return headers

def get_images(page_no, max_page, image_directory):
    headers = get_headers()

    for page_no in range(page_no, max_page, 30):
        image_url = f"""https://image.baidu.com/search/acjson?tn=resultjson_com&logid=11372884182457470742&ipn=rj&ct=201326592&is=&fp=result&fr=&word=%E7%BE%8E%E5%A5%B3&cg=girl&queryWord=%E7%BE%8E%E5%A5%B3&cl=2&lm=&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn={page_no}&rn=30&gsm=5a&1740658639617="""

        try:
            res = requests.get(url=image_url, headers=headers)
            print(f"第{page_no - 30}~{page_no}条数据，处理结果：{res.status_code}\n")

            images_json = res.json()
            for num, image_data in enumerate(images_json["data"]):
                if not image_data:
                    continue

                image_type = image_data["type"]
                image_url = image_data["middleURL"]

                save_image_to_file(image_url, image_type, image_directory)
        except Exception as ex:
            print(ex)
            print(res)
            print(image_data)


if __name__ == "__main__":
    image_directory = "./images"
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)
    get_images(page_no=30, max_page=15000, image_directory=image_directory)
