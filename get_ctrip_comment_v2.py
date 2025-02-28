from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # 设置页面加载超时时间
    page.set_default_timeout(15000)

    page.goto("https://hotels.ctrip.com/hotels/99048095.html#ibu_hotel_review")

    try:
        # 处理可能的隐私条款弹窗（新版本网站常见）
        page.wait_for_selector('button:has-text("同意")', timeout=5000).click()
    except:
        pass

    # 增强版评论点击逻辑
    for _ in range(3):  # 最多重试3次
        try:
            # 等待评论标签可交互
            page.wait_for_selector(".comment-tab", state="attached", timeout=50000)
            page.click(".comment-tab")

            # 检测登录弹窗（携程新版常见登录框）
            login_frame = page.query_selector('iframe[id="loginFrame"]')
            if login_frame:
                print("检测到登录要求，正在处理...")

                # 切换到登录iframe
                with page.expect_popup() as popup_info:
                    # 选择账号密码登录方式
                    page.click("text=密码登录")

                # 获取弹出窗口（实际可能需要根据最新页面结构调整）
                popup = popup_info.value

                # 填写登录信息（需要替换为实际账号）
                popup.fill("#username", "your_username")
                popup.fill("#password", "your_password")

                # 处理验证码（需要人工干预或对接打码平台）
                popup.wait_for_selector(".captcha-img")
                captcha = input("请输入验证码：")
                popup.fill("#captcha", captcha)

                # 提交登录
                popup.click("#submit-btn")

                # 等待登录完成
                popup.wait_for_event("close")
                print("登录完成")

            break
        except Exception as e:
            print(f"点击评论失败，重试中... ({str(e)})")
            time.sleep(2)

    # 滚动加载优化（添加容错机制）
    load_attempt = 0
    while load_attempt < 5:  # 最多尝试加载5次
        try:
            load_more = page.locator(".load-more:visible")
            if load_more.count() == 0:
                break

            load_more.click()
            # 使用等待网络空闲代替固定延迟
            page.wait_for_load_state("networkidle")
            load_attempt = 0  # 重置尝试计数器
        except Exception as e:
            print(f"加载失败：{str(e)}")
            load_attempt += 1
            page.wait_for_timeout(2000)

    # 更健壮的评论解析
    comments = page.locator(".comment-item").all()
    for i, comment in enumerate(comments):
        try:
            print(f"评论 {i + 1}:")
            print(comment.inner_text())
            print("-" * 50)
        except:
            print(f"第 {i + 1} 条评论解析失败")

    browser.close()
