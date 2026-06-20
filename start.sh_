#!/bin/bash

# 1. 启动后台服务 (使用 $HOME 自动匹配用户名)
source $HOME/myaichat/venv/bin/activate
cd $HOME/myaichat
python3 main.py > $HOME/myaichat/app.log 2>&1 &

# 2. 隐藏鼠标（屏蔽报错，防止该命令失败导致后续不执行）
unclutter -idle 0.1 -root > /dev/null 2>&1 &

# 3. 稍微多等一会儿，确保桌面系统完全渲染完毕
sleep 10

# 4. 启动浏览器并记录日志
chromium --kiosk --disable-infobars --noerrdialogs --disable-translate --incognito http://127.0.0.1:8000 > $HOME/myaichat/browser.log 2>&1
