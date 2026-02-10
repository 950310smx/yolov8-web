@echo off
REM 切换到你的项目根目录
cd "D:\yolov8-cpu-project"
 
REM 激活虚拟环境
REM 虚拟环境在另一个位置，需要提供其完整路径
call "D:\yolov8-cpu-demo-2\.venv\Scripts\activate.bat"
 
REM 启动 Streamlit 应用，并通过 start 命令在新窗口中运行，不阻塞当前脚本
REM 'app.py' 是你的主 Streamlit 应用文件，现在 cd 到了正确的目录，所以可以直接找到
start "" cmd /k streamlit run app.py
 
REM 可以选择性地添加一些延迟，确保 Streamlit 有时间启动
REM ping 127.0.0.1 -n 5 > nul
 
REM 脚本到此结束，因为 Streamlit 会一直运行直到手动关闭其窗口