# config.py
DATA_SOURCE_TYPE = 'FILE'
# 在 'FILE', 'SERIAL', 'NETWORK' 中选择一种
# 'FILE':    用于从本地文件进行测试
# 'SERIAL':  用于现场从串口读取数据
# 'NETWORK': 用于现场从网络TCP端口读取数据

WARNING_DISTANCE_METERS = 200.0
INPUT_DATA_FILE = '2点_ais.txt' # <--- 使用你的数据文件名
OUTPUT_LOG_FILE = 'ais_data_log.jsonl'

# --- 浮标初始位置配置 ---
# !!! 确保这两行存在并且有值 !!!
BUOY_LAT = 19.564408  # 您的示例坐标
BUOY_LON = 110.822627 # 您的示例坐标

# --- 串口配置 (现在这些配置将生效！) ---
# !!! 这里的串口号需要您根据第二步来确认和修改 !!!
SERIAL_PORT = '/dev/ttyS1'  # 示例: Linux/RK3576 上的硬件串口
# SERIAL_PORT = '/dev/ttyUSB0'  # 示例: 使用USB转串口适配器时

# 波特率必须与您的AIS接收机设置完全一致（根据手册: 115200, 8-N-1）
SERIAL_BAUDRATE = 115200
SERIAL_BYTESIZE = 8     # 8 数据位
SERIAL_PARITY = 'N'     # 无校验 (None)
SERIAL_STOPBITS = 1     # 1 停止位

# --- 网络配置 ---
NETWORK_HOST = '192.168.1.10'
NETWORK_PORT = 8001