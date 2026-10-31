# main.py
# 核心主程序：负责调度配置、组件和工具，完成AIS数据处理和告警任务。

import json
import time
import serial
import socket
from pyais import decode

# 从我们自己创建的文件中导入所有需要的模块
from config import *
from components import AlarmSystem, Tracker
from plot import run_visualization

def extract_nmea_and_timestamp(line: str):
    """
    从可能带有前缀的原始行中，智能提取时间戳和纯NMEA报文。
    返回 (timestamp, nmea_sentence)，如果无效则返回 (None, None)。
    """
    timestamp = None
    nmea_sentence = None

    # 兼容格式如: [133981662129140000]!AIVDM...
    if ']' in line and ('!' in line or '$' in line):
        try:
            parts = line.split(']', 1)
            timestamp_str = parts[0].strip('[')
            if timestamp_str.isdigit():
                timestamp = int(timestamp_str)
            nmea_sentence = parts[1].strip()
        except (ValueError, IndexError):
            return None, None
    # 兼容纯NMEA格式
    elif line.startswith(('!', '$')):
        nmea_sentence = line.strip()
        # 如果没有外部时间戳，就使用系统当前时间
        timestamp = int(time.time() * 1e9)

    return timestamp, nmea_sentence


def pretty_print_decoded_info(msg, dist, in_zone, level):
    """
    一个专门用于在控制台清晰打印解码和告警信息的函数。
    """
    mmsi = msg.get("mmsi", "N/A")
    lat = msg.get("lat", "N/A")
    lon = msg.get("lon", "N/A")

    print("-" * 50)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] 收到位置报文:")
    print(f"  -> MMSI: {mmsi}")
    print(f"  -> 坐标: (Lat: {lat}, Lon: {lon})")

    if dist is not None:
        print(f"  -> 与浮标距离: {round(dist, 1)} 米")
        if in_zone:
            level_label = {1: "一级", 2: "二级", 3: "三级"}.get(level, "")
            print("\n    #######################################")
            print(f"    #   !!! {level_label}告警 !!! 船只进入预警区   #")
            print("    #######################################\n")


def process_line(line, outfile, alarm_outfile, alarm_system, tracker):
    """
    处理单行NMEA报文的核心逻辑函数。
    """
    timestamp, nmea = extract_nmea_and_timestamp(line)
    if not nmea:
        return

    try:
        msg = decode(nmea).asdict()
        msg_type = msg.get("msg_type")

        # --- 逻辑核心 ---
        # 1. TODO: 如果需要GPS更新浮标位置，在这里添加对GPS报文的处理逻辑
        #    (需要使用 pynmea2，因为 pyais 可能不支持GPS)
        #    例如: if msg.sentence_type == 'RMC': alarm_system.update_buoy_location(...)

        # 2. 如果是AIS位置报文
        if msg_type in (1, 2, 3, 18, 19):
            lat = msg.get("lat")
            lon = msg.get("lon")
            mmsi = msg.get("mmsi")

            if not all([lat, lon, mmsi]):
                return

            # 更新追踪器
            tracker.update(mmsi, lat, lon, timestamp)

            # 检查告警（返回: 是否告警, 距离, 告警等级）
            in_zone, dist, level = alarm_system.check(mmsi, lat, lon)

            # 在屏幕上打印实时信息
            pretty_print_decoded_info(msg, dist, in_zone, level)

            # 写入日志文件
            log_record = {
                'log_timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'raw_nmea': nmea,
                'decoded_data': msg,
                'analysis': {
                    'distance_to_buoy_m': dist,
                    'warning_triggered': in_zone,
                    'alarm_level': level
                }
            }
            outfile.write(json.dumps(log_record) + '\n')
            outfile.flush()

            # 如果在告警区内，单独写入告警日志文件
            if in_zone:
                alarm_record = {
                    'alarm_timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'alarm_level': level,
                    'mmsi': mmsi,
                    'lat': lat,
                    'lon': lon,
                    'distance_to_buoy_m': dist,
                    'raw_nmea': nmea,
                    'decoded_data': msg
                }
                alarm_outfile.write(json.dumps(alarm_record) + '\n')
                alarm_outfile.flush()

    except Exception as e:
        # print(f"解析或处理时发生错误: {e}") # 调试时可以打开
        pass


def run():
    """
    主运行函数，根据配置选择数据源并启动无限循环处理。
    """
    print("--- AIS 监测与告警系统启动 ---")

    # 初始化组件
    # 注意：这里的浮标坐标是静态的初始值，实际系统应由GPS动态更新
    alarm_system = AlarmSystem()
    tracker = Tracker()

    # 告警日志文件名带启动时间后缀
    alarm_filename = f"ais_alarm_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"数据源: {DATA_SOURCE_TYPE}")
    print(f"数据日志将保存到: {OUTPUT_LOG_FILE}")
    print(f"告警日志将保存到: {alarm_filename}")

    with open(OUTPUT_LOG_FILE, 'a', encoding='utf-8') as outfile, \
         open(alarm_filename, 'w', encoding='utf-8') as alarm_outfile:

        # --- 根据配置选择数据源，并进入主循环 ---
        if DATA_SOURCE_TYPE == 'FILE':
            print(f"正在从文件读取: {INPUT_DATA_FILE}")
            try:
                with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        process_line(line, outfile, alarm_outfile, alarm_system, tracker)
                print("--- 文件处理完成，程序退出 ---")
            except FileNotFoundError:
                print(f"错误: 找不到输入文件 '{INPUT_DATA_FILE}'")

        elif DATA_SOURCE_TYPE == 'SERIAL':
            print(f"正在监听串口: {SERIAL_PORT} @ {SERIAL_BAUDRATE}bps")
            while True:
                try:
                    with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE,
                                       bytesize=SERIAL_BYTESIZE,
                                       parity=SERIAL_PARITY,
                                       stopbits=SERIAL_STOPBITS,
                                       timeout=1) as ser:
                        print("串口已连接。")
                        while True:
                            line_bytes = ser.readline()
                            if line_bytes:
                                process_line(line_bytes.decode('utf-8', errors='ignore'),
                                             outfile, alarm_outfile, alarm_system, tracker)
                except serial.SerialException as e:
                    print(f"串口错误: {e}。5秒后重试...")
                    time.sleep(5)

        elif DATA_SOURCE_TYPE == 'NETWORK':
            print(f"正在连接网络主机: {NETWORK_HOST}:{NETWORK_PORT}")
            while True:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((NETWORK_HOST, NETWORK_PORT))
                        print("网络已连接。")
                        with s.makefile('r', encoding='utf-8', errors='ignore') as f:
                            while True:
                                line = f.readline()
                                if not line:
                                    print("网络连接已断开。")
                                    break
                                process_line(line, outfile, alarm_outfile, alarm_system, tracker)
                except Exception as e:
                    print(f"网络错误: {e}。5秒后重试...")
                    time.sleep(5)


if __name__ == "__main__":
    run()
    run_visualization()