# components.py
import time
from utils import calculate_distance  # 从我们的工具文件导入距离计算函数
from config import BUOY_LAT, BUOY_LON, WARNING_DISTANCE_METERS

class AlarmSystem:
    def __init__(self):
        self.buoy_lat = BUOY_LAT
        self.buoy_lon = BUOY_LON
        self.warning_distance = WARNING_DISTANCE_METERS
        # 用于防止对同一艘船在短时间内重复报警
        self.last_alarm_time = {}

    def update_buoy_location(self, lat, lon):
        """用于实时更新浮标自身的位置"""
        self.buoy_lat = lat
        self.buoy_lon = lon

    @staticmethod
    def get_alarm_level(distance):
        """根据距离返回告警等级: 1(一级,≤100m) / 2(二级,≤150m) / 3(三级,≤200m) / None"""
        if distance is None:
            return None
        if distance <= 100:
            return 1
        elif distance <= 150:
            return 2
        elif distance <= 200:
            return 3
        return None

    def check(self, mmsi, lat, lon):
        """检查目标是否在告警区内，并返回(是否告警, 距离, 告警等级)"""
        distance = calculate_distance(self.buoy_lat, self.buoy_lon, lat, lon)
        level = self.get_alarm_level(distance)

        if distance is not None and distance < self.warning_distance:
            # 告警冷却：5分钟内对同一艘船只报一次警
            now = time.time()
            if mmsi not in self.last_alarm_time or (now - self.last_alarm_time[mmsi]) > 300:
                self.last_alarm_time[mmsi] = now
                return True, distance, level

        return False, distance, level


class Tracker:
    def __init__(self):
        # 字典的键是 mmsi，值是另一个字典，包含该船的最新信息
        self.tracked_ships = {}

    def update(self, mmsi, lat, lon, timestamp):
        """更新或添加一艘船的追踪信息"""
        self.tracked_ships[mmsi] = {
            'latitude': lat,
            'longitude': lon,
            'last_seen_ts': timestamp,
            'update_time': time.time()
        }

    def get_ship_info(self, mmsi):
        """获取特定船只的信息"""
        return self.tracked_ships.get(mmsi)

    def get_all_ships(self):
        """获取所有正在追踪的船只"""
        return self.tracked_ships
