import json
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D  #  用于创建自定义图例图标
import numpy as np
from scipy.interpolate import splprep, splev
import os

# 从配置中导入设置
try:
    from config import BUOY_LAT, BUOY_LON, WARNING_DISTANCE_METERS, OUTPUT_LOG_FILE
except ImportError:
    BUOY_LAT, BUOY_LON = 19.564408, 110.822627
    WARNING_DISTANCE_METERS = 200
    OUTPUT_LOG_FILE = 'ais_data_log.jsonl'


def latlon_to_xy(lat, lon, lat0, lon0):
    x = math.radians(lon - lon0) * 6371000 * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * 6371000
    return x, y


def run_visualization():
    if not os.path.exists(OUTPUT_LOG_FILE):
        print(f"错误：找不到文件 {OUTPUT_LOG_FILE}")
        return

    tracks = {}
    record_count = 0

    with open(OUTPUT_LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                ais_info = record.get('decoded_data') or record.get('data')
                if not ais_info: continue

                mmsi = ais_info.get('mmsi')
                lat = ais_info.get('lat') or ais_info.get('latitude')
                lon = ais_info.get('lon') or ais_info.get('longitude')

                if mmsi and lat is not None and lon is not None and abs(lat) > 0.1:
                    x, y = latlon_to_xy(lat, lon, BUOY_LAT, BUOY_LON)
                    tracks.setdefault(mmsi, []).append((x, y))
                    record_count += 1
            except:
                continue

    if not tracks:
        print("未提取到任何有效的轨迹。")
        return

    all_pts = np.array([pt for p_list in tracks.values() for pt in p_list])
    max_coord = np.max(np.abs(all_pts))
    limit = max(WARNING_DISTANCE_METERS * 1.5, max_coord * 1.2)

    arrow_size = limit * 0.04
    head_w = arrow_size * 0.8
    head_l = arrow_size

    plt.figure(figsize=(10, 10))
    try:
        cmap = matplotlib.colormaps.get_cmap('tab10')
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(tracks)))

    # --- 准备图例列表 ---
    legend_elements = []

    for i, (mmsi, points) in enumerate(tracks.items()):
        pts = np.array(points)
        _, indices = np.unique(pts, axis=0, return_index=True)
        pts = pts[np.sort(indices)]
        xs, ys = pts[:, 0], pts[:, 1]
        c = colors[i]

        plt.scatter(xs, ys, s=20, color=c, alpha=0.4, zorder=2)

        # 绘制航迹线并获取句柄
        line_label = f"MMSI {mmsi}"
        if len(xs) >= 4:
            try:
                tck, u = splprep([xs, ys], s=2)
                u_new = np.linspace(0, 1, 300)
                l_xs, l_ys = splev(u_new, tck)
                line, = plt.plot(l_xs, l_ys, color=c, linewidth=2, label=line_label, zorder=3)
            except:
                line, = plt.plot(xs, ys, color=c, linewidth=2, linestyle='--', label=line_label, zorder=3)
        else:
            line, = plt.plot(xs, ys, color=c, linewidth=2, linestyle='--', label=line_label, zorder=3)

        # 将船只航迹加入图例列表
        legend_elements.append(line)

        # 绘制箭头
        step = max(len(xs) // 5, 1)
        for j in range(1, len(xs), step):
            dx, dy = xs[j] - xs[j - 1], ys[j] - ys[j - 1]
            if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                plt.arrow(xs[j - 1], ys[j - 1], dx, dy, head_width=head_w, head_length=head_l, fc=c, ec=c,
                          length_includes_head=True, zorder=4)

    # 1. 绘制地图上的大浮标（不加 label）
    plt.scatter(0, 0, facecolors='none', edgecolors='black', s=250, marker='o', linewidth=2.5, zorder=5)

    # 2. 创建一个专门用于图例的小浮标图标
    buoy_legend_icon = Line2D([0], [0], marker='o', color='black', label='Buoy',
                              markerfacecolor='none', markersize=8,  # <--- 在这里控制图例图标的大小
                              markeredgewidth=1.5, linestyle='None')

    # 3. 创建警戒圈的图例图标
    zone_legend_icon = Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                              label=f'{int(WARNING_DISTANCE_METERS)}m Zone')

    # 绘制地图上的警戒圈（不加 label）
    circle = plt.Circle((0, 0), WARNING_DISTANCE_METERS, color='red', fill=False, linestyle='--', linewidth=2,
                        alpha=0.6, zorder=1)
    plt.gca().add_patch(circle)

    # 4. 重新组合图例
    # 把浮标和警戒圈放在最前面
    all_handles = [buoy_legend_icon, zone_legend_icon] + legend_elements
    plt.legend(handles=all_handles, loc='upper right', fontsize='x-small', ncol=2, framealpha=0.8)

    # 图形修饰
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.title(f"Scientific AIS Track Analysis\nRecords: {record_count}, Vessels: {len(tracks)}", fontsize=12)
    plt.xlabel("East-West Distance (m)")
    plt.ylabel("North-South Distance (m)")

    plt.savefig('ais_track_report_pro.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_visualization()