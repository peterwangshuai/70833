def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 二阶贝塞尔平滑函数
    def smooth_curve(p0, pm, p1, seg_num=22):
        curve_pts = []
        for t in np.linspace(0, 1, seg_num):
            lat = (1-t)**2 * p0[0] + 2*(1-t)*t * pm[0] + t**2 * p1[0]
            lon = (1-t)**2 * p0[1] + 2*(1-t)*t * pm[1] + t**2 * p1[1]
            curve_pts.append((lat, lon))
        return curve_pts

    # 直接飞越平滑弧线
    mid_point = ((start[0]+end[0])/2, (start[1]+end[1])/2)
    routes["直接飞越"] = smooth_curve(start, mid_point, end, seg_num=12)

    # 读取障碍物最大高度
    max_obs_height = 0
    for idx in range(len(obstacle_list)):
        current_height = obstacle_heights.get(idx, 50)
        if current_height > max_obs_height:
            max_obs_height = current_height

    # 高度足够只保留直飞
    if fly_height > max_obs_height or not obstacle_list:
        return routes

    # 合并全部障碍物
    all_polygons = []
    for obs_coords in obstacle_list:
        all_polygons.append(Polygon(obs_coords))
    merged_obs = unary_union(all_polygons)
    center_lat = np.mean([p[0] for obs in obstacle_list for p in obs])
    center_lon = np.mean([p[1] for obs in obstacle_list for p in obs])
    center_point = Point(center_lon, center_lat)

    lat_off, lon_off = meter_to_latlon_offset(center_lat, safe_radius)
    safe_buffer = merged_obs.buffer(safe_radius / 111319.9)

    # ========== 按你的图纸修改：左=西侧空地、右=东侧马路，锚点大幅外扩 ==========
    offset_scale = 9.2      # 初始偏移放大，绕到建筑外围空地
    left_ok = False
    right_ok = False
    left_waypoint = None
    right_waypoint = None
    max_try = 28
    step_add = 3.5

    for attempt in range(max_try):
        # 左绕行：建筑群【西侧】空地（对应图纸蓝虚线左段）
        left_waypoint = (center_point.y + lat_off * offset_scale, center_point.x - lon_off * offset_scale)
        # 右绕行：建筑群【东侧】马路（图纸右侧空地绕行）
        right_waypoint = (center_point.y - lat_off * offset_scale, center_point.x + lon_off * offset_scale)

        left_line = LineString([start, left_waypoint, end])
        right_line = LineString([start, right_waypoint, end])

        if not left_line.intersects(safe_buffer):
            left_ok = True
        if not right_line.intersects(safe_buffer):
            right_ok = True
        if left_ok and right_ok:
            break
        offset_scale += step_add

    # 生成平滑绕行曲线
    if left_ok:
        routes["左侧绕行"] = smooth_curve(start, left_waypoint, end, seg_num=22)
    if right_ok:
        routes["右侧绕行"] = smooth_curve(start, right_waypoint, end, seg_num=22)

    # 最优航线筛选（自动选距离短的那条，贴合示例蓝线）
    min_dist = float("inf")
    best_route = None
    best_name = ""
    for name, pts in routes.items():
        if name in ("左侧绕行", "右侧绕行"):
            dist = 0
            for i in range(len(pts)-1):
                dist += latlon_to_meter(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
            if dist < min_dist:
                min_dist = dist
                best_route = pts
                best_name = name
    if best_route is not None:
        routes[f"最优航线（{best_name}）"] = best_route

    return routes
