def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 直飞平滑参考线
    mid_zhifei = ((start[0]+end[0])/2, (start[1]+end[1])/2)
    def full_smooth(p0,pm,p1,n=12):
        arr=[]
        for t in np.linspace(0,1,n):
            la=(1-t)**2*p0[0]+2*(1-t)*t*pm[0]+t**2*p1[0]
            lo=(1-t)**2*p0[1]+2*(1-t)*t*pm[1]+t**2*p1[1]
            arr.append((la,lo))
        return arr
    routes["直接飞越"] = full_smooth(start,mid_zhifei,end,12)

    max_obs_height = 0
    for idx in range(len(obstacle_list)):
        h = obstacle_heights.get(idx,50)
        if h>max_obs_height:
            max_obs_height=h
    if fly_height>max_obs_height or not obstacle_list:
        return routes

    # 合并障碍物+生成安全缓冲区（航线与建筑最小距离固定=safe_radius）
    all_poly = []
    for co in obstacle_list:
        all_poly.append(Polygon(co))
    merged = unary_union(all_poly)
    buf = merged.buffer(safe_radius / 111319.9)
    buf_bound = buf.boundary

    # -------------------- 左侧多段折线（沿缓冲区外边缘，多拐点、最短绕行） --------------------
    def get_multi_polyline(side_offset):
        clat = np.mean([p[0] for obs in obstacle_list for p in obs])
        clon = np.mean([p[1] for obs in obstacle_list for p in obs])
        cpoint = Point(clon,clat)
        lat_off,lon_off = meter_to_latlon_offset(clat,safe_radius)
        mid_anchor = (cpoint.y + lat_off*side_offset, cpoint.x - lon_off*side_offset)
        # 拆分3段折线：起点→近障拐点→远障拐点→终点，多段折、沿安全边线
        p1 = mid_anchor
        p2 = ((p1[0]+e_lat)/2.1, (p1[1]+e_lon)/1.9)
        poly_pts = [start, p1, p2, end]
        check_line = LineString(poly_pts)
        if not check_line.intersects(buf):
            return poly_pts
        else:
            # 偏移放大重新生成点位
            return get_multi_polyline(side_offset+1.8)

    left_line_pts = get_multi_polyline(7.2)
    right_line_pts = get_multi_polyline(-7.2)

    routes["左侧绕行"] = left_line_pts
    routes["右侧绕行"] = right_line_pts

    # 计算两条多段折线总里程，总距离最短自动设为最优航线
    def calc_total_len(pts):
        dist=0
        for i in range(len(pts)-1):
            dist += latlon_to_meter(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
        return dist
    len_left = calc_total_len(left_line_pts)
    len_right = calc_total_len(right_line_pts)

    if len_left <= len_right:
        routes[f"最优航线（左侧最短）"] = left_line_pts
    else:
        routes[f"最优航线（右侧最短）"] = right_line_pts

    return routes
