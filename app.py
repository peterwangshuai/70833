def generate_routes(start, end, obstacle_list, obstacle_heights, fly_height, safe_radius):
    routes = {}
    s_lat, s_lon = start
    e_lat, e_lon = end

    # 局部贝塞尔：只用首尾两段做圆弧，中间纯直线
    def start_arc(p0, pm, seg=14):
        """起点侧向外平滑拐弯"""
        pts = []
        for t in np.linspace(0, 0.38, seg):
            la = (1-t)**2 * p0[0] + 2*(1-t)*t * pm[0] + t**2 * e_lat
            lo = (1-t)**2 * p0[1] + 2*(1-t)*t * pm[1] + t**2 * e_lon
            pts.append((la, lo))
        return pts

    def end_arc(p_midline, p_end, seg=10):
        """终点处小幅向内收弯"""
        pts = []
        mid_p = ((p_midline[0]+p_end[0])/2, (p_midline[1]+p_end[1])/2)
        for t in np.linspace(0.65, 1.0, seg):
            la = (1-t)**2 * p_midline[0] + 2*(1-t)*t * mid_p[0] + t**2 * p_end[0]
            lo = (1-t)**2 * p_midline[1] + 2*(1-t)*t * mid_p[1] + t**2 * p_end[1]
            pts.append((la, lo))
        return pts

    # 直飞全程浅平滑弧线不变
    mid_zhifei = ((start[0]+end[0])/2, (start[1]+end[1])/2)
    def full_smooth(p0,pm,p1,n=12):
        arr=[]
        for t in np.linspace(0,1,n):
            la=(1-t)**2*p0[0]+2*(1-t)*t*pm[0]+t**2*p1[0]
            lo=(1-t)**2*p0[1]+2*(1-t)*t*pm[1]+t**2*p1[1]
            arr.append((la,lo))
        return arr
    routes["直接飞越"] = full_smooth(start,mid_zhifei,end,12)

    # 障碍物高度判定
    max_obs_height = 0
    for idx in range(len(obstacle_list)):
        h = obstacle_heights.get(idx,50)
        if h>max_obs_height:
            max_obs_height=h
    if fly_height>max_obs_height or not obstacle_list:
        return routes

    # 合并障碍物轮廓
    all_poly = []
    for co in obstacle_list:
        all_poly.append(Polygon(co))
    merged = unary_union(all_poly)
    clat = np.mean([p[0] for o in obstacle_list for p in o])
    clon = np.mean([p[1] for o in obstacle_list for p in o])
    cpoint = Point(clon,clat)
    lat_off,lon_off = meter_to_latlon_offset(clat,safe_radius)
    buf = merged.buffer(safe_radius/111319.9)

    # 控制点：左绕行锚点精准在建筑西侧空地（图纸蓝线路径）
    offset_scale =7.9
    left_ok=False
    right_ok=False
    lp=None
    rp=None
    maxtry=30
    add=2.1
    for _ in range(maxtry):
        # 左控制点：往西+往北偏移，匹配图西侧空地
        lp=(cpoint.y+lat_off*offset_scale*1.18, cpoint.x-lon_off*offset_scale*1.35)
        # 右控制点：东侧马路
        rp=(cpoint.y-lat_off*offset_scale, cpoint.x+lon_off*offset_scale*1.22)
        lline=LineString([start,lp,end])
        rline=LineString([start,rp,end])
        if not lline.intersects(buf):left_ok=True
        if not rline.intersects(buf):right_ok=True
        if left_ok and right_ok:break
        offset_scale+=add

    # 【核心构造：三段拼接 = 图纸样式：起弧+长直线+尾弧】
    def build_route(p_start,p_anchor,p_end):
        # 1、起点平滑拐弯段
        arc_start = start_arc(p_start,p_anchor,14)
        # 2、中间长直线（去掉首尾，中间直线路径）
        line_mid_start = arc_start[-1]
        line_mid_end = p_end
        # 3、终点收弯段
        arc_end = end_arc(line_mid_start,line_mid_end,10)
        # 拼接：起点弧 + 中间直线 + 末端收弧
        full = arc_start
        full.append(line_mid_end)
        full += arc_end[1:]
        return full

    if left_ok:
        routes["左侧绕行"]=build_route(start,lp,end)
    if right_ok:
        routes["右侧绕行"]=build_route(start,rp,end)

    # 择优：左绕距离最短=最优蓝色（和图纸一致）
    min_d=float("inf")
    best_r=None
    best_n=""
    for name,pts in routes.items():
        if name in ("左侧绕行","右侧绕行"):
            d=0
            for i in range(len(pts)-1):
                d+=latlon_to_meter(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1])
            if d<min_d:
                min_d=d
                best_r=pts
                best_n=name
    if best_r is not None:
        routes[f"最优航线（{best_n}）"]=best_r

    return routes
