# -*- coding: utf-8 -*-
import os, re, json
import numpy as np
import cv2, open3d as o3d
from glob import glob

# ========= 路径 =========
BASE = os.path.abspath("data")   # /mnt/newpart/project_space/data
CALIB = os.path.join(BASE, "calib.json")
OUT_MERGED = os.path.join(BASE, "vis_merged")
OUT_PAIRS  = os.path.join(BASE, "vis_lidar_pairs")
os.makedirs(OUT_MERGED, exist_ok=True)
os.makedirs(OUT_PAIRS,  exist_ok=True)

# ========= 子雷达与相机的配对（保持你的关系）=========
PAIRS = [("0","9"), ("1","0"), ("2","3"), ("3","6")]

# ========= 可选：是否做旧代码里的坐标轴重映射 (x,y,z)->(z,-y,x) =========
AXIS_REMAP = False   # 默认 False；如你的 pcd 需要旧约定，可改 True

# ========= 常用函数 =========
def rodrigues_to_R(rvec3):
    r = np.asarray(rvec3, dtype=np.float64).reshape(3)
    R,_ = cv2.Rodrigues(r)
    return R

def read_pcd_xyz(path, max_points=None):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if AXIS_REMAP and xyz.size:
        x,y,z = xyz[:,0].copy(), xyz[:,1].copy(), xyz[:,2].copy()
        xyz[:,0], xyz[:,1], xyz[:,2] = z, -y, x
    if max_points and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
    return xyz

def draw_overlay(img, uv, depth):
    out = img.copy()
    if uv.size == 0:
        return out, 0
    h,w = out.shape[:2]
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colors = cv2.applyColorMap((d*255).astype(np.uint8), cv2.COLORMAP_JET)
    n_in = 0
    for (u,v), c in zip(uv, colors):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            n_in += 1
            cv2.circle(out, (ui,vi), 1, tuple(int(x) for x in c[0]), -1)
    return out, n_in

def project_points(xyz_obj, R, t, K, dist, is_fisheye):
    """
    xyz_obj: 物点在上游坐标（Virtual 或 Lidar 合成后）的坐标 (N,3)
    Xc = R * X + t
    """
    Pc = (R @ xyz_obj.T).T + t.reshape(1,3)
    front = Pc[:,2] > 0.1
    if not np.any(front):
        return np.zeros((0,2)), np.zeros((0,)), Pc, front
    obj = xyz_obj[front].astype(np.float64).reshape(-1,1,3)
    rvec,_ = cv2.Rodrigues(R.astype(np.float64))
    tvec   = t.astype(np.float64).reshape(3,1)
    if is_fisheye and dist is not None and len(dist) == 4:
        uv,_ = cv2.fisheye.projectPoints(obj, rvec, tvec, K.astype(np.float64), dist.astype(np.float64))
    else:
        uv,_ = cv2.projectPoints(obj, rvec, tvec, K.astype(np.float64), None if dist is None else dist.astype(np.float64))
    uv = uv.reshape(-1,2)
    depth = Pc[front,2]
    return uv, depth, Pc, front

def discover_timestamps_pairs():
    """从 cam*_*.png 与 lidar*_*.pcd 交集里找可用时间戳"""
    cam_ts = set(re.findall(r'_(\d+)\.png$', p)[0] for p in glob(os.path.join(BASE, "cam*_*.png")))
    lid_ts = set(re.findall(r'_(\d+)\.pcd$', p)[0] for p in glob(os.path.join(BASE, "lidar*_*.pcd")))
    return sorted(cam_ts & lid_ts)

def discover_timestamps_merged():
    """从 <ts>.pcd 找 merged 的时间戳"""
    ts = []
    for p in glob(os.path.join(BASE, "*.pcd")):
        name = os.path.basename(p)
        m = re.match(r'^(\d+)\.pcd$', name)
        if m:
            ts.append(m.group(1))
    # 也要求该 ts 下至少有一张 cam 图存在
    cam_ts = set(re.findall(r'_(\d+)\.png$', p)[0] for p in glob(os.path.join(BASE, "cam*_*.png")))
    return sorted([t for t in ts if t in cam_ts])

# ========= 模式 A：merged.pcd（VirtualLidar系）→ 各相机 =========
def project_merged_to_all_cams(calib, ts):
    merged_pcd = os.path.join(BASE, f"{ts}.pcd")
    if not os.path.exists(merged_pcd):
        print(f"[skip][merged] {ts} 缺少 {os.path.basename(merged_pcd)}")
        return
    xyzV = read_pcd_xyz(merged_pcd)   # merged 已在 VirtualLidar
    if xyzV.size == 0:
        print(f"[warn][merged] {ts} 空点云")
        return

    for cam_id, C in calib["camera"].items():
        img_path = os.path.join(BASE, f"cam{cam_id}_{ts}.png")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)

        # 取 V->C 外参
        R_V2C = rodrigues_to_R(C["virtualLidarToCam"]["rotate"])
        t_V2C = np.asarray(C["virtualLidarToCam"]["trans"], dtype=np.float64).reshape(3,1)

        K    = np.asarray(C["intri"], dtype=np.float64).reshape(3,3)
        dist = np.asarray(C.get("distor", []), dtype=np.float64).reshape(-1) if "distor" in C else None
        is_f = bool(C.get("isFish", 0))

        uv, depth, Pc, front = project_points(xyzV, R_V2C, t_V2C, K, dist, is_f)
        vis, n_in = draw_overlay(img, uv, depth)

        out_png = os.path.join(OUT_MERGED, f"merged_cam{cam_id}_{ts}.png")
        cv2.imwrite(out_png, vis)

        detR = float(np.linalg.det(R_V2C))
        ortho = float(np.linalg.norm(R_V2C.T @ R_V2C - np.eye(3), ord='fro'))
        print(f"[merged] ts={ts} V->C{cam_id}  z>0={int((Pc[:,2]>0.1).sum())}  in_img={n_in}  det={detR:+.4f}  ortho={ortho:.2e}  save={os.path.basename(out_png)}")

# ========= 模式 B：四个 lidar{i}.pcd（LiDAR系）→ 各自相机 =========
def project_lidar_pairs(calib, ts):
    for lidar_id, cam_id in PAIRS:
        img_path = os.path.join(BASE, f"cam{cam_id}_{ts}.png")
        pcd_path = os.path.join(BASE, f"lidar{lidar_id}_{ts}.pcd")
        if not (os.path.exists(img_path) and os.path.exists(pcd_path)):
            print(f"[skip][pair] ts={ts} 缺 {os.path.basename(img_path)} 或 {os.path.basename(pcd_path)}")
            continue
        img = cv2.imread(img_path)
        xyzL = read_pcd_xyz(pcd_path)
        if xyzL.size == 0:
            print(f"[warn][pair] ts={ts} L{lidar_id} 空点云")
            continue

        L = calib["lidar"][lidar_id]["lidarToVirtualLidar"]
        C = calib["camera"][cam_id]

        R_L2V = np.asarray(L["rotateMatrix"], dtype=np.float64).reshape(3,3)
        t_L2V = np.asarray(L["trans"], dtype=np.float64).reshape(3,1)

        R_V2C = rodrigues_to_R(C["virtualLidarToCam"]["rotate"])
        t_V2C = np.asarray(C["virtualLidarToCam"]["trans"], dtype=np.float64).reshape(3,1)

        # 合成 L->C
        R_L2C = R_V2C @ R_L2V
        t_L2C = R_V2C @ t_L2V + t_V2C

        K    = np.asarray(C["intri"], dtype=np.float64).reshape(3,3)
        dist = np.asarray(C.get("distor", []), dtype=np.float64).reshape(-1) if "distor" in C else None
        is_f = bool(C.get("isFish", 0))

        uv, depth, Pc, front = project_points(xyzL, R_L2C, t_L2C, K, dist, is_f)
        vis, n_in = draw_overlay(img, uv, depth)

        out_png = os.path.join(OUT_PAIRS, f"lidar{lidar_id}_cam{cam_id}_{ts}.png")
        cv2.imwrite(out_png, vis)

        detR = float(np.linalg.det(R_L2C))
        ortho = float(np.linalg.norm(R_L2C.T @ R_L2C - np.eye(3), ord='fro'))
        print(f"[pair ] ts={ts} L{lidar_id}->C{cam_id}  z>0={int((Pc[:,2]>0.1).sum())}  in_img={n_in}  det={detR:+.4f}  ortho={ortho:.2e}  save={os.path.basename(out_png)}")

def main():
    # 读取标定
    with open(CALIB, "r") as f:
        calib = json.load(f)

    # 简要输出相机内参与模型
    for cam_id, C in calib["camera"].items():
        K = np.asarray(C["intri"], dtype=np.float64).reshape(3,3)
        is_f = bool(C.get("isFish", 0))
        print(f"[K] cam{cam_id} isFish={is_f} fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    # 先跑 merged → cams
    ts_merged = discover_timestamps_merged()
    if ts_merged:
        print("\n=== 模式A：merged.pcd → 各相机 ===")
        for ts in ts_merged:
            project_merged_to_all_cams(calib, ts)
    else:
        print("\n[warn] 没发现 <ts>.pcd（merged），跳过模式A。")

    # 再跑 lidar{i}.pcd → 各自相机
    ts_pairs = discover_timestamps_pairs()
    if ts_pairs:
        print("\n=== 模式B：lidar{i}.pcd → 各自相机 ===")
        for ts in ts_pairs:
            project_lidar_pairs(calib, ts)
    else:
        print("\n[warn] 没发现 cam*_*.png 与 lidar*_*.pcd 的共同时间戳，跳过模式B。")

if __name__ == "__main__":
    main()
