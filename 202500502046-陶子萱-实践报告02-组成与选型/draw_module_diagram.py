# -*- coding: utf-8 -*-
"""绘制 课题六“行为克隆游戏智能体”系统模块图（实践报告02 用）。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---- 配色 ----
C_RES   = '#E3F2E5'   # 外部资源：浅绿
C_USER  = '#E9EDF6'   # 外部使用者：浅蓝灰
C_MOD   = '#DCEBF7'   # 内部模块：浅蓝
C_OPT   = '#FBE8D2'   # 可选扩展：浅橙
C_EDGE  = '#44546A'   # 边框
C_ARROW = '#2F3B52'   # 箭头
C_TEXT  = '#1A202C'   # 文字

fig, ax = plt.subplots(figsize=(13.2, 10.8), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 13.6)
ax.axis('off')


def box(x, y, w, h, text, fc, fs=12, bold=False, ec=C_EDGE, lw=1.4, ls='-'):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.12",
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=C_TEXT, zorder=4)


def arrow(p1, p2, label='', lp=None, rad=0.0, ls='-', lw=1.5, color=C_ARROW):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=16,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=6)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        tx, ty = (mx, my) if lp is None else lp
        ax.text(tx, ty, label, ha='center', va='center', fontsize=11,
                color=C_TEXT, zorder=7,
                bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.88))


# ================= 标题 =================
ax.text(7, 13.25, '行为克隆游戏智能体（Open Pixel2Play）· 系统模块图',
        ha='center', va='center', fontsize=18, fontweight='bold', color=C_TEXT)

# ================= 系统边界（虚线框） =================
bound = FancyBboxPatch((0.95, 3.5), 11.5, 6.55,
                       boxstyle="round,pad=0.05,rounding_size=0.18",
                       linewidth=1.5, edgecolor='#8A97A6', facecolor='#F7FAFC',
                       linestyle=(0, (6, 4)), zorder=0)
ax.add_patch(bound)
ax.text(1.18, 9.72, '系统内部（本项目开发范围）', fontsize=12.5, color='#5A6B7B',
        ha='left', va='center', zorder=2)

# ================= 外部资源 =================
ax.text(7, 12.18, '外部资源', fontsize=13, color='#5A6B7B', ha='center', fontweight='bold')
box(2.2, 11.3, 3.5, 1.05, '官方仓库\n+ 150M 权重', C_RES, fs=12)
box(6.0, 11.3, 3.3, 1.05, 'HuggingFace\ntoy 数据', C_RES, fs=12)
box(9.9, 11.3, 3.3, 1.05, '学校 NVIDIA\n服务器（GPU）', C_RES, fs=12)

# ================= 内部模块 =================
box(5.6, 9.55, 4.7, 1.2, '① 数据准备\n下载权重 + toy 数据，切 200 帧测试集', C_MOD, fs=12, bold=True)
box(5.6, 7.95, 3.5, 1.05, '② 推理\n150M 模型 → 键鼠动作', C_MOD, fs=12, bold=True)
box(5.6, 6.4, 4.7, 1.2, '③ 评测\n按键准确率 + 鼠标相关系数', C_MOD, fs=12, bold=True)
box(5.6, 4.85, 4.7, 1.05, '④ 演示\n可视化对比 + 第 5 天录屏', C_MOD, fs=12, bold=True)
box(10.85, 7.95, 3.1, 1.3, '⑤ 微调（扩展）\n1 款游戏 ≤2h 微调', C_OPT, fs=11.5, bold=True)

# ================= 外部使用者 =================
ax.text(7, 2.85, '外部使用者', fontsize=13, color='#5A6B7B', ha='center', fontweight='bold')
box(2.0, 2.85, 2.7, 1.0, '本人（开发者）\n驱动全流程', C_USER, fs=12)
box(7.0, 2.85, 3.3, 1.0, '课程验收方（老师）\n接收交付物', C_USER, fs=12)

# ================= 箭头 =================
# 外部资源 → ①
arrow((2.2, 10.75), (4.6, 10.02), '脚本+权重')
arrow((6.0, 10.75), (5.6, 10.15), 'toy 数据')
arrow((9.9, 10.75), (7.0, 10.02), '算力')

# 主路径：① → ② → ③ → ④
arrow((5.6, 8.95), (5.6, 8.5), '权重+测试帧', lp=(6.75, 8.72))
arrow((5.6, 7.42), (5.6, 7.02), '预测动作', lp=(6.75, 7.22))
arrow((5.6, 5.8), (5.6, 5.4), '指标+对比', lp=(6.85, 5.6))

# ① → ③：人类标注（左侧长箭头）
arrow((3.25, 8.95), (3.25, 7.02), '帧+人类标注\n(user_action)', lp=(2.35, 8.0))

# ④ → 验收方
arrow((5.6, 4.32), (6.2, 3.35), '录屏 + 报告')

# 本人 → 系统（驱动，虚线，跨入系统边界）
arrow((2.0, 3.35), (2.0, 4.85), '操作 / 驱动全流程', lp=(3.35, 4.05), ls='--', lw=1.3)

# 可选分支：① → ⑤ → ②
arrow((7.95, 9.0), (9.95, 8.5), '1 款游戏 ≤2h 数据', lp=(9.35, 9.15), rad=0.18)
arrow((9.3, 7.95), (7.35, 7.95), '微调后权重', lp=(8.35, 7.55))

# ================= 图例 =================
lx, ly = 9.0, 1.35
for i, (c, lab) in enumerate([(C_RES, '外部资源'), (C_USER, '外部使用者'),
                              (C_MOD, '内部模块'), (C_OPT, '可选扩展')]):
    x = lx + i * 1.45
    ax.add_patch(Rectangle((x - 0.55, ly - 0.22), 1.1, 0.44,
                           facecolor=c, edgecolor=C_EDGE, linewidth=0.8, zorder=3))
    ax.text(x, ly - 0.55, lab, ha='center', va='center', fontsize=9.5, color=C_TEXT, zorder=4)

plt.tight_layout()
out = r'c:\Users\tao_z\Desktop\mini项目_6\模块图.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print('SAVED:', out)
