# -*- coding: utf-8 -*-
"""生成《面试项目资料_人工智能研发工程师助理_v1.pdf》。

用途：2026-10-21（周三）双选会，目标岗位「人工智能研发工程师助理」。
内容全部取自本项目已有产物：
  report/实验结果与分析.md
  report/答辩准备.md
  runs/yolo11s_896/args.yaml, results.csv
  runs/yolo11s_896/*.png（训练曲线、混淆矩阵）
  report/figures/gui_*.png（上位机截图）
  max_det 口径追查（本机实测，见附录 C）

不重新训练、不改动任何实验产物。

用法：
  D:/yolo-env/Scripts/python.exe report/make_interview_pdf.py
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame,
                                Image, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'report', 'figures')
OUT = os.path.join(BASE, 'report',
                   '面试项目资料_人工智能研发工程师助理_v1.pdf')

INK = colors.HexColor('#1F3864')
BLUE = colors.HexColor('#2E5395')
LIGHT = colors.HexColor('#D9E2F3')
PALE = colors.HexColor('#F3F6FB')
ORANGE = colors.HexColor('#C55A11')
GREEN = colors.HexColor('#548235')
GRAY = colors.HexColor('#595959')
GRID = colors.HexColor('#B4C6E7')
WHITE = colors.white

FONT = 'MSYH'
FONT_BD = 'MSYHBD'
pdfmetrics.registerFont(TTFont(FONT, r'C:\Windows\Fonts\msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont(FONT_BD, r'C:\Windows\Fonts\msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_BD,
                              italic=FONT, boldItalic=FONT_BD)

PW, PH = A4

BODY = ParagraphStyle('body', fontName=FONT, fontSize=9.6, leading=15.6,
                      textColor=colors.HexColor('#262626'), alignment=TA_JUSTIFY,
                      wordWrap='CJK', spaceAfter=5, rightIndent=8,
                      allowWidows=0, allowOrphans=0)
BULLET = ParagraphStyle('bullet', parent=BODY, leftIndent=13, bulletIndent=3,
                        bulletFontName=FONT, bulletFontSize=9.6, spaceAfter=3.5)
TINY = ParagraphStyle('tiny', parent=BODY, fontSize=7.8, leading=11.5,
                      textColor=GRAY)
H1S = ParagraphStyle('h1', fontName=FONT_BD, fontSize=17, leading=22,
                     textColor=INK, spaceBefore=2, spaceAfter=3)
H2S = ParagraphStyle('h2', fontName=FONT_BD, fontSize=11.6, leading=16,
                     textColor=BLUE, spaceBefore=10, spaceAfter=4)
QUOTE = ParagraphStyle('quote', parent=BODY, fontSize=9.3, leading=14.8,
                       textColor=colors.HexColor('#333333'), leftIndent=10,
                       rightIndent=9, spaceAfter=4)
CELL = ParagraphStyle('cell', fontName=FONT, fontSize=8.0, leading=11.0,
                      textColor=colors.HexColor('#262626'), alignment=TA_CENTER,
                      wordWrap='CJK')
CELL_L = ParagraphStyle('cell_l', parent=CELL, alignment=TA_LEFT)


def P(text, style=BODY, **kw):
    return Paragraph(text, style, **kw)


def cell(text, fs=8.0, left=False, bold=False, color=None):
    st = ParagraphStyle('c%s' % fs, parent=CELL_L if left else CELL,
                        fontSize=fs, leading=fs + 3.0)
    if bold:
        st.fontName = FONT_BD
    if color is not None:
        st.textColor = color
    return Paragraph(str(text), st)


def h1(num, title):
    bar = Table([[Paragraph('<font color="#FFFFFF">%s</font>' % num,
                            ParagraphStyle('n', fontName=FONT_BD, fontSize=12,
                                           textColor=WHITE, alignment=TA_CENTER)),
                  Paragraph(title, H1S)]],
                colWidths=[30, None], rowHeights=[25])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), ORANGE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (0, 0), 0), ('RIGHTPADDING', (0, 0), (0, 0), 0),
        ('LEFTPADDING', (1, 0), (1, 0), 8),
        ('LINEBELOW', (0, 0), (-1, -1), 1.1, INK),
    ]))
    return [CondPageBreak(115), Spacer(1, 9), bar, Spacer(1, 7)]


def h2(text):
    return Paragraph(text, H2S)


def bullets(items):
    return [Paragraph(x, BULLET, bulletText='•') for x in items]


def numbered(items):
    out = []
    for i, x in enumerate(items, 1):
        st = ParagraphStyle('nb', parent=BULLET, bulletFontName=FONT_BD)
        out.append(Paragraph(x, st, bulletText='%d.' % i))
    return out


def table(data, widths, fs=8.0, header=True, left_cols=0, align='CENTER',
          row_bg=None, hdr_bg=INK, pad=3.2):
    rows = []
    for i, row in enumerate(data):
        r = []
        for j, v in enumerate(row):
            if isinstance(v, (Paragraph, Image, Table)):
                r.append(v)
            elif header and i == 0:
                r.append(cell(v, fs=fs, left=(j < left_cols), bold=True,
                              color=WHITE))
            else:
                r.append(cell(v, fs=fs, left=(j < left_cols)))
        rows.append(r)
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign=align)
    style = [
        ('GRID', (0, 0), (-1, -1), 0.4, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), pad),
        ('BOTTOMPADDING', (0, 0), (-1, -1), pad),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [('BACKGROUND', (0, 0), (-1, 0), hdr_bg)]
    if row_bg:
        for idx, c in row_bg.items():
            style.append(('BACKGROUND', (0, idx), (-1, idx), c))
    t.setStyle(TableStyle(style))
    return t


def note(title, text, color=ORANGE):
    inner = [Paragraph('<font color="#%s">%s</font>' % (
        color.hexval()[2:], title), ParagraphStyle(
            'nt', fontName=FONT_BD, fontSize=9.2, leading=13.5, wordWrap='CJK')),
        Spacer(1, 2), Paragraph(text, QUOTE)]
    t = Table([[inner]], colWidths=[500])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE),
        ('LINEBEFORE', (0, 0), (0, -1), 3, color),
        ('BOX', (0, 0), (-1, -1), 0.3, GRID),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def code_block(lines):
    txt = '<br/>'.join(x.replace('&', '&amp;').replace('<', '&lt;')
                       .replace('>', '&gt;') for x in lines)
    st = ParagraphStyle('code', fontName='Courier', fontSize=7.6, leading=11.2,
                        textColor=colors.HexColor('#1A1A1A'), wordWrap='CJK')
    t = Table([[Paragraph(txt, st)]], colWidths=[500])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F2F2F2')),
        ('BOX', (0, 0), (-1, -1), 0.4, colors.HexColor('#BFBFBF')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return t


def figure(path, width, caption=None):
    from PIL import Image as PILImage
    im = PILImage.open(path)
    h = width * im.height / im.width
    out = [Image(path, width=width, height=h)]
    if caption:
        out.append(Paragraph(caption, TINY))
    return out


def cover_page(c, doc):
    c.saveState()
    c.setFillColor(INK)
    c.rect(0, PH - 320, PW, 320, stroke=0, fill=1)
    c.setFillColor(ORANGE)
    c.rect(0, PH - 326, PW, 6, stroke=0, fill=1)
    c.setFillColor(colors.HexColor('#EEF2F9'))
    c.rect(0, 0, PW, 74, stroke=0, fill=1)
    c.restoreState()


def body_page(c, doc):
    c.saveState()
    c.setStrokeColor(GRID)
    c.setLineWidth(0.6)
    c.line(46, PH - 42, PW - 46, PH - 42)
    c.setFont(FONT, 8)
    c.setFillColor(GRAY)
    c.drawString(46, PH - 34, '无人机视角目标检测 · 面试项目资料')
    c.drawRightString(PW - 46, PH - 34, '目标岗位：人工智能研发工程师助理')
    c.line(46, 44, PW - 46, 44)
    c.setFont(FONT, 8)
    c.drawString(46, 30, '所有数字均来自本项目日志与实测脚本；口径见附录 C，请勿跨口径混用。')
    c.setFont(FONT_BD, 8.5)
    c.setFillColor(INK)
    c.drawRightString(PW - 46, 30, '第 %d 页' % c.getPageNumber())
    c.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4,
                      leftMargin=46, rightMargin=46, topMargin=58, bottomMargin=52,
                      title='面试项目资料：无人机视角目标检测（YOLO11s）',
                      author='YHAJ36',
                      subject='人工智能研发工程师助理 · 项目面试资料')
frame_cover = Frame(46, 64, PW - 92, PH - 64 - 60, id='cover',
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
frame_body = Frame(46, 52, PW - 92, PH - 52 - 58, id='body',
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([
    PageTemplate(id='cover', frames=[frame_cover], onPage=cover_page),
    PageTemplate(id='body', frames=[frame_body], onPage=body_page),
])

story = []

# ================================================================ COVER
story.append(Spacer(1, 52))
story.append(Paragraph('PROJECT INTERVIEW DOSSIER · 2026-10-19',
                       ParagraphStyle('cv_k', fontName=FONT_BD, fontSize=8.5,
                                      textColor=colors.HexColor('#F4B183'))))
story.append(Spacer(1, 12))
story.append(Paragraph('无人机视角<br/>小目标检测', ParagraphStyle(
    'cv_t', fontName=FONT_BD, fontSize=34, leading=44,
    textColor=colors.HexColor('#F2F5FA'))))
story.append(Spacer(1, 8))
story.append(Paragraph('VisDrone2019-DET 上的 YOLO11s 训练、误差归因与工程化记录',
                       ParagraphStyle('cv_s', fontName=FONT, fontSize=12.5,
                                      leading=18, textColor=LIGHT)))
story.append(Spacer(1, 18))
story.append(Paragraph('目标岗位：人工智能研发工程师助理　|　2026 年 10 月 21 日（周三）双选会',
                       ParagraphStyle('cv_s2', fontName=FONT, fontSize=9.2,
                                      leading=14, textColor=colors.HexColor('#D9E2F3'))))

story.append(Spacer(1, 62))
cards = [
    ('7,019', '图像样本', 'train 6,471 / val 548'),
    ('381,963', '标注框', '10 类，不均衡 42.1 : 1'),
    ('9.43 M', '模型参数', 'YOLO11s，21.7 GFLOPs'),
    ('0.478 / 0.489', 'mAP50 两种口径', 'max_det 300 / 902（见附录 C）'),
]
rows = []
for i in range(0, 4, 2):
    rows.append([Paragraph('<font size="18" color="#1F3864"><b>%s</b></font><br/>'
                           '<font size="8" color="#C55A11">%s</font><br/>'
                           '<font size="7" color="#595959">%s</font>' % cc,
                           ParagraphStyle('card', fontName=FONT, leading=13,
                                          alignment=TA_LEFT, wordWrap='CJK'))
                 for cc in cards[i:i + 2]])
t = Table(rows, colWidths=[250, 250], rowHeights=[62, 62])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), PALE),
    ('BOX', (0, 0), (-1, -1), 0.5, GRID),
    ('INNERGRID', (0, 0), (-1, -1), 0.5, WHITE),
    ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ('TOPPADDING', (0, 0), (-1, -1), 7),
]))
story.append(t)
story.append(Spacer(1, 24))
story.append(Paragraph('这份材料最值得记的三句话',
                       ParagraphStyle('cv_k2', fontName=FONT_BD, fontSize=10,
                                      textColor=ORANGE, spaceAfter=5)))
for _s in [
    '瓶颈在「小」不在「密」：<16px 目标召回 34.5%，≥96px 有 90.7%；控制尺寸后密度影响消失。',
    '别拿「场景太密」当结论：那只是密集场景里小目标更多造成的成分效应，交叉表能拆开。',
    '先报口径再报数：同一权重 max_det 300 报 0.478、902 报 0.489，差 1.06 pt 而 P/R 一点没变。',
]:
    story.append(Paragraph(_s, ParagraphStyle(
        'cv_l', fontName=FONT, fontSize=9.6, leading=15,
        textColor=colors.HexColor('#262626'), leftIndent=11,
        bulletIndent=1, wordWrap='CJK', spaceAfter=3), bulletText='•'))

story.append(Spacer(1, 14))
story.append(Paragraph('个人项目 · 仅供面试使用，不随代码仓库发布',
                       ParagraphStyle('cv_f', fontName=FONT, fontSize=8,
                                      textColor=GRAY)))

story.append(NextPageTemplate('body'))
story.append(PageBreak())

# ================================================================ 00
story += h1('00', '这份材料怎么用')
story.append(P('这是一份<b>给自己看</b>的材料，不是给面试官看的简历。目标是：让任何一句「为什么」'
               '都能在 3 秒内找到答案，并且答案带数字。'))
story.append(h2('材料地图'))
story.append(table([
    ['你现在的处境', '翻到'],
    ['只有 1 分钟，要背点东西', '第 01 章（30 秒版）+ 首页三句话'],
    ['要讲清项目，怕被追问', '第 01 章 2 分钟版 + 第 09 章'],
    ['被问「为什么效果不好」', '第 09 章（尺寸 vs 密度）+ 第 08 章（不均衡）'],
    ['被问「你这些数可信吗」', '第 10 章 + 附录 C（口径与雷区）'],
    ['被问「你踩过什么坑」', '第 11 章（负结果）+ 第 12 章（工程实践）'],
    ['面试前一晚', '第 13 章（高频追问）+ 第 15 章（不要做的事）'],
    ['要投简历', '第 14 章（三档写法，直接抄）'],
], [150, 350], fs=8.2, left_cols=1))
story.append(Spacer(1, 8))
story.append(note('读法建议',
                  '先读第 01 章和第 09 章——这两章是整份材料的骨架。第 09 章那个「密度是伪变量」的'
                  '交叉表，是这个项目里唯一别人不容易复述出来的东西，务必讲到。'))

# ================================================================ 01
story += h1('01', '项目速览与自我介绍')
story.append(h2('1.1 一句话版本'))
story.append(Paragraph('「用 YOLO11s 在 VisDrone2019-DET 上做无人机视角的小目标检测，'
                       'mAP50 0.478（放开 max_det 后 0.489），并把它做成了一个带 PyQt5 界面的'
                       '可演示系统；重点是做了误差归因，证明瓶颈是目标尺寸而不是场景密度。」', QUOTE))

story.append(h2('1.2 我的工作范围（如实说，别拔高）'))
story.append(table([
    ['环节', '我做了什么'],
    ['数据处理', '自己写 VisDrone→YOLO 标注转换，处理了 4 类真实数据异常；结果与参考实现逐字节比对零差异'],
    ['训练', '自己写训练脚本；超参选择、显存预算、时间盒控制、收敛判据都由我定并记录'],
    ['误差分析', '自己写 13 个分析脚本，做尺寸/密度分桶与交叉表，把「密」证伪成「小」'],
    ['界面', '自己写 PyQt5 上位机（图片/视频/摄像头），含工作线程与检测接入'],
    ['评测口径', '发现并实测了 max_det 截断对 mAP 的影响（附录 C），统一对外报数口径'],
], [62, 438], fs=8.2, left_cols=1))

story.append(h2('1.3 关键数字（先记这张表）'))
story.append(table([
    ['项', '数值', '口径 / 备注'],
    ['mAP@0.5', '0.478 / 0.489', 'max_det=300（报告口径） / 902（放开截断）'],
    ['mAP@0.5:0.95', '0.288 / 0.292', '同上'],
    ['Precision / Recall', '0.598 / 0.469', '两口径完全一致'],
    ['训练规模', '64 epochs / 7.5 h', '≈48,500 次梯度更新，809 iter/epoch'],
    ['显存峰值', '6.04 / 8.00 GB', '余量 1.96 GB，未 OOM'],
    ['推理速度', '10.8 ms/张', '896 输入，含前后处理约 15 ms'],
    ['数据', '6,471 / 548 张', '381,963 个标注框，10 类'],
    ['最大单图目标数', '902（train）/ 317（val）', 'max_det 必须 ≥ 902 才不截断'],
], [86, 118, 296], fs=8.2, left_cols=2))

story.append(h2('1.4 30 秒版'))
story.append(Paragraph('「项目是无人机视角的目标检测。用的是 VisDrone 这个公开数据集，'
                       '十类目标、三十八万个标注框，特点是目标特别小——三成目标不到 16 像素。'
                       '我用 YOLO11s 训到 mAP50 0.478，同时写了个 PyQt5 界面可以演示图片、视频和摄像头。'
                       '这个项目我花最多力气的地方是误差分析：把失败案例按尺寸和密度分桶，'
                       '开了一张交叉表，证明性能损失来自小目标，而不是场景拥挤。」', QUOTE))

story.append(h2('1.5 2 分钟版（建议背熟）'))
for _s in [
    '先说背景：无人机巡检、交通监测这类场景，摄像头在天上，目标在画面里很小，'
    '所以工业界一直用小目标检测这套方法。VisDrone 就是这个场景的标准公开数据集。',
    '数据这块：训练集 6471 张、验证集 548 张，一共 38 万个框。原始标注有四个坑——'
    '要丢掉「忽略区域」和「others」两类、有一个文件行尾多了逗号、图片有 11 种不同分辨率'
    '必须逐张读真实尺寸、还有 1 个零面积框。我把这些处理完，最后和一份已知正确的实现'
    '在 7019 个文件上逐字节比，零差异，24 秒跑完。',
    '训练这块：YOLO11s，896 输入，batch 8，SGD 加余弦退火，显存峰值 6.04 GB 留了 1.96 GB 余量。'
    '用 7.5 小时时间盒跑完 64 轮，大约 4 万 8 千次梯度更新，学习率退火到底，是收敛的。',
    '结果：mAP50 0.478。但我后来发现这个数是被截断的——验证集单图最多 317 个目标，'
    '而推理默认最多输出 300 个，我把上限放开到 902 再测，是 0.489，涨了 1.06 个点，'
    '而精确率和召回率完全没变，说明纯粹是密集图的排序损失。',
    '最有价值的是误差分析：目标尺寸从 96 像素以上到 16 像素以下，召回率从 90.7% 掉到 34.5%，'
    '差 2.6 倍。而按场景密度分桶看，好像越密越差；但我把尺寸固定住再看交叉表，'
    '32 到 96 像素这一列四个密度桶的召回率是 81.6、81.0、84.3、81.4，几乎不动。'
    '所以「密集」不是独立原因，是密集场景里小目标占比更高的成分效应。',
    '下一步我会做两件事：一是推理分辨率再往上走或者用切片推理，针对小目标；'
    '二是给稀少类别加权，因为 awning-tricycle 只有 532 个框，AP 只有 0.173。',
]:
    story.append(P(_s, BULLET, bulletText='•'))
story.append(Spacer(1, 4))
story.append(note('讲的时候的节奏',
                  '2 分钟版不要一口气背完。讲到「我把上限放开到 902 再测」停一下——'
                  '这是最容易被追问、也最容易加分的地方。等对方问「为什么会差」，再展开。',
                  color=GREEN))

# ================================================================ 02
story += h1('02', '业务背景与任务定义')
story.append(h2('2.1 应用场景'))
story.append(P('无人机视角的视觉检测用在：电力/管线巡检、交通流量与违章监测、'
               '大型活动与边检安防、农业估产、灾害搜救。共同点是<b>相机在高处、俯视角、'
               '目标在画面里占的像素很少</b>，与手持/车载视角的检测问题在分布上明显不同。'))
story.append(h2('2.2 三类难点'))
story.append(table([
    ['难点', '在这个项目里的具体表现'],
    ['目标极小', '30.8% 的目标短边 < 16 px；≥96 px 的只占 2.8%'],
    ['场景极密', '单图最多 902 个目标；默认 max_det=300 会直接截断召回'],
    ['类别极不均衡', 'car 158,930 框 vs awning-tricycle 3,778 框，42.1 : 1'],
], [88, 412], fs=8.2, left_cols=1))
story.append(h2('2.3 目标、约束与验收口径'))
story.append(table([
    ['项', '设定'],
    ['检测类别', 'VisDrone 10 类：pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor'],
    ['硬件约束', '单卡 RTX 4060 Laptop 8 GB，必须不 OOM'],
    ['时间约束', '7.5 小时时间盒（不是按轮数停，是按墙钟时间停）'],
    ['验收指标', 'mAP@0.5 为主，mAP@0.5:0.95 为辅；附加要求：可演示'],
    ['明确不追求', '不追 SOTA。VisDrone test-dev 无公开标注，本项目不做无偏泛化声称'],
], [78, 422], fs=8.2, left_cols=1))
story.append(Spacer(1, 6))
story.append(note('这一章的作用',
                  '面试官问「这个项目解决什么问题」时，答不到「小目标 + 密集 + 不均衡」'
                  '这三个词，就说明没真的做过。'))

# ================================================================ 03
story += h1('03', '数据集、标注转换与四类数据异常')
story.append(h2('3.1 数据规模与类别'))
story.append(table([
    ['划分', '图像数', '标注框数', '说明'],
    ['train', '6,471', '343,204', '训练实际使用'],
    ['val', '548', '38,759', '仅用于验证与选权重'],
    ['合计', '7,019', '381,963', '转换后有效框'],
], [88, 88, 108, 216], fs=8.2, left_cols=3))

story.append(h2('3.2 VisDrone 标注格式（8 字段，位次是坑）'))
story.append(code_block([
    'bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion',
    '索引:       0          1           2            3        4        5          6           7',
    '',
    '# 最容易错的地方：',
    '#   第 5 个字段（0-based 索引 4）是 score，不是类别！',
    '#   第 6 个字段（0-based 索引 5）才是 category。',
    '#   把 score 当类别 → 所有框都变成同一类，因为 score 通常都是 1。',
]))
story.append(P('转换为 YOLO 5 字段：<font name="Courier" size="8">class_id x_center y_center width height</font>，'
               '归一化公式：<font name="Courier" size="8">x_c=(x+w/2)/W</font>、'
               '<font name="Courier" size="8">y_c=(y+h/2)/H</font>、'
               '<font name="Courier" size="8">w_n=w/W</font>、'
               '<font name="Courier" size="8">h_n=h/H</font>。'))

story.append(h2('3.3 四类真实数据异常（都被处理了，且都有计数）'))
story.append(table([
    ['#', '异常', '实测规模', '处理'],
    ['1', 'category 0 = 忽略区域', '丢弃 10,191 框', '跳过（nc=10 不含此类）'],
    ['2', 'category 11 = others', '丢弃 1,564 框', '跳过'],
    ['3', '行尾多余逗号', '34 行，集中在 9999974_00000_d_0000053.txt', '先剥掉末尾空字段再判长度'],
    ['4', '退化框（宽或高 ≤ 0）', '1 个', '丢弃'],
], [22, 168, 190, 120], fs=8.2, left_cols=1, row_bg={3: colors.HexColor('#FBF3E8')}))
story.append(Spacer(1, 6))
story.append(P('还有第五个坑不算「异常」但更致命：<b>图片有 11 种不同分辨率</b>'
               '（1400×1050 有 2498 张、1400×788 有 1299 张、1360×765 有 1151 张、'
               '2000×1500 有 772 张、1916×1078 有 537 张……）。归一化必须除以'
               '<b>该图真实的宽高</b>，用固定值会让大部分框错位。'))
story.append(note('被追问「你怎么知道要处理这些」',
                  '不是拍脑袋：转换时统计了「字段数 ≠ 8 的行数」得到 34，再定位到单个文件；'
                  '统计了所有图片的 (w,h) 去重得到 11 种分辨率。每个异常都有一个计数可以报。'))

story.append(h2('3.4 转换结果的验证方法（最硬的一招）'))
story.append(P('和一份<b>已知正确的参考实现</b>在全部 7,019 个标签文件上<b>逐字节比对</b>：'
               'train 6,471 个、val 548 个，内容完全相同，零差异；文件名集合一致；'
               '有效框合计 381,963 与统计值吻合。整轮耗时 <b>24 秒</b>。'))
story.append(note('面试加分点',
                  '「我没法证明自己的实现是对的，所以找了个已知正确的实现做逐字节等价比对。」'
                  '——这句话比任何「我测试过了」都有说服力。'))

# ================================================================ 04
story += h1('04', 'YOLO11s 架构与选型')
story.append(h2('4.1 为什么是 YOLO11s'))
story.append(table([
    ['候选', '不选的理由'],
    ['YOLO11n', '更省显存，但容量不足以撑 10 类小目标；n 在 VisDrone 上明显吃紧'],
    ['YOLO11m', '38.8 MB 权重，896 输入 + batch 8 在 8 GB 卡上余量不足'],
    ['YOLO11s', '9.43 M 参数 / 21.7 GFLOPs，896+batch8 实测峰值 6.04 GB，留 1.96 GB 余量 → 选它'],
    ['更大的两阶段模型', '推理 10.8 ms 的实时性目标达不到'],
], [110, 390], fs=8.2, left_cols=1))

story.append(h2('4.2 三个检测尺度（896 输入下）'))
story.append(table([
    ['输出层', '下采样', '特征图', '主要覆盖的目标'],
    ['P3', '1/8', '112 × 112', '小目标（本项目的主战场）'],
    ['P4', '1/16', '56 × 56', '中等目标'],
    ['P5', '1/32', '28 × 28', '大目标'],
], [62, 62, 108, 268], fs=8.2, left_cols=3))
story.append(P('结构上是「CSP 主干 + PAN 式特征融合 + 三尺度解耦检测头」，'
               '无锚框（anchor-free），标签分配用 TAL。'
               '在 896 输入下 P3 的网格是 112×112，一个 16 像素的目标大约占 2×2 个格子；'
               '换到 640 输入，同样目标只剩 1.25 个格子——<b>这才是 896 的意义</b>。'))

# ================================================================ 05
story += h1('05', '输入分辨率 896：唯一的大杠杆')
story.append(P('这个项目里我做过的最重要的一个判断是：<b>在 8 GB 显存预算内，'
               '把分辨率顶到 896 比换模型、加模块都划算</b>。理由不是「越大越好」，'
               '而是三条约束取交集。'))
story.append(h2('5.1 三条约束的交点'))
story.append(table([
    ['约束', '数值', '推出的结论'],
    ['显存', '896 + batch 8 → 峰值 6.04 / 8.00 GB', '还能再加 batch 或换 m 模型？余量只剩 1.96 GB，加不动'],
    ['时间', '7.03 min/epoch，7.5 h 时间盒', '分辨率再翻倍会让单轮时间超预算'],
    ['任务', '30.8% 目标 < 16 px', '分辨率是唯一能直接改善这 30.8% 的旋钮'],
], [58, 216, 226], fs=8.2, left_cols=2))
story.append(h2('5.2 一句话的因果链'))
story.append(Paragraph('目标小 → 在特征图上占的格子少 → P3 尺度上的响应弱 → 召回上不去；'
                       '所以要么把小目标「放大」（提高输入分辨率 / 切片推理），'
                       '要么把 P3 做得更细（加 P2 检测头，但显存和耗时都会涨）。'
                       '我选前者，因为它在现有显存下就能吃到收益。', QUOTE))
story.append(note('诚实声明（很重要）',
                  '本项目<b>没有</b>在 640 上跑对照组，所以不能说「896 比 640 高多少 pt」。'
                  '如果面试官问「你怎么证明 896 更好」，正确回答是：'
                  '「我没有跑 640 对照，这是本项目的欠缺；我给的依据是显存预算内取最大值这个工程判断，'
                  '以及 P3 网格密度对 16 像素目标是否够用的几何推导，不是实测对照。」'
                  '—— 认下这个缺口，比编一个数字安全得多。'))

# ================================================================ 06
story += h1('06', '训练配置与评测协议')
story.append(h2('6.1 环境'))
story.append(table([
    ['项', '版本 / 配置'],
    ['Python', '3.12.10（项目内独立虚拟环境，与系统环境隔离）'],
    ['torch / torchvision', '2.11.0+cu128 / 0.26.0+cu128'],
    ['ultralytics', '8.4.178'],
    ['显卡 / 驱动', 'NVIDIA RTX 4060 Laptop 8 GB / 610.88'],
    ['其他', 'Pillow 12.3.0、OpenCV 5.0.0、numpy 2.5.2、PyQt5 5.15.11'],
], [110, 390], fs=8.2, left_cols=1))

story.append(h2('6.2 训练超参（完整）'))
story.append(table([
    ['项', '值', '项', '值'],
    ['模型', 'YOLO11s（COCO 预训练）', 'epochs 上限', '100（实际 64）'],
    ['imgsz', '896', '时间盒 time', '7.5 小时'],
    ['batch', '8', 'patience', '30'],
    ['optimizer', 'SGD', 'lr0 / lrf', '0.01 / 0.005'],
    ['momentum', '0.937', 'weight_decay', '0.0005'],
    ['cos_lr', 'True（余弦退火）', 'warmup_epochs', '3.0'],
    ['mosaic / close_mosaic', '0.8 / 5', 'mixup / copy_paste', '0.1 / 0.1'],
    ['scale / fliplr', '0.5 / 0.5', 'hsv_h/s/v', '0.02 / 0.8 / 0.5'],
    ['amp', 'True', 'deterministic', 'True（seed=0）'],
    ['workers', '0', 'cache', 'False'],
], [96, 150, 104, 150], fs=7.8, left_cols=0))
story.append(Spacer(1, 4))
story.append(P('<b>workers=0</b> 不是省事：Windows 上 DataLoader 用 spawn 起子进程会重新 import 主模块，'
               '造成重复初始化和偶发卡死，设 0 走主进程加载最稳。代价是数据加载慢一点，'
               '本项目验证过不是瓶颈。'))

story.append(h2('6.3 为什么用时间盒而不是按轮数停'))
story.append(P('训练日志里最常见的谎言是「我训了 100 轮」。<b>轮数不是资源，墙钟时间才是</b>。'
               '本项目用 <font name="Courier" size="8">time=7.5</font>（单位是小时）设墙钟上限，'
               'epochs=100 只是上限，谁先到算谁。实际停在 64 轮、27014.8 秒。'
               '好处是：可以在任何机器上给出「我需要多久」的准确承诺。'))
story.append(note('这一点在工程岗很吃香',
                  '面试官如果做过多卡训练，一定被「跑一晚上结果没跑完」坑过。'
                  '「我用墙钟时间约束训练预算」这句话会直接对上他的经验。'))

# ================================================================ 07
story += h1('07', '主实验结果与收敛性')
story.append(h2('7.1 总体指标'))
story.append(table([
    ['指标', 'best.pt', 'last.pt', '差值', '结论'],
    ['mAP@0.5', '0.478', '0.475', '+0.003', '选择偏置很小'],
    ['mAP@0.5:0.95', '0.288', '0.286', '+0.002', '同上'],
    ['mAP@0.75', '0.292', '—', '—', '严格 IoU 下明显下滑'],
    ['Precision', '0.598', '0.600', '−0.002', '—'],
    ['Recall', '0.469', '0.465', '+0.004', '—'],
], [96, 74, 74, 70, 186], fs=8.2, left_cols=4, row_bg={1: colors.HexColor('#FBF3E8')}))
story.append(P('上表为 <b>max_det=300</b> 口径（即报告口径）。放开到 902 后 '
               'best.pt 为 mAP50 <b>0.489</b> / mAP50-95 <b>0.292</b>，P/R 不变。'
               '详见第 10 章与附录 C。'))

story.append(h2('7.2 收敛性（三个证据，缺一个都别说过拟合/欠拟合）'))
story.append(table([
    ['epoch', 'mAP@0.5', 'mAP@0.5:0.95', '学习率 lr'],
    ['1', '0.2933', '0.1686', '3.33e-3'],
    ['8', '0.3850', '0.2222', '9.59e-3'],
    ['16', '0.4269', '0.2518', '8.50e-3'],
    ['24', '0.4441', '0.2616', '6.90e-3'],
    ['32', '0.4672', '0.2769', '5.03e-3'],
    ['40', '0.4767', '0.2838', '3.06e-3'],
    ['48', '0.4841', '0.2877', '1.55e-3'],
    ['56', '0.4874', '0.2904', '4.41e-4'],
    ['64', '0.4840', '0.2883', '5.60e-5'],
], [62, 96, 106, 96], fs=8.0))
story.append(Spacer(1, 2))
for _s in [
    '<b>证据 1</b>：最优 epoch 是第 58 轮（mAP50 0.488），此后波动只有 ±0.004。',
    '<b>证据 2</b>：第 40 → 64 轮只涨 0.7 个百分点（0.4767 → 0.4840），边际收益趋零。',
    '<b>证据 3</b>：余弦退火把学习率从 0.01 压到 5.6e-5，退火日程完整执行完毕。',
]:
    story.append(P(_s, BULLET, bulletText='•'))
story.append(Spacer(1, 4))
story.append(note('被问「才 64 轮够吗」',
                  '够。关键不是轮数而是梯度更新次数：6471 张 ÷ batch 8 = 809 iter/epoch，'
                  '64 轮 ≈ 48,500 次更新。同样 64 轮如果数据集只有 1400 张，那是 11,200 次，'
                  '差了 4 倍多。<b>「数据量大不等于需要更多轮」</b>，因为每轮见过的样本更多。'))

story.append(h2('7.3 训练曲线'))
_p = os.path.join(FIG, 'results.png')
if os.path.exists(_p):
    story += figure(_p, 400, '图 7-1　训练过程曲线（损失与指标随 epoch 变化）')
_p = os.path.join(FIG, 'confusion_matrix_normalized.png')
if os.path.exists(_p):
    story += figure(_p, 330, '图 7-2　归一化混淆矩阵：car 与 van/truck 的互混最明显')

# ================================================================ 08
story += h1('08', '类别不均衡，和一个隐藏变量')
story.append(h2('8.1 不均衡有多严重'))
story.append(P('训练集里 <b>car 有 158,930 个框，awning-tricycle 只有 3,778 个，相差 42.1 倍</b>。'
               '这种量级的差异一定会体现在指标上。'))
story.append(h2('8.2 逐类指标（val 集，10 类）'))
story.append(table([
    ['类别', 'val GT 框', 'AP@0.5', 'AP@0.5:0.95', '尺寸中位', '<16px 占比', '召回率'],
    ['car', '14,064', '0.843', '0.600', '32.5 px', '19.4%', '83.4%'],
    ['bus', '251', '0.639', '0.472', '35.2 px', '14.7%', '56.2%'],
    ['motor', '4,886', '0.562', '0.260', '20.0 px', '35.1%', '55.6%'],
    ['pedestrian', '8,844', '0.551', '0.263', '17.2 px', '44.8%', '51.5%'],
    ['van', '1,975', '0.516', '0.370', '33.6 px', '16.6%', '48.7%'],
    ['truck', '750', '0.451', '0.309', '38.6 px', '12.1%', '43.7%'],
    ['people', '5,125', '0.415', '0.164', '16.7 px', '47.1%', '41.4%'],
    ['tricycle', '1,045', '0.352', '0.204', '28.4 px', '18.4%', '34.6%'],
    ['bicycle', '1,287', '0.246', '0.116', '20.8 px', '30.8%', '26.0%'],
    ['awning-tricycle', '532', '0.173', '0.104', '29.3 px', '16.7%', '21.1%'],
], [86, 66, 62, 76, 62, 62, 52], fs=7.8, left_cols=0,
    row_bg={1: colors.HexColor('#FBF3E8'), 10: colors.HexColor('#F2F2F2')}))
story.append(Spacer(1, 4))
story.append(P('整体排序基本跟着样本量走：car 最多也最好（0.843），awning-tricycle 最少也最差（0.173）。'
               '但表里有两个「不听话」的：<b>van 和 truck 的尺寸中位比 car 还大</b>'
               '（33.6 / 38.6 px vs car 32.5 px），召回却只有 48.7% / 43.7%。'))
story.append(P('这说明尺寸之外还有第二个限制因素：<b>样本稀缺 + 与 car 的外观相似导致的类间混淆</b>。'
               '混淆矩阵里 car、van、truck 之间的互混是最明显的一块。'
               '这也直接决定了改进优先级——补 van/truck 的样本，比继续堆 car 有用。'))

story.append(h2('8.3 一个容易被漏掉的隐藏变量：部分类头复用'))
story.append(P('本版 ultralytics 在跨数据集微调时，会<b>按类名</b>重排预训练分类头。'
               'VisDrone 里 <font name="Courier" size="8">bicycle / car / bus / truck</font> 与 COCO 同名，'
               '所以这 4 类的分类头<b>起始于 COCO 的真实权重，而不是随机初始化</b>。训练日志实测：'))
story.append(code_block([
    'Remapped 4/10 cls head rows from pretrained weights by class name',
    'Transferred 499/499 items from pretrained weights',
]))
story.append(note('这条为什么重要',
                  '它意味着 <b>car 的高分有一部分来自更好的初始化起点，不只是数据量多</b>。'
                  '如果面试官问「car 为什么最好」，只答「因为 car 样本多」是不完整的——'
                  'van 有 1,975 个框也不算极少，但它没有 COCO 同名类头可用。'
                  '能答出「部分类头按名复用」这个机制，说服力完全不同。'))

# ================================================================ 09
story += h1('09', '误差分析：把「小」和「密」拆开')
story.append(P('这一章是整个项目最有价值的部分，也是唯一别人不容易复述的内容。'
               '结论先说：<b>性能瓶颈是目标尺寸，不是场景密度。</b>'
               '而「密度有影响」是个成分效应假象。'))
story.append(h2('9.1 方法'))
story.append(P('在 val 集上按 <b>IoU ≥ 0.5 的同类匹配</b>逐框判定 TP/FP/FN，'
               '然后按两个维度分桶统计召回率：<b>目标短边尺寸</b>与<b>单图目标总数</b>。'
               '关键是第三步——做<b>两维交叉表</b>，把其中一个维度固定住再看另一个。'))

story.append(h2('9.2 按尺寸：决定性因素'))
story.append(table([
    ['目标尺寸桶', '召回率', '占全部目标比例', '解读'],
    ['≥ 96 px', '90.7%', '2.8%', '基本能用'],
    ['32 ~ 96 px', '82.7%', '28.7%', '可用'],
    ['16 ~ 32 px', '62.1%', '37.7%', '明显退化'],
    ['< 16 px', '34.5%', '30.8%', '主要损失来源'],
], [104, 74, 108, 214], fs=8.2, left_cols=3,
    row_bg={4: colors.HexColor('#FBF3E8')}))
story.append(P('尺寸从 ≥96 px 掉到 <16 px，召回率相差 <b>2.6 倍</b>。而接近三分之一的目标'
               '（30.8%）就落在 <16 px 这一档——这就是全部性能损失的主体。'))

story.append(h2('9.3 按密度：看起来有影响'))
story.append(table([
    ['单图目标数', '< 20', '20 ~ 50', '50 ~ 100', '≥ 100'],
    ['召回率', '70.7%', '68.7%', '62.2%', '55.0%'],
], [104, 96, 96, 96, 96], fs=8.2, left_cols=0))
story.append(P('单看这张表，很容易得出「越密越差、密度是独立因素」的结论。'
               '<b>这个结论是错的。</b>'))

story.append(h2('9.4 交叉表：把尺寸固定住，密度影响就消失了'))
story.append(table([
    ['密度 ＼ 尺寸', '< 16 px', '16 ~ 32 px', '32 ~ 96 px', '≥ 96 px'],
    ['< 20', '31.5%', '57.7%', '81.6%', '86.3%'],
    ['20 ~ 50', '44.2%', '69.0%', '81.0%', '94.4%'],
    ['50 ~ 100', '36.5%', '64.9%', '84.3%', '90.0%'],
    ['≥ 100', '30.2%', '57.4%', '81.4%', '91.1%'],
], [104, 96, 96, 96, 96], fs=8.2, left_cols=0,
    row_bg=None, hdr_bg=INK))
story.append(Spacer(1, 4))
story.append(note('关键读数',
                  '看 <b>32 ~ 96 px 这一列</b>：四个密度桶分别是 <b>81.6 / 81.0 / 84.3 / 81.4</b>，'
                  '几乎完全不随密度变化。16~32 px 列同理（57.7 / 69.0 / 64.9 / 57.4，'
                  '无单调趋势）。<b>所以在控制尺寸之后，密度的影响基本消失。</b>'))
story.append(P('<b>结论的因果链</b>：密集场景里小目标占比本来就更高 → 按密度分桶时'
               '「密集桶」里塞了更多 <16 px 的目标 → 平均召回被拉低 → 看起来像「密度有害」。'
               '这叫<b>成分效应（composition effect）</b>，本质是混杂变量没控制住。'))
story.append(Paragraph('「先分桶、再交叉——把看起来的原因按住，看另一个是否还成立。」'
                       '这一招可以直接迁移到任何「A 和 B 都在变，到底谁起作用」的场景。', QUOTE))

story.append(h2('9.5 这段话面试时怎么讲（建议照说）'))
story.append(P('「我先按目标尺寸分桶，召回率从 96 像素以上的 90.7% 掉到 16 像素以下的 34.5%，'
               '而后者占了全部目标的 30.8%，所以瓶颈很明确是小目标。'
               '但我一开始也以为是场景太密——按单图目标数分桶，确实越密越差，'
               '从 70.7% 掉到 55.0%。后来我开了一张尺寸乘密度的交叉表，'
               '把尺寸固定住，32 到 96 像素这一列四个密度桶是 81.6、81.0、84.3、81.4，'
               '基本不动。所以密度是伪相关，真正的原因是密集场景里小目标占比更高。」', QUOTE))

# ================================================================ 10
story += h1('10', '评估口径与选择偏置')
story.append(P('这一章是我做完之后才补的，也是最容易在面试里加分的一章：'
               '<b>同一个模型，换个口径能差 1 个多点的 mAP。</b>'))

story.append(h2('10.1 选择偏置：best.pt 报 val 是不是作弊'))
story.append(P('best.pt 是按 val 的 fitness（0.1·mAP50 + 0.9·mAP50-95）选出来的，'
               '再拿它回报 val，<b>技术上确实有乐观偏差</b>。而且 VisDrone2019-DET 的 '
               'test-dev / test-challenge <b>没有公开标注</b>，无法做留出集评估。'))
story.append(P('所以我没有回避，而是<b>量化它</b>：同时评估 best.pt 与 last.pt，'
               'mAP@0.5 分别是 0.478 与 0.475，<b>差 0.0034 < 0.01</b>，说明选择偏置很小。'
               '报告里对这一点写了明确的口径声明。'))

story.append(h2('10.2 max_det 截断：0.478 与 0.489 的真实原因'))
story.append(P('训练日志里有一个警告：'))
story.append(code_block([
    'Dataset images contain up to 902 objects (train=902, val=317), but max_det=300.',
    'Setting max_det=902 to match the observed maximum.',
]))
story.append(P('训练时框架把上限自动抬到 902 了，但<b>保存下来的权重里 max_det 字段仍是 300</b>。'
               '所以用常规方式独立复评时，密集图的后处理结果会被截断。我做了对照实测：'))
story.append(table([
    ['max_det', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall'],
    ['300（实际被抬到 317）', '0.47820', '0.28822', '0.59749', '0.46893'],
    ['902（数据集真实上限）', '0.48879', '0.29209', '0.59749', '0.46893'],
    ['差值', '+0.01059', '+0.00387', '0', '0'],
], [140, 92, 104, 82, 82], fs=8.2, left_cols=0,
    row_bg={3: colors.HexColor('#FBF3E8')}))
story.append(Spacer(1, 4))
story.append(note('为什么 P/R 一点没变而 mAP 变了',
                  '因为 P/R 是在固定置信度阈值下算的，截断丢掉的是<b>低置信度的尾部框</b>，'
                  '不改变阈值以上的统计；而 mAP 是对整条 PR 曲线积分，密集图里被砍掉的'
                  '中低分框会让高召回段的曲线塌下去。<b>「P/R 不变但 mAP 掉」是排序损失的典型指纹</b>。',
                  color=GREEN))
story.append(P('对外报数我统一用 <b>0.478</b>（与交付报告一致，属保守下限），'
               '同时明确说明放开截断后是 <b>0.489</b>。两个数都报，反而比只报一个大的更可信。'))

story.append(h2('10.3 报数规则（三条，建议背）'))
for _s in [
    '<b>先说口径再说数</b>：数据集划分、输入尺寸、max_det、NMS IoU、置信度阈值，缺一个数就不能比。',
    '<b>不同口径的数不能跨表比较</b>：本材料所有表格都标注了口径，混用会得出错误结论。',
    '<b>保守数对外</b>：宁可报 0.478 并解释，也不要报 0.489 却不提截断。',
]:
    story.append(P(_s, BULLET, bulletText='•'))

# ================================================================ 11
story += h1('11', '负结果与工程判断')
story.append(P('下面这些是「做了但没按预期走」的记录。面试里讲负结果，比讲成功经验更显得真做过。'))
story.append(table([
    ['#', '一开始的直觉', '实测 / 想通之后', '怎么讲'],
    ['1', '场景太密所以检不出',
     '控制尺寸后密度影响消失（交叉表 81.6/81.0/84.3/81.4）', '「我被自己的第一直觉骗过一次」'],
    ['2', 'mAP 0.478 就是模型上限',
     '是 max_det=300 截断，放开 902 后 0.489', '「同一个模型换个口径差 1 个点」'],
    ['3', '64 轮太少，肯定欠拟合',
     '809 iter/epoch × 64 ≈ 48,500 次更新，lr 已退火到 5.6e-5', '「轮数不是资源，更新次数才是」'],
    ['4', 'car 分数高是因为 car 样本多',
     '还因为 bicycle/car/bus/truck 与 COCO 同名，类头按名复用', '「有个隐藏变量叫 cls_remap」'],
    ['5', 'van/truck 尺寸够大应该不难',
     '召回仅 48.7% / 43.7%，卡在样本稀缺 + 与 car 类间混淆', '「尺寸不是唯一因素」'],
    ['6', '验证集上最优就是泛化最优',
     'best vs last 差 0.0034，偏置小但不为零，且 test 无标注', '「我量化了它，没有回避」'],
], [18, 128, 210, 144], fs=7.8, left_cols=1))

# ================================================================ 12
story += h1('12', '工程实践与部署考虑')
story.append(h2('12.1 三个真实的坑（都有明确报错信息）'))
story.append(table([
    ['坑', '现象', '原因与解法'],
    ['配置目录污染',
     '多项目共存时 ultralytics 配置互相覆盖',
     '<font name="Courier" size="7.5">YOLO_CONFIG_DIR</font> 必须在 '
     '<font name="Courier" size="7.5">import ultralytics</font> 之前设置，指到项目内目录'],
    ['DLL 加载失败',
     '<font name="Courier" size="7.5">OSError [WinError 1114] Error loading torch/lib/c10.dll</font>',
     'PyQt5 必须先于 torch 加载会冲突；改成 <b>ultralytics(torch) 先 import，PyQt5 后 import</b>'],
    ['DataLoader 卡死',
     'Windows 上偶发挂起',
     'spawn 子进程会重新 import 主模块；<font name="Courier" size="7.5">workers=0</font> 走主进程'],
], [86, 176, 238], fs=7.8, left_cols=2))
story.append(note('第 2 条特别值得讲',
                  '这不是「查文档看到的」，是实测复现出来的：只要把 import 顺序反过来就稳定报错，'
                  '换回来就正常。面试官会认可这种<b>「能稳定复现才算定位」</b>的判断标准。'))

story.append(h2('12.2 路径可移植：不用绝对路径'))
story.append(P('所有脚本统一用 <font name="Courier" size="8">Path(__file__).resolve().parents[N]</font> '
               '推导项目根，再拼相对路径。好处是仓库 clone 到任何位置都能直接跑，'
               '而不是「在我机器上能跑」。'))
story.append(code_block([
    'from pathlib import Path',
    '',
    '# 本脚本位于 code/train/，向上 2 级即项目根',
    'ROOT    = Path(__file__).resolve().parents[2]',
    'DATA    = str(ROOT / "dataset_visdrone" / "yolo" / "VisDrone.yaml")',
    'WEIGHTS = str(ROOT / "models" / "yolo11s.pt")',
    'PROJECT = str(ROOT / "runs")',
]))

story.append(h2('12.3 上位机：为什么必须开线程'))
story.append(P('PyQt5 的界面在主线程里跑事件循环。<font name="Courier" size="8">cv2.VideoCapture.read()</font> '
               '是阻塞调用，直接在主线程里 while 循环会让窗口变成「未响应」。'
               '所以把视频/摄像头读取与推理放进 <font name="Courier" size="8">QThread</font>，'
               '通过信号回传帧，主线程只负责刷新显示。'))
_p = os.path.join(FIG, 'gui_主界面.png')
if os.path.exists(_p):
    story += figure(_p, 300, '图 12-1　PyQt5 上位机主界面（图片检测 / 视频检测 / 实时摄像）')
_p = os.path.join(FIG, 'gui_图片检测.png')
if os.path.exists(_p):
    story += figure(_p, 400, '图 12-2　单张图片检测效果，右侧为识别结果与类别统计')

story.append(h2('12.4 部署到边缘设备（Jetson）的思路）'))
for _s in [
    '<b>先导出再谈优化</b>：<font name="Courier" size="8">model.export(format="engine", half=True)</font> '
    '导出 TensorRT FP16；输入分辨率 896 是静态 shape，导出时可固定。',
    '<b>分辨率是最大的算力杠杆</b>：Jetson 上 896 的算力开销约为 640 的 2 倍（面积比），'
    '要在召回和帧率之间重新取点。',
    '<b>小目标场景优先考虑切片推理</b>：把 896 切成 2×2 的 448 子图分别推理再合并，'
    '等效把目标放大 2 倍，代价是耗时约 3~5 倍。',
    '<b>INT8 校准要小心</b>：小目标响应本来就弱，量化误差会进一步压掉低置信度框，'
    '必须用真数据校准集并单独验证 mAP 掉多少。',
]:
    story.append(P(_s, BULLET, bulletText='•'))

# ================================================================ 13
story += h1('13', '高频追问与回答')
story.append(P('下面按<b>面试官真实的追问链</b>组织：他会从结果问到原因，再问到方法可靠性。'
               '每条给「一句话 + 展开」。'))

QA = [
    ('你这个 mAP 0.478 怎么算的？可信吗？',
     '先说口径：VisDrone val 集 548 张、896 输入、max_det=300、NMS IoU 0.7，'
     'best.pt 上测得 0.478。要说可信度的话，我做过两件事：一是 best.pt 和 last.pt 差 0.0034，'
     '说明选权重带来的偏置很小；二是我发现 max_det=300 会截断密集图，放开到 902 实际是 0.489，'
     '所以我报的是<b>保守下限</b>。'),
    ('为什么不用 test 集？',
     'VisDrone2019-DET 的 test-dev 和 test-challenge 没有公开标注，'
     '要评估必须提交到官方服务器。本地无法做无偏评估，'
     '这一点我在报告里明确声明了，没有假装 val 就是泛化性能。'),
    ('为什么选 YOLO11s，不选 m 或者 n？',
     'n 容量不够撑 10 类小目标；m 是 38.8 MB 权重，896 加 batch 8 在 8 GB 卡上余量不足。'
     's 是 9.43 M 参数、21.7 GFLOPs，实测显存峰值 6.04 GB，留 1.96 GB 余量，是能跑满 896 的最大档。'),
    ('为什么用 896 不用 640？',
     '这个问题我要诚实回答：<b>我没有跑 640 的对照组</b>，所以给不出「高多少 pt」。'
     '我的依据是工程判断——三成目标不到 16 像素，896 输入下 P3 网格是 112×112，'
     '16 像素目标还能占到 2×2 格；640 下只剩 1.25 格。同时显存还有 1.96 GB 余量。'
     '如果重做，我会补 640 对照。'),
    ('64 个 epoch 够吗？会不会欠拟合？',
     '关键不是轮数而是梯度更新次数。6471 张除以 batch 8 是 809 次迭代每轮，'
     '64 轮约 48,500 次更新。另外三个收敛证据：最优轮是第 58 轮，之后只波动 ±0.004；'
     '第 40 到 64 轮只涨 0.7 个点；学习率已经余弦退火到 5.6e-5。'),
    ('car 为什么最好，别的类为什么这么差？',
     '两个原因。第一是尺寸：car 的中位尺寸 32.5 像素，而全数据 30.8% 的目标不到 16 像素，'
     '主要集中在行人、自行车这些小类。第二是个容易被忽略的机制——'
     '这一版 ultralytics 会按类名复用预训练分类头，bicycle、car、bus、truck 与 COCO 同名，'
     '所以这 4 类的分类头是从 COCO 权重起步的，不是随机初始化。'),
    ('van 和 truck 尺寸也不小，为什么召回还是低？',
     '这正是尺寸不是唯一因素的证据。它们中位尺寸 33.6 和 38.6 像素，比 car 还大，'
     '但召回只有 48.7% 和 43.7%。原因是样本稀缺（truck 只有 750 个框、'
     'van 1,975 个）加上与 car 外观相似造成的类间混淆，混淆矩阵里能看到。'),
    ('你怎么知道误差来自小目标而不是场景拥挤？',
     '我开了交叉表。单看密度分桶确实越密越差，70.7% 掉到 55.0%；'
     '但把尺寸固定住，32 到 96 像素这一列四个密度桶是 81.6、81.0、84.3、81.4，'
     '几乎不动。所以密度是成分效应——密集场景里小目标更多，把平均值拉低了。'),
    ('如果让你继续优化，你会做什么？',
     '按性价比排序：第一是切片推理（SAHI 那类），直接针对小目标，代价是耗时 3~5 倍；'
     '第二是提高稀有类采样权重，改善 van、truck、awning-tricycle；'
     '第三是补 640 的对照实验，把分辨率的收益量化出来。'
     '延长训练我反而排最后，因为收敛证据已经够了。'),
    ('你在这个项目里最大的收获是什么？',
     '学会了<b>不要相信第一直觉</b>。「场景太密所以检不出」听起来很合理，'
     '但它经不起交叉表的检验。做完这个项目我对「相关性不等于因果」这句话'
     '有了具体的手感——不是背概念，是真的被自己的数据打脸过一次。'),
    ('代码是你自己写的吗？',
     '是我自己写的。数据处理、训练、推理、界面四部分都是我一行行写的，'
     '写完都跑通了；分析脚本 13 个也是我自己写的。'
     '唯一需要说明的是标注转换这块，我用了一份已知正确的实现做<b>逐字节比对</b>来验证我的实现，'
     '7,019 个文件零差异——这是验证手段，不是照抄。'),
]

for _q, _a in QA:
    story.append(Paragraph('Q　' + _q, ParagraphStyle(
        'q', fontName=FONT_BD, fontSize=9.6, leading=14.5, textColor=ORANGE,
        spaceBefore=7, spaceAfter=2, wordWrap='CJK')))
    story.append(Paragraph(_a, QUOTE))
    story.append(Spacer(1, 1))

# ================================================================ 14
story += h1('14', '简历写法（可直接用）')
story.append(h2('14.1 详细版（5 条，适合项目栏）'))
for _s in [
    '<b>数据处理</b>：手写 VisDrone→YOLO 标注转换，识别并处理 4 类数据异常'
    '（忽略区域/others 类共丢 11,755 框、行尾多余逗号 34 行、11 种分辨率、1 个退化框）；'
    '转换结果与参考实现逐字节比对 7,019 文件零差异，耗时 24 s。',
    '<b>模型训练</b>：YOLO11s @ 896 输入、batch 8、SGD+余弦退火，'
    '在 RTX 4060 Laptop 8 GB 上以墙钟时间盒 7.5 h 完成 64 epoch / ≈48,500 次梯度更新，'
    '显存峰值 6.04 GB（余量 1.96 GB），mAP@0.5 <b>0.478</b>、mAP@0.5:0.95 <b>0.288</b>。',
    '<b>误差归因</b>：自建分桶 + 交叉表分析流程，证伪「场景密度是主因」的直觉——'
    '控制目标尺寸后密度影响消失（32~96 px 列 81.6/81.0/84.3/81.4），'
    '定位真实瓶颈为小目标（&lt;16 px 召回 34.5% vs ≥96 px 90.7%，占全量 30.8%）。',
    '<b>评测严谨性</b>：量化 best/last 选择偏置（ΔmAP50 = 0.0034）；'
    '发现默认 max_det=300 截断密集图，实测放开至 902 后 mAP@0.5 由 0.478 升至 0.489'
    '（P/R 不变，纯排序损失），统一对外报数口径并写入声明。',
    '<b>工程化</b>：PyQt5 上位机支持图片/视频/摄像头检测，用 QThread 隔离阻塞推理；'
    '全脚本以 <font name="Courier" size="7.5">Path(__file__)</font> 推导项目根，'
    'clone 即可运行；13 个分析脚本覆盖数据勘察、密度/尺寸分桶、显存与耗时标定。',
]:
    story.append(P(_s, BULLET, bulletText='▸'))

story.append(h2('14.2 精简版（3 条，适合一页简历）'))
for _s in [
    'VisDrone2019-DET 无人机视角目标检测，YOLO11s @ 896，mAP@0.5 <b>0.478</b>'
    '（放开 max_det 后 0.489），显存峰值 6.04 GB / 7.5 h 训练预算内收敛。',
    '误差归因：尺寸/密度双维分桶 + 交叉表，证伪密度假设，定位小目标瓶颈'
    '（&lt;16 px 召回 34.5%，占全量 30.8%）。',
    '工程化：手写标注转换（逐字节验证零差异）+ PyQt5 上位机（图片/视频/摄像头实时检测）。',
]:
    story.append(P(_s, BULLET, bulletText='▸'))

story.append(h2('14.3 一句话版（自我介绍用）'))
story.append(Paragraph('「我做过一个无人机视角的小目标检测项目，用 YOLO11s 在 VisDrone 上'
                       '做到 mAP50 0.478，最有意思的部分是我用交叉表证明了'
                       '「场景太密导致检不出」这个直觉是错的，真正的原因是目标太小。」', QUOTE))

# ================================================================ 15
story += h1('15', '现场演示与材料清单')
story.append(h2('15.1 带什么'))
story.append(table([
    ['材料', '状态', '说明'],
    ['本 PDF（打印 1 份）', '必须', '面试时自己看，不要递给对方'],
    ['手机/笔记本里的演示录屏', '必须', '现场网络与 GPU 都不可靠，录屏兜底'],
    ['3~5 张典型检测结果图', '必须', '挑一张大目标成功、一张密集小目标失败的对比'],
    ['关键指标截图（训练曲线）', '建议', '被问「真的训过吗」时直接给看'],
], [150, 60, 290], fs=8.2, left_cols=2))

story.append(h2('15.2 2 分钟演示流程'))
for _s in [
    '打开上位机，先点「图片检测」，选一张车辆密集的图 → 讲「这就是 VisDrone 的典型场景，'
    '小目标多、密度高」。',
    '把结果列表拉出来，指出「检测到的基本都是中近距离的车，远处的小目标漏了」。',
    '切到「视频检测」，放 10 秒 → 讲「实时性 10.8 ms 一张，视频能跑起来」。',
    '最后切到指标截图，讲交叉表那一条结论（这是压轴）。',
]:
    story.append(P(_s, BULLET, bulletText='•'))

story.append(h2('15.3 不要做的事'))
for _s in [
    '<b>不要临场跑训练</b>：7.5 小时的活，跑不出来只会暴露没准备。',
    '<b>不要主动说「我用了 XX 框架所以很快」</b>：那是框架的功劳，讲你自己的判断。',
    '<b>不要报 0.489 却不提口径</b>：被追问「怎么比报告高」会很被动。统一报 0.478 + 说明。',
    '<b>不要硬答没做过的事</b>：本项目没跑 640 对照、没做 test 集评估，'
    '被问到就直说是缺口。',
    '<b>不要背不出数字</b>：6471 / 548 / 381,963 / 0.478 / 64 / 7.5h / 6.04GB / 10.8ms，'
    '这八个必须张口就来。',
]:
    story.append(P(_s, BULLET, bulletText='✗'))

story.append(h2('面试前 5 分钟检查清单'))
story.append(table([
    ['□', '八个关键数字能一口气说完'],
    ['□', '三句话（小/密/口径）能不看稿讲出来'],
    ['□', '交叉表那四个数 81.6 / 81.0 / 84.3 / 81.4 记得'],
    ['□', '「我没跑 640 对照」这句话已经能坦然说出来'],
    ['□', '录屏能打开，且不需要网络'],
], [24, 476], fs=8.4, left_cols=1, header=False))

# ================================================================ A
story += h1('A', '复现命令')
story.append(h2('A.1 环境'))
story.append(code_block([
    'pip install -r requirements.txt',
    '# Python 3.12 / torch 2.11.0+cu128 / ultralytics 8.4.178 / PyQt5 5.15.11',
]))
story.append(h2('A.2 数据与转换'))
story.append(code_block([
    '# 1) 从 VisDrone 官方下载 VisDrone2019-DET，解压到 dataset_visdrone/',
    '#    得到 raw_train/ 与 raw_val/',
    'python tools/convert_visdrone.py',
]))
story.append(h2('A.3 训练与评估'))
story.append(code_block([
    '# 2) 训练（时间盒 7.5 小时）',
    'python code/train/train.py',
    '',
    '# 3) 验证（best.pt 与 last.pt 对照，量化选择偏置）',
    'python tools/val.py',
    '',
    '# 4) 推理（图片 / 文件夹 / 视频 / 摄像头）',
    'python tools/predict_demo.py <源路径>',
    '',
    '# 5) 上位机',
    'python code/app/main.py',
]))
story.append(note('注意 max_det',
                  '若要复现放开截断的口径，评估时需显式传 '
                  '<font name="Courier" size="7.5">max_det=902</font>，'
                  '否则会得到 0.478 而非 0.489。这是附录 C 的第 1 条雷区。'))

# ================================================================ B
story += h1('B', '术语表')
story.append(table([
    ['术语', '一句话解释'],
    ['mAP@0.5', 'IoU 阈值 0.5 下各类 AP 的平均值；比 P/R 更能反映整体排序质量'],
    ['mAP@0.5:0.95', 'IoU 从 0.5 到 0.95 每隔 0.05 取一次再平均，对框的精度要求高得多'],
    ['IoU', '预测框与真实框的交集除以并集'],
    ['TP / FP / FN', '正确检出 / 误检 / 漏检'],
    ['NMS', '非极大值抑制，同一目标只保留分数最高的框'],
    ['max_det', '单图最多输出多少个框；设小了会在密集图上截断召回'],
    ['AP（逐类）', '某一类在整条 PR 曲线下的面积'],
    ['fitness', 'ultralytics 用来选 best.pt 的综合分：0.1·mAP50 + 0.9·mAP50-95'],
    ['cls_remap', '微调时按类名把预训练分类头的同名行搬过来'],
    ['seed 波动', '固定其余条件下换随机种子造成的指标抖动，衡量结论是否稳定'],
], [104, 396], fs=8.0, left_cols=1))

# ================================================================ C
story += h1('C', '数字口径与雷区')
story.append(P('这一章是<b>防翻车</b>用的。任何数字被问到，先在这里查口径，再回答。'))
story.append(table([
    ['#', '雷区', '正确说法'],
    ['1', '把 0.478 当成模型上限',
     '0.478 是 <b>max_det=300</b> 口径；放开到 902 是 <b>0.489</b>。P/R 两口径相同（0.598/0.469）'],
    ['2', '用 val 指标声称泛化性能',
     'best.pt 按 val 选出再报 val 含选择偏置；test-dev 无公开标注，无法做无偏评估'],
    ['3', '说「场景密度导致漏检」',
     '控制尺寸后密度影响消失；真实原因是密集场景中小目标占比更高（成分效应）'],
    ['4', '说「896 比 640 高 X 个点」',
     '本项目<b>未做</b> 640 对照，只能说工程判断与几何推导，不能说 pt 差'],
    ['5', '把 car 的高分全归因于数据量',
     '还有 cls_remap：bicycle/car/bus/truck 与 COCO 同名，4/10 行类头复用预训练权重'],
    ['6', '说「64 轮太少」',
     '809 iter/epoch × 64 ≈ 48,500 次更新；lr 已退火到 5.6e-5，最优轮在第 58 轮'],
    ['7', '把 P 和 R 混为「准确率」',
     'Precision 0.598 / Recall 0.469 是两个不同数，别合并成一个「准确率」'],
    ['8', '训练内 val 与独立复评混用',
     '训练日志每轮 mAP50（第 64 轮 0.484）与独立复评（0.478 / 0.489）'
     '分属不同流程，不可跨表比较'],
], [18, 130, 352], fs=7.8, left_cols=2))

story.append(Spacer(1, 8))
story.append(Paragraph('数字速查（全部原始出处）',
                       ParagraphStyle('cx', fontName=FONT_BD, fontSize=10,
                                      textColor=BLUE, spaceBefore=6, spaceAfter=3)))
story.append(table([
    ['数字', '含义', '出处'],
    ['0.478 / 0.489', 'mAP@0.5（max_det 300 / 902）', '本机实测 max_det 对照'],
    ['0.288 / 0.292', 'mAP@0.5:0.95（同口径）', '同上'],
    ['0.598 / 0.469', 'Precision / Recall', '同上（两口径一致）'],
    ['64 / 27014.8 s', '实际轮数 / 总训练秒数', 'runs/yolo11s_896/results.csv'],
    ['6.04 / 8.00 GB', '显存峰值 / 显卡容量', '训练日志'],
    ['10.8 ms', '单张推理耗时（896）', 'tools/analyze_timing.py'],
    ['7,019 / 381,963', '图像总数 / 有效框总数', 'tools/convert_visdrone.py 统计'],
    ['902 / 317', '单图最大目标数（train / val）', '训练日志警告'],
], [104, 216, 180], fs=8.0, left_cols=1))

# ================================================================ D
story += h1('D', '项目文件索引')
story.append(table([
    ['路径', '内容'],
    ['code/train/convert.py', 'VisDrone → YOLO 标注转换（含 4 类异常处理）'],
    ['code/train/train.py', '训练脚本（时间盒、余弦退火、显存预算）'],
    ['code/train/detect.py', '单图/批量推理封装'],
    ['code/train/_check_train.py', '训练前配置自检'],
    ['code/app/main.py', 'PyQt5 上位机（图片/视频/摄像头 + QThread）'],
    ['tools/convert_visdrone.py', '带统计输出的转换工具（生成 VisDrone.yaml）'],
    ['tools/verify_yolo.py', '转换结果校验（数量/字段/坐标）'],
    ['tools/val.py', '验证 + 逐类 AP + best/last 选择偏置'],
    ['tools/analyze_density.py', '按目标密度分桶统计召回率'],
    ['tools/analyze_perclass.py', '逐类尺寸与召回率'],
    ['tools/analyze_annots.py', '标注分布勘察（类别 × score 联合分布）'],
    ['tools/analyze2.py', '异常行与越界框勘察'],
    ['tools/analyze_timing.py', '训练耗时标注'],
    ['tools/show_spec.py', '打印模型与数据集清单'],
    ['tools/smoke_test.py', '烧机标定（3% 数据验显存与流水线）'],
    ['report/实验结果与分析.md', '完整实验分析（本材料的底稿）'],
    ['report/figures/', '训练曲线、混淆矩阵、界面截图'],
], [190, 310], fs=8.0, left_cols=1))

story.append(Spacer(1, 10))
story.append(Paragraph('结语', ParagraphStyle(
    'end', fontName=FONT_BD, fontSize=11.6, textColor=BLUE, spaceBefore=8,
    spaceAfter=4)))
story.append(Paragraph('这个项目最有价值的不是 0.478 这个数，而是'
                       '「知道它为什么不是 0.55、以及哪些看起来很合理的原因其实只是假象」的判断过程。'
                       '面试时把这条讲清楚，比多背几个指标有用得多。', QUOTE))

doc.build(story)
print('已生成: ' + OUT)
print('页数  : %d' % doc.page)

