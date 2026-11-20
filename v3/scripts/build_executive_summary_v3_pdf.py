"""
v3/scripts/build_executive_summary_v3_pdf.py
Compiles EXECUTIVE_SUMMARY_REPORT_V3.pdf in the workspace root using ReportLab.
Consolidates all Retraining Round v3 discoveries, experimental tables, and embedded high-res figures.
Engineered precisely for a clean, professional 5-page layout.
Oriented as the Executive Research Briefing & Manuscript Foundation for a New International Conference Paper.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (on pages 2 and later)
        # Using carefully measured string lengths to prevent any text overlap
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 26, "RESEARCH BRIEFING: AUDIT TARGET LEAKAGE PIMA (EDISI v3)")
            self.setFont("Helvetica", 7.5)
            self.drawRightString(8.5 * inch - 36, 11 * inch - 26, "M. M. Affandi & Dr. R. Y. N. Fu'adah (IEEE Conference Paper)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Footer (on all pages)
        page_str = f"Halaman {self._pageNumber} dari {page_count}"
        self.setFont("Helvetica", 7.5)
        self.drawRightString(8.5 * inch - 36, 20, page_str)
        self.drawString(36, 20, "Repositori: github.com/mitcheltastic/DiabeticAnalysis (v3 Final) | Basis Paper Konferensi Internasional")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * inch - 36, 30)
        self.restoreState()


def build_pdf(filename="EXECUTIVE_SUMMARY_REPORT_V3.pdf"):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = os.path.join(repo_root, filename)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Color Palette
    c_primary = colors.HexColor("#0F172A")     # Slate 900
    c_secondary = colors.HexColor("#1E40AF")   # Blue 800
    c_accent = colors.HexColor("#0284C7")      # Sky 600
    c_dark = colors.HexColor("#1E293B")        # Slate 800
    c_muted = colors.HexColor("#475569")       # Slate 600
    c_border = colors.HexColor("#CBD5E1")      # Slate 300
    c_bg_light = colors.HexColor("#F8FAFC")    # Slate 50
    c_green = colors.HexColor("#166534")       # Green 800
    c_red = colors.HexColor("#991B1B")         # Red 800

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15.5,
        leading=18.5,
        textColor=c_primary,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=c_secondary,
        spaceAfter=5
    )

    meta_style = ParagraphStyle(
        'MetaBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.8,
        textColor=c_dark
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.8,
        leading=12.5,
        textColor=c_primary,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=c_dark,
        spaceAfter=3
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.8,
        textColor=c_dark,
        leftIndent=8,
        firstLineIndent=-5,
        spaceAfter=2
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.8,
        textColor=colors.HexColor("#1E3A8A")
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=6.8,
        leading=8.5,
        textColor=c_muted,
        alignment=1,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.2,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=7.8,
        textColor=c_dark
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=7.8,
        textColor=c_primary
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=7.8,
        textColor=c_dark,
        alignment=1
    )

    story = []

    # =========================================================================
    # HALAMAN 1: HEADER, METADATA, RINGKASAN EKSEKUTIF, & AUDIT LEAKAGE E1
    # =========================================================================
    story.append(Paragraph("LAPORAN RINGKASAN EKSEKUTIF RISET (EDISI v3 FINAL)", title_style))
    story.append(Paragraph("Audit Forensik Target Leakage, Replikasi Presisi Baseline, & Standarisasi Metodologi Klinis Murni (Kohort PIMA)", subtitle_style))

    meta_data = [
        [
            Paragraph("<b>Peneliti / Penulis Utama:</b> Mitchel Mohamad Affandi<br/>"
                      "<b>Dosen Pembimbing / Co-Author:</b> Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.<br/>"
                      "<b>Tujuan Dokumen:</b> Basis Riset & Briefing Naskah Konferensi Baru", meta_style),
            Paragraph("<b>Target Luaran:</b> Paper Konferensi Internasional (IEEE / Scopus)<br/>"
                      "<b>Protokol Resmi:</b> v3/PROTOCOL.md (Amendemen 1 & 2 Terdaftar Apriori)<br/>"
                      "<b>Repositori GitHub:</b> <code>mitcheltastic/DiabeticAnalysis</code> (Cabang <code>main</code>)", meta_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Ringkasan Eksekutif: Nilai Jual Ilmiah & Relevansi Paper Konferensi Baru", h1_style))
    story.append(Paragraph(
        "Dalam riset biomedical machine learning, <b>kebenaran metodologis dan kebaruan audit forensik adalah nilai jual ilmiah tertinggi</b>. Angka akurasi tinggi yang dibangun di atas kebocoran data (<i>target leakage</i>) atau penalaan pada data uji (<i>test-set tuning</i>) adalah ilusi yang langsung runtuh di hadapan peer reviewer internasional. Makalah yang berhasil mereplikasi hasil terdahulu, membongkar kebocoran data secara matematis, dan menegakkan standar pengujian murni memiliki <b>posisi ilmiah prestisius dan potensi sitasi tinggi</b>. Edisi v3 menyempurnakan seluruh siklus riset menjadi <b>benteng ilmiah yang kedap air (airtight)</b>:",
        body_style
    ))

    flow_box = [
        [Paragraph(
            "<b>Peta Jalur & Temuan Kunci Retraining Round v3:</b><br/>"
            "• <b>E1 (Audit Kebocoran 3-Tier):</b> Bukti OLS objektif: t = 26.77, p = 3.97e-88. Mengoreksi nama metode menjadi <i>Rough-Set Relations</i>.<br/>"
            "• <b>E2 (Tangga Inflasi 100 Seed):</b> Mendekomposisi lonjakan: Leak-Free (75.25%) -> RLTR Bocor (83.69%) -> Test-Tuned (86.45%) -> Seed Ekstrem (89.61%).<br/>"
            "• <b>E2b (Replikasi Presisi Paper):</b> Mereplikasi exact matriks Paper 1 (90,10,13,41 = 85.06%) & Paper 2 (92,8,12,42 = 87.01%).<br/>"
            "• <b>E3 (Uji Kesetaraan TOST):</b> 50-fold CV membuktikan Imputasi Median setara secara TOST dengan MICE & MissForest (p_TOST < 0.0001).<br/>"
            "• <b>E4 (Parsimoni Seleksi Model):</b> LogReg-L2 Default menang atas pohon boosting tertuning (p_Holm = 1.0000; depth saturated pada batas bawah).<br/>"
            "• <b>E5 (Skrining Klinis & DCA):</b> Nested Policy B (Sensitivitas 80.43%, Spesifisitas 68.48%) terbukti memberi Net Benefit positif pada DCA.<br/>"
            "• <b>E6 (Uji Subgrup Headline):</b> RLTR runtuh 16.25 pp pd insulin asli (p < 10^-8); LogReg leak-free mengungguli RLTR +0.0386 AUC di biologi riil!<br/>"
            "• <b>E7 (Stabilitas SHAP 200 Refit):</b> Glukosa (#1) & BMI (#2) menguasai 100% bootstrap resamples; Insulin terbukti non-esensial.",
            callout_style
        )]
    ]
    t_flow = Table(flow_box, colWidths=[540])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_flow)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Topologi Data PIMA & Audit Kebocoran Target 3-Tier (Eksperimen E1)", h1_style))
    story.append(Paragraph(
        "Kohort PIMA (N = 768, 500 kontrol, 268 diabetes) memiliki 374 nilai insulin hilang (48.70%) dan 227 lipatan kulit hilang (29.56%). Pasien dengan data biologis murni lengkap berjumlah 392 orang. "
        "Audit E1 mengoreksi nomenklatur literatur lama dari 'regresi parametrik' menjadi <b>Metode Relasi Toleransi Rough-Set (TR, LTR, RLTR, NSSR, SIM)</b> dan membuktikan kebocoran label target pada file RLTR:",
        body_style
    ))

    e1_data = [
        [
            Paragraph("Tingkatan Bukti (*Tier*)", table_header_style),
            Paragraph("Metrik Uji Statistik", table_header_style),
            Paragraph("Data Biologis Asli (N=392-394)", table_header_style),
            Paragraph("File RLTR Imputed (N=768)", table_header_style),
            Paragraph("Signifikansi & Vonis Ilmiah", table_header_style),
        ],
        [
            Paragraph("<b>Tier 1: Bivariat</b>", table_cell_bold),
            Paragraph("Korelasi Point-Biserial r(Insulin, Outcome)", table_cell_style),
            Paragraph("r = +0.2163 [0.120, 0.308]", table_cell_center),
            Paragraph("<b>r = +0.6729</b> [0.632, 0.710]", table_cell_center),
            Paragraph("Fisher z = 10.05 (p = 4.41e-24). Melonjak 3x lipat batas alami.", table_cell_style),
        ],
        [
            Paragraph("<b>Tier 2: OLS Kondisional</b>", table_cell_bold),
            Paragraph("Koefisien Outcome (mengontrol 7 fitur lain)", table_cell_style),
            Paragraph("beta = -6.16 mg/dL<br/>t = -0.484, p = 0.6285", table_cell_center),
            Paragraph("<b>beta = +28.68 mg/dL<br/>t = 26.77, p = 3.97e-88</b>", table_cell_center),
            Paragraph("<b>Delta-beta = +34.84 mg/dL</b> (Target disuntikkan secara masif ke insulin!).", table_cell_style),
        ],
        [
            Paragraph("<b>Tier 3: Rekonsiliasi</b>", table_cell_bold),
            Paragraph("Formula Relasi Aproksimasi Kasar", table_cell_style),
            Paragraph("Data observasional murni", table_cell_center),
            Paragraph("Matriks relasi melibatkan kelas target", table_cell_center),
            Paragraph("Algoritma aproksimasi ketetanggaan mengikat kelas target.", table_cell_style),
        ]
    ]
    t_e1 = Table(e1_data, colWidths=[75, 115, 105, 105, 140])
    t_e1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_e1)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "• <b>Audit 5 File Imputasi:</b> TR (t=0.74, p=0.46), SIM (t=0.45, p=0.66), dan NSSR (t=-1.15, p=0.25) bebas kebocoran target; LTR (t=12.18, p=1.01e-30) dan RLTR (t=26.77, p=3.97e-88) terbukti bocor target artifisial masif.",
        bullet_style
    ))

    # =========================================================================
    # HALAMAN 2: TANGGA INFLASI E2, REPLIKASI E2b, KESETARAAN E3, PARSIMONI E4
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("3. Tangga Inflasi Metodologis (E2) & Replikasi Presisi Baseline Paper (E2b)", h1_style))
    story.append(Paragraph(
        "Eksperimen E2 mengevaluasi 100 stratified splits independen (80/20) untuk memetakan bagaimana performa terinflasi secara artifisial, dan E2b mereplikasi hasil paper terdahulu secara presisi matematika:",
        body_style
    ))

    e2_data = [
        [
            Paragraph("Rung", table_header_style),
            Paragraph("Kondisi Metodologis", table_header_style),
            Paragraph("Akurasi 100-Seed (Mean±SD)", table_header_style),
            Paragraph("ROC-AUC 100-Seed", table_header_style),
            Paragraph("Delta vs R0 (p_NB)", table_header_style),
            Paragraph("Seed 42 Acc", table_header_style),
            Paragraph("Seed 42 Matrix", table_header_style),
        ],
        [
            Paragraph("<b>R0</b>", table_cell_bold),
            Paragraph("<b>Leak-Free Baseline</b> (In-fold med+scale, XGB Ref)", table_cell_style),
            Paragraph("<b>75.25% ± 3.06%</b>", table_cell_center),
            Paragraph("0.8163 ± 0.0296", table_cell_center),
            Paragraph("Baseline Acuan", table_cell_center),
            Paragraph("75.32%", table_cell_center),
            Paragraph("(84, 16, 22, 32)", table_cell_center),
        ],
        [
            Paragraph("<b>R1</b>", table_cell_style),
            Paragraph("Pre-split MinMax Scaling Leakage", table_cell_style),
            Paragraph("75.23% ± 3.09%", table_cell_center),
            Paragraph("0.8166 ± 0.0297", table_cell_center),
            Paragraph("-0.01 pp (p=0.941)", table_cell_center),
            Paragraph("75.32%", table_cell_center),
            Paragraph("(84, 16, 22, 32)", table_cell_center),
        ],
        [
            Paragraph("<b>R3a_RLTR</b>", table_cell_bold),
            Paragraph("<b>RLTR As-Is (Baseline Paper Kakak Kelas)</b>", table_cell_style),
            Paragraph("<b>83.69% ± 2.65%</b>", table_cell_center),
            Paragraph("<b>0.8998 ± 0.0217</b>", table_cell_center),
            Paragraph("<b>+8.45 pp (p=3.9e-7)</b>", table_cell_center),
            Paragraph("<b>85.06%</b>", table_cell_center),
            Paragraph("<b>(90, 10, 13, 41)</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>R4-50</b>", table_cell_bold),
            Paragraph("<b>Test-Set Objective Tuning (Optuna 50 trials)</b>", table_cell_style),
            Paragraph("<b>85.86% ± 2.19%</b>", table_cell_center),
            Paragraph("0.8930 ± 0.0223", table_cell_center),
            Paragraph("<b>+10.61 pp (p=8.2e-10)</b>", table_cell_center),
            Paragraph("<b>87.01%</b>", table_cell_center),
            Paragraph("<b>(92, 8, 12, 42)</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>R5</b>", table_cell_style),
            Paragraph("Maximum Seed Selection Effect (Seed 12)", table_cell_style),
            Paragraph("89.61% (Single Split Max)", table_cell_center),
            Paragraph("0.9231", table_cell_center),
            Paragraph("+14.36 pp", table_cell_center),
            Paragraph("—", table_cell_center),
            Paragraph("(94, 6, 10, 44)", table_cell_center),
        ],
    ]
    t_e2 = Table(e2_data, colWidths=[40, 175, 95, 75, 75, 45, 35])
    t_e2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_e2)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "• <b>Bukti Replikasi Paper 1 (Saputra dkk.):</b> R3a RLTR mereplikasi exact angka Paper 1 Tabel II (XGBoost): Acc <b>85.06%</b>, Prec 0.80, Rec 0.76, Spec 0.90, F1 0.78, Matrix <b>(90, 10, 13, 41)</b>.<br/>"
        "• <b>Bukti Replikasi Paper 2 (Pratama dkk.):</b> R4-50 Test-Tuned mereplikasi exact Paper 2 (RLTR XGB+Optuna): Acc <b>87.01%</b>, Prec 86.90%, Rec 87.01%, F1 86.89%, Matrix <b>(92, 8, 12, 42)</b>.<br/>"
        "• <b>Sensitivitas True Defaults:</b> Di bawah default asli xgboost 3.4 (lr=0.3), inflasi RLTR tetap masif: <b>+9.21 pp (p = 3.34e-9)</b> dan <b>+0.0981 AUC (p = 6.88e-12)</b>.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Kesetaraan Imputasi (E3) & Kemenangan Parsimoni Model Akhir (E4)", h1_style))
    story.append(Paragraph(
        "Eksperimen E3 menguji 6 metode imputasi pada 50-fold nested CV. Hasil uji Two-One-Sided Tests (TOST) membuktikan <b>semua metode setara secara statistik dengan Imputasi Median sederhana pada margin ±0.015 akurasi (p_TOST < 0.0001)</b>. "
        "Di bawah aturan terdaftar Protokol Amendemen 2 (E4), model diseleksi pada 50 outer test folds:",
        body_style
    ))

    e4_data = [
        [
            Paragraph("Model Kandidat", table_header_style),
            Paragraph("Status Tuning", table_header_style),
            Paragraph("Outer ROC-AUC (Mean±SD)", table_header_style),
            Paragraph("Outer Akurasi (Mean±SD)", table_header_style),
            Paragraph("Brier Score", table_header_style),
            Paragraph("Cal. Slope", table_header_style),
            Paragraph("p_Holm vs LogReg", table_header_style),
            Paragraph("TOST Equiv. (±0.02)", table_header_style),
        ],
        [
            Paragraph("<b>LogReg-L2 Default</b>", table_cell_bold),
            Paragraph("Untuned Baseline (C=1.0)", table_cell_style),
            Paragraph("<b>0.8357 ± 0.0381</b>", table_cell_center),
            Paragraph("<b>0.7745 ± 0.0366</b>", table_cell_center),
            Paragraph("0.1601", table_cell_center),
            Paragraph("0.9859", table_cell_center),
            Paragraph("Baseline Acuan", table_cell_center),
            Paragraph("<b>Baseline Terpilih</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>LogReg-EN</b>", table_cell_style),
            Paragraph("Tuned ElasticNet (50 trials)", table_cell_style),
            Paragraph("0.8369 ± 0.0383", table_cell_center),
            Paragraph("0.7745 ± 0.0361", table_cell_center),
            Paragraph("0.1598", table_cell_center),
            Paragraph("0.9822", table_cell_center),
            Paragraph("p = 1.0000", table_cell_center),
            Paragraph("p < 0.0001 (Setara)", table_cell_style),
        ],
        [
            Paragraph("<b>XGBoost</b>", table_cell_style),
            Paragraph("Tuned (50 trials)", table_cell_style),
            Paragraph("0.8379 ± 0.0396", table_cell_center),
            Paragraph("0.7600 ± 0.0401", table_cell_center),
            Paragraph("0.1610", table_cell_center),
            Paragraph("0.9634", table_cell_center),
            Paragraph("p = 1.0000", table_cell_center),
            Paragraph("p < 0.0001 (Setara)", table_cell_style),
        ],
        [
            Paragraph("<b>CatBoost</b>", table_cell_style),
            Paragraph("Tuned (50 trials)", table_cell_style),
            Paragraph("0.8417 ± 0.0363", table_cell_center),
            Paragraph("0.7748 ± 0.0371", table_cell_center),
            Paragraph("0.1582", table_cell_center),
            Paragraph("0.9712", table_cell_center),
            Paragraph("p = 1.0000", table_cell_center),
            Paragraph("p = 0.0014 (Setara)", table_cell_style),
        ],
    ]
    t_e4 = Table(e4_data, colWidths=[90, 95, 95, 85, 45, 45, 45, 40])
    t_e4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_e4)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "• <b>Keputusan Parsimoni:</b> Tidak ada model kompleks yang mengungguli LogReg-L2 (semua p_Holm = 1.00). <b>LogReg-L2 Default ditetapkan sebagai Model Akhir Resmi Penelitian</b>.<br/>"
        "• <b>Saturasi Kedalaman Pohon:</b> XGBoost memilih max_depth=2 pada 78% fold (39/50), dan CatBoost memilih depth=3 pada 68% fold (34/50). Pada data tabular kecil, pohon dipaksa sangat dangkal mendekati linier.",
        bullet_style
    ))

    # =========================================================================
    # HALAMAN 3: SKRINING KLINIS E5, SUBGRUP HEADLINE E6, SHAP STABILITY E7
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("5. Kebijakan Skrining Klinis, Kalibrasi Probabilitas, & DCA (Eksperimen E5)", h1_style))
    story.append(Paragraph(
        "Untuk menghindari leakage seleksi threshold v2 (tau=0.37 dievaluasi di data sama), E5 mengoptimasi threshold secara <b>nested murni</b> di dalam inner CV dan mengevaluasi di outer fold serta holdout beku (Seed 42, N=154):",
        body_style
    ))

    e5_data = [
        [
            Paragraph("Kebijakan Skrining Klinis", table_header_style),
            Paragraph("Aturan Optimasi Inner Loop", table_header_style),
            Paragraph("Mean Threshold (tau)", table_header_style),
            Paragraph("Sensitivitas Outer", table_header_style),
            Paragraph("Spesifisitas Outer", table_header_style),
            Paragraph("Outer F1-Score", table_header_style),
            Paragraph("Seed 42 Holdout Matrix", table_header_style),
        ],
        [
            Paragraph("<b>Default</b>", table_cell_style),
            Paragraph("Standar cut-off probabilitas", table_cell_style),
            Paragraph("0.500", table_cell_center),
            Paragraph("0.5580 ± 0.0827", table_cell_center),
            Paragraph("0.8788 ± 0.0416", table_cell_center),
            Paragraph("0.6242 ± 0.0638", table_cell_center),
            Paragraph("(82, 18, 27, 27) - Acc 70.78%", table_cell_center),
        ],
        [
            Paragraph("<b>Policy A</b>", table_cell_style),
            Paragraph("Maksimalkan F1-Score", table_cell_style),
            Paragraph("0.297 ± 0.041", table_cell_center),
            Paragraph("0.7747 ± 0.0934", table_cell_center),
            Paragraph("0.7020 ± 0.0664", table_cell_center),
            Paragraph("0.6646 ± 0.0526", table_cell_center),
            Paragraph("(78, 22, 22, 32) - Acc 71.43%", table_cell_center),
        ],
        [
            Paragraph("<b>Policy B (Rekomendasi)</b>", table_cell_bold),
            Paragraph("<b>Sensitivitas >= 0.80 & Max Spesifisitas</b>", table_cell_bold),
            Paragraph("<b>0.280 ± 0.038</b>", table_cell_center),
            Paragraph("<b>0.8043 ± 0.0888</b>", table_cell_center),
            Paragraph("<b>0.6848 ± 0.0716</b>", table_cell_center),
            Paragraph("<b>0.6725 ± 0.0531</b>", table_cell_center),
            Paragraph("<b>(67, 33, 9, 45) - Sens 83.33%</b>", table_cell_bold),
        ],
    ]
    t_e5 = Table(e5_data, colWidths=[100, 115, 65, 75, 75, 55, 55])
    t_e5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_e5)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "• <b>Kalibrasi & DCA:</b> Kalibrasi model sangat prima (Slope 0.7608, CITL -0.0062, Brier 0.1754). Decision Curve Analysis (DCA) membuktikan Net Benefit positif pada seluruh rentang probabilitas 0.10 - 0.55. Pada pt = 0.20, Net Benefit mencapai 0.2458 (mendeteksi 25 kasus positif murni per 100 pasien tanpa intervensi palsu).",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Uji Subgrup Headline & Bukti Fatalitas RLTR pada Biologi Asli (Eksperimen E6)", h1_style))
    story.append(Paragraph(
        "Eksperimen E6 memecah data uji 100 seed menjadi dua subgrup: pasien dengan data insulin asli laboratorium (n ≈ 70) vs pasien tanpa insulin yang diimputasi (n ≈ 84). Hasilnya mengungkap bukti terkuat mengapa RLTR cacat fatal:",
        body_style
    ))

    e6_data = [
        [
            Paragraph("Arsitektur Model Evaluasi", table_header_style),
            Paragraph("Subgrup Pasien Data Uji", table_header_style),
            Paragraph("Akurasi 100-Seed (Mean±SD)", table_header_style),
            Paragraph("ROC-AUC 100-Seed", table_header_style),
            Paragraph("Pergeseran (Asli vs Imputasi)", table_header_style),
            Paragraph("Signifikansi (p_NB)", table_header_style),
        ],
        [
            Paragraph("<b>RLTR R3a Reference XGBoost</b><br/>(Baseline Paper Kakak Kelas)", table_cell_style),
            Paragraph("Insulin Diimputasi Buatan (n ≈ 84)<br/><b>Insulin Asli Laboratorium (n ≈ 70)</b>", table_cell_style),
            Paragraph("92.00% ± 3.31%<br/><b>75.75% ± 3.87%</b>", table_cell_center),
            Paragraph("0.9700 ± 0.0174<br/><b>0.8060 ± 0.0449</b>", table_cell_center),
            Paragraph("<b>Akurasi Anjlok -16.25 pp</b><br/><b>AUC Turun -0.1640</b>", table_cell_center),
            Paragraph("<b>p = 2.25 x 10^-9</b><br/><b>p = 2.10 x 10^-9</b>", table_cell_center),
        ],
        [
            Paragraph("<b>Leak-Free LogReg-L2</b><br/>(Model Akhir Riset v3)", table_cell_bold),
            Paragraph("Insulin Diimputasi Median (n ≈ 84)<br/><b>Insulin Asli Laboratorium (n ≈ 70)</b>", table_cell_style),
            Paragraph("75.43% ± 3.64%<br/><b>77.68% ± 3.63%</b>", table_cell_center),
            Paragraph("0.8316 ± 0.0364<br/><b>0.8446 ± 0.0362</b>", table_cell_center),
            Paragraph("Stabil / Naik +2.26 pp<br/>Stabil / Naik +0.0130", table_cell_center),
            Paragraph("p = 0.3871 (Konsisten)<br/>p = 0.6085 (Konsisten)", table_cell_center),
        ],
    ]
    t_e6 = Table(e6_data, colWidths=[120, 125, 95, 75, 70, 55])
    t_e6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_e6)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "• <b>Keunggulan di Pasien Nyata:</b> Pada pasien berinsulin asli laboratorium, LogReg-L2 leak-free kita <b>mengungguli model RLTR kakak kelas sebesar +0.0386 ROC-AUC (p = 0.0295)</b>.<br/>"
        "• <b>Interaksi Difference-in-Differences (DiD):</b> Uji kontras DiD membuktikan inflasi performa RLTR terkonsentrasi total pada baris insulin sintetis: <b>Delta-Delta Acc = -18.51 pp (p = 8.98e-11)</b> dan <b>Delta-Delta AUC = -0.1770 (p = 2.15e-10)</b>.<br/>"
        "• <b>Ablasi Fitur Retrained:</b> Menghapus kolom Insulin dan Skinfold menghasilkan akurasi 76.49% dan AUC 0.8395, identik tanpa perbedaan signifikan vs 8 fitur penuh (p > 0.49). Kolom data hilang ini redundan.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("7. Restorasi Fisiologis Medis & Stabilitas SHAP Bootstrap (Eksperimen E7)", h1_style))
    story.append(Paragraph(
        "Evaluasi 200 ulangan bootstrap refit pada 614 pasien latih membongkar ilusi lama bahwa insulin adalah fitur utama:",
        body_style
    ))

    e7_data = [
        [
            Paragraph("Fitur Klinis", table_header_style),
            Paragraph("LogReg-L2 Mean |SHAP|", table_header_style),
            Paragraph("LogReg-L2 Median Rank [IQR]", table_header_style),
            Paragraph("LogReg Top-3 Freq", table_header_style),
            Paragraph("Tuned XGBoost Mean |SHAP|", table_header_style),
            Paragraph("XGBoost Median Rank [IQR]", table_header_style),
            Paragraph("XGB Top-3 Freq", table_header_style),
        ],
        [
            Paragraph("<b>Glucose</b>", table_cell_bold),
            Paragraph("1.0535 ± 0.1088", table_cell_center),
            Paragraph("<b>1.0 [0.0]</b>", table_cell_center),
            Paragraph("<b>100.0%</b>", table_cell_center),
            Paragraph("0.8407 ± 0.0926", table_cell_center),
            Paragraph("<b>1.0 [0.0]</b>", table_cell_center),
            Paragraph("<b>100.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>BMI</b>", table_cell_bold),
            Paragraph("0.5809 ± 0.1066", table_cell_center),
            Paragraph("<b>2.0 [0.0]</b>", table_cell_center),
            Paragraph("<b>100.0%</b>", table_cell_center),
            Paragraph("0.4835 ± 0.0845", table_cell_center),
            Paragraph("<b>2.0 [0.0]</b>", table_cell_center),
            Paragraph("<b>100.0%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>Pregnancies</b>", table_cell_style),
            Paragraph("0.3298 ± 0.1198", table_cell_center),
            Paragraph("3.0 [0.0]", table_cell_center),
            Paragraph("77.0%", table_cell_center),
            Paragraph("0.1178 ± 0.0587", table_cell_center),
            Paragraph("6.0 [2.0]", table_cell_center),
            Paragraph("3.5%", table_cell_center),
        ],
        [
            Paragraph("<b>Insulin</b>", table_cell_style),
            Paragraph("0.0902 ± 0.0612", table_cell_center),
            Paragraph("<b>6.0 [2.0]</b>", table_cell_center),
            Paragraph("<b>1.0%</b>", table_cell_center),
            Paragraph("0.1421 ± 0.0641", table_cell_center),
            Paragraph("<b>5.0 [2.0]</b>", table_cell_center),
            Paragraph("<b>3.5%</b>", table_cell_center),
        ],
        [
            Paragraph("<b>SkinThickness</b>", table_cell_style),
            Paragraph("0.0615 ± 0.0455", table_cell_center),
            Paragraph("7.0 [2.0]", table_cell_center),
            Paragraph("0.0%", table_cell_center),
            Paragraph("0.0656 ± 0.0463", table_cell_center),
            Paragraph("7.0 [2.0]", table_cell_center),
            Paragraph("0.5%", table_cell_center),
        ],
    ]
    t_e7 = Table(e7_data, colWidths=[80, 80, 85, 60, 80, 85, 70])
    t_e7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_e7)
    story.append(Spacer(1, 3))
    story.append(Paragraph("<b>Kesimpulan Biologis:</b> Glukosa dan BMI menguasai peringkat 1 dan 2 pada 100% bootstrap refits di seluruh arsitektur model. Insulin terpuruk di peringkat 5-6 (hanya 1.0% masuk Top-3). Dominasi insulin pada karya terdahulu murni artefak kebocoran data.", body_style))

    # =========================================================================
    # HALAMAN 4: AUDIT LOG 12 KOREKSI, 5 ARGUMEN PAPER, & KESIAPAN REPO
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("8. Rekapitulasi 12 Koreksi Dokumen Terdahulu (Audit Log)", h1_style))
    story.append(Paragraph(
        "Integritas ilmiah Edisi v3 didasarkan pada audit transparan terhadap dokumen terdahulu (dicatat formal di <code>v3/results/CORRECTIONS_TO_V2.md</code>):",
        body_style
    ))

    corrections_table_data = [
        [
            Paragraph("No", table_header_style),
            Paragraph("Topik / Lokasi Dokumen Lama", table_header_style),
            Paragraph("Pernyataan / Angka di v2", table_header_style),
            Paragraph("Koreksi Ilmiah & Realitas Teknis di v3 Final", table_header_style),
        ],
        [
            Paragraph("1", table_cell_center),
            Paragraph("Signifikansi OLS RLTR", table_cell_style),
            Paragraph("p = 4.24 x 10^-77", table_cell_center),
            Paragraph("<b>p = 3.97 x 10^-88</b> (df=365). Presisi floating-point scipy.", table_cell_style),
        ],
        [
            Paragraph("2", table_cell_center),
            Paragraph("Kovariat t = 26.77", table_cell_style),
            Paragraph("Kontrol Glukosa & BMI", table_cell_style),
            Paragraph("<b>Model Penuh 8 Fitur</b> (kontrol Glukosa+BMI saja: t=27.64).", table_cell_style),
        ],
        [
            Paragraph("3", table_cell_center),
            Paragraph("Baseline Biologi Murni", table_cell_style),
            Paragraph("Evaluasi data N=394", table_cell_center),
            Paragraph("<b>Data mentah N=392 tanpa imputasi</b> (beta = -6.16, p=0.6285).", table_cell_style),
        ],
        [
            Paragraph("4", table_cell_center),
            Paragraph("Parameter Toleransi RLTR", table_cell_style),
            Paragraph("Ambiguitas epsilon", table_cell_style),
            Paragraph("Mendokumentasikan transisi <b>eps=0.15</b> (paper) dan <b>eps=0.25</b>.", table_cell_style),
        ],
        [
            Paragraph("5", table_cell_center),
            Paragraph("Nomenklatur Imputasi", table_cell_style),
            Paragraph("Regresi tren parametrik linier/spline", table_cell_style),
            Paragraph("<b>Metode Relasi Toleransi Rough-Set (TR, LTR, RLTR, NSSR, SIM)</b>.", table_cell_bold),
        ],
        [
            Paragraph("6", table_cell_center),
            Paragraph("Batas R_max Footnote", table_cell_style),
            Paragraph("R_max = 0.5299 batas korelasi", table_cell_style),
            Paragraph("<b>Ditarik formal</b> karena ruang fitur yang dibandingkan berbeda.", table_cell_style),
        ],
        [
            Paragraph("7", table_cell_center),
            Paragraph("Keunggulan MissForest", table_cell_style),
            Paragraph("+1.12 pp di atas Median", table_cell_center),
            Paragraph("<b>Terkoreksi setara median</b> (p_TOST < 0.0001) pd classifier tetap.", table_cell_style),
        ],
        [
            Paragraph("8", table_cell_center),
            Paragraph("Threshold Skrining 0.37", table_cell_style),
            Paragraph("Sens 79.85%, Spec 75.20%", table_cell_center),
            Paragraph("Post-hoc leakage; diganti <b>Nested Policy B (Sens 80.43%, Spec 68.48%)</b>.", table_cell_style),
        ],
        [
            Paragraph("9", table_cell_center),
            Paragraph("Peringkat SHAP Insulin", table_cell_style),
            Paragraph("Insulin peringkat #2", table_cell_center),
            Paragraph("200 bootstrap refits membuktikan <b>Insulin peringkat 6/8</b>.", table_cell_style),
        ],
        [
            Paragraph("10", table_cell_center),
            Paragraph("Akurasi Holdout Seed 42", table_cell_style),
            Paragraph("Ensemble 74.68%", table_cell_center),
            Paragraph("LogReg-L2 seed-42: <b>70.78%</b>; rata-rata 100 seed: <b>76.60% ± 2.62%</b>.", table_cell_style),
        ],
        [
            Paragraph("11", table_cell_center),
            Paragraph("Kedalaman Pohon Optuna", table_cell_style),
            Paragraph("Belum terdokumentasi", table_cell_center),
            Paragraph("Saturasi batas bawah: <b>78% XGB (depth=2), 68% CatBoost (depth=3)</b>.", table_cell_style),
        ],
        [
            Paragraph("12", table_cell_center),
            Paragraph("Label Default XGBoost", table_cell_style),
            Paragraph("Dinamai 'default XGBoost'", table_cell_style),
            Paragraph("Relabeled <b>'XGBoost reference config (lr=0.1)'</b>; true default lr=0.3.", table_cell_style),
        ],
    ]
    t_corr = Table(corrections_table_data, colWidths=[20, 130, 140, 250])
    t_corr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_corr)
    story.append(Spacer(1, 4))

    story.append(Paragraph("9. Lima Kontribusi Inti Paper Konferensi & Panduan Menjawab Reviewer", h1_style))
    pilar_box = [
        [Paragraph(
            "<b>Jika Reviewer Konferensi Bertanya: 'Mengapa Akurasi Akhir 76% - 77%, Bukan 85% - 87% Seperti Paper Terdahulu?'</b><br/>"
            "1. <b>Replikasi Presisi Sempurna:</b> Kita membuktikan eksak asal-usul angka 85.06% (Paper 1, matriks 90,10,13,41) dan 87.01% (Paper 2, matriks 92,8,12,42 via test tuning). Kita membuktikan angka tersebut dapat direproduksi secara deterministik.<br/>"
            "2. <b>Bukti Matematis Kebocoran Target:</b> Angka tinggi tersebut hanya terjadi pada file RLTR di mana insulinnya disuntik label target (OLS t = 26.77, p = 3.97e-88).<br/>"
            "3. <b>Model Literatur Terdahulu Runtuh pada Pasien Nyata (Headline Finding):</b> Pada pasien berinsulin asli laboratorium, performa RLTR anjlok 16.25 pp (p < 10^-8). Sebaliknya, model LogReg leak-free kita mengalahkan model RLTR sebesar +0.0386 AUC (p = 0.0295) pada data biologis riil!<br/>"
            "4. <b>Standar Emas Metodologis Internasional:</b> Riset ini memenuhi standar kedokteran dan informatika tertinggi: 100-seed split, 50-fold nested CV, uji Nadeau-Bengio, uji TOST, kalibrasi probabilitas, dan DCA.<br/>"
            "5. <b>Menyelamatkan Literatur Ilmiah (High Novelty):</b> Mengungkap <i>target leakage</i> dan menegakkan benchmark murni adalah kontribusi ilmiah langka bernilai tinggi yang sangat dicari konferensi dan jurnal internasional bereputasi.",
            callout_style
        )]
    ]
    t_pilar = Table(pilar_box, colWidths=[540])
    t_pilar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86EFAC")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_pilar)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Status Kesiapan Teknis Repositori:</b> Seluruh eksperimen v3 (E1-E7) didukung oleh skrip deterministik di <code>v3/scripts/</code>, data lengkap di <code>v3/results/</code>, dan grafik resolusi tinggi di <code>v3/figures/</code>. Dokumen ini siap menjadi acuan penulisan naskah paper konferensi IEEE.",
        body_style
    ))

    # =========================================================================
    # HALAMAN 5: LAMPIRAN GALERI VISUAL RESOLUSI TINGGI (5 EXHIBITS)
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("10. Lampiran Galeri Visual Ilmiah Utama (Key Visual Exhibits v3)", h1_style))
    story.append(Paragraph("Berikut adalah 5 visualisasi resolusi tinggi dari folder <code>v3/figures/</code> yang siap digunakan sebagai gambar utama (camera-ready figures) pada naskah paper konferensi:", body_style))
    story.append(Spacer(1, 3))

    # Exhibit 1: Waterfall
    p_waterfall = os.path.join(repo_root, "v3/figures/E2_inflation_ladder_waterfall.png")
    img_w_full = 510
    img_h_waterfall = img_w_full / 2.67

    t_waterfall = Table([
        [RLImage(p_waterfall, width=img_w_full, height=img_h_waterfall)],
        [Paragraph("<b>Exhibit 1:</b> Dekomposisi Tangga Inflasi Metodologis (Waterfall Ladder R0 ke R5 pada 100 Seed)", caption_style)]
    ], colWidths=[540])
    t_waterfall.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 1),
        ('TOPPADDING', (0,1), (-1,1), 1),
    ]))
    story.append(t_waterfall)
    story.append(Spacer(1, 4))

    # Exhibit 2 & 3: DCA & Reliability Diagram (Side-by-Side)
    p_dca = os.path.join(repo_root, "v3/figures/E5_decision_curve_analysis.png")
    p_cal = os.path.join(repo_root, "v3/figures/E5_reliability_diagram_seed42.png")

    img_w_half = 250
    img_h_dca = img_w_half / 1.33
    img_h_cal = img_w_half / 1.17

    row_clinical = [
        [
            RLImage(p_dca, width=img_w_half, height=img_h_dca),
            RLImage(p_cal, width=img_w_half, height=img_h_cal)
        ],
        [
            Paragraph("<b>Exhibit 2:</b> Decision Curve Analysis (Net Benefit Positif pt 0.10 - 0.55)", caption_style),
            Paragraph("<b>Exhibit 3:</b> Diagram Reliabilitas Kalibrasi Seed 42 (CITL -0.0062, Slope 0.7608)", caption_style)
        ]
    ]
    t_clinical = Table(row_clinical, colWidths=[270, 270])
    t_clinical.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 1),
        ('TOPPADDING', (0,1), (-1,1), 1),
    ]))
    story.append(t_clinical)
    story.append(Spacer(1, 4))

    # Exhibit 4 & 5: Feature Importance Ranks & SHAP Comparison
    p_ranks = os.path.join(repo_root, "v3/figures/E7_feature_importance_ranks.png")
    p_shap = os.path.join(repo_root, "v3/figures/E7_shap_importance_comparison.png")

    img_h_ranks = img_w_half / 1.67
    img_h_shap = img_w_half / 2.33

    row_shap = [
        [
            RLImage(p_ranks, width=img_w_half, height=img_h_ranks),
            RLImage(p_shap, width=img_w_half, height=img_h_shap)
        ],
        [
            Paragraph("<b>Exhibit 4:</b> Distribusi Ranking 200 Bootstrap Refits (Glukosa #1, BMI #2 di 100% resample)", caption_style),
            Paragraph("<b>Exhibit 5:</b> Perbandingan SHAP Global LogReg-L2 vs Tuned XGBoost", caption_style)
        ]
    ]
    t_shap = Table(row_shap, colWidths=[270, 270])
    t_shap.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 1),
        ('TOPPADDING', (0,1), (-1,1), 1),
    ]))
    story.append(t_shap)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {out_path}")
    return out_path


if __name__ == "__main__":
    build_pdf()
