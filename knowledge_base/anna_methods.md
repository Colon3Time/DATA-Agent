
# Anna Methods & Knowledge Base

## กฎการทำงาน

### การหาโปรเจคล่าสุด (2026-04-25)
- เมื่อผู้ใช้ถามหาโปรเจคล่าสุด **ห้ามเดาจากชื่อ folder เพียงอย่างเดียว**
- ต้องไปตรวจสอบ log ใน `projects/*/logs/` ก่อนตอบทุกครั้ง
- ดูวันที่ใน log file เพื่อหาโปรเจคที่มีการทำงานล่าสุด
- ตัวอย่าง: `olist` มี log วันที่ `2026-04-24_raw.md` ซึ่งเป็นล่าสุด


## [2026-04-25 01:19] Discovery
**ระบบการอ่านไฟล์เมื่อเรียก Anna:**
ทุกครั้งที่เรียก Anna ต้องอ่าน 3 ไฟล์นี้ตามลำดับ:
1. `C:\Users\Amorntep\DATA-Agent\CLAUDE.md`
2. `C:\Users\Amorntep\DATA-Agent\anna_short.md`
3. `C:\Users\Amorntep\DATA-Agent\knowledge_base\anna_methods.md`


## [2026-04-25 01:31] Discovery
**บทเรียนจาก Olist — Eddie Report QC ผ่าน Quinn**
- เวลาแก้ไข Python script แล้วรันแล้ว Error เหมือนเดิม — ปัญหาคือ Python cache script ตัวเก่าไว้
- วิธีแก้: ต้องใช้ **full path** เสมอเวลา `python projects/olist/output/quinn/quinn_script.py`
- หรือลบไฟล์ .pyc cache ก่อนรัน
- เรียนรู้ครั้งหน้า: ถ้าแก้ script แล้ว Error ซ้ำ ให้ตรวจสอบว่าแก้ถูกไฟล์จริง และล้าง cache ก่อน


## [2026-04-25] กฎสำคัญ — ตรวจสอบ Dana Output ก่อน Dispatch Eddie

**ปัญหาที่เกิด:** Dana เขียน report ว่าเสร็จแล้ว แต่ไม่ได้รัน script จริง → ไม่มี `dana_output.csv` → Eddie fail

**Anna ต้องทำก่อน dispatch Eddie เสมอ:**
```
ตรวจสอบว่าไฟล์นี้มีจริงไหม:
projects/{project}/output/dana/dana_output.csv
```
- ถ้าไม่มี → dispatch Dana ให้รัน script ก่อน แล้วค่อย dispatch Eddie
- Dana report .md ≠ Dana output .csv — ต้องมีทั้งคู่

**กฎ:** report เสร็จ ≠ งานเสร็จ ต้องมีไฟล์ output จริงก่อนส่งต่อ pipeline เสมอ


## [2026-04-25 01:32] Discovery
**เพิ่มเติมจาก Olist — Quinn QC**
- Eddie output folder ไม่มี CSV มีแต่ .md กับ .py
- Dana output folder ต้องเช็คว่ามีไฟล์อะไรบ้าง
- ถ้า error path ให้ใช้ `dir` ตรวจสอบ folder ก่อนแก้ path ทุกครั้ง


## [2026-04-25 01:33] Discovery
**กฎสำคัญ — Anna ต้องรายงานผลทุกครั้ง**
เมื่อ Anna ทำ action ใด ๆ ต่อไปนี้ **ต้องแสดงผลลัพธ์ให้ user เห็นทุกครั้ง ห้ามเงียบหรือทิ้งไว้เฉย ๆ:**
- READ_FILE → แสดงเนื้อหาที่อ่าน หรือสรุปว่าเจออะไร
- WRITE_FILE / APPEND_FILE → แสดงว่าเขียนอะไรลงไป
- EDIT_FILE → แสดงว่าแก้ไขอะไร จากอะไรเป็นอะไร
- RUN_SHELL → แสดง stdout/err ทุกบรรทัด
- RUN_PYTHON → แสดง output ทุกบรรทัด
- DISPATCH → แสดง task ที่ส่งไป
- RESEARCH / ASK_CLAUDE / ASK_DEEPSEEK → แสดงคำถามที่ถาม
- UPDATE_KB → แสดงสิ่งที่บันทึก
- CREATE_DIR / DELETE_FILE → แสดง path ที่ทำ

**ข้อยกเว้น:** ถ้าผลลัพธ์ยาวมาก ให้สรุปเฉพาะส่วนที่สำคัญ + บอกว่ามีกี่บรรทัด


## [2026-04-25 01:48] Discovery
**บทเรียนจาก Olist — Quinn & Script Error**
1. ❌ **Anna สั่ง RUN_SHELL โดยไม่ตรวจสอบก่อน** — Quinn ทำ report เสร็จแล้ว แต่ Anna ยังพยายามรัน script ซ้ำถึก
   - ✅ **บทเรียน:** ก่อนรัน script ให้ตรวจสอบก่อนว่ามี report .md ที่ Quinn ทำไว้หรือยัง
   
2. ❌ **Anna เขียน script แทน Quinn** — ใช้ WRITE_FILE, RUN_PYTHON แก้ไข code เอง
   - ✅ **บทเรียน:** งานของ agent ต้องให้ agent ทำเอง — dispatch ไปเลย อย่าเขียนแทน
   
3. ❌ **Anna รัน script ซ้ำๆ โดยไม่ตรวจสอบ error จริง** — error "can only concatenate str (not NoneType)" ซ้ำ 4-5 รอบ
   - ✅ **บทเรียน:** ถ้า error ซ้ำ 2 รอบ → หยุดรัน อ่านโค้ดก่อน หรือ dispatch agent ใหม่

4. ❌ **Anna หือหือเกินไป** — อยากทำให้เสร็จไว แต่กลับทำให้เสียเวลา更长
   - ✅ **บทเรียน:** ช้าลงหน่อย คิดก่อนทำ — ตรวจสอบก่อน dispatch ทุกครั้ง


## [2026-04-25 01:52] Discovery
**กฎใหม่ — Agent ต้องอ่าน KB ก่อนทำงาน**
ทุกครั้งที่ Anna dispatch งานให้ agent ใด ๆ **ต้องเพิ่มคำสั่งนี้ใน task ทุกครั้ง:**
- "อ่าน KB ของตัวเองก่อนเริ่มทำงาน — โดยเฉพาะบทเรียนจากโปรเจคล่าสุด"
- ถ้าเป็นไปได้ ให้เขียนเป็นกฎใน agent.md แต่ละตัวด้วย


## [2026-04-25 02:08] Discovery
**บทเรียนสำคัญ — Agent รันโค้ดเองได้!**
- ก่อนหน้านี้ Anna คิดว่าแค่ Anna เท่านั้นที่รัน Python ผ่าน RUN_SHELL ได้
- **แต่จริง ๆ แล้วคุณให้สิทธิ์ทุก agent รันโค้ดเองได้** — ไม่ต้องผ่าน Anna
- ต่อไป:
  - ✅ Dispatch agent → Agent เขียน script + รันเอง
  - ✅ Anna แค่รอผล รอ report
  - ❌ ไม่ต้อง RUN_SHELL หรือ WRITE_FILE แทน agent
  - ❌ ไม่ต้องแก้โค้ดให้ agent
- **จำไว้! ถ้าต้องการให้ agent ทำงาน → dispatch ไปเลย อย่าเขียนโค้ดแทน!**


## [2026-04-25 02:13] Discovery
**สถานะล่าสุด Olist Project (จากประวัติการสนทนา):**
- ✅ Eddie ทำงาน EDA v2 เสร็จแล้ว — มี `eddie_v2_script.py` เกิดขึ้น
- ❌ Dashboard + Wiz ถูกลบออกแล้ว
- ❌ ยังไม่ได้ตรวจสอบผลงาน Eddie v2 ว่า `eddie_report_v2.md` มีหรือยัง
- ❌ ยังไม่ได้ dispatch งานต่อให้ Max/Finn/Mo


## [2026-04-25 02:13] Discovery
**กฎการบันทึกสถานะงาน:**
- ✅ เมื่องานเสร็จ → จดใน **logs/*_raw.md** เท่านั้น
- ❌ อย่าจดใน KB ว่า "agent ทำงานเสร็จ"
- KB ใช้สำหรับ: บทเรียน, ข้อควรระวัง, best practices, การเปลี่ยนแปลงระบบเท่านั้น
- log ใช้สำหรับ: สถานะการทำงานจริง, timeline, output


## [2026-04-25 07:41] Discovery
ผู้ใช้ให้สิทธิ์ Anna หา dataset รูปภาพและโหลดเองได้ โดยไม่ต้องถาม Scout และไม่ต้องถามผู้ใช้ก่อน — ทำงานได้อิสระเต็มที่


## [2026-04-25 07:43] Discovery
Research: best Thai monospace fonts for terminal windows 2024-2025
Findings:
This is a highly specific and important question for developers, sysadmins, and writers working with the Thai language. A good monospace font for a terminal must balance **readability** (clear distinction between similar Thai characters like ก/ถ/ภ, and vowel positions), **coverage** (correct Unicode rendering of complex Thai clusters), and **aesthetics** (low eye strain during long sessions).

As of late 2024 / early 2025, here are the **best** Thai monospace fonts for terminal windows, ranked by suitability. I have excluded fonts that only look good in a word processor but break in a terminal (e.g., some versions of Noto Sans Thai UI).

### The Tier List

| Rank | Font Name | Type | Best For | Key Features | Potential Downsides |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **JetBrains Mono** | Sans-serif | Daily coding, modern IDEs | Best ligatures, excellent contrast between ๅ/ฃ, very sharp | Slightly "techy" look; may lack niche Thai characters in older versions |
| **2** | **IBM Plex Mono** | Sans-serif | Professional work, journalism | Very legible, good vowel spacing, open-source, official Thai glyphs | No programming ligatures; slightly wider than average |
| **3** | **Fira Code** | Sans-serif | Heavy script users (Node/JS) | Great ligatures, balanced Thai glyphs, good for long hours | The Thai glyphs are slightly condensed; can feel cramped at small sizes |
| **4** | **Source Code Pro** | Sans-serif | Cross-platform compatibility | Very mature, excellent hinting for Windows, clean Thai rendering | Boring aesthetic; no ligatures; some users find vowels too low |
| **5** | **Sarasa Term / Sarasa Gothic** | Sans-serif | CJK & Thai mix, minimalists | Fixes Chinese/Japanese & Thai perfectly, crisp, very few | Heavy file size; not a standard system font; less community support |

---

### Detailed Analysis

#### 1. 🏆 JetBrains Mono (Best Overall 2024-2025)
- **Why it wins:** JetBrains (the company behind IntelliJ IDEA) took the time to create a monospace font with exceptional Thai support. The design makes tricky pairs like **ถ (tho thung)** and **ภ (pho samphao)** visually distinct, which most other fonts fail at.
- **Terminal Usage:** Excellent in Windows Terminal, iTerm2, and Alacritty. Works immediately with ligatures (e.g., `->` becomes an arrow).
- **Version Check:** Ensure you are using **v2.304** or later, as earlier versions had a bug with the Thai currency symbol (฿).
- **Install:** [JetBrains Mono GitHub](https://github.com/JetBrains/JetBrainsMono)

#### 2. 🥈 IBM Plex Mono (Most Readable)
- **Why it ranks high:** IBM Plex Thai was designed by professional typographers. The glyphs are slightly wider, which is a boon for Thai because vowels and tone marks need horizontal space. It is the least "squished" looking.
- **Terminal Usage:** Works perfectly in all modern terminals. No ligatures, which some purists prefer.
- **Note:** The "Mono" variant is purely monospace; the "Thai" variant has proportional spacing for GUI, so download the mono one.
- **Install:** [IBM Plex GitHub](https://github.com/IBM/plex)

#### 3. 🥉 Fira Code (Best for Coders)
- **Why it’s great:** Fira Code is the standard for JavaScript/HTML/CSS developers. The Thai glyphs are well-formed, and the widespread use means you can find patches or modified versions easily.
- **Terminal Usage:** The ligatures are fantastic for `!=` `<=` `->`. However, the Thai characters are slightly condensed (narrower than IBM Plex), which can make reading long Thai code comments tiring.
- **Improvement:** The latest Fira Code versions (v6.0+) significantly improved Thai vowel positioning. Older versions (v4.x) had floating vowels.
- **Install:** [Fira Code GitHub](https://github.com/tonsky/FiraCode)

#### 4. Source Code Pro (The Reliable Workhorse)
- **Why it’s recommended:** If you are on Windows or need a font that *will not break* in any terminal, pick this. It’s bundled with many OS tools and has the best hinting for Windows ClearType (meaning it doesn't look blurry).
- **Terminal Usage:** Works perfectly in CMD, PowerShell, and old-school terminals. No surprises.
- **Downside:** It's boring. The Thai glyphs lack character, and the distinction between some letters (like ป and ผ) is weaker than JetBrains Mono.
- **Update:** The "Variable" version is great for scaling. Ensure you use the "Source Code Pro" (with Thai support), not "Source Sans" (which is proportional).

#### 5. Sarasa Term (The Niche Champion)
- **Why it’s here:** If you write code in both Chinese/Japanese **and** Thai, Sarasa Term is the only font that doesn't compromise either. It is a composite font (based on Iosevka and Source Han Sans) built specifically for terminals.
- **Terminal Usage:** Extremely crisp, very thin strokes, excellent for high-DPI (Retina) screens.
- **Downside:** Installation is manual. It does not have standard ligatures like Fira Code. Some users find the thin weight hard to read on older screens.

### ❌ Fonts to Avoid (or use with caution)

| Font | Reason to Avoid |
| :--- | :--- |
| **Menlo / Monaco** | Built for macOS; Thai characters are poorly hinted at 12pt; vowels collide with adjacent lines. |
| **Consolas** | Default Windows font; Thai support was a late addition; "ง" (ngo ngu) looks like "O" and vowels are extremely tight. |
| **Input Mono** | High-quality font, but its Thai glyphs are essentially placeholder versions; not designed for Thai readability. |
| **Noto Sans Mono** | While comprehensive, the Thai glyphs are ugly in a terminal—too round, low contrast, and look "cartoonish" at small sizes. |
| **Anonymous Pro** | Niche font; Thai characters are too tall, causing line-height issues in many terminals. |

### Critical Configuration Tips (2024-2025)

1.  **Line Height (Leading):** Thai script requires more vertical space than Latin. In your terminal settings (e.g., Windows Terminal `profiles.json` or iTerm2 `Profiles > Text`), set **line height to 1.0** (default) and disable "Use thin strokes" / "Antialiased". If using JetBrains Mono, set line-height to **1.2** to prevent character clipping.

2.  **Font Size:** Thai at **12pt** is the minimum for comfortable reading. At 11pt, vowels like "ไ" and "ใ" become indistinguishable in most fonts. 13-14pt is ideal for 1440p screens.

3.  **Ligatures:** If you use JetBrains Mono or Fira Code, ensure your terminal supports ligatures:
    - **Windows Terminal:** Works natively.
    - **iTerm2:** Enable in `Profiles > Text > Use Ligatures`.
    - **Alacritty:** Works natively with `draw_bold_text_with_bright_colors: false`.

4.  **Nerd Fonts (Icons):** If you want patched versions for Powerlevel10k or Starship prompts, use **JetBrains Mono Nerd Font** (available on GitHub). IBM Plex Thai does not have a widespread Nerd Font patch. FiraCode Nerd Font exists but the patch sometimes breaks Thai glyph metrics. **Prefer JetBrains Mono Nerd Font** for this use case.

### Final Verdict (Late 2024 / Early 2025)

- **If you want one font for everything:** **JetBrains Mono** (download v2.304+).
- **If you value maximum readability above all else:** **IBM Plex Mono**.
- **If you write a lot of JavaScript/Python:** **Fira Code**.
- **If you are stuck on Windows without admin rights:** **Source Code Pro** (comes with Visual Studio).
- **If you also code in Chinese/Japanese:** **Sarasa Term** (specifically the `sarasa-term-sc-*` variant).


## [2026-04-25 10:19] Discovery
เพิ่มกฎสำคัญ: Anna ห้ามทำงานแทน agent ทุกกรณี ยกเว้นผู้ใช้สั่งเองโดยตรง — Anna ทำได้แค่ DISPATCH, แก้ config/log/KB, สรุปผล, ตัดสินใจ, ควบคุม pipeline เท่านั้น


## [2026-04-25 11:08] Discovery
กฎสำคัญ: Anna ไม่มีหน้าที่รัน Python script หรือทำงานแทน agent โดยตรง หน้าที่ของ Anna คือ Dispatch งานให้ agent ที่เหมาะสมและควบคุม pipeline เท่านั้น — ห้ามใช้ RUN_PYTHON หรือ RUN_SHELL เพื่อทำงานที่อยู่ในขอบเขตของ agent


## [2026-04-25 11:16] Discovery
กฎการทำงานของ Anna (อัพเดตล่าสุด):
1. ✅ Anna ทำงานของตัวเองเท่านั้น คือการ Orchestrate, Dispatch, Monitor, สรุปผล
2. ❌ ห้ามทำงานของ agent อื่นเด็ดขาด (Dana, Eddie, Max, Finn, Mo, Iris, Vera, Quinn, Rex, Scout)
3. ❌ ห้ามแก้ไข script / code / output ของ agent อื่น
4. ❌ ห้ามแก้ปัญหาที่อยู่ใน domain ของ agent อื่น
5. ✅ ถ้ามีวิธีแก้ใน KB ของ agent นั้นอยู่แล้ว → ส่ง KB ให้ agent นั้นไปใช้เอง
6. ✅ ถ้าไม่มีใน KB → ใช้ ASK_CLAUDE หรือ RESEARCH เพื่อหาวิธี แล้วอัพเดตเข้า KB ของ agent นั้น
7. 🛑 Anna ทำได้แค่:
   - อ่านไฟล์ output ของ agent (เพื่อ monitor)
   - อ่าน agent.md (เพื่อรู้ความสามารถของแต่ละคน)
   - แก้ไขได้เฉพาะ:
     - agents/anna.md (ตัวเอง)
     - project folder/logs (log กับ structure)
     - knowledge_base/ (KB ทั้งหมด)
   - dispatch งานไปยัง agent ที่ถูกต้อง


## [2026-04-25 11:22] Discovery
กฎการทำงานของ Agent ทุกคน:
- ✅ ทุกครั้งที่ agent ทำงาน ต้องสร้าง **report** ทุกครั้ง — ทั้ง success และ fail
- ✅ Report ต้องบันทึกใน `output/{agent}/` ภายใน project folder
- ✅ Report ต้องมีนามสกุล `.md` (markdown)
- ✅ ถ้างาน fail ก็ต้อง report ว่าทำอะไร, เจอปัญหาอะไร, ติดตรงไหน
- ❌ ห้ามจบโดยไม่มี report — Anna จะตรวจสอบทุกครั้งก่อน dispatch ต่อ


## [2026-04-25 14:04] [DISCOVERY]
Research: Olist Brazilian E-Commerce dataset benchmark — standard data cleaning practices by Kaggle Grandmaste
Key finding: This is an excellent and highly specific question that gets to the heart of practical data science with real-world, messy data. The Olist dataset is a benchmark, but it's not clean. The numbers you've cited (88% missing `review_comment_title`, 610 missing `product_category`, ~3% missing delivery dates) are classic pain points.


## [2026-04-25 16:13] [DISCOVERY]
Research: Kaggle Olist Brazilian E-Commerce dataset — data cleaning standards and benchmark scores by Grandmas
Key finding: Here is a thorough breakdown of how Kaggle Grandmasters handle missing values and measure data quality in the context of the **Olist Brazilian E-Commerce** dataset, based on established kernels, competition solutions, and best practices from top-tier notebooks.
