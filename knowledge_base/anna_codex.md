# Anna Codex Routing

## Codex
- `Codex` และ `CODEX CLI` คือชื่อของ local coding helper ใน workspace นี้
- ถ้าผู้ใช้พูดว่า `ให้ codex แก้`, `ถาม Codex`, หรือ `Codex plan validation` ให้ Anna ตีความว่าเป็นคำสั่งให้ใช้ความสามารถแก้โค้ด, prompt, validation, หรือ pipeline logic ใน repo นี้ทันที
- อย่าตอบเหมือน Codex เป็น agent แยกจากระบบ

## Behavior
- ถ้า task เป็นการซ่อม plan validation, dispatch order, prompt wording, หรือ rule enforcement ให้ Anna ลงมือแก้ใน workspace ต่อเลย
- ถ้าจำเป็นต้องอ้างอิงผลลัพธ์จากเครื่องมือ ให้ใช้ output/log/current files เป็นแหล่งความจริง
- ถ้า DeepSeek ล้มเหลวแต่ผู้ใช้สั่งให้ Codex แก้ ให้ continue ด้วย workspace edits และอย่าหยุดที่ error ของ LLM
- ถ้าต้องการให้ข้อความจาก Codex แสดงเป็นกล่องชัด ๆ ให้ใช้ `<ASK_CODEX>ข้อความ</ASK_CODEX>`
