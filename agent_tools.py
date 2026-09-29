# -*- coding: utf-8 -*-
# Multi-tool AI Agent مع Ollama — يقرر بنفسه أي أداة يستخدم.
# الأدوات: CALCULATOR, SEARCH_FILE, READ_DATA.
import os
import difflib
import requests
import pandas as pd

MODEL = "llama3.2:3b"
CHAT_URL = "http://localhost:11434/api/chat"
BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# الجزء 2 — أداة الحاسبة
# ============================================================

# تنفّذ عملية حسابية نصية وتعيد الناتج، أو رسالة خطأ واضحة عند الفشل.
def calculator_tool(expression):
    allowed = "0123456789+-*/(). %"
    expr = "".join(ch for ch in expression if ch in allowed).strip()
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # مدخل من النموذج، منظّف مسبقًا
        return str(result)
    except Exception as e:
        return f"خطأ بالعملية الحسابية: {e}"


# ============================================================
# الجزء 3 — أداة البحث في ملف نصي
# ============================================================

# تبحث عن كلمة مفتاحية في company_info.txt وتعيد الأسطر المطابقة (غير حساسة لحالة الأحرف).
def search_file_tool(keyword):
    path = os.path.join(BASE, "company_info.txt")
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "الملف غير موجود."
    key = keyword.strip().lower()
    matches = [ln.strip() for ln in lines if key in ln.lower()]
    return "\n".join(matches) if matches else "ما لقيت معلومة."


# ============================================================
# الجزء 4 — أداة قراءة بيانات المنتجات (CSV)
# ============================================================

# تبحث عن منتج (بحث تقريبي) في products.csv وتعيد سعره وكميته، أو رسالة عدم وجود.
def read_data_tool(product_name):
    path = os.path.join(BASE, "products.csv")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return "ملف المنتجات غير موجود."
    query = product_name.strip().lower()
    names = df["product_name"].astype(str).tolist()
    # مطابقة تقريبية: احتواء جزئي في أي اتجاه، ثم أقرب تطابق نصي
    idx = None
    for i, name in enumerate(names):
        nl = name.lower()
        if query in nl or nl in query:
            idx = i
            break
    if idx is None:
        close = difflib.get_close_matches(query, [n.lower() for n in names], n=1, cutoff=0.5)
        if close:
            idx = [n.lower() for n in names].index(close[0])
    if idx is None:
        return f"المنتج '{product_name}' غير موجود."
    row = df.iloc[idx]
    return f"المنتج: {row['product_name']} | السعر: {row['price']} | الكمية المتوفرة: {row['quantity_in_stock']}"


# ============================================================
# الجزء 5 — الـ Agent الكامل (Router)
# ============================================================

SYSTEM_PROMPT = (
    "أنت مساعد ذكي لديك ثلاث أدوات. اقرأ سؤال المستخدم واختر الأداة المناسبة، "
    "ثم اكتب سطرًا واحدًا فقط بالصيغة (اسم الأداة ثم نقطتان ثم المدخل):\n"
    "- CALCULATOR: <تعبير حسابي>  لأي عملية حسابية. مثال: CALCULATOR: 340 * 12\n"
    "- SEARCH_FILE: <كلمة مفتاحية>  للبحث عن معلومة عن الشركة (التأسيس، الموظفين، الخدمات، المقر). مثال: SEARCH_FILE: المقر\n"
    "- READ_DATA: <اسم المنتج>  لمعرفة سعر أو كمية منتج. مثال: READ_DATA: لابتوب\n"
    "اكتب سطر الأداة فقط بدون أي شرح، ثم توقّف."
)

TOOLS = {
    "CALCULATOR": calculator_tool,
    "SEARCH_FILE": search_file_tool,
    "READ_DATA": read_data_tool,
}


# يرسل رسائل للنموذج ويعيد ردًّا من سطر واحد (لاختيار الأداة).
def _tool_line(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    r = requests.post(CHAT_URL, json={"model": MODEL, "messages": messages, "stream": False,
                                      "options": {"temperature": 0, "stop": ["\n"]}}, timeout=180)
    return r.json()["message"]["content"]


# يطلب من النموذج صياغة جواب نهائي بلغة طبيعية بناءً على نتيجة الأداة.
def _compose(question, tool, observation):
    prompt = (
        f"سؤال المستخدم: {question}\n"
        f"نتيجة الأداة ({tool}): {observation}\n"
        "صُغ جوابًا نهائيًا واضحًا بالعربية بجملة واحدة، بلا مقدمات."
    )
    r = requests.post(CHAT_URL, json={"model": MODEL,
                                      "messages": [{"role": "user", "content": prompt}],
                                      "stream": False, "options": {"temperature": 0}}, timeout=180)
    return r.json()["message"]["content"].strip()


# يحدّد الأداة ومدخلها من سطر رد النموذج.
def _parse_tool(reply):
    for name in TOOLS:
        idx = reply.find(name)
        if idx != -1:
            rest = reply[idx + len(name):].lstrip(": ").strip()
            tool_input = rest.splitlines()[0] if rest else ""
            return name, tool_input
    return None, None


# الدالة الكاملة: توجّه السؤال للأداة المناسبة وتعيد جوابًا نهائيًا.
def route_and_respond(user_question, verbose=True):
    reply = _tool_line(user_question)
    tool, tool_input = _parse_tool(reply)
    if tool is None:
        if verbose:
            print(f"  [الـ Agent] لم يطلب أداة، رد مباشر: {reply.strip()}")
        return {"tool": None, "input": None, "observation": None, "answer": reply.strip()}
    observation = TOOLS[tool](tool_input)
    if verbose:
        print(f"  [الأداة المختارة] {tool}: {tool_input}")
        print(f"  [نتيجة الأداة] {observation}")
    answer = _compose(user_question, tool, observation)
    return {"tool": tool, "input": tool_input, "observation": observation, "answer": answer}


# ============================================================
# التشغيل والاختبارات
# ============================================================

if __name__ == "__main__":
    print("########## الجزء 2: اختبار الحاسبة ##########")
    for expr in ["340 * 12", "1500 / 7", "10 / 0"]:
        print(f"  calculator_tool('{expr}') = {calculator_tool(expr)}")

    print("\n########## الجزء 3: اختبار البحث بالملف ##########")
    for kw in ["التأسيس", "الموظفين", "الخدمات"]:
        print(f"  search_file_tool('{kw}') -> {search_file_tool(kw)}")

    print("\n########## الجزء 4: اختبار قراءة المنتجات ##########")
    for p in ["لابتوب", "ماوس", "شاشة", "تلفزيون"]:
        print(f"  read_data_tool('{p}') -> {read_data_tool(p)}")

    print("\n########## الجزء 5: الـ Agent الكامل (6 أسئلة) ##########")
    questions = [
        "كم ناتج 340 ضرب 12؟",
        "لو وزعت 1500 على 7 كم لكل واحد؟",
        "وين مقر الشركة؟",
        "كم عدد موظفي الشركة؟",
        "كم سعر الماوس؟",
        "كم القطع المتوفرة من اللابتوب؟",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\nسؤال {i}: {q}")
        result = route_and_respond(q)
        print(f"  >> الجواب النهائي: {result['answer']}")
