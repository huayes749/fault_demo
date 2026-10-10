import streamlit as st
import sqlite3
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================【内置设备说明书知识库，系统自带，永久存在】======================
default_manual_data = [
    {
        "fault_text": "电机过热",
        "solution": "1.停机断电；2.检查散热风扇；3.清理灰尘；4.检测负载是否超标",
        "source": "设备说明书"
    },
    {
        "fault_text": "传感器无信号",
        "solution": "1.检查线路接头是否松动；2.更换传感器供电；3.校验传感器标定",
        "source": "设备说明书"
    },
    {
        "fault_text": "泵体异响",
        "solution": "1.停机检查轴承；2.检查润滑油液位；3.排查管路气蚀问题",
        "source": "设备说明书"
    },
    {
        "fault_text": "电机无法启动",
        "solution": "1.检查供电电源；2.查看接触器状态；3.检测电机绕组绝缘",
        "source": "设备说明书"
    },
    {
        "fault_text": "管道压力异常",
        "solution": "1.检查阀门开闭状态；2.排查管道堵塞；3.校验压力变送器",
        "source": "设备说明书"
    },
    {
        "fault_text": "风机振动过大",
        "solution": "1.停机检查地脚螺栓；2.检查叶轮积灰；3.检测轴承磨损情况",
        "source": "设备说明书"
    }
]

# ---------------------- 数据库：存放用户后期手动新增故障案例 ----------------------
def init_db():
    conn = sqlite3.connect("fault_db.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS fault_records
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         fault_text TEXT,
         solution TEXT,
         source TEXT)
    ''')
    conn.commit()
    conn.close()

def get_db_faults():
    conn = sqlite3.connect("fault_db.db")
    c = conn.cursor()
    c.execute("SELECT fault_text, solution, source FROM fault_records")
    res = c.fetchall()
    conn.close()
    list_db = []
    for ft, sol, src in res:
        list_db.append({"fault_text": ft, "solution": sol, "source": src})
    return list_db

def add_new_fault(fault_content, sol_content, src_type):
    conn = sqlite3.connect("fault_db.db")
    c = conn.cursor()
    c.execute("INSERT INTO fault_records (fault_text, solution, source) VALUES (?, ?, ?)",
              (fault_content, sol_content, src_type))
    conn.commit()
    conn.close()

# ---------------------- 检索：TF-IDF 向量化 + 余弦相似度 ----------------------
def search_fault(keyword, threshold=0.0, top_k=5):
    db_data = get_db_faults()
    all_data = default_manual_data + db_data

    if not all_data:
        return []

    all_fault_texts = [item["fault_text"] for item in all_data]

    tokenized_docs = [" ".join(jieba.lcut(text)) for text in all_fault_texts]
    tokenized_query = " ".join(jieba.lcut(keyword))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs + [tokenized_query])

    user_vec = tfidf_matrix[-1]
    sim_matrix = cosine_similarity(tfidf_matrix[:-1], user_vec)

    tfidf_result = []
    for idx, score in enumerate(sim_matrix):
        score_val = float(score[0])
        if score_val >= threshold:
            tfidf_result.append((all_data[idx], score_val))

    tfidf_result.sort(key=lambda x: x[1], reverse=True)
    return tfidf_result[:top_k]

# ---------------------- 网页页面布局 ----------------------
st.set_page_config(page_title="工业故障检索系统", layout="wide")
st.title("工业故障案例&说明书检索系统")
st.divider()

init_db()

col_left, col_right = st.columns(2)

with col_left:
    st.header("查询故障解决方案（TF-IDF 智能检索）")
    input_fault = st.text_input("请输入故障现象描述：", placeholder="例如：电机过热 / 泵噪音大")
    search_btn = st.button("开始检索")

    if search_btn:
        if input_fault.strip() == "":
            st.warning("请输入故障描述！")
        else:
            res_list = search_fault(input_fault)
            if not res_list:
                st.info("未找到匹配故障，你可以在右侧录入新故障案例")
            else:
                st.success(f"找到 {len(res_list)} 条匹配记录")
                for item, score in res_list:
                    st.subheader(f"故障：{item['fault_text']}")
                    st.write(f"来源：{item['source']}")
                    st.write(f"匹配相似度：{score:.4f}")
                    st.write(f"解决方法：{item['solution']}")
                    st.divider()

with col_right:
    st.header("新增故障案例入库（后期补充）")
    st.info("⚠️ 设备基础说明书内容已经内置，无需在这里录入")
    new_fault = st.text_input("故障现象：")
    new_solution = st.text_area("处理解决方案：")
    source_select = st.selectbox("来源类型", ["设备说明书", "历史故障案例"])
    save_btn = st.button("保存记录到数据库")

    if save_btn:
        if new_fault.strip() and new_solution.strip():
            add_new_fault(new_fault, new_solution, source_select)
            st.success("✅ 保存成功！可在左侧检索该故障")
        else:
            st.error("故障描述和解决方案不能为空！")
