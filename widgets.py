"""
拉桿 + 數字輸入框 雙向同步元件。

Streamlit 沒有原生「可輸入數字的拉桿」，這裡用同一個 canonical key
搭配拉桿/輸入框各自的 widget key 做雙向同步。這支檔案直接操作
st.session_state，跟 sidebar 版面高度耦合，所以獨立成 widgets.py
而不是跟 utils.py 放在一起（utils.py 只放純函式）。
"""

import streamlit as st


def synced_slider(label, min_value, max_value, key, step=1, fmt=None,
                   unit="", on_user_change=None, help_text=None):
    if key not in st.session_state:
        st.session_state[key] = min_value

    slider_key = f"{key}__slider"
    input_key = f"{key}__input"

    def _from_slider():
        v = st.session_state[slider_key]
        st.session_state[key] = v
        # 【修正 bug】反向同步：拉桿變動時，也要把數字寫回 input_key，
        # 否則畫面上的數字輸入框會停在舊值，跟拉桿顯示的位置對不上
        # （跟下面 _from_input() 的問題是同一個成因，只是方向相反）。
        st.session_state[input_key] = v
        if on_user_change:
            on_user_change()

    def _from_input():
        v = st.session_state[input_key]
        v = min(max(v, min_value), max_value)
        st.session_state[key] = v
        # 【修正 bug】原本這裡只更新了 canonical key（st.session_state[key]），
        # 沒有寫回 slider 自己的 widget key（slider_key）。Streamlit 的 widget
        # 只要帶了 key=，重新渲染時會優先讀 session_state 裡該 key 的既有值，
        # 而不是這次傳入的 value= 參數——所以在數字框輸入新值後，拉桿的視覺
        # 位置會停在原地不動，即使 canonical 值其實已經正確更新。這裡額外寫回
        # slider_key（若使用者輸入超出範圍被 clamp，也一併把 input_key 修正成
        # clamp 後的值，避免數字框顯示的數字跟實際生效的值不一致）。
        st.session_state[slider_key] = v
        st.session_state[input_key] = v
        if on_user_change:
            on_user_change()

    display_label = f"{label}（{unit}）" if unit else label

    # 滑桿本身就帶標籤，數字框收窄並靠右對齊，減少元件堆疊感
    col_s, col_n = st.columns([4, 1.1], gap="small")
    with col_s:
        st.slider(
            display_label, min_value, max_value,
            value=st.session_state[key], step=step,
            key=slider_key, on_change=_from_slider,
            help=help_text,
        )
    with col_n:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        st.number_input(
            display_label, min_value, max_value,
            value=st.session_state[key], step=step,
            key=input_key, on_change=_from_input,
            label_visibility="collapsed", format=fmt,
        )

    return st.session_state[key]